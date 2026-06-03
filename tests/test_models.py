"""Tests for models.py — provider routing, JSON parsing, and retry logic."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import models


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

def test_registry_has_expected_keys():
    assert set(models.MODEL_REGISTRY) == {"gpt-4.1", "llama-4-maverick"}


def test_call_model_rejects_unknown_key():
    with pytest.raises(ValueError):
        models.call_model("nope", "sys", "user", "questions")


def test_call_model_rejects_unknown_schema():
    with pytest.raises(ValueError):
        models.call_model("gpt-4.1", "sys", "user", "nope")


# ---------------------------------------------------------------------------
# Retry helper — covers the spec's "exponential backoff, 3 retries" requirement
# ---------------------------------------------------------------------------

class _Transient(Exception):
    """Stand-in for openai.RateLimitError / botocore throttling."""
    pass


def _is_transient(exc):
    return isinstance(exc, _Transient)


def test_with_retries_succeeds_after_two_transient_failures():
    # Two failures then a success: assert the sleep sequence is 1s, 2s.
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Transient("rate limited")
        return "ok"

    sleeps: list[float] = []
    result = models._with_retries(fn, _is_transient, sleep=sleeps.append)
    assert result == "ok"
    assert calls["n"] == 3
    assert sleeps == [1, 2]


def test_with_retries_gives_up_after_max_retries():
    # 4 attempts total (1 + 3 retries) then re-raise.
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _Transient("still failing")

    sleeps: list[float] = []
    with pytest.raises(_Transient):
        models._with_retries(fn, _is_transient, sleep=sleeps.append)
    assert calls["n"] == models.MAX_RETRIES + 1
    assert sleeps == [1, 2, 4]


def test_with_retries_does_not_retry_non_transient():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("programmer error")

    sleeps: list[float] = []
    with pytest.raises(ValueError):
        models._with_retries(fn, _is_transient, sleep=sleeps.append)
    assert calls["n"] == 1  # no retries for non-transient errors
    assert sleeps == []


# ---------------------------------------------------------------------------
# OpenAI provider — mocked client, JSON parsing
# ---------------------------------------------------------------------------

def _mock_openai_response(content: str):
    """Build a fake OpenAI response mimicking chat.completions.create()."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_call_openai_parses_structured_output(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_openai_response(
        json.dumps({"questions": ["q1", "q2"]})
    )
    monkeypatch.setattr(models, "_get_openai_client", lambda: fake_client)

    result = models.call_model("gpt-4.1", "sys", "user", "questions")

    assert result == {"questions": ["q1", "q2"]}
    # Check the OpenAI call was wired correctly.
    args, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4.1-2025-04-14"
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "questions_list"


# ---------------------------------------------------------------------------
# Bedrock provider — mocked Converse client, JSON parsing & code-fence stripping
# ---------------------------------------------------------------------------

def _bedrock_converse_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


def test_call_bedrock_parses_clean_json(monkeypatch):
    fake_client = MagicMock()
    fake_client.converse.return_value = _bedrock_converse_response(
        json.dumps({"passage": "An excerpt from the text."})
    )
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: fake_client)

    result = models.call_model("llama-4-maverick", "sys", "user", "passage")

    assert result == {"passage": "An excerpt from the text."}
    # Bedrock Converse arguments wired correctly.
    args, kwargs = fake_client.converse.call_args
    assert kwargs["modelId"] == "us.meta.llama4-maverick-17b-instruct-v1:0"
    assert kwargs["inferenceConfig"]["temperature"] == 1.0
    assert kwargs["inferenceConfig"]["topP"] == 1.0


def test_call_bedrock_recovers_from_markdown_fences(monkeypatch):
    # Some Bedrock models wrap JSON in ```json ... ``` fences. The first parse
    # falls back to _extract_json's fence-stripper.
    fenced = '```json\n{"summary": "A short summary."}\n```'
    fake_client = MagicMock()
    fake_client.converse.return_value = _bedrock_converse_response(fenced)
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: fake_client)

    result = models.call_model("llama-4-maverick", "sys", "user", "summary")

    assert result == {"summary": "A short summary."}
    # Only one call needed — the fence stripper handles it.
    assert fake_client.converse.call_count == 1


def test_call_bedrock_retries_with_stricter_prompt_on_bad_json(monkeypatch):
    # First response is unparseable garbage; second response (stricter) is valid.
    fake_client = MagicMock()
    fake_client.converse.side_effect = [
        _bedrock_converse_response("totally not json at all"),
        _bedrock_converse_response(json.dumps({"summary": "ok"})),
    ]
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: fake_client)

    result = models.call_model("llama-4-maverick", "sys", "user", "summary")

    assert result == {"summary": "ok"}
    assert fake_client.converse.call_count == 2
    # The second call should include the stricter "ONLY the JSON object" hint.
    second_kwargs = fake_client.converse.call_args_list[1][1]
    stricter_text = second_kwargs["system"][0]["text"]
    assert "ONLY the JSON object" in stricter_text


def test_call_bedrock_wraps_bare_string_for_passage(monkeypatch):
    """LLaMA sometimes returns a bare JSON string instead of {"passage": "..."}."""
    fake_client = MagicMock()
    fake_client.converse.return_value = _bedrock_converse_response(
        json.dumps("just the passage text without the wrapper")
    )
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: fake_client)

    result = models.call_model("llama-4-maverick", "sys", "user", "passage")

    assert result == {"passage": "just the passage text without the wrapper"}
    # One call: shape coercion shouldn't trigger a retry.
    assert fake_client.converse.call_count == 1


def test_call_bedrock_wraps_bare_list_for_questions(monkeypatch):
    """A bare list also gets wrapped to match the questions schema."""
    fake_client = MagicMock()
    fake_client.converse.return_value = _bedrock_converse_response(
        json.dumps(["q1", "q2", "q3"])
    )
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: fake_client)

    result = models.call_model("llama-4-maverick", "sys", "user", "questions")

    assert result == {"questions": ["q1", "q2", "q3"]}


def test_call_bedrock_accepts_control_chars_in_strings(monkeypatch):
    """LLaMA emits literal newlines inside string values; strict=False handles that."""
    # Raw text with a literal newline inside the string value — invalid in strict JSON.
    raw = '{"summary": "line one\nline two"}'
    fake_client = MagicMock()
    fake_client.converse.return_value = _bedrock_converse_response(raw)
    monkeypatch.setattr(models, "_get_bedrock_client", lambda: fake_client)

    result = models.call_model("llama-4-maverick", "sys", "user", "summary")

    assert result == {"summary": "line one\nline two"}


# ---------------------------------------------------------------------------
# JSON schema shapes
# ---------------------------------------------------------------------------

def test_json_schemas_have_expected_root_fields():
    assert "questions" in models.JSON_SCHEMAS["questions"]["schema"]["properties"]
    assert "requirements" in models.JSON_SCHEMAS["requirements"]["schema"]["properties"]
    assert "passage" in models.JSON_SCHEMAS["passage"]["schema"]["properties"]
    assert "summary" in models.JSON_SCHEMAS["summary"]["schema"]["properties"]
