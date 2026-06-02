"""End-to-end tests for pipeline.py with a mocked model client."""

import json
import re

import pytest

import io_utils
import pipeline


# ---------------------------------------------------------------------------
# Fixtures: tiny fake corpus and a deterministic mock call_model
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    """Two-text corpus on disk; meta_rows and a temp_dir for caches.

    The mock caller below returns the literal string "selected passage by
    <model_key>" for passage requests. The fixture files contain that exact
    phrase so the pipeline's verbatim check (substring against the source)
    passes without triggering a re-prompt.
    """
    plaintext = tmp_path / "plaintext"
    plaintext.mkdir()
    (plaintext / "01_a.txt").write_text(
        "This is text A. Contains selected passage by gpt-4.1 and "
        "selected passage by llama-4-maverick.",
        encoding="utf-8",
    )
    (plaintext / "02_b.txt").write_text(
        "This is text B. Contains selected passage by gpt-4.1 and "
        "selected passage by llama-4-maverick.",
        encoding="utf-8",
    )

    # Point io_utils.load_text at our fixture directory.
    monkeypatch.setattr(io_utils, "PLAINTEXT_DIR", plaintext)

    meta_rows = [
        {"TEXT_ID": "01", "AUTHOR": "Author A", "TITLE": "Book A",
         "GENRE": "novel", "FILENAME": "01_a.txt"},
        {"TEXT_ID": "02", "AUTHOR": "Author B", "TITLE": "Book B",
         "GENRE": "poetry", "FILENAME": "02_b.txt"},
    ]
    temp_dir = tmp_path / "temp"
    return meta_rows, temp_dir


_LIST_COUNT_RE = re.compile(r"list of (\d+) (high-level|specific)")


def _make_mock_caller():
    """Return a call_model stand-in that returns canned JSON by schema name.

    For list responses we honor the N requested in the prompt so tests can
    drive different selection/scene/global counts.
    """

    def fake_call(model_key, system, user, schema_name):
        if schema_name in {"questions", "requirements"}:
            m = _LIST_COUNT_RE.search(user)
            assert m, f"could not find list count in prompt: {user[:200]}"
            n = int(m.group(1))
            label = "q" if schema_name == "questions" else "req"
            items = [f"{label}{i} for {model_key}" for i in range(1, n + 1)]
            return {schema_name: items}
        if schema_name == "passage":
            return {"passage": f"selected passage by {model_key}"}
        if schema_name == "summary":
            return {"summary": f"summary by {model_key}"}
        raise ValueError(schema_name)

    return fake_call


# ---------------------------------------------------------------------------
# Stage 4a
# ---------------------------------------------------------------------------

def test_passage_selection_record_count_and_ids(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=2,
        call_model=_make_mock_caller(),
        temp_dir=temp_dir,
    )

    # 2 texts × 2 questions = 4 passages.
    assert len(passages) == 4
    passage_ids = [p["passage_id"] for p in passages]
    assert passage_ids == [
        "p_01_gpt41_01", "p_01_gpt41_02",
        "p_02_gpt41_01", "p_02_gpt41_02",
    ]
    # JSONL cache mirrors the in-memory list.
    cache = io_utils.read_jsonl(temp_dir / "gpt-4.1_passages.jsonl")
    assert len(cache) == 4
    assert cache[0]["model"] == "gpt-4.1"
    assert cache[0]["question"].startswith("q1")
    assert cache[0]["passage_text"].startswith("selected passage")


def test_passage_selection_with_multiple_models(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1", "llama-4-maverick"],
        selection_n=2,
        call_model=_make_mock_caller(),
        temp_dir=temp_dir,
    )
    # 2 models × 2 texts × 2 questions = 8 passages.
    assert len(passages) == 8
    # Each model writes its own cache file.
    assert (temp_dir / "gpt-4.1_passages.jsonl").exists()
    assert (temp_dir / "llama-4-maverick_passages.jsonl").exists()
    # Same text + counter but different model -> different passage_id.
    p01_gpt = next(p["passage_id"] for p in passages
                   if p["model"] == "gpt-4.1" and p["text_id"] == "01"
                   and p["passage_n"] == 1)
    p01_llama = next(p["passage_id"] for p in passages
                     if p["model"] == "llama-4-maverick" and p["text_id"] == "01"
                     and p["passage_n"] == 1)
    assert p01_gpt == "p_01_gpt41_01"
    assert p01_llama == "p_01_llama4m_01"


# ---------------------------------------------------------------------------
# Stage 4b and 4c
# ---------------------------------------------------------------------------

def test_scene_summaries_counts_and_id_infix(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=2,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    scene = pipeline.run_scene_summaries(
        passages=passages,
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        scene_n=2,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    # 4 passages × 2 requirements = 8 scene summaries.
    assert len(scene) == 8
    # IDs should carry the "scene" infix and the model short tag.
    assert all("scene" in s["summary_id"] for s in scene)
    assert all("gpt41" in s["summary_id"] for s in scene)
    # IDs should be unique and well-formed.
    assert len({s["summary_id"] for s in scene}) == 8
    # First few IDs match the expected pattern.
    expected_first = [
        "s_01_01_gpt41_scene_01", "s_01_01_gpt41_scene_02",
        "s_01_02_gpt41_scene_01", "s_01_02_gpt41_scene_02",
    ]
    actual_first = [s["summary_id"] for s in scene[:4]]
    assert actual_first == expected_first


def test_global_summaries_use_global_infix(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    global_summaries = pipeline.run_global_summaries(
        passages=passages,
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        global_n=2,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    # 1 model × 2 texts × 1 passage × 2 requirements = 4 global summaries.
    assert len(global_summaries) == 4
    assert all("global" in s["summary_id"] for s in global_summaries)


def test_summary_stage_only_summarizes_own_model_passages(fake_corpus):
    """A model summarizes only passages it selected itself."""
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1", "llama-4-maverick"],
        selection_n=1,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    # Run scene summaries with only one of the two models. The other model's
    # passages should be skipped.
    scene = pipeline.run_scene_summaries(
        passages=passages,
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        scene_n=1,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    # 2 texts × 1 passage × 1 requirement = 2 scene summaries, all from gpt-4.1.
    assert len(scene) == 2
    assert all(s["model"] == "gpt-4.1" for s in scene)


# ---------------------------------------------------------------------------
# Record shape (so the CSV columns line up downstream)
# ---------------------------------------------------------------------------

def test_passage_record_has_expected_fields(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=_make_mock_caller(),
        temp_dir=temp_dir,
    )
    expected_fields = {
        "passage_id", "passage_n", "text_id", "title", "author",
        "model", "question", "passage_text",
    }
    assert set(passages[0].keys()) == expected_fields


def test_summary_record_has_expected_fields(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows, model_keys=["gpt-4.1"], selection_n=1,
        call_model=fake_call, temp_dir=temp_dir,
    )
    scene = pipeline.run_scene_summaries(
        passages=passages, meta_rows=meta_rows, model_keys=["gpt-4.1"],
        scene_n=1, call_model=fake_call, temp_dir=temp_dir,
    )
    expected_fields = {
        "summary_id", "passage_id", "text_id", "title", "author",
        "model", "requirement", "summary_text",
    }
    assert set(scene[0].keys()) == expected_fields


# ---------------------------------------------------------------------------
# Verbatim check on selected passages
# ---------------------------------------------------------------------------

def test_is_verbatim_excerpt_basic_match():
    src = "Hello, world. The quick brown fox jumps over the lazy dog."
    assert pipeline.is_verbatim_excerpt("the quick brown fox", src)


def test_is_verbatim_excerpt_tolerates_whitespace_and_case():
    src = "Line one.\nLine two.\nLine three."
    # Multiple spaces, capitalized, line break in source becomes space.
    assert pipeline.is_verbatim_excerpt("Line   ONE.\nLine two.", src)


def test_is_verbatim_excerpt_rejects_added_commentary():
    src = "Just the source text here."
    # Commentary that paraphrases is not a substring.
    assert not pipeline.is_verbatim_excerpt(
        "This passage exemplifies a key theme of the text.", src
    )


def test_verbatim_check_passes_through_when_response_is_substring(fake_corpus):
    """A response that is a verbatim substring should not trigger a retry."""
    meta_rows, temp_dir = fake_corpus

    # Track every call so we can assert on retry behavior.
    calls: list[tuple[str, str]] = []

    def call(model_key, system, user, schema_name):
        calls.append((schema_name, user))
        if schema_name == "questions":
            return {"questions": ["q1"]}
        if schema_name == "passage":
            # Substring of both 01_a.txt and 02_b.txt fixtures.
            return {"passage": "selected passage by gpt-4.1"}
        raise ValueError(schema_name)

    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=call,
        temp_dir=temp_dir,
    )
    # 2 texts × 1 question → 2 passages, with 1 questions call + 1 passage call
    # per text (no retries). Total schemas: ["questions","passage"] × 2.
    assert len(passages) == 2
    assert [s for s, _ in calls] == ["questions", "passage", "questions", "passage"]


def test_verbatim_check_re_prompts_on_non_substring(fake_corpus):
    """Non-substring response should trigger one retry with stricter instruction."""
    meta_rows, temp_dir = fake_corpus

    # First passage call returns commentary, second returns a clean substring.
    state = {"passage_call_count": 0}

    def call(model_key, system, user, schema_name):
        if schema_name == "questions":
            return {"questions": ["q1"]}
        if schema_name == "passage":
            state["passage_call_count"] += 1
            if state["passage_call_count"] % 2 == 1:
                # First attempt: bad (commentary).
                return {"passage": "This poem exemplifies a key theme."}
            else:
                # Second attempt: clean substring of the source.
                return {"passage": "selected passage by gpt-4.1"}
        raise ValueError(schema_name)

    # Limit to one text so we can count calls deterministically.
    one_text = [meta_rows[0]]
    passages = pipeline.run_passage_selection(
        meta_rows=one_text,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=call,
        temp_dir=temp_dir,
    )

    # One bad + one good passage call.
    assert state["passage_call_count"] == 2
    assert len(passages) == 1
    assert passages[0]["passage_text"] == "selected passage by gpt-4.1"


def test_verbatim_check_accepts_response_when_retry_also_fails(fake_corpus, capsys):
    """If the retry also fails, accept the response but warn."""
    meta_rows, temp_dir = fake_corpus

    def call(model_key, system, user, schema_name):
        if schema_name == "questions":
            return {"questions": ["q1"]}
        if schema_name == "passage":
            # Always commentary, never a substring.
            return {"passage": "Still adding commentary, not a substring."}
        raise ValueError(schema_name)

    one_text = [meta_rows[0]]
    passages = pipeline.run_passage_selection(
        meta_rows=one_text,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=call,
        temp_dir=temp_dir,
    )

    # The pipeline accepts the second response even though it failed the check.
    assert len(passages) == 1
    assert passages[0]["passage_text"] == "Still adding commentary, not a substring."
    # And it warned on stdout.
    captured = capsys.readouterr().out
    assert "re-prompting" in captured
    assert "still not verbatim" in captured
