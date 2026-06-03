"""End-to-end tests for pipeline.py with a mocked model client."""

import json
import re

import pytest

import io_utils
import pipeline


# A deterministic 150-word passage used by the fake_corpus fixture below.
# Sits comfortably inside the 100-300 word window enforced by
# pipeline._passage_failures, so the mock caller's passage responses pass
# both quality checks without triggering a re-prompt.
PASSAGE_150 = " ".join(f"w{i}" for i in range(1, 151))


# ---------------------------------------------------------------------------
# Fixtures: tiny fake corpus and a deterministic mock call_model
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_corpus(tmp_path, monkeypatch):
    """Two-text corpus on disk; meta_rows and a temp_dir for caches.

    Both source files include PASSAGE_150 verbatim, so the mock caller can
    return that string for "passage" requests and pass both the verbatim
    substring check and the 100-300 word range check.
    """
    plaintext = tmp_path / "plaintext"
    plaintext.mkdir()
    (plaintext / "01_a.txt").write_text(
        f"Front matter for text A.\n\n{PASSAGE_150}\n\nMore text A.",
        encoding="utf-8",
    )
    (plaintext / "02_b.txt").write_text(
        f"Front matter for text B.\n\n{PASSAGE_150}\n\nMore text B.",
        encoding="utf-8",
    )

    # Point io_utils.load_text at our fixture directory.
    monkeypatch.setattr(io_utils, "PLAINTEXT_DIR", plaintext)

    meta_rows = [
        {"text_id": "1", "author": "Author A", "title": "Book A",
         "genre": "novel", "filename": "01_a.txt"},
        {"text_id": "2", "author": "Author B", "title": "Book B",
         "genre": "poetry", "filename": "02_b.txt"},
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
            return {"passage": PASSAGE_150}
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

    # 2 texts × 2 requirements = 4 passages.
    assert len(passages) == 4
    # passage_id is an integer counter that resets per (text, model).
    ids_per_text = [
        (p["text_id"], p["passage_id"]) for p in passages
    ]
    assert ids_per_text == [("1", 1), ("1", 2), ("2", 1), ("2", 2)]
    # JSONL cache mirrors the in-memory list.
    cache = io_utils.read_jsonl(temp_dir / "gpt-4.1_passages.jsonl")
    assert len(cache) == 4
    assert cache[0]["model"] == "gpt-4.1"
    assert cache[0]["requirement"].startswith("q1")
    assert cache[0]["passage_text"] == PASSAGE_150


def test_passage_selection_with_multiple_models(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1", "llama-4-maverick"],
        selection_n=2,
        call_model=_make_mock_caller(),
        temp_dir=temp_dir,
    )
    # 2 models × 2 texts × 2 requirements = 8 passages.
    assert len(passages) == 8
    # Each model writes its own cache file.
    assert (temp_dir / "gpt-4.1_passages.jsonl").exists()
    assert (temp_dir / "llama-4-maverick_passages.jsonl").exists()
    # passage_id counter resets per (text, model): both models have
    # passage_id=1 for their first selection on text 01.
    p01_gpt = next(p["passage_id"] for p in passages
                   if p["model"] == "gpt-4.1" and p["text_id"] == "1"
                   and p["passage_id"] == 1)
    p01_llama = next(p["passage_id"] for p in passages
                     if p["model"] == "llama-4-maverick" and p["text_id"] == "1"
                     and p["passage_id"] == 1)
    assert p01_gpt == 1 and p01_llama == 1
    # Uniqueness comes from the (text_id, model, passage_id) triple.
    triples = {(p["text_id"], p["model"], p["passage_id"]) for p in passages}
    assert len(triples) == 8


# ---------------------------------------------------------------------------
# Stage 4b and 4c
# ---------------------------------------------------------------------------

def test_scene_summaries_counts_and_ids(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=2,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    scene = pipeline.run_summaries(
        kind="scene",
        passages=passages,
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        summary_n=2,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    # 4 passages × 2 requirements = 8 scene summaries.
    assert len(scene) == 8
    # summary_type marks every record as "scene".
    assert all(s["summary_type"] == "scene" for s in scene)
    # summary_id counter resets per (text, model, passage_id, summary_type).
    # First passage of text 01 has summary_ids [1, 2]; same for second passage.
    first_passage_ids = [s["summary_id"] for s in scene
                          if s["text_id"] == "1" and s["passage_id"] == 1]
    assert first_passage_ids == [1, 2]
    second_passage_ids = [s["summary_id"] for s in scene
                           if s["text_id"] == "1" and s["passage_id"] == 2]
    assert second_passage_ids == [1, 2]
    # Uniqueness: (text_id, model, passage_id, summary_type, summary_id).
    keys = {(s["text_id"], s["model"], s["passage_id"], s["summary_type"], s["summary_id"])
            for s in scene}
    assert len(keys) == 8


def test_global_summaries_are_marked_global(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    global_summaries = pipeline.run_summaries(
        kind="global",
        passages=passages,
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        summary_n=2,
        call_model=fake_call,
        temp_dir=temp_dir,
    )
    # 1 model × 2 texts × 1 passage × 2 requirements = 4 global summaries.
    assert len(global_summaries) == 4
    assert all(s["summary_type"] == "global" for s in global_summaries)
    # summary_id counter restarts at 1 for each passage.
    text1_ids = [s["summary_id"] for s in global_summaries if s["text_id"] == "1"]
    assert text1_ids == [1, 2]


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
    scene = pipeline.run_summaries(
        kind="scene",
        passages=passages,
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        summary_n=1,
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
        "text_id", "model", "passage_id", "requirement", "passage_text",
    }
    assert set(passages[0].keys()) == expected_fields


def test_summary_record_has_expected_fields(fake_corpus):
    meta_rows, temp_dir = fake_corpus
    fake_call = _make_mock_caller()
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows, model_keys=["gpt-4.1"], selection_n=1,
        call_model=fake_call, temp_dir=temp_dir,
    )
    scene = pipeline.run_summaries(
        kind="scene", passages=passages, meta_rows=meta_rows,
        model_keys=["gpt-4.1"], summary_n=1,
        call_model=fake_call, temp_dir=temp_dir,
    )
    expected_fields = {
        "text_id", "model", "passage_id",
        "summary_type", "summary_id",
        "requirement", "summary_text",
    }
    assert set(scene[0].keys()) == expected_fields


# ---------------------------------------------------------------------------
# Passage quality checks (verbatim + word-count)
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


def test_count_words_basic():
    assert pipeline.count_words("the quick brown fox") == 4
    assert pipeline.count_words("  spaces   collapsed   to   four ") == 4
    assert pipeline.count_words("") == 0


def test_is_in_word_range_inclusive_bounds():
    # 100 and 200 are both accepted (inclusive bounds).
    assert pipeline.is_in_word_range(" ".join(["w"] * 100), 100, 200)
    assert pipeline.is_in_word_range(" ".join(["w"] * 200), 100, 200)
    assert not pipeline.is_in_word_range(" ".join(["w"] * 99), 100, 200)
    assert not pipeline.is_in_word_range(" ".join(["w"] * 201), 100, 200)


def test_passage_check_passes_through_when_response_is_clean(fake_corpus):
    """A response that is verbatim AND in the word range should not retry."""
    meta_rows, temp_dir = fake_corpus

    calls: list[tuple[str, str]] = []

    def call(model_key, system, user, schema_name):
        calls.append((schema_name, user))
        if schema_name == "questions":
            return {"questions": ["q1"]}
        if schema_name == "passage":
            return {"passage": PASSAGE_150}
        raise ValueError(schema_name)

    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=call,
        temp_dir=temp_dir,
    )
    # 2 texts × 1 requirement → 2 passages; no retries.
    assert len(passages) == 2
    assert [s for s, _ in calls] == ["questions", "passage", "questions", "passage"]


def test_passage_check_re_prompts_on_non_substring(fake_corpus):
    """Non-substring response should trigger one retry."""
    meta_rows, temp_dir = fake_corpus

    state = {"passage_call_count": 0}

    def call(model_key, system, user, schema_name):
        if schema_name == "questions":
            return {"questions": ["q1"]}
        if schema_name == "passage":
            state["passage_call_count"] += 1
            if state["passage_call_count"] % 2 == 1:
                # First attempt: commentary (non-substring AND too short).
                return {"passage": "This poem exemplifies a key theme."}
            else:
                return {"passage": PASSAGE_150}
        raise ValueError(schema_name)

    one_text = [meta_rows[0]]
    passages = pipeline.run_passage_selection(
        meta_rows=one_text,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=call,
        temp_dir=temp_dir,
    )
    assert state["passage_call_count"] == 2
    assert len(passages) == 1
    assert passages[0]["passage_text"] == PASSAGE_150


def test_passage_check_re_prompts_on_overlong_passage(fake_corpus, capsys):
    """A verbatim but over-max response should also trigger a retry, and the
    retry instruction should mention the actual word count."""
    meta_rows, temp_dir = fake_corpus
    # Build a 400-word passage (over the 300-word max). Append it to text A
    # so the verbatim check still passes for the overlong response.
    overlong = " ".join(f"x{i}" for i in range(1, 401))  # 400 words
    target = pipeline.io_utils.PLAINTEXT_DIR / meta_rows[0]["filename"]
    target.write_text(target.read_text() + "\n\n" + overlong, encoding="utf-8")

    state = {"passage_call_count": 0}
    captured_retry_prompt: list[str] = []

    def call(model_key, system, user, schema_name):
        if schema_name == "questions":
            return {"questions": ["q1"]}
        if schema_name == "passage":
            state["passage_call_count"] += 1
            if state["passage_call_count"] == 1:
                return {"passage": overlong}  # 400 words, verbatim
            # The second call should have the length retry instruction appended.
            captured_retry_prompt.append(user)
            return {"passage": PASSAGE_150}
        raise ValueError(schema_name)

    one_text = [meta_rows[0]]
    passages = pipeline.run_passage_selection(
        meta_rows=one_text,
        model_keys=["gpt-4.1"],
        selection_n=1,
        call_model=call,
        temp_dir=temp_dir,
    )

    assert state["passage_call_count"] == 2
    assert len(passages) == 1
    # The retry prompt should mention the actual count (400).
    assert "400 words" in captured_retry_prompt[0]
    # And the warning log should name the length failure.
    captured = capsys.readouterr().out
    assert "length:400" in captured


def test_passage_check_accepts_response_when_retry_also_fails(fake_corpus, capsys):
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
    assert "still failing" in captured
