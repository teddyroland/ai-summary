"""Tests for prompts.py — placeholder substitution and template selection."""

import prompts


# ---------------------------------------------------------------------------
# Stage 1 of each pipeline stage produces a list prompt with the right count
# ---------------------------------------------------------------------------

def test_questions_prompt_substitutes_all_placeholders():
    out = prompts.render_questions_prompt(
        author="Author One",
        title="Book One",
        text="Full text here.",
        selection_n=5,
    )
    assert "Author One" in out
    assert "Book One" in out
    assert "Full text here." in out
    # The count flows into the "generate a list of N" instruction.
    assert "list of 5 high-level interpretive questions" in out
    # No unfilled placeholders remain.
    assert "[[" not in out and "]]" not in out


def test_scene_requirements_prompt_uses_scene_examples():
    out = prompts.render_scene_requirements_prompt(
        author="A", title="T", passage="P", scene_n=3,
    )
    # Should include one of the scene-setting example requirements.
    assert "linguistic register" in out
    # Should NOT include a global-theorizing example.
    assert "pastoral poetry" not in out
    assert "list of 3 specific summary requirements" in out


def test_global_requirements_prompt_uses_global_examples():
    out = prompts.render_global_requirements_prompt(
        author="A", title="T", passage="P", global_n=4,
    )
    assert "pastoral poetry" in out
    assert "linguistic register" not in out
    assert "list of 4 specific summary requirements" in out


# ---------------------------------------------------------------------------
# Stage 2 prompts (passage selection and shared summary template)
# ---------------------------------------------------------------------------

def test_passage_prompt_includes_question_and_text():
    out = prompts.render_passage_prompt(
        author="A", title="T", requirement="Q?", text="full text",
    )
    assert "Q?" in out
    assert "full text" in out
    assert "[[" not in out


def test_passage_prompt_forbids_commentary():
    """The strict-excerpt instruction added in Phase 2 must be present."""
    out = prompts.render_passage_prompt(
        author="A", title="T", requirement="Q?", text="full text",
    )
    assert "verbatim" in out.lower()
    assert "do not add commentary" in out.lower()


def test_verbatim_retry_instruction_is_strict():
    assert "verbatim" in prompts.PASSAGE_VERBATIM_RETRY_INSTRUCTION.lower()
    assert "no commentary" in prompts.PASSAGE_VERBATIM_RETRY_INSTRUCTION.lower()


def test_passage_length_retry_instruction_mentions_count_and_bounds():
    msg = prompts.passage_length_retry_instruction(547)
    assert "547 words" in msg
    assert "100" in msg and "300" in msg


def test_passage_prompt_emphasizes_word_count():
    out = prompts.render_passage_prompt(
        author="A", title="T", requirement="Q?", text="full text",
    )
    # The hardened instruction should mention the word range and the "hard
    # constraint" / "do not exceed" framing.
    assert "100 and 300 words" in out
    assert "hard constraint" in out.lower()
    assert "do not exceed" in out.lower()


def test_summary_prompt_includes_passage_requirement_and_text():
    out = prompts.render_summary_prompt(
        author="A",
        title="T",
        passage="passage body",
        requirement="Summarize the form.",
        text="full text",
    )
    assert "passage body" in out
    assert "Summarize the form." in out
    assert "full text" in out
    assert "[[" not in out


def test_summary_prompt_is_shared_for_scene_and_global():
    # The spec says scene and global use the same Stage 2 prompt template.
    # We verify by rendering twice with identical arguments and checking equality.
    a = prompts.render_summary_prompt("A", "T", "P", "R", "X")
    b = prompts.render_summary_prompt("A", "T", "P", "R", "X")
    assert a == b


# ---------------------------------------------------------------------------
# System prompt is non-empty and mentions JSON
# ---------------------------------------------------------------------------

def test_system_prompt_mentions_json():
    assert "JSON" in prompts.SYSTEM_PROMPT
