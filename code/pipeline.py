"""Orchestrate the three stages of the summarization pipeline.

Each stage runs two prompts per (model, text-or-passage):

  Stage 4a — passage selection
    Prompt 1: produce a list of interpretive questions
    Prompt 2: select a passage for each question

  Stage 4b — scene-setting summaries
    Prompt 1: produce a list of scene-setting requirements
    Prompt 2: write a summary for each requirement

  Stage 4c — global-theorizing summaries
    Prompt 1: produce a list of global-theorizing requirements
    Prompt 2: write a summary for each requirement

For each stage, every model summarizes its own selections. (We do not
cross-apply: one model never summarizes a passage that another model selected.
This keeps the per-model pipeline coherent for comparative analysis.)

Every API response is appended as a JSON line to a per-stage cache in temp/,
so a crash mid-run loses nothing. main.py is responsible for compiling those
caches into the final CSVs in results/.
"""

from collections.abc import Callable
from pathlib import Path

import io_utils
import models
import prompts

# Summary types we support (used as the `summary_type` column value and to pick
# the correct Stage-1 prompt). Centralized so a typo in pipeline.py is caught
# early.
SUMMARY_TYPES = ("scene", "global")


# Type aliases for readability.
MetaIndex = dict[str, dict]
ModelCaller = Callable[[str, str, str, str], dict]


def _resolve_caller(call_model: ModelCaller | None) -> ModelCaller:
    """Return the explicit caller if given, else the current models.call_model.

    We look up models.call_model at *call* time (not as a default argument)
    so tests can monkeypatch it and have the patch take effect.
    """
    return call_model if call_model is not None else models.call_model


# ---------------------------------------------------------------------------
# Quality checks for selected passages
# ---------------------------------------------------------------------------
#
# Stage 4a asks the model to "select a poem or passage" of 100-300 words from
# the source. In practice models sometimes add editorial commentary or write
# a much longer excerpt than the prompt asks for. We check the response for
# two things — that it is a verbatim substring of the source, and that it is
# in the requested word range — then re-prompt once if either fails. We don't
# raise on a second failure; long runs should not break on one uncooperative
# response. Both failures are logged so the operator can find the rows later.

PASSAGE_MIN_WORDS = 100
PASSAGE_MAX_WORDS = 300


def is_verbatim_excerpt(excerpt: str, source: str) -> bool:
    """True if `excerpt` appears as a substring of `source` after whitespace
    and case normalization.

    Normalizing collapses runs of whitespace to single spaces and lowercases
    both inputs. This tolerates trivial reformatting (line breaks, repeated
    spaces, case differences) while still catching genuine commentary.
    """
    def normalize(s: str) -> str:
        return " ".join(s.split()).lower()
    return normalize(excerpt) in normalize(source)


def count_words(text: str) -> int:
    """Split on whitespace and return the number of tokens.

    Naive but consistent with how a human would count words in a passage.
    """
    return len(text.split())


def is_in_word_range(text: str, min_words: int, max_words: int) -> bool:
    """True if `text` has between `min_words` and `max_words` words inclusive."""
    n = count_words(text)
    return min_words <= n <= max_words


def _passage_failures(passage: str, source: str) -> list[str]:
    """Return labels for any quality checks that the passage fails.

    Each label is one of: "verbatim", or "length:<actual_count>".
    Empty list means the passage passes every check.
    """
    failures: list[str] = []
    if not is_verbatim_excerpt(passage, source):
        failures.append("verbatim")
    n = count_words(passage)
    if not (PASSAGE_MIN_WORDS <= n <= PASSAGE_MAX_WORDS):
        failures.append(f"length:{n}")
    return failures


def _build_retry_instruction(failures: list[str]) -> str:
    """Assemble the follow-up instruction for whichever checks failed."""
    parts: list[str] = []
    if "verbatim" in failures:
        parts.append(prompts.PASSAGE_VERBATIM_RETRY_INSTRUCTION)
    for f in failures:
        if f.startswith("length:"):
            actual = int(f.split(":", 1)[1])
            parts.append(prompts.passage_length_retry_instruction(actual))
    return "".join(parts)


def _select_validated_passage(
    model_key: str,
    user_prompt: str,
    source_text: str,
    call_model: ModelCaller,
    passage_label: str,
) -> str:
    """Call the model for a passage; if it fails any quality check, retry once.

    Both verbatim and word-count checks run together; the retry instruction
    addresses whichever checks failed. `passage_label` is used only in the
    warning messages so the operator can match the warning to a CSV row.
    """
    response = call_model(
        model_key, prompts.SYSTEM_PROMPT, user_prompt, "passage"
    )
    passage_text = response["passage"]

    failures = _passage_failures(passage_text, source_text)
    if not failures:
        return passage_text

    # Re-prompt once with the appropriate stricter follow-up.
    print(f"  [passage check] {passage_label}: re-prompting (failed: {failures})")
    retry_prompt = user_prompt + _build_retry_instruction(failures)
    response = call_model(
        model_key, prompts.SYSTEM_PROMPT, retry_prompt, "passage"
    )
    passage_text = response["passage"]

    failures = _passage_failures(passage_text, source_text)
    if failures:
        print(
            f"  [passage check] {passage_label}: re-prompt still failing "
            f"({failures}); accepting response anyway"
        )
    return passage_text


def build_meta_index(meta_rows: list[dict]) -> MetaIndex:
    """Map text_id -> {"row": meta_row, "text": loaded plaintext}.

    Texts are loaded once and reused across all stages and models so we don't
    re-read big files for every call.
    """
    index: MetaIndex = {}
    for row in meta_rows:
        index[row["text_id"]] = {
            "row": row,
            "text": io_utils.load_text(row["filename"]),
        }
    return index


# ---------------------------------------------------------------------------
# Stage 4a — Passage selection
# ---------------------------------------------------------------------------

def run_passage_selection(
    meta_rows: list[dict],
    model_keys: list[str],
    selection_n: int,
    *,
    call_model: ModelCaller | None = None,
    temp_dir: Path | None = None,
) -> list[dict]:
    """Generate questions and select a passage for each question.

    Returns a list of passage records (same shape as the JSONL cache rows).
    """
    call_model = _resolve_caller(call_model)
    if temp_dir is None:
        temp_dir = io_utils.TEMP_DIR
    meta_index = build_meta_index(meta_rows)
    passages: list[dict] = []

    for model_key in model_keys:
        cache_path = temp_dir / f"{model_key}_passages.jsonl"
        for text_id, entry in meta_index.items():
            row = entry["row"]
            text = entry["text"]

            # Prompt 1 — generate the list of interpretive requirements.
            questions_prompt = prompts.render_questions_prompt(
                author=row["author"],
                title=row["title"],
                text=text,
                selection_n=selection_n,
            )
            questions_response = call_model(
                model_key, prompts.SYSTEM_PROMPT, questions_prompt, "questions"
            )
            # The questions schema still returns a "questions" field. We treat
            # each entry as a "requirement" in the downstream record and CSV.
            requirements = questions_response["questions"]

            # Prompt 2 — one passage per requirement, validated as a verbatim
            # substring of the source. passage_id is a 1-based counter that
            # resets per (text, model); uniqueness is (text_id, model,
            # passage_id).
            for i, requirement in enumerate(requirements, start=1):
                passage_prompt = prompts.render_passage_prompt(
                    author=row["author"],
                    title=row["title"],
                    requirement=requirement,
                    text=text,
                )
                passage_text = _select_validated_passage(
                    model_key=model_key,
                    user_prompt=passage_prompt,
                    source_text=text,
                    call_model=call_model,
                    passage_label=f"{text_id}/{model_key}/p{i}",
                )
                record = {
                    "text_id": text_id,
                    "model": model_key,
                    "passage_id": i,
                    "requirement": requirement,
                    "passage_text": passage_text,
                }
                io_utils.append_jsonl(cache_path, record)
                passages.append(record)

    return passages


# ---------------------------------------------------------------------------
# Stages 4b and 4c — Scene-setting and global-theorizing summaries
# ---------------------------------------------------------------------------

def run_summaries(
    kind: str,
    passages: list[dict],
    meta_rows: list[dict],
    model_keys: list[str],
    summary_n: int,
    *,
    call_model: ModelCaller | None = None,
    temp_dir: Path | None = None,
) -> list[dict]:
    """Generate scene-setting or global-theorizing summaries for each passage.

    The two summary stages share the same iteration structure and Prompt 2
    template (`prompts.render_summary_prompt`); they differ only in which
    Prompt 1 template they use and in the `summary_type` column written to the
    record. `kind` is "scene" or "global".

    summary_id is a 1-based counter that resets per
    (text_id, model, passage_id, summary_type). Uniqueness within a row is
    (text_id, model, passage_id, summary_type, summary_id).
    """
    if kind not in SUMMARY_TYPES:
        raise ValueError(f"kind must be one of {SUMMARY_TYPES}, got {kind!r}")
    call_model = _resolve_caller(call_model)
    if temp_dir is None:
        temp_dir = io_utils.TEMP_DIR
    render_requirements = (
        prompts.render_scene_requirements_prompt if kind == "scene"
        else prompts.render_global_requirements_prompt
    )

    meta_index = build_meta_index(meta_rows)
    summaries: list[dict] = []

    for model_key in model_keys:
        cache_path = temp_dir / f"{model_key}_{kind}.jsonl"
        # Only summarize passages that this model selected.
        own_passages = [p for p in passages if p["model"] == model_key]
        for passage in own_passages:
            text_id = passage["text_id"]
            entry = meta_index[text_id]
            row = entry["row"]
            text = entry["text"]

            # Prompt 1 — requirements list.
            req_prompt = render_requirements(
                row["author"], row["title"], passage["passage_text"], summary_n
            )
            req_response = call_model(
                model_key, prompts.SYSTEM_PROMPT, req_prompt, "requirements"
            )
            requirements = req_response["requirements"]

            # Prompt 2 — one summary per requirement (shared template).
            for j, requirement in enumerate(requirements, start=1):
                summary_prompt = prompts.render_summary_prompt(
                    author=row["author"],
                    title=row["title"],
                    passage=passage["passage_text"],
                    requirement=requirement,
                    text=text,
                )
                summary_response = call_model(
                    model_key, prompts.SYSTEM_PROMPT, summary_prompt, "summary"
                )
                record = {
                    "text_id": text_id,
                    "model": model_key,
                    "passage_id": passage["passage_id"],
                    "summary_type": kind,
                    "summary_id": j,
                    "requirement": requirement,
                    "summary_text": summary_response["summary"],
                }
                io_utils.append_jsonl(cache_path, record)
                summaries.append(record)

    return summaries
