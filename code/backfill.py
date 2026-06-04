"""Backfill failed/missing rows after a partial pipeline run.

Every stage of the pipeline can lose rows in two ways during a failed run:

1. **Marker rows.** A per-item call (passage or summary) failed after all
   retries; the pipeline wrote a row whose text field starts with
   "[FAILED ...]" and moved on.
2. **Silently absent rows.** The Stage-1 list call (questions for passage
   selection; requirements for summaries) failed, so the pipeline skipped
   the whole (text, model) pair or whole passage and no rows were written
   for any of its N expected items.

This script identifies both kinds of gaps for one (model, stage) combination
and re-runs only the failed slots:

- For absent rows, it regenerates the Stage-1 list and runs the missing
  Stage-2 calls. The new list will differ from the original (the questions
  / requirements are themselves model-generated), so the pairing between
  conditions and items in those slots is reset.
- For marker rows, it re-runs only the failed Stage-2 call using the
  condition (requirement) preserved in each marker row. Other (good) rows
  for the same parent are untouched, so the original condition→item
  pairings stay intact wherever they exist.

New records are appended to the existing JSONL cache; the final CSV is then
recompiled with the standard dedup (last-wins) so marker rows are
overwritten by their backfilled counterparts.

Usage from the project root:

    # Backfill missing/marker passages for one model
    python code/backfill.py --model llama-4-maverick --stage passages

    # Backfill missing/marker summaries for one model and one summary type
    python code/backfill.py --model llama-4-maverick --stage global
    python code/backfill.py --model gpt-4.1 --stage scene
"""

import argparse
from pathlib import Path

import pandas as pd

import io_utils
import main
import models
import pipeline
import prompts


MARKER_PREFIX = "[FAILED"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _existing_rows(csv_path: Path, model_key: str, summary_type: str | None) -> pd.DataFrame:
    """Return the subset of `csv_path` rows for one (model[, summary_type])."""
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    mask = df["model"] == model_key
    if summary_type is not None:
        mask &= df["summary_type"] == summary_type
    return df[mask]


def _id_gaps(rows: pd.DataFrame, id_column: str, text_column: str, n: int) -> tuple[pd.DataFrame, list[int]]:
    """Return (marker_rows, missing_ids) for a group of rows expected to cover
    integer ids 1..n in `id_column`. `text_column` is the field we inspect
    for the [FAILED ...] marker (passage_text or summary_text)."""
    if rows.empty:
        return pd.DataFrame(), list(range(1, n + 1))
    is_marker = rows[text_column].astype(str).str.startswith(MARKER_PREFIX)
    marker_rows = rows[is_marker]
    non_marker_ids = set(rows.loc[~is_marker, id_column].astype(int))
    present_ids = non_marker_ids | set(marker_rows[id_column].astype(int))
    missing_ids = sorted(set(range(1, n + 1)) - present_ids)
    return marker_rows, missing_ids


def _recompile(stage: str) -> None:
    """Rebuild the final CSV for one stage from every model's JSONL cache."""
    if stage == "passages":
        csv_path = io_utils.RESULTS_DIR / "passages.csv"
        cols = main.PASSAGE_CSV_COLUMNS
        dedup = main.PASSAGE_DEDUP_BY
        sort_by = main.PASSAGE_SORT_BY
        jsonl_paths = list(io_utils.TEMP_DIR.glob("*_passages.jsonl"))
    else:
        csv_path = io_utils.RESULTS_DIR / f"{stage}_summaries.csv"
        cols = main.SUMMARY_CSV_COLUMNS
        dedup = main.SUMMARY_DEDUP_BY
        sort_by = main.SUMMARY_SORT_BY
        jsonl_paths = list(io_utils.TEMP_DIR.glob(f"*_{stage}.jsonl"))
    n_rows = io_utils.compile_csv(
        jsonl_paths=jsonl_paths, csv_path=csv_path,
        columns=cols, dedup_by=dedup, sort_by=sort_by,
    )
    print(f"Recompiled {n_rows} rows into {csv_path}")


# ---------------------------------------------------------------------------
# Stage 4a — passages
# ---------------------------------------------------------------------------

def _call_passage(text_id: str, entry: dict, requirement: str,
                  passage_id: int, model_key: str) -> dict:
    """Build one passage record. Uses _select_validated_passage so verbatim
    and word-count checks (with one stricter re-prompt) still apply."""
    row = entry["row"]
    text = entry["text"]
    passage_prompt = prompts.render_passage_prompt(
        author=row["author"], title=row["title"],
        requirement=requirement, text=text,
    )
    try:
        passage_text = pipeline._select_validated_passage(
            model_key=model_key,
            user_prompt=passage_prompt,
            source_text=text,
            call_model=models.call_model,
            passage_label=f"{text_id}/{model_key}/p{passage_id}",
        )
    except Exception as e:
        print(f"    [FAIL] passage call: {type(e).__name__}: {str(e)[:200]}")
        passage_text = pipeline._failure_marker("passage", e)
    return {
        "text_id": text_id,
        "model": model_key,
        "passage_id": passage_id,
        "requirement": requirement,
        "passage_text": passage_text,
    }


def _backfill_text_passages(text_id: str, entry: dict, marker_rows: pd.DataFrame,
                            missing_ids: list[int], model_key: str,
                            selection_n: int, cache_path: Path) -> int:
    """Backfill missing/marker passages for one (text, model) pair."""
    row = entry["row"]
    text = entry["text"]
    label = f"{text_id}/{model_key}"
    n_written = 0

    # Regenerate the questions list for any missing passage_ids — the
    # original list is gone (it was only ever held in memory).
    if missing_ids:
        print(f"  [{label}] {len(missing_ids)} passages missing — regenerating questions")
        questions_prompt = prompts.render_questions_prompt(
            author=row["author"], title=row["title"],
            text=text, selection_n=selection_n,
        )
        try:
            questions_response = models.call_model(
                model_key, prompts.SYSTEM_PROMPT, questions_prompt, "questions"
            )
            requirements = questions_response["questions"]
        except Exception as e:
            print(f"  [{label}] [FAIL] questions: {type(e).__name__}: {e}; skipping (text,model)")
            return 0
        for pid in missing_ids:
            if pid - 1 >= len(requirements):
                print(f"  [{label}/p{pid}] [SKIP] questions list too short")
                continue
            record = _call_passage(text_id, entry, requirements[pid - 1], pid, model_key)
            io_utils.append_jsonl(cache_path, record)
            n_written += 1
            print(f"  [{label}/p{pid}] backfilled")

    # Marker rows: re-attempt with the stored requirement so the
    # requirement→passage pairing matches the original run wherever possible.
    for _, mrow in marker_rows.iterrows():
        pid = int(mrow["passage_id"])
        requirement = mrow["requirement"]
        record = _call_passage(text_id, entry, requirement, pid, model_key)
        io_utils.append_jsonl(cache_path, record)
        n_written += 1
        print(f"  [{label}/p{pid}] re-attempted (was marker)")

    return n_written


def backfill_passages(model_key: str, selection_n: int) -> None:
    csv_path = io_utils.RESULTS_DIR / "passages.csv"
    existing = _existing_rows(csv_path, model_key, summary_type=None)
    meta_index = pipeline.build_meta_index(io_utils.load_metadata())
    cache_path = io_utils.TEMP_DIR / f"{model_key}_passages.jsonl"

    print(f"Backfilling {model_key} passages (target: {selection_n}/text)")
    total_written = 0
    for text_id, entry in meta_index.items():
        rows = existing[existing["text_id"].astype(str) == str(text_id)] if not existing.empty else pd.DataFrame()
        marker_rows, missing_ids = _id_gaps(
            rows, id_column="passage_id", text_column="passage_text", n=selection_n,
        )
        if marker_rows.empty and not missing_ids:
            continue
        total_written += _backfill_text_passages(
            text_id, entry, marker_rows, missing_ids,
            model_key, selection_n, cache_path,
        )
    print(f"Wrote {total_written} new records to {cache_path}")
    _recompile("passages")


# ---------------------------------------------------------------------------
# Stages 4b / 4c — summaries
# ---------------------------------------------------------------------------

def _call_summary(passage: dict, requirement: str, summary_id: int,
                  summary_type: str, model_key: str, meta_index: dict) -> dict:
    """Build one summary record. Substitutes a marker if the call fails."""
    text_id = passage["text_id"]
    row = meta_index[text_id]["row"]
    text = meta_index[text_id]["text"]
    summary_prompt = prompts.render_summary_prompt(
        author=row["author"], title=row["title"],
        passage=passage["passage_text"], requirement=requirement, text=text,
    )
    try:
        response = models.call_model(
            model_key, prompts.SYSTEM_PROMPT, summary_prompt, "summary"
        )
        summary_text = response["summary"]
    except Exception as e:
        print(f"    [FAIL] summary call still failed: {type(e).__name__}: {str(e)[:200]}")
        summary_text = pipeline._failure_marker(f"{summary_type}_summary", e)
    return {
        "text_id": text_id,
        "model": model_key,
        "passage_id": passage["passage_id"],
        "summary_type": summary_type,
        "summary_id": summary_id,
        "requirement": requirement,
        "summary_text": summary_text,
    }


def _backfill_passage_summaries(passage: dict, marker_rows: pd.DataFrame,
                                missing_ids: list[int], meta_index: dict,
                                model_key: str, summary_type: str,
                                summary_n: int, cache_path: Path) -> int:
    """Backfill missing/marker summaries for one (text, model, passage)."""
    text_id = passage["text_id"]
    pid = passage["passage_id"]
    label = f"{text_id}/{model_key}/p{pid}"
    n_written = 0

    if missing_ids:
        print(f"  [{label}] {len(missing_ids)} summaries missing — regenerating requirements")
        render_fn = (
            prompts.render_scene_requirements_prompt if summary_type == "scene"
            else prompts.render_global_requirements_prompt
        )
        row = meta_index[text_id]["row"]
        req_prompt = render_fn(
            row["author"], row["title"], passage["passage_text"], summary_n,
        )
        try:
            req_response = models.call_model(
                model_key, prompts.SYSTEM_PROMPT, req_prompt, "requirements"
            )
            requirements = req_response["requirements"]
        except Exception as e:
            print(f"  [{label}] [FAIL] requirements: {type(e).__name__}: {e}; skipping passage")
            return 0
        for sid in missing_ids:
            if sid - 1 >= len(requirements):
                print(f"  [{label}/s{sid}] [SKIP] requirements list too short")
                continue
            record = _call_summary(
                passage, requirements[sid - 1], sid,
                summary_type, model_key, meta_index,
            )
            io_utils.append_jsonl(cache_path, record)
            n_written += 1
            print(f"  [{label}/s{sid}] backfilled")

    for _, mrow in marker_rows.iterrows():
        sid = int(mrow["summary_id"])
        requirement = mrow["requirement"]
        record = _call_summary(
            passage, requirement, sid, summary_type, model_key, meta_index,
        )
        io_utils.append_jsonl(cache_path, record)
        n_written += 1
        print(f"  [{label}/s{sid}] re-attempted (was marker)")

    return n_written


def backfill_summaries(model_key: str, summary_type: str, summary_n: int) -> None:
    passages = io_utils.read_jsonl(io_utils.TEMP_DIR / f"{model_key}_passages.jsonl")
    if not passages:
        print(f"No cached passages for {model_key}; nothing to backfill.")
        return
    csv_path = io_utils.RESULTS_DIR / f"{summary_type}_summaries.csv"
    existing = _existing_rows(csv_path, model_key, summary_type=summary_type)
    meta_index = pipeline.build_meta_index(io_utils.load_metadata())
    cache_path = io_utils.TEMP_DIR / f"{model_key}_{summary_type}.jsonl"

    print(f"Backfilling {model_key} {summary_type} summaries (target: {summary_n}/passage)")
    total_written = 0
    for passage in passages:
        if not existing.empty:
            rows = existing[
                (existing["text_id"].astype(str) == str(passage["text_id"]))
                & (existing["passage_id"].astype(int) == int(passage["passage_id"]))
            ]
        else:
            rows = pd.DataFrame()
        marker_rows, missing_ids = _id_gaps(
            rows, id_column="summary_id", text_column="summary_text", n=summary_n,
        )
        if marker_rows.empty and not missing_ids:
            continue
        total_written += _backfill_passage_summaries(
            passage, marker_rows, missing_ids, meta_index,
            model_key, summary_type, summary_n, cache_path,
        )
    print(f"Wrote {total_written} new records to {cache_path}")
    _recompile(summary_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Model key (e.g. llama-4-maverick or gpt-4.1)")
    parser.add_argument("--stage", required=True,
                        choices=["passages", "scene", "global"],
                        help="Which stage to backfill")
    parser.add_argument("--n", type=int, default=5,
                        help="Expected items per parent (passages per text, "
                             "or summaries per passage). Default: 5")
    args = parser.parse_args()
    if args.stage == "passages":
        backfill_passages(args.model, args.n)
    else:
        backfill_summaries(args.model, args.stage, args.n)
