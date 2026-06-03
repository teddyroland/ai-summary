"""Command-line entry point for the AI summarization pipeline.

Usage from the project root:

    python code/main.py --selection 5 --scene 5 --global 5 \\
        --models gpt-4.1,llama-4-maverick

All flags are optional; defaults are 1 of each and the gpt-4.1 model. After
the three stages run, the per-call JSONL caches in temp/ are compiled into
three CSV files in results/.
"""

import argparse
import sys
from pathlib import Path

import io_utils
import models
import pipeline


# ---------------------------------------------------------------------------
# Final CSV columns and sort keys
# ---------------------------------------------------------------------------

PASSAGE_CSV_COLUMNS = [
    "text_id", "model", "passage_id", "requirement", "passage_text",
]
PASSAGE_DEDUP_BY = ["text_id", "model", "passage_id"]
PASSAGE_SORT_BY = ["model", "text_id", "passage_id"]

SUMMARY_CSV_COLUMNS = [
    "text_id", "model", "passage_id",
    "summary_type", "summary_id",
    "requirement", "summary_text",
]
SUMMARY_DEDUP_BY = ["text_id", "model", "passage_id", "summary_type", "summary_id"]
SUMMARY_SORT_BY = ["model", "text_id", "passage_id", "summary_id"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI argument parser and return parsed args."""
    parser = argparse.ArgumentParser(
        description="Run the AI summarization pipeline (selection + scene + global).",
    )
    parser.add_argument(
        "--selection",
        type=int,
        default=1,
        help="Number of passages to select per text (default: 1).",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=1,
        help="Number of scene-setting summaries per passage (default: 1).",
    )
    parser.add_argument(
        "--global",
        dest="global_n",
        type=int,
        default=1,
        help="Number of global-theorizing summaries per passage (default: 1).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt-4.1",
        help="Comma-separated model keys (default: gpt-4.1). "
             "Choose from: " + ", ".join(sorted(models.MODEL_REGISTRY)),
    )
    return parser.parse_args(argv)


def _parse_model_keys(models_arg: str) -> list[str]:
    """Split the --models string and validate each key against MODEL_REGISTRY."""
    keys = [m.strip() for m in models_arg.split(",") if m.strip()]
    unknown = [k for k in keys if k not in models.MODEL_REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown model key(s): {unknown}. "
            f"Choose from: {sorted(models.MODEL_REGISTRY)}"
        )
    return keys


def _gather_jsonl_paths(model_keys: list[str], stage: str, temp_dir: Path) -> list[Path]:
    """Return one JSONL path per model for a given stage suffix."""
    return [temp_dir / f"{key}_{stage}.jsonl" for key in model_keys]


def run(
    selection_n: int,
    scene_n: int,
    global_n: int,
    model_keys: list[str],
    *,
    temp_dir: Path | None = None,
    results_dir: Path | None = None,
    meta_path: Path | None = None,
) -> dict[str, int]:
    """Run the three stages end-to-end and write final CSVs.

    Returns row counts for the three output CSVs.
    """
    temp_dir = temp_dir or io_utils.TEMP_DIR
    results_dir = results_dir or io_utils.RESULTS_DIR

    meta_rows = io_utils.load_metadata(meta_path)

    # Stage 4a — passage selection.
    print(f"[1/3] Selecting passages with: {', '.join(model_keys)}")
    passages = pipeline.run_passage_selection(
        meta_rows=meta_rows,
        model_keys=model_keys,
        selection_n=selection_n,
        temp_dir=temp_dir,
    )
    print(f"      wrote {len(passages)} passage records to temp/")

    # Stage 4b — scene-setting summaries.
    print(f"[2/3] Generating scene-setting summaries")
    scene = pipeline.run_summaries(
        kind="scene",
        passages=passages,
        meta_rows=meta_rows,
        model_keys=model_keys,
        summary_n=scene_n,
        temp_dir=temp_dir,
    )
    print(f"      wrote {len(scene)} scene-summary records to temp/")

    # Stage 4c — global-theorizing summaries.
    print(f"[3/3] Generating global-theorizing summaries")
    global_summaries = pipeline.run_summaries(
        kind="global",
        passages=passages,
        meta_rows=meta_rows,
        model_keys=model_keys,
        summary_n=global_n,
        temp_dir=temp_dir,
    )
    print(f"      wrote {len(global_summaries)} global-summary records to temp/")

    # Compile the JSONL caches into final CSVs.
    counts = {}
    counts["passages"] = io_utils.compile_csv(
        jsonl_paths=_gather_jsonl_paths(model_keys, "passages", temp_dir),
        csv_path=results_dir / "passages.csv",
        columns=PASSAGE_CSV_COLUMNS,
        dedup_by=PASSAGE_DEDUP_BY,
        sort_by=PASSAGE_SORT_BY,
    )
    counts["scene"] = io_utils.compile_csv(
        jsonl_paths=_gather_jsonl_paths(model_keys, "scene", temp_dir),
        csv_path=results_dir / "scene_summaries.csv",
        columns=SUMMARY_CSV_COLUMNS,
        dedup_by=SUMMARY_DEDUP_BY,
        sort_by=SUMMARY_SORT_BY,
    )
    counts["global"] = io_utils.compile_csv(
        jsonl_paths=_gather_jsonl_paths(model_keys, "global", temp_dir),
        csv_path=results_dir / "global_summaries.csv",
        columns=SUMMARY_CSV_COLUMNS,
        dedup_by=SUMMARY_DEDUP_BY,
        sort_by=SUMMARY_SORT_BY,
    )
    print(
        f"Compiled CSVs in results/: "
        f"{counts['passages']} passages, "
        f"{counts['scene']} scene summaries, "
        f"{counts['global']} global summaries."
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    model_keys = _parse_model_keys(args.models)
    run(
        selection_n=args.selection,
        scene_n=args.scene,
        global_n=args.global_n,
        model_keys=model_keys,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
