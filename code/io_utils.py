"""Read inputs from data/ and write outputs to temp/ and results/.

The pipeline produces a lot of intermediate records. To keep things simple and
recoverable, each call to a model is appended as one line to a JSONL file in
temp/. At the end of a run, all JSONL files for a given stage are compiled
into a single CSV in results/, sorted and deduplicated by ID.

This two-step write pattern means:
- We never lose work if the pipeline crashes mid-run (records are already on
  disk).
- Reruns can append without overwriting; compile_csv keeps the most recent
  record for each ID.
"""

import json
from pathlib import Path

import pandas as pd

# The project root is two directories above this file (code/ -> project root).
# Every read or write resolves paths against ROOT so the code works regardless
# of the caller's working directory.
ROOT = Path(__file__).resolve().parent.parent

# Convenience paths used throughout the project.
DATA_DIR = ROOT / "data"
PLAINTEXT_DIR = DATA_DIR / "plaintext"
META_PATH = DATA_DIR / "meta.csv"
TEMP_DIR = ROOT / "temp"
RESULTS_DIR = ROOT / "results"


def load_metadata(path: Path | None = None) -> list[dict]:
    """Load meta.csv as a list of plain dicts.

    Using a list of dicts (rather than a DataFrame) makes downstream code
    easier to read for students who are new to pandas: each row is just a
    Python dict with the column names as keys.

    `path` defaults to the module-level META_PATH constant. We look the
    constant up at call time so tests can override it.
    """
    if path is None:
        path = META_PATH
    df = pd.read_csv(path, dtype=str)
    # to_dict(orient="records") returns one dict per row.
    return df.to_dict(orient="records")


def load_text(filename: str, plaintext_dir: Path | None = None) -> str:
    """Load one plaintext file by filename (UTF-8)."""
    if plaintext_dir is None:
        plaintext_dir = PLAINTEXT_DIR
    return (plaintext_dir / filename).read_text(encoding="utf-8")


def list_plaintext_files(plaintext_dir: Path | None = None) -> list[Path]:
    """Return all .txt files in data/plaintext, sorted by filename.

    Anything that isn't a .txt file (notably .DS_Store on macOS) is skipped.
    """
    if plaintext_dir is None:
        plaintext_dir = PLAINTEXT_DIR
    return sorted(p for p in plaintext_dir.iterdir() if p.suffix == ".txt")


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a single line to a JSONL file.

    The parent directory is created if missing so callers don't have to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Returns [] if the file is missing."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compile_csv(
    jsonl_paths: list[Path],
    csv_path: Path,
    columns: list[str],
    dedup_by: list[str],
    sort_by: list[str],
) -> int:
    """Compile multiple JSONL caches into one final CSV.

    Steps:
    1. Read every JSONL file (some may not exist yet — that's fine).
    2. Deduplicate by the compound `dedup_by` columns, keeping the last
       occurrence (so reruns overwrite earlier records).
    3. Sort by `sort_by` columns.
    4. Select only the requested `columns` for the final CSV.

    The compound dedup matters because the same passage_id can appear for
    multiple models (each model selects its own passage labelled p_01_01),
    so we typically dedup by (model, passage_id) rather than passage_id alone.

    Returns the number of rows written.
    """
    # Collect every record from every JSONL file.
    all_records: list[dict] = []
    for path in jsonl_paths:
        all_records.extend(read_jsonl(path))

    if not all_records:
        # Still write an empty CSV with just the header, so downstream tools
        # know the file exists and what shape it has.
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        return 0

    df = pd.DataFrame(all_records)

    # Deduplicate by the compound key, keeping the latest record (last-wins).
    df = df.drop_duplicates(subset=dedup_by, keep="last")

    # Sort by the requested columns.
    df = df.sort_values(by=sort_by, kind="stable").reset_index(drop=True)

    # Keep only the requested columns, in the requested order.
    df = df[columns]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return len(df)
