"""Tests for io_utils.py — loading data and writing intermediate outputs."""

import csv

import pandas as pd
import pytest

import io_utils


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def test_load_metadata_returns_list_of_dicts(tmp_path):
    meta = tmp_path / "meta.csv"
    meta.write_text(
        "TEXT_ID,AUTHOR,TITLE,GENRE,FILENAME\n"
        "01,Author One,Book One,novel,01_one.txt\n"
        "02,Author Two,Book Two,poetry,02_two.txt\n",
        encoding="utf-8",
    )
    rows = io_utils.load_metadata(meta)
    assert len(rows) == 2
    assert rows[0]["TEXT_ID"] == "01"
    assert rows[0]["AUTHOR"] == "Author One"
    assert rows[1]["FILENAME"] == "02_two.txt"


# ---------------------------------------------------------------------------
# Plaintext loading and listing
# ---------------------------------------------------------------------------

def test_load_text_reads_utf8(tmp_path):
    (tmp_path / "sample.txt").write_text("hello — café", encoding="utf-8")
    assert io_utils.load_text("sample.txt", plaintext_dir=tmp_path) == "hello — café"


def test_list_plaintext_files_skips_non_txt(tmp_path):
    # Two valid text files plus a .DS_Store that should be ignored.
    (tmp_path / "01_a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "02_b.txt").write_text("b", encoding="utf-8")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x00")
    files = io_utils.list_plaintext_files(tmp_path)
    assert [p.name for p in files] == ["01_a.txt", "02_b.txt"]


# ---------------------------------------------------------------------------
# JSONL append / read round-trip
# ---------------------------------------------------------------------------

def test_jsonl_round_trip_creates_parent(tmp_path):
    path = tmp_path / "new_dir" / "records.jsonl"
    io_utils.append_jsonl(path, {"id": "p_01_01", "model": "gpt-4.1"})
    io_utils.append_jsonl(path, {"id": "p_01_02", "model": "gpt-4.1"})
    records = io_utils.read_jsonl(path)
    assert records == [
        {"id": "p_01_01", "model": "gpt-4.1"},
        {"id": "p_01_02", "model": "gpt-4.1"},
    ]


def test_read_jsonl_missing_file_returns_empty(tmp_path):
    assert io_utils.read_jsonl(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# CSV compilation: dedup by ID and sort order
# ---------------------------------------------------------------------------

def test_compile_csv_dedupes_keeping_latest_and_sorts(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    # Same (model, passage_id) appears twice; later record should win.
    io_utils.append_jsonl(a, {"passage_id": "p_01_02", "model": "gpt-4.1", "passage_text": "old"})
    io_utils.append_jsonl(b, {"passage_id": "p_01_01", "model": "gpt-4.1", "passage_text": "first"})
    io_utils.append_jsonl(b, {"passage_id": "p_01_02", "model": "gpt-4.1", "passage_text": "new"})

    out_csv = tmp_path / "out.csv"
    rows_written = io_utils.compile_csv(
        jsonl_paths=[a, b],
        csv_path=out_csv,
        columns=["passage_id", "model", "passage_text"],
        dedup_by=["model", "passage_id"],
        sort_by=["model", "passage_id"],
    )

    assert rows_written == 2
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["passage_id", "model", "passage_text"]
    # Sorted by (model, passage_id) and the dedup kept the latest value.
    assert df.iloc[0]["passage_id"] == "p_01_01"
    assert df.iloc[0]["passage_text"] == "first"
    assert df.iloc[1]["passage_id"] == "p_01_02"
    assert df.iloc[1]["passage_text"] == "new"


def test_compile_csv_keeps_same_id_across_models(tmp_path):
    """Two models may produce passages with the same passage_id.

    The compound (model, passage_id) dedup key keeps both rows so that
    cross-model results are preserved in the final CSV.
    """
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    io_utils.append_jsonl(a, {"passage_id": "p_01_01", "model": "gpt-4.1", "passage_text": "from gpt"})
    io_utils.append_jsonl(b, {"passage_id": "p_01_01", "model": "llama-4-maverick", "passage_text": "from llama"})

    out_csv = tmp_path / "out.csv"
    rows = io_utils.compile_csv(
        jsonl_paths=[a, b],
        csv_path=out_csv,
        columns=["passage_id", "model", "passage_text"],
        dedup_by=["model", "passage_id"],
        sort_by=["model", "passage_id"],
    )
    assert rows == 2  # both kept because models differ
    df = pd.read_csv(out_csv)
    assert set(df["model"]) == {"gpt-4.1", "llama-4-maverick"}


def test_compile_csv_writes_empty_when_no_records(tmp_path):
    out_csv = tmp_path / "empty.csv"
    rows = io_utils.compile_csv(
        jsonl_paths=[tmp_path / "missing.jsonl"],
        csv_path=out_csv,
        columns=["passage_id", "model"],
        dedup_by=["model", "passage_id"],
        sort_by=["model", "passage_id"],
    )
    assert rows == 0
    # The CSV exists and has just the header.
    with out_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        with pytest.raises(StopIteration):
            next(reader)
    assert header == ["passage_id", "model"]
