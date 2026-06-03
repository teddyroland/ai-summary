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
        "text_id,author,title,genre,filename\n"
        "1,Author One,Book One,novel,1_one.txt\n"
        "2,Author Two,Book Two,poetry,2_two.txt\n",
        encoding="utf-8",
    )
    rows = io_utils.load_metadata(meta)
    assert len(rows) == 2
    assert rows[0]["text_id"] == "1"
    assert rows[0]["author"] == "Author One"
    assert rows[1]["filename"] == "2_two.txt"


# ---------------------------------------------------------------------------
# Plaintext loading and listing
# ---------------------------------------------------------------------------

def test_load_text_reads_utf8(tmp_path):
    (tmp_path / "sample.txt").write_text("hello — café", encoding="utf-8")
    assert io_utils.load_text("sample.txt", plaintext_dir=tmp_path) == "hello — café"


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
    # Same (text_id, model, passage_id) appears twice; later record should win.
    io_utils.append_jsonl(a, {"text_id": "01", "model": "gpt-4.1", "passage_id": 2, "passage_text": "old"})
    io_utils.append_jsonl(b, {"text_id": "01", "model": "gpt-4.1", "passage_id": 1, "passage_text": "first"})
    io_utils.append_jsonl(b, {"text_id": "01", "model": "gpt-4.1", "passage_id": 2, "passage_text": "new"})

    out_csv = tmp_path / "out.csv"
    rows_written = io_utils.compile_csv(
        jsonl_paths=[a, b],
        csv_path=out_csv,
        columns=["text_id", "model", "passage_id", "passage_text"],
        dedup_by=["text_id", "model", "passage_id"],
        sort_by=["model", "text_id", "passage_id"],
    )

    assert rows_written == 2
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["text_id", "model", "passage_id", "passage_text"]
    # Sorted by (model, text_id, passage_id) and dedup kept the latest value.
    assert df.iloc[0]["passage_id"] == 1
    assert df.iloc[0]["passage_text"] == "first"
    assert df.iloc[1]["passage_id"] == 2
    assert df.iloc[1]["passage_text"] == "new"


def test_compile_csv_keeps_same_id_across_models(tmp_path):
    """Two models produce passages with the same integer passage_id.

    The compound (text_id, model, passage_id) dedup key keeps both rows.
    """
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    io_utils.append_jsonl(a, {"text_id": "01", "model": "gpt-4.1", "passage_id": 1, "passage_text": "from gpt"})
    io_utils.append_jsonl(b, {"text_id": "01", "model": "llama-4-maverick", "passage_id": 1, "passage_text": "from llama"})

    out_csv = tmp_path / "out.csv"
    rows = io_utils.compile_csv(
        jsonl_paths=[a, b],
        csv_path=out_csv,
        columns=["text_id", "model", "passage_id", "passage_text"],
        dedup_by=["text_id", "model", "passage_id"],
        sort_by=["model", "text_id", "passage_id"],
    )
    assert rows == 2  # both kept because models differ
    df = pd.read_csv(out_csv)
    assert set(df["model"]) == {"gpt-4.1", "llama-4-maverick"}


def test_compile_csv_writes_empty_when_no_records(tmp_path):
    out_csv = tmp_path / "empty.csv"
    rows = io_utils.compile_csv(
        jsonl_paths=[tmp_path / "missing.jsonl"],
        csv_path=out_csv,
        columns=["text_id", "model", "passage_id"],
        dedup_by=["text_id", "model", "passage_id"],
        sort_by=["model", "text_id", "passage_id"],
    )
    assert rows == 0
    # The CSV exists and has just the header.
    with out_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        with pytest.raises(StopIteration):
            next(reader)
    assert header == ["text_id", "model", "passage_id"]
