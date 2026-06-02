"""Smoke test for main.run() — end-to-end with mocked model calls."""

import csv

import pytest

import io_utils
import main
import models
import pipeline


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    """Stage a workspace with two texts, plus a mock call_model."""
    # data layout
    data = tmp_path / "data"
    plaintext = data / "plaintext"
    plaintext.mkdir(parents=True)
    # Source text contains the mock's passage response so the pipeline's
    # verbatim-substring check passes without triggering a re-prompt.
    (plaintext / "01_a.txt").write_text(
        "Text A body. Contains passage-by-gpt-4.1.", encoding="utf-8",
    )
    (plaintext / "02_b.txt").write_text(
        "Text B body. Contains passage-by-gpt-4.1.", encoding="utf-8",
    )
    meta_path = data / "meta.csv"
    meta_path.write_text(
        "TEXT_ID,AUTHOR,TITLE,GENRE,FILENAME\n"
        "01,Author A,Book A,novel,01_a.txt\n"
        "02,Author B,Book B,poetry,02_b.txt\n",
        encoding="utf-8",
    )

    # Point io_utils at the fake workspace.
    monkeypatch.setattr(io_utils, "PLAINTEXT_DIR", plaintext)
    monkeypatch.setattr(io_utils, "META_PATH", meta_path)

    # Replace the real model client with the same kind of mock the pipeline
    # tests use, but installed at the call_model entry point so main.py picks
    # it up automatically.
    def fake_call(model_key, system, user, schema_name):
        if schema_name == "questions":
            return {"questions": ["q1", "q2"]}
        if schema_name == "requirements":
            return {"requirements": ["r1", "r2"]}
        if schema_name == "passage":
            return {"passage": f"passage-by-{model_key}"}
        if schema_name == "summary":
            return {"summary": f"summary-by-{model_key}"}
        raise ValueError(schema_name)

    monkeypatch.setattr(models, "call_model", fake_call)
    monkeypatch.setattr(pipeline.models, "call_model", fake_call)

    temp_dir = tmp_path / "temp"
    results_dir = tmp_path / "results"
    return {
        "meta_path": meta_path,
        "temp_dir": temp_dir,
        "results_dir": results_dir,
    }


def test_run_produces_three_csvs_with_correct_shapes(fake_workspace):
    counts = main.run(
        selection_n=2,
        scene_n=2,
        global_n=2,
        model_keys=["gpt-4.1"],
        temp_dir=fake_workspace["temp_dir"],
        results_dir=fake_workspace["results_dir"],
        meta_path=fake_workspace["meta_path"],
    )

    # Expected: 2 texts × 2 passages = 4 passages.
    # 4 passages × 2 scene reqs = 8 scene summaries.
    # 4 passages × 2 global reqs = 8 global summaries.
    assert counts == {"passages": 4, "scene": 8, "global": 8}

    # Verify the passages CSV columns and one row.
    with (fake_workspace["results_dir"] / "passages.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == main.PASSAGE_CSV_COLUMNS
    assert len(rows) == 4
    assert rows[0]["model"] == "gpt-4.1"
    # IDs sort correctly (lexicographic across the zero-padded passage_id).
    assert [r["passage_id"] for r in rows] == [
        "p_01_gpt41_01", "p_01_gpt41_02", "p_02_gpt41_01", "p_02_gpt41_02"
    ]

    # Verify scene_summaries CSV.
    with (fake_workspace["results_dir"] / "scene_summaries.csv").open() as f:
        scene_rows = list(csv.DictReader(f))
    assert len(scene_rows) == 8
    assert all("scene" in r["summary_id"] for r in scene_rows)

    # Verify global_summaries CSV.
    with (fake_workspace["results_dir"] / "global_summaries.csv").open() as f:
        global_rows = list(csv.DictReader(f))
    assert len(global_rows) == 8
    assert all("global" in r["summary_id"] for r in global_rows)


def test_parse_args_defaults():
    args = main._parse_args([])
    assert args.selection == 1
    assert args.scene == 1
    assert args.global_n == 1
    assert args.models == "gpt-4.1"


def test_parse_args_custom():
    args = main._parse_args(["--selection", "5", "--scene", "3", "--global", "2",
                             "--models", "gpt-4.1,llama-4-maverick"])
    assert args.selection == 5
    assert args.scene == 3
    assert args.global_n == 2
    assert args.models == "gpt-4.1,llama-4-maverick"


def test_parse_model_keys_rejects_unknown():
    with pytest.raises(SystemExit):
        main._parse_model_keys("gpt-4.1,not-a-real-model")


def test_parse_model_keys_accepts_known():
    assert main._parse_model_keys("gpt-4.1, llama-4-maverick") == [
        "gpt-4.1", "llama-4-maverick"
    ]
