"""Tests for ids.py — passage and summary ID formatting."""

import pytest

import ids


def test_passage_id_format():
    assert ids.passage_id("01", "gpt41", 3) == "p_01_gpt41_03"


def test_passage_id_pads_single_digit():
    # 1 -> "01" so IDs sort lexicographically.
    assert ids.passage_id("02", "llama4m", 1) == "p_02_llama4m_01"


def test_passage_ids_disambiguate_across_models():
    # Same text and counter, different model -> different ID.
    a = ids.passage_id("02", "gpt41", 1)
    b = ids.passage_id("02", "llama4m", 1)
    assert a != b


def test_summary_id_scene():
    assert ids.summary_id("01", 3, "gpt41", "scene", 2) == "s_01_03_gpt41_scene_02"


def test_summary_id_global():
    assert ids.summary_id("01", 3, "llama4m", "global", 2) == "s_01_03_llama4m_global_02"


def test_summary_id_rejects_unknown_kind():
    with pytest.raises(ValueError):
        ids.summary_id("01", 3, "gpt41", "scene-setting", 2)


def test_summary_ids_sort_lexicographically_with_zero_padding():
    # If we forgot the padding, "s_01_01_gpt41_scene_10" would sort before
    # "s_01_01_gpt41_scene_2" — this test catches that regression.
    ordered = [
        ids.summary_id("01", 1, "gpt41", "scene", n) for n in (1, 2, 10)
    ]
    assert ordered == sorted(ordered)
