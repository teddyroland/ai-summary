"""Generate unique IDs for passages and summaries.

The pipeline produces many records that need to be traced back to the text,
model, and passage they came from. We use compact, human-readable IDs so that
anyone scanning a CSV can tell at a glance which text and which model each row
belongs to.

Examples:
    passage_id("01", "gpt41", 3)
        -> "p_01_gpt41_03"
    summary_id("01", 3, "gpt41", "scene", 2)
        -> "s_01_03_gpt41_scene_02"
    summary_id("01", 3, "llama4m", "global", 2)
        -> "s_01_03_llama4m_global_02"

All counters are zero-padded to two digits. This is enough for the default
pipeline settings (1 or 5 passages/summaries per text). If counts ever exceed
99 the IDs will sort incorrectly and the padding should be widened.
"""

# Counters in the IDs are padded so they sort lexicographically the same way
# they sort numerically. Two digits handle counts up to 99.
COUNTER_WIDTH = 2

# Allowed values for the summary kind. Keeping this as a constant catches typos
# in callers (e.g. passing "scene-setting" instead of "scene").
SUMMARY_KINDS = {"scene", "global"}


def passage_id(text_id: str, model_short: str, n: int) -> str:
    """Build a passage ID like "p_01_gpt41_03".

    Args:
        text_id: The TEXT_ID from meta.csv (already a zero-padded string).
        model_short: Short tag for the model that selected the passage; see
            models.MODEL_REGISTRY[...]["short"].
        n: The 1-based index of the passage within (text, model).
    """
    return f"p_{text_id}_{model_short}_{n:0{COUNTER_WIDTH}d}"


def summary_id(
    text_id: str, passage_n: int, model_short: str, kind: str, n: int
) -> str:
    """Build a summary ID like "s_01_03_gpt41_scene_02".

    Args:
        text_id: The TEXT_ID from meta.csv.
        passage_n: The 1-based index of the passage this summary belongs to.
        model_short: Short tag for the model that wrote the summary.
        kind: Either "scene" or "global"; encoded as an infix in the ID so
            scene and global summaries can be distinguished by ID alone.
        n: The 1-based index of the summary within the passage.
    """
    if kind not in SUMMARY_KINDS:
        raise ValueError(f"summary kind must be one of {SUMMARY_KINDS}, got {kind!r}")
    return (
        f"s_{text_id}"
        f"_{passage_n:0{COUNTER_WIDTH}d}"
        f"_{model_short}"
        f"_{kind}"
        f"_{n:0{COUNTER_WIDTH}d}"
    )
