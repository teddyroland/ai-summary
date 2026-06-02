# AI Summary

A research pipeline that studies **controllable summarization** of literary texts: rather than generating a single "general" summary, the pipeline produces many summaries per text, each conditioned on a domain-specific requirement drawn from literary studies.

## Overview

For each text in the corpus, the pipeline runs three stages:

1. **Passage selection.** The model generates a short list of high-level interpretive questions about the text, then selects one passage that best addresses each question.
2. **Scene-setting summaries.** For each selected passage, the model generates a list of summary requirements focused on the passage itself (linguistic register, characters, setting, form, etc.) and writes a summary for each requirement.
3. **Global-theorizing summaries.** For each selected passage, the model generates a list of summary requirements that connect the passage to its author's larger body of work, its genre, period, or historical context, and writes a summary for each requirement.

Both prompts at every stage — the requirement-generation prompt and the passage/summary prompt — are run through the model, so all conditions are themselves model-generated rather than hand-written.

The pipeline supports multiple language models via a single abstraction. The current prototype targets OpenAI's GPT-4.1 (`gpt-4.1-2025-04-14`) and Meta's LLaMA 4 Maverick via Amazon Bedrock.

## Data

The corpus is not redistributed in this repository — populate `data/` locally before running the pipeline. See [`data/README.md`](data/README.md) for the expected file layout and `meta.csv` schema.

- `data/meta.csv` — one row per text, with columns `TEXT_ID, AUTHOR, TITLE, GENRE, FILENAME`.
- `data/plaintext/*.txt` — one plaintext file per text. Books of poetry are stored as a single file with individual poems separated by asterisks.

## Project layout

```
ai-summary/
├── data/             # corpus (meta.csv + plaintext files)
├── results/          # final per-stage CSVs (created at runtime)
├── temp/             # per-call JSONL caches (created at runtime)
├── code/             # all Python source (main.py + modules)
├── tests/            # pytest suite + fixtures
├── requirements.txt
├── README.md
├── TODO.md
└── CLAUDE.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

From the project root:

```bash
python code/main.py --selection 5 --scene 5 --global 5 \
  --models gpt-4.1,llama-4-maverick
```

Defaults are `--selection 1 --scene 1 --global 1 --models gpt-4.1`, useful for quick smoke tests. The pipeline writes per-call JSONL caches into `temp/` and compiles them into three CSVs in `results/`:

- `results/passages.csv` — every selected passage.
- `results/scene_summaries.csv` — every scene-setting summary.
- `results/global_summaries.csv` — every global-theorizing summary.

All CSVs include a `model` column and are sorted by `(model, passage_id, summary_id)`. The `temp/` cache is append-only across runs and not cleared automatically — see `TODO.md` for the cleanup step between debugging and full-dataset runs.

## Running the tests

```bash
pytest
```

Tests run with mocked model clients and do not make API calls.

## Style

The code is written to be read by humanities researchers and students who may not be familiar with advanced algorithms or uncommon libraries. We favor standard libraries (`pandas`, `pathlib`, `argparse`, `json`, `unittest.mock`), small intuitive functions, and interlinear comments. Implementation choices that are not obvious from the code — model IDs, JSON schema decisions, environment variables — are documented in `CLAUDE.md`.
