# CLAUDE.md

Implementation context for future sessions. Things that are not obvious from reading the code, not duplicated in `README.md` or `TODO.md`.

## Goals and style

- Audience: humanities researcher (user) and students adapting the code in a classroom setting.
- Priorities: **interpretability first, then efficiency**. Standard libraries, small functions, interlinear comments. No `tenacity`, `pydantic`, or other less-familiar dependencies unless there's a strong reason.
- The code should make the *pipeline structure* visible — three stages, two prompts per stage, JSON outputs at every step.

## Environment file

`.env` lives in the project root with three variables:

```
OPENAI_API_KEY=sk-proj-...
AWS_BEARER_TOKEN_BEDROCK=ABSK...
AWS_REGION=us-west-2
```

The Bedrock variable name **must** be `AWS_BEARER_TOKEN_BEDROCK` — boto3 picks it up automatically when constructing a `bedrock-runtime` client. `AWS_API_KEY` is not recognized.

`code/models.py` loads `.env` via `dotenv.load_dotenv()` at import. No `.env.example` is checked in (per user preference) and the README does not document setup.

## Model registry

| Key | Provider | Model ID | Short tag | Status (as of 2026-06-02) |
|---|---|---|---|---|
| `gpt-4.1` | OpenAI | `gpt-4.1-2025-04-14` | `gpt41` | **Verified working.** Snapshot ID used (not the rolling `gpt-4.1` alias) for reproducibility. |
| `llama-4-maverick` | Bedrock | `us.meta.llama4-maverick-17b-instruct-v1:0` | `llama4m` | **Verified working.** On-demand requires the **cross-region inference profile** ID (`us.` prefix), not the bare regional ID (`meta.llama4-...`). Bedrock returns `ValidationException` if the regional ID is used on-demand. |

The `short` tag is used as an infix in passage and summary IDs (see "ID formats" below) so that records from different models are distinguishable by ID alone.

Generation parameters across all models: `temperature=1.0`, `top_p=1.0`. No seed.

## JSON outputs

All four prompt types return JSON. Single field per response, single shape:

- `questions` step (Stage 1 of selection): `{"questions": ["...", "..."]}`
- `requirements` step (Stage 1 of scene and global summaries): `{"requirements": ["...", "..."]}`
- `passage` step (Stage 2 of selection): `{"passage": "..."}`
- `summary` step (Stage 2 of scene and global summaries): `{"summary": "..."}`

OpenAI calls use the SDK's `response_format={"type": "json_schema", "json_schema": ...}` (Structured Outputs) for guaranteed-valid JSON.

Bedrock calls use the **Converse API** (`bedrock-runtime.converse`) and instruct JSON via the system prompt. The Converse API is the unified way to talk to Anthropic and Meta models on Bedrock — same request shape, same response shape. We parse with `json.loads`; on parse failure we retry once with a stricter "JSON only, no prose" instruction.

## ID formats

The model's short tag is part of every ID so that records from different models can be distinguished by ID alone (otherwise both gpt-4.1 and llama-4-maverick would label their first text-01 selection `p_01_01`).

- Passage ID: `p_{TEXT_ID}_{model_short}_{NN}` — e.g. `p_01_gpt41_03`, `p_01_llama4m_03`.
- Scene summary ID: `s_{TEXT_ID}_{passage_NN}_{model_short}_scene_{NN}` — e.g. `s_01_03_gpt41_scene_02`.
- Global summary ID: `s_{TEXT_ID}_{passage_NN}_{model_short}_global_{NN}` — e.g. `s_01_03_llama4m_global_02`.

Counters are zero-padded to two digits. If counts ever exceed 99 (i.e. `--selection 100+`), the IDs lose their nice lexicographic sort and we'll need to widen the padding. Not a concern at the defaults (1) or planned settings (5).

## `temp/` and `results/`

Per-call API responses are appended to JSONL files in `temp/` as the pipeline runs:

- `temp/{model_key}_passages.jsonl`
- `temp/{model_key}_scene.jsonl`
- `temp/{model_key}_global.jsonl`

`main.py` does **not** clear `temp/` at startup. Reruns append, and `io_utils.compile_csv()` dedupes by the compound `(model, id)` key (last-wins) when producing the final CSVs in `results/`. Strictly speaking, the model short is already baked into every ID, so a single-column dedup on `passage_id` (or `summary_id`) would also work. We keep the compound key as a safety net in case the model short is ever changed or omitted.

This lets us iterate during Phase 2 debugging without losing earlier records.

Between Phase 2 (prototype) and Phase 4 (full run), the user manually clears `temp/` (`rm temp/*.jsonl`) so the production run is clean. See `TODO.md`.

## Passage selection: verbatim check

Stage 4a (passage selection) is the only place where the model is supposed to copy from the source. Two safeguards reduce the chance of editorial commentary leaking into the `passage` field:

1. **Stricter prompt language.** `PASSAGE_PROMPT` ends with: *"Return ONLY a verbatim excerpt — a literal, consecutive copy of words taken directly from the source. Do not add commentary, analysis, framing, headings, or explanation."* This is the closest instruction to where the model generates, which is where LLMs attend most.
2. **Substring validation with one re-prompt.** After each passage response, `pipeline._select_passage_with_verbatim_check()` normalizes whitespace and case on both the response and the source, then tests whether the response is a substring. If not, it appends `PASSAGE_VERBATIM_RETRY_INSTRUCTION` to the prompt and re-calls once. If the second response also fails, the pipeline accepts it but prints a warning to stdout (no exception — long runs should not break on a single uncooperative model). Operators can grep the run log for `[verbatim check]` to find rows that failed both attempts.

Normalization is intentionally loose (whitespace + case only). It will not catch punctuation modernization (curly → straight quotes, em-dash → hyphen). If real false rejections turn up, extend `_normalize_for_verbatim_check`.

## Pipeline shape

Three stage functions in `code/pipeline.py`, all taking the same iteration shape:

1. `run_passage_selection(meta_rows, model_keys, n)` → SELECTION_NUMBER passages per (model, text).
2. `run_scene_summaries(passages, meta_rows, model_keys, n)` → SCENE_NUMBER scene summaries per (model, passage).
3. `run_global_summaries(passages, meta_rows, model_keys, n)` → GLOBAL_NUMBER global summaries per (model, passage).

All three first call the model to produce a list of conditions (questions or requirements), then iterate over that list calling the model a second time to produce the passage or summary.

Stage 2 of (scene) and Stage 2 of (global) use **the same prompt template** — `prompts.render_summary_prompt()` is reused.

## Code directory

All Python lives in `code/` including `main.py`. Invocation is `python code/main.py` from the project root. Modules import each other with plain `from prompts import ...` — Python adds the script's parent directory to `sys.path[0]` when you run a script directly. No `__init__.py`; we're treating `code/` as a script directory, not a package.

Side note: `code` shadows Python's stdlib `code` module if treated as a package. Fine here since we never `import code` from the stdlib. Don't rename without good reason.

Path resolution: each module computes the project root once via `ROOT = Path(__file__).resolve().parent.parent` so `data/`, `results/`, `temp/` work regardless of the caller's working directory.
