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

| Key | Provider | Model ID | Status (as of 2026-06-03) |
|---|---|---|---|
| `gpt-4.1` | OpenAI | `gpt-4.1-2025-04-14` | **Verified working.** Snapshot ID used (not the rolling `gpt-4.1` alias) for reproducibility. |
| `llama-4-maverick` | Bedrock | `us.meta.llama4-maverick-17b-instruct-v1:0` | **Verified working.** On-demand requires the **cross-region inference profile** ID (`us.` prefix), not the bare regional ID (`meta.llama4-...`). Bedrock returns `ValidationException` if the regional ID is used on-demand. |

Generation parameters across all models: `temperature=1.0`, `top_p=1.0`. No seed.

## JSON outputs

All four prompt types return JSON. Single field per response, single shape:

- `questions` step (Stage 1 of selection): `{"questions": ["...", "..."]}`
- `requirements` step (Stage 1 of scene and global summaries): `{"requirements": ["...", "..."]}`
- `passage` step (Stage 2 of selection): `{"passage": "..."}`
- `summary` step (Stage 2 of scene and global summaries): `{"summary": "..."}`

OpenAI calls use the SDK's `response_format={"type": "json_schema", "json_schema": ...}` (Structured Outputs) for guaranteed-valid JSON.

Bedrock calls use the **Converse API** (`bedrock-runtime.converse`) and instruct JSON via the system prompt. The Converse API is the unified way to talk to Anthropic and Meta models on Bedrock — same request shape, same response shape. Three defensive layers in `code/models.py` handle real-world Bedrock quirks:

1. `_extract_json` parses with `json.loads(..., strict=False)` so literal control characters (raw newlines inside string values, common from LLaMA) don't blow up parsing. It also strips ```` ```json ``` ```` fences and regex-extracts the first `{...}` block as a last resort.
2. `_ensure_bedrock_shape` coerces bare values into the expected single-key dict. LLaMA sometimes returns `"passage text"` instead of `{"passage": "..."}`; we wrap it rather than retrying — the content is correct, only the wrapper is missing.
3. On a true parse / shape failure, the call retries once with a stricter "JSON only, no prose, no code fences" instruction.

## ID formats

`passage_id` and `summary_id` are **1-based integers**, not unique strings. Uniqueness is established by the combination of columns, not the ID alone.

- `passage_id` counter resets per `(text_id, model)`. Uniqueness within a row: `(text_id, model, passage_id)`.
- `summary_id` counter resets per `(text_id, model, passage_id, summary_type)`. Uniqueness within a row: `(text_id, model, passage_id, summary_type, summary_id)`.
- `summary_type` is `"scene"` or `"global"`.

`compile_csv` dedupes on these compound keys (`PASSAGE_DEDUP_BY` / `SUMMARY_DEDUP_BY` in `code/main.py`).

## `temp/` and `results/`

Per-call API responses are appended to JSONL files in `temp/` as the pipeline runs:

- `temp/{model_key}_passages.jsonl`
- `temp/{model_key}_scene.jsonl`
- `temp/{model_key}_global.jsonl`

`main.py` does **not** clear `temp/` at startup. Reruns append, and `io_utils.compile_csv()` dedupes by the compound key (`PASSAGE_DEDUP_BY` / `SUMMARY_DEDUP_BY`, last-wins) when producing the final CSVs in `results/`. Compound dedup is necessary because passage_id and summary_id are no longer unique strings — `passage_id=1` appears across `(text_id, model)` pairs.

This lets us iterate during Phase 2 debugging without losing earlier records.

Between Phase 2 (prototype) and Phase 4 (full run), the user manually clears `temp/` (`rm temp/*.jsonl`) so the production run is clean. See `TODO.md`.

## Passage selection: quality checks

Stage 4a (passage selection) is the only place where the model is supposed to copy from the source, and the only place we ask for a specific length. Two safeguards run together after every passage response:

1. **Verbatim substring check.** `pipeline.is_verbatim_excerpt` normalizes whitespace and case on both response and source, then tests whether the response is a substring. Catches editorial commentary ("This poem exemplifies…") because commentary won't be in the source.
2. **Word-count check.** `pipeline.is_in_word_range` requires `PASSAGE_MIN_WORDS=100` ≤ words ≤ `PASSAGE_MAX_WORDS=300`. The prompt asks for "a 100-300 word passage"; this enforces it. The upper bound is generous enough to accommodate novel scenes, which often run a few hundred words.

Both checks run in `pipeline._select_validated_passage()`. If either fails, the function appends the relevant follow-up instruction(s) — `PASSAGE_VERBATIM_RETRY_INSTRUCTION` and/or `passage_length_retry_instruction(actual_count)` — and re-prompts once. If the second response still fails, the pipeline accepts it and prints a warning. Grep run logs for `[passage check]` to find rows that failed.

Two reinforcements help the model behave on the first try:
- `PASSAGE_PROMPT` ends with the hard-constraint framing: *"…between 100 and 300 words. The length is a hard constraint: count the words and do not exceed 300. Return only a verbatim excerpt…"* — placed at the end of the user prompt, where models attend most.
- The length retry instruction includes the actual word count the model emitted, so it can correct in the right direction.

Normalization for the verbatim check is intentionally loose (whitespace + case only). It will not catch punctuation modernization (curly → straight quotes, em-dash → hyphen). If real false rejections turn up, extend the `normalize` helper inside `is_verbatim_excerpt`.

## Pipeline shape

Two stage functions in `code/pipeline.py`, both with the same iteration shape:

1. `run_passage_selection(meta_rows, model_keys, selection_n)` → `selection_n` passages per (model, text).
2. `run_summaries(kind, passages, meta_rows, model_keys, summary_n)` → `summary_n` summaries per (model, passage). `kind` is `"scene"` or `"global"`; it selects the Stage-1 requirements prompt and is recorded in the `summary_type` column. Both summary types share the Stage-2 prompt template (`prompts.render_summary_prompt`), so a single function covers Stages 4b and 4c.

Both functions first call the model to produce a list of conditions (interpretive questions or summary requirements), then iterate over that list calling the model a second time to produce the passage or summary.

## Code directory

All Python lives in `code/` including `main.py`. Invocation is `python code/main.py` from the project root. Modules import each other with plain `from prompts import ...` — Python adds the script's parent directory to `sys.path[0]` when you run a script directly. No `__init__.py`; we're treating `code/` as a script directory, not a package.

Side note: `code` shadows Python's stdlib `code` module if treated as a package. Fine here since we never `import code` from the stdlib. Don't rename without good reason.

Path resolution: each module computes the project root once via `ROOT = Path(__file__).resolve().parent.parent` so `data/`, `results/`, `temp/` work regardless of the caller's working directory.
