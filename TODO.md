# TODO

Status legend: `[ ]` pending, `[~]` in progress, `[x]` completed.

## Phase 1 — Data (user)

- [x] Create `data/meta.csv` and `data/plaintext/*.txt`.
- [x] Place `.env` in project root with `OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION`.

## Phase 2 — Prototype (assistant)

- [x] Scaffold files: `code/`, `tests/`, `results/`, `temp/`, `README.md`, `TODO.md`, `CLAUDE.md`, `requirements.txt`.
- [x] `code/ids.py` — passage_id / summary_id helpers + tests.
- [x] `code/io_utils.py` — metadata loading, plaintext loading, JSONL append, CSV compilation + tests.
- [x] `code/prompts.py` — six prompt templates + render functions + tests.
- [x] `code/models.py` — multi-provider `call_model()` with retries + mocked tests.
- [x] `code/pipeline.py` — three stage orchestrators + mocked end-to-end test.
- [x] `code/main.py` — CLI entry point.
- [x] Mocked smoke test: 44 tests green; `python code/main.py` with defaults produces correct CSVs in `results/`.
- [x] Real API debug: GPT-4.1 on the poetry text with `--selection 1 --scene 1 --global 1` — verified.
- [x] Real API debug: Bedrock — LLaMA 4 Maverick verified working after switching to the cross-region inference profile ID (`us.meta.llama4-...`). See `CLAUDE.md`.

## Phase 3 — Review (user)

- [ ] Review prototype output and prompts; request revisions if needed.

## Transition to Phase 4

- [ ] **Manual step**: clear `temp/` (`rm temp/*.jsonl`) before the full production run so it starts from a clean cache.

## Phase 4 — Full run (assistant)

- [ ] Run `python code/main.py --selection 5 --scene 5 --global 5 --models gpt-4.1,llama-4-maverick` against the full corpus.
- [ ] Verify final CSVs in `results/`.
