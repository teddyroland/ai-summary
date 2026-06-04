# TODO

Status legend: `[ ]` pending, `[~]` in progress, `[x]` completed.

## Phase 1 — Data (user)

- [x] Create `data/meta.csv` and `data/plaintext/*.txt`.
- [x] Place `.env` in project root with `OPENAI_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION`.

## Phase 2 — Prototype (assistant)

- [x] Scaffold files: `code/`, `tests/`, `results/`, `temp/`, `README.md`, `TODO.md`, `CLAUDE.md`, `requirements.txt`.
- [x] `code/io_utils.py` — metadata loading, plaintext loading, JSONL append, CSV compilation + tests.
- [x] `code/prompts.py` — six prompt templates + render functions + tests.
- [x] `code/models.py` — multi-provider `call_model()` with retries + mocked tests.
- [x] `code/pipeline.py` — three stage orchestrators (passage_id / summary_id are 1-based integer counters set inline) + mocked end-to-end test.
- [x] `code/main.py` — CLI entry point.
- [x] Mocked smoke test: 49 tests green; `python code/main.py` with defaults produces correct CSVs in `results/`.
- [x] Real API debug: GPT-4.1 on the poetry text — verified.
- [x] Real API debug: Bedrock — LLaMA 4 Maverick verified working after switching to the cross-region inference profile ID (`us.meta.llama4-...`). See `CLAUDE.md`.

## Phase 3 — Review (user)

- [ ] Review prototype output and prompts; request revisions if needed.

## Transition to Phase 4

- [ ] **Manual step**: clear `temp/` (`rm temp/*.jsonl`) before the full production run so it starts from a clean cache.

## Phase 4 — Full run (assistant)

- [x] Run `python code/main.py --selection 5 --scene 5 --global 5 --models gpt-4.1,llama-4-maverick` against the full corpus.
- [x] Verify final CSVs in `results/`. 20 passages, 100 scene summaries, 90 global summaries on first pass; 12 LLaMA global slots lost to Bedrock throttling.
- [x] Backfill the throttled LLaMA global slots with `python code/backfill.py --model llama-4-maverick --stage global`. Final state: 20 passages, 100 scene summaries, 100 global summaries; no marker rows in any CSV.
