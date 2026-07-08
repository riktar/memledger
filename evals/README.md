# Evals

## LoCoMo

`evals/run_locomo.py` runs MemLedger on the official LoCoMo 10-conversation QA
benchmark.

What the runner does:

- downloads `locomo10.json` from the official repository if it is missing;
- ingests LoCoMo conversations into MemLedger sessions, optionally batching
  multiple source sessions into one checkpoint;
- checkpoints through the normal MemLedger path, with optional reflection
  disabled for faster ingest;
- answers each QA item from retrieved memories plus their original evidence turns;
- scores predictions with the official LoCoMo QA logic, including Porter-stemmed
  token F1, category-wise accuracy, retrieval recall by category, memory-distance
  buckets and context-length buckets;
- logs progress with elapsed time and ETA during both ingest and QA.

Quick smoke run:

```bash
. .venv/bin/activate
python evals/run_locomo.py \
  --memory-model 'openai-compat:http://localhost:11434/v1|qwen3:4b' \
  --sample-limit 1 \
  --question-limit 20 \
  --checkpoint-every 2 \
  --disable-reflection \
  --disable-retrieval-rerank \
  --disable-retrieval-log \
  --log-every-questions 5 \
  --output-file evals/locomo/results/smoke.json
```

Full 10-conversation run:

```bash
. .venv/bin/activate
python evals/run_locomo.py \
  --memory-model 'openai-compat:http://localhost:11434/v1|qwen3:4b' \
  --checkpoint-every 3 \
  --log-every-questions 25 \
  --output-file evals/locomo/results/locomo10-qwen3.json
```

Useful flags:

- `--qa-model`: use a different model for final QA answers than for memory extraction.
- `--checkpoint-every`: group multiple LoCoMo source sessions into one MemLedger checkpoint to reduce ingest cost.
- `--disable-reflection`: skip the second LLM pass used for semantic merge, supersede and promotion proposals.
- `--disable-retrieval-rerank`: skip the per-question LLM reranker inside `session.recall()`.
- `--disable-retrieval-log`: stop writing `recalled` events during QA.
- `--retrieval-k`: change how many memories are retrieved per question.
- `--sample-ids conv-26,conv-31`: run only specific conversations.
- `--categories 2,5`: restrict evaluation to selected LoCoMo QA categories.
- `--reingest`: ignore cached SQLite ledgers and rebuild them from scratch.
- `--stats-file`: write the official LoCoMo metrics summary to a separate JSON file.
- `--log-every-questions`: control how often the runner prints QA progress and ETA.

Notes:

- Quote `openai-compat:...|...` model specs because `|` is a shell pipe.
- Install benchmark dependencies with `pip install -e '.[evals]'` if your env does
  not already include `nltk`.
- A full run is expensive: 10 conversations means roughly 2k QA items plus all
  checkpoint-time extraction and reflection calls.
- `--checkpoint-every > 1` is the main speed lever for ingest, but it also changes
  session granularity inside MemLedger, so quarantine and repeated-session signals
  are measured per batch instead of per original LoCoMo source session.
- `--disable-reflection` means extraction still happens, quarantine still happens,
  and QA still runs; what you lose is the reflection pass that proposes semantic
  merges, contradiction cleanup and instinct promotions.
- `--disable-retrieval-rerank` is usually the next biggest speed lever after ingest optimizations,
  because otherwise every QA can pay for both a retrieval LLM call and an answer LLM call.
- `--disable-retrieval-log` is a smaller optimization: it reduces SQLite writes during QA,
  but also removes `recalled` audit events from the ledger.
- The runner writes two files by default: the detailed benchmark output and a
  sibling `*_stats.json` file containing the official aggregate metrics.
