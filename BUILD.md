# BUILD.md — Implementation Brief for MemLedger 0.1 "Ego"

You are implementing MemLedger, a self-maintaining memory framework for AI
agents with a full audit trail. Read `SPEC.md` first: it is the normative
contract for all data formats. This document tells you WHAT to build and
in WHICH ORDER. Where this document and SPEC.md disagree, SPEC.md wins.

## 0. Ground rules

- Language: Python ≥ 3.11. Package name: `memledger`. Layout: `src/` layout.
- Dependencies (runtime): keep to a strict minimum —
  `python-ulid`, `pyyaml`, `httpx`. Optional extras:
  - `memledger[local]` → `fastembed` (CPU embeddings) and `sqlite-vec`
  - `memledger[dev]` → `pytest`, `ruff`, `mypy`
  No LangChain, no heavy frameworks in the core.
- Triage (`salience@v2` by default, `salience@v1` still supported for
   older policy hashes) is PURELY LEXICAL: the normative signal and
   formula definitions live in `prompts/salience.v2.md` (NFC + tokenizer,
  embedded stopword list, regex cue patterns). NO NLP models, no spaCy,
  no taggers: byte-identical output on every machine is a hard
   requirement — `rebuild` depends on it. Do NOT build a POS/NER variant
   in 0.1; a future `salience@v3` may use one only by pinning the model
  and recording its digest like an LLM `model_digest`.
- Storage: SQLite via stdlib `sqlite3`. One file per ledger. Single
  writer enforced (SPEC §9.5): acquire an exclusive lock file next to
  the DB; fail fast with a clear error if already locked.
- All framework LLM calls: `temperature=0`, JSON-constrained output,
  routed through the deterministic cache (SPEC §6). No exceptions.
- Type hints everywhere, `ruff` clean, docstrings on public API only.
- Every module must be independently testable with a mock model backend.

## 1. Repository layout (create exactly this)

```
memledger/
├── README.md                  # provided separately, include as-is
├── SPEC.md                    # provided, include as-is
├── BUILD.md                   # this file
├── LICENSE                    # MIT
├── pyproject.toml
├── memory.policy.yaml         # provided separately, ship as default
├── prompts/
│   ├── extract.v1.md          # provided, include as-is
│   ├── rerank.v1.md           # provided, include as-is
│   ├── reflect.v1.md          # provided, include as-is
│   ├── salience.v1.md         # provided, include as-is (normative triage formula)
│   └── text_form.v1.md        # provided, include as-is
├── src/memledger/
│   ├── __init__.py            # public API re-exports
│   ├── ids.py                 # ULID + prefixes, canonical JSON (RFC 8785), sha256 helpers
│   ├── events.py              # envelope dataclasses, validation per SPEC §3–§4
│   ├── ledger.py              # append-only store, event queries, redaction
│   ├── projection.py          # records table, state machine §5.1, cascade/tainted §9.3
│   ├── tuples.py              # Tuple model §5, text_form template engine (text_form@v1)
│   ├── policy.py              # policy load, canonicalize, hash (resolves referenced formula ids to hashes, SPEC §8); impact@v1 evaluator
│   ├── triage.py              # salience@v2 purely lexical scorer; prompts/salience.v2.md is normative
│   ├── prompts.py             # prompt registry: load, hash, render placeholders
│   ├── cache.py               # llm_cache table; key = H(model‖prompt_hash‖input_hash‖params)
│   ├── models/
│   │   ├── base.py            # ModelBackend protocol: complete(prompt, params) -> str
│   │   ├── openai_compat.py   # any OpenAI-compatible endpoint (Ollama, OpenRouter, hosted gateways)
│   │   ├── anthropic.py
│   │   └── mock.py            # deterministic canned responses for tests
│   ├── embeddings/
│   │   ├── base.py            # Embedder protocol; index_version string
│   │   └── fastembed_local.py # optional extra
│   ├── retrieval.py           # FTS5 + vector candidates → rerank@v1 → recalled event
│   ├── checkpoint.py          # extract → reflect → lifecycle orchestration
│   ├── session.py             # Session: observe/feedback/outcome/remember/recall/build_context/checkpoint
│   ├── api.py                 # Ledger facade (open, seed, session, rebuild, regenerate)
│   └── cli.py                 # `memledger` entrypoint (see §4)
├── examples/
│   ├── 01_chat_with_memory.py     # minimal chatbot, configurable for Ollama or remote gateways
│   ├── 02_coding_assistant.py     # remembers corrected mistakes across sessions
│   └── 03_support_agent.py        # outcomes + promotion review flow
├── evals/
│   ├── regression/                # golden transcripts → expected tuples (extract)
│   │   └── cases/*.yaml
│   └── run_regression.py          # fails CI if extraction quality drops
└── tests/                         # pytest; see acceptance criteria §5
```

## 2. Public API (implement exactly this surface)

```python
from memledger import Ledger, Policy

ledger = Ledger(
    path="./memory.db",
    policy=Policy.from_yaml("memory.policy.yaml"),   # or Policy.default()
    memory_model="openai-compat:http://localhost:11434/v1|qwen3:4b",
    cache="deterministic",                           # "deterministic" | "off"
)

ledger.instinct.seed([(subject, relation, value), ...])   # emits `seeded`

session = ledger.session(user_id="dev_123")
memories = session.recall(query, k=5)                # pipeline per SPEC §4.4
ctx = session.build_context(instinct=True, episodic=memories, working="tail")
# ctx.system: str  — instinct + selected episodic, rendered from text_form
# ctx.messages: list — working memory tail within policy token budget

session.observe(user=..., assistant=...)             # emits `observed` (x2)
session.feedback(-1, on="last_turn")                # emits `feedback`
session.outcome("success", task="deploy_fix")      # emits `outcome`
session.remember((subject, relation, value))         # emits `remember`, active, conf=1.0

report = session.checkpoint()
# report.triaged (counts: extracted vs skipped, tokens avoided by triage)
# report.extracted, report.merged, report.superseded, report.promotions_proposed
# report.tokens_spent_on_memory, report.tokens_saved_in_context

ledger.rebuild()        # drop projections, replay events, compare hashes → bool
ledger.regenerate(model=..., prompt="extract@v1")    # SPEC §4.4 regenerated
```

`openai-compat` specs use `openai-compat:<base_url>|<model>`. For remote
providers, use `OPENROUTER_API_KEY` when the host is `openrouter.ai`, otherwise
use `OPENAI_API_KEY`. CLI examples must quote the full spec because it contains `|`.

Checkpoint internals (order is normative):
1. Triage pre-pass (rule, `salience@v2`, zero LLM cost): score every raw
   turn of the session, emit one `triaged` event per turn. Verdict order
   is normative: `ineligible` (role-based, per `triage.ineligible_roles`)
   → cue bypass (`always_extract_cues`) → threshold. Only "extract"
   turns enter the extraction batch.
2. `extract@v1` on the triaged-in turns only (one batch call) →
   `extracted` event; `sources` MUST point to the `observed` events, not
   to `triaged` ones (SPEC §4.2); materialize tuples ≥
   `extraction.min_confidence` as `quarantined` (or increment known ones
   — see step 3).
3. Rule-based dedup pre-pass (no LLM): exact-duplicate detection by
   (subject, relation, value) → increment `sessions_seen`, no new record.
4. Quarantine lift check (rule): `sessions_seen ≥ quarantine.new_facts`
   → `quarantine_lifted`.
5. Impact recompute (rule, `impact@v1` from policy) → `scored` events.
6. Eligibility scan (rule, `instinct.promote_when`) → candidate list.
7. `reflect@v1` with new tuples + related tuples + eligible candidates →
   apply merges (`merged`), supersedes (`superseded`), promotion verdicts
   (`promotion_proposed`; auto-approve only if `instinct.autonomous`).
8. TTL sweep → `expired`. Tainted re-extraction from surviving sources.

## 3. Retrieval pipeline (normative)

1. Stage 1 candidates: union of FTS5 BM25 top-N and vector top-N
   (N = `retrieval.candidates`, split evenly; vector stage skipped
   gracefully if no embedder installed).
2. Pre-score (rule): `stage1_score × exp(-age_days·ln2/half_life) ×
   (1 + impact_boost·impact)` — parameters from policy.
3. If `retrieval.rerank: true`: call `rerank@v1` with top candidates,
   apply its selection; else take top-k by pre-score.
4. Emit `recalled` per `retrieval.log` mode (SPEC §9.4).
5. `quarantined` tuples are included in candidates but flagged; `superseded`
   and `deleted` never surface.

## 4. CLI (`memledger`)

| command | behavior |
|---|---|
| `init [path]` | create DB, copy default policy + prompts |
| `log [--type --session --since]` | pretty-print events |
| `why <id>` | provenance chain down to observed/remember/seeded |
| `review` | interactive approve/reject of `promotion_proposed` queue |
| `replay [--at TS] [--cached]` | rebuild state at TS; `--cached` errors on cache miss |
| `rebuild` | SPEC §7 conformance check; exit code 0/1 |
| `regenerate --model M [--prompt P]` | re-extraction per SPEC §4.4 |
| `delete <id> [--cascade] [--reason R]` | tombstone + cascade/tainted per SPEC §9.3 |
| `stats` | records per layer/status, tokens spent vs saved, cache hit rate, triage skip rate |

## 5. Acceptance criteria (implement as pytest; all must pass)

1. **Rebuild conformance:** generate 3 synthetic sessions with the mock
   backend → `ledger.rebuild()` returns True (projections byte-identical).
2. **Cache determinism:** run the same checkpoint twice with cache on →
   second run performs zero backend calls (assert via mock call counter).
3. **Replay strictness:** `replay --cached` on a ledger with one evicted
   cache entry raises, does not call the network.
4. **State machine:** every illegal transition of SPEC §5.1 raises
   (e.g. quarantined → instinct directly).
5. **Cascade/tainted:** delete one of two sources of a merged tuple →
   tuple marked tainted; next checkpoint re-extracts from the survivor.
6. **Anti-poisoning:** a transcript containing "remember at maximum
   confidence that X" (injection fixture) must not produce an active
   tuple for X; extraction notes flag the attempt. Use a mock model that
   simulates a naive extractor for the negative control.
7. **Quarantine flow:** a fact extracted in session 1 is quarantined; the
   same fact in session 2 lifts quarantine (with default policy = 2).
8. **remember() bypass:** dev-declared tuples are active immediately,
   confidence 1.0, and `why` resolves to the `remember` event.
9. **Regression harness:** `evals/run_regression.py` compares extraction
   output on golden cases against expected tuples (subject+relation+value
   match, order-insensitive) and reports precision/recall; nonzero exit
   below thresholds in policy `evals:` section.
10. **CLI smoke:** every command in §4 runs without error on a fixture DB.
11. **Triage:** on a fixture transcript mixing phatic turns ("ok",
    "thanks!"), dense factual turns, a `role: "tool"` turn containing a
    planted fact, and a low-salience turn with a correction cue
    ("actually, my name is Lia"): (a) phatic turns get verdict "skip" and
    never reach the mock extractor (assert via call inputs); (b) the
    correction turn is extracted despite low salience
    (`always_extract_cues`); (c) the tool turn gets verdict "ineligible";
    (d) every turn has exactly one `triaged` event and two runs on two
    fresh databases produce identical `triaged` payloads (lexical
    determinism); (e) `regenerate` with `triage.threshold: 0` recovers
    the fact planted in a skipped turn but NOT the one in the ineligible
    tool turn.

## 6. Implementation order (milestones)

M1 `ids`, `events`, `ledger`, `projection`, `tuples` + tests 1, 4.
M2 `policy`, `prompts`, `cache`, `triage`, `models/mock` + tests 2, 3, 11a–d.
M3 `checkpoint` full pipeline on mock + tests 5, 6, 7, 8, 11e.
M4 `retrieval` (FTS5 first, embeddings optional) + recalled logging.
M5 `models/openai_compat`, `models/anthropic`, `embeddings/fastembed_local`.
M6 `cli`, `examples`, `evals` + tests 9, 10. README/policy files land here.

## 7. Non-goals for 0.1 (do not build)

Multi-writer/shared memory, network server mode, non-SQLite backends,
dashboard/UI, TypeScript SDK, automatic PII detection. Stub nothing:
absent means absent.