# SPEC.md — MemLedger Ledger Format v0.1 "Ego"

**Profile: Ego** — single agent, single file, single writer. This version
covers the construction and maintenance of one agent's memory. Shared
multi-agent memory is out of scope (see §9.5) and will define the next
profile.

This document is the **contract** for the data format. Any SDK (Python,
TypeScript, other) that reads/writes this format is a conforming client.
Code implements the spec, never the other way around.

---

## 1. Principles

1. **The ledger is the source of truth.** An append-only event log.
   Memory state (working / episodic / instinct) is a projection,
   reconstructible by replaying events.
2. **No event is ever modified or removed.** Deletion itself is an event
   (`deleted`) plus a tombstone on the projection.
3. **Every event has an explicit cause** (`cause`): a policy, a rule,
   a developer signal, or a fully described LLM call.
4. **Cache determinism:** every framework LLM call is identified by
   `cache_key = H(model ∥ prompt_hash ∥ input_hash ∥ params)`.
   Same key ⇒ same output (served from cache). Replaying the ledger with
   a warm cache issues zero new calls.
5. **Full provenance:** every derived record points to its sources, down
   to raw turns. Deletion can cascade.

---

## 2. Identifiers and hashing

- **Event ID:** ULID (time-sortable, 26 chars). Readability prefixes:
  - `ev_` event · `tu_` tuple · `ep_` episodic record · `in_` instinct record
  - `se_` session · `pr_` prompt · `po_` policy
- **Hashes:** SHA-256, lowercase hex, truncatable to 12 chars in CLI output.
- **Timestamps:** ISO-8601 UTC with milliseconds (`2026-07-07T14:03:22.117Z`).
- **Canonicalization for hashing:** JSON with sorted keys, UTF-8, no
  whitespace (JCS / RFC 8785). Applies to `input_hash`, `policy_hash`,
  `prompt_hash`.

---

## 3. Event envelope

Every event is a JSON object with this envelope. Fields marked * are
required.

```json
{
  "id": "ev_01J9ZK3M8Q...",
  "ts": "2026-07-07T14:03:22.117Z",
  "session": "se_01J9ZK...",
  "user": "dev_123",
  "type": "extracted",
  "actor": "llm",
  "cause": {
    "kind": "llm",
    "ref": "extract@v1",
    "detail": "checkpoint end-of-session"
  },
  "policy_hash": "a3f2c1…",
  "spec_version": "0.1",
  "payload": { ... },
  "llm": { ... },
  "sources": ["ev_...", "tu_..."]
}
```

Rule: `sources` is **required** for every event that creates or modifies a
derived record (`extracted`, `merged`, `promoted`, `compacted`,
`superseded`, `regenerated`).

---

## 4. Event types

### 4.1 Ingestion and triage (never LLM)
All events in this section have `actor: "dev"` or `"rule"` — never LLM.
Ingestion events are emitted per-turn; `triaged` is emitted at
checkpoint, one per raw turn.
| type       | payload                                            | notes |
|------------|----------------------------------------------------|-------|
| `observed` | `{role: "user"\|"assistant"\|"tool", text, turn}`  | raw turn in working memory |
| `triaged`  | `{turn: ev_id, salience, signals: {...}, verdict: "extract"\|"skip"\|"ineligible", formula: "salience@v1"}` | rule-based pre-filter, zero LLM cost. `salience@v1` is normatively defined in the registry (`prompts/salience.v1.md`, §8) and its hash is folded into `policy_hash`. Verdicts: `skip` = below threshold this time (recoverable via `regenerate`); `ineligible` = this class of turns is never extracted by policy (e.g. `role: "tool"`), not recovered by threshold changes. Skipped and ineligible turns remain in the ledger as provenance and regeneration material — **triage prunes LLM work, never memory.** |
| `feedback` | `{value: -1\|+1, on: event_id}`                    | explicit signal |
| `outcome`  | `{status: "success"\|"failure", task, on: [ids]}`  | outcome declared by the dev |
| `remember` | `{tuple: {...}}` (§5)                              | fact declared via API by the dev; born `active`, skips quarantine. **Warning:** never expose `remember` to end-user input — it bypasses anti-poisoning. |

### 4.2 Checkpoint (LLM, batch)

The extraction batch contains only turns whose `triaged` verdict is
"extract". Turns matching `triage.always_extract_cues` (constraints,
corrections, preferences) are always extracted regardless of salience.

Provenance rule: the `sources` of an `extracted` event point to the
`observed` turns themselves — the semantic raw material — never to
`triaged` events. `why` therefore reaches raw turns directly. The triage
pass is referenced in `cause.detail` (the checkpoint id), so "why was
this turn in the batch" remains answerable by querying the `triaged`
events of that checkpoint, without provenance chains traversing them.

| type        | payload |
|-------------|---------|
| `extracted` | `{tuples: [Tuple...], rejected: [{tuple, confidence}]}` — tuples below `min_confidence` are logged but not materialized |
| `merged`    | `{into: tu_id, from: [tu_ids], resulting_value}` — semantic dedup |
| `superseded`| `{old: tu_id, new: tu_id, reason}` — contradiction resolution |
| `compacted` | `{summary_of: [ev_ids], record: ep_id}` — summary of raw turns |
| `scored`    | `{record: id, impact_delta, impact_total, formula: "impact@v1"}` |

### 4.3 Lifecycle
| type                 | payload |
|----------------------|---------|
| `promotion_proposed` | `{record: id, to: "instinct", rationale}` |
| `promotion_approved` | `{proposal: ev_id, by: "dev"\|"auto"}` — `auto` only if `instinct.autonomous: true` |
| `promotion_rejected` | `{proposal: ev_id, by, reason}` |
| `quarantine_lifted`  | `{record: id, confirmed_in_sessions: n}` |
| `expired`            | `{record: id, ttl: "90d"}` |
| `deleted`            | `{record: id, cascade: [ids], reason: "gdpr"\|"poisoning"\|"manual"}` |
| `seeded`             | `{tuples: [...]}` — initial instinct from the dev |

### 4.4 Retrieval and regeneration
| type          | payload |
|---------------|---------|
| `recalled`    | `{query_hash, candidates: [ids], selected: [ids], reasons: {id: str}, index_version}` |
| `regenerated` | `{replaces: [tu_ids], with: [tu_ids], from_events: [ev_ids]}` — re-run of extraction with a new model/prompt; old records become `superseded`, never deleted. Regeneration re-runs triage first with the current policy: a lowered threshold recovers turns with verdict "skip"; "ineligible" turns re-enter only if the policy's `ineligible_roles` changes. |

Note on `recalled`: it is the only high-frequency event. Logging is
configurable (`retrieval.log: full | sampled | off`) — see §9.4.

---

## 5. Tuple schema

The tuple is the semantic unit of the episodic and instinct layers.

```json
{
  "id": "tu_01J9ZK...",
  "subject": "user",
  "relation": "preferred_language",
  "value": "python",
  "qualifiers": {
    "when": "2026-07",
    "context": "memledger_project"
  },
  "confidence": 0.92,
  "layer": "episodic",
  "status": "quarantined",
  "impact": 3.5,
  "ttl": "90d",
  "sessions_seen": ["se_...", "se_..."],
  "sources": ["ev_...", "ev_..."],
  "text_form": "The user prefers Python as their language."
}
```

### 5.1 States (state machine)

```
quarantined → active → superseded
     │           │
     └───────────┴────────→ deleted (tombstone)
active(episodic) → active(instinct)   [only via promotion_approved]
```

- `quarantined`: retrievable (flagged), **never** promotable; leaves
  quarantine after `quarantine.new_facts` confirming sessions.
- `superseded`: excluded from default retrieval, kept for audit.
- `deleted`: tombstone; content may be physically removed from the
  projection, the `deleted` event stays in the ledger (for GDPR the
  payload of source events can be *redacted*: replaced by its hash).

### 5.2 Raw turns

Turns (`observed`) stay in the ledger with their own TTL
(`episodic.raw_retention`, default 30d) and serve as provenance and raw
material for `regenerate`. Past the TTL they may be compacted
(`compacted`) or redacted — never silently dropped: always via an event.

---

## 6. LLM block

Required in every event with `actor: "llm"`.

```json
{
  "model": "ollama/qwen3:4b",
  "model_digest": "sha256:…",
  "prompt": "extract@v1",
  "prompt_hash": "9f1b…",
  "params": {"temperature": 0, "max_tokens": 2000, "schema": "tuples@v1"},
  "input_hash": "c44a…",
  "output_hash": "77de…",
  "cache_key": "H(model ∥ prompt_hash ∥ input_hash ∥ params)",
  "cache_hit": false,
  "tokens": {"in": 1840, "out": 312}
}
```

Requirements: `temperature: 0` by default for all framework calls; output
always constrained to a JSON schema (`schema` in params, versioned).

**Cache:** table keyed on `cache_key` → verbatim output. Replay
(`memledger replay --cached`) fails with an explicit error on a missing
key — it never silently hits the network.

---

## 7. Storage (SQLite profile, default)

One `.db` file. Minimal tables:

| table        | contents | notes |
|--------------|----------|-------|
| `events`     | full envelope (JSON) + extracted columns: id, ts, session, type, actor | append-only; indexes on (session, ts) and (type, ts) |
| `records`    | current tuple projection (§5) | reconstructible; columns on layer, status, subject, relation |
| `fts`        | FTS5 over `records.text_form` + raw turns | index, reconstructible |
| `vectors`    | per-record embeddings (sqlite-vec) + `index_version` | **disposable**: droppable and regenerable |
| `llm_cache`  | cache_key → output, tokens, ts | exportable/shareable |
| `meta`       | spec_version, current policy_hash, prompt registry | |

Invariant: `records`, `fts`, `vectors` must be fully reconstructible from
`events` (+ `llm_cache` for determinism). `memledger rebuild` verifies
this: rebuild and compare hashes. Passing rebuild is the conformance test.

---

## 8. Policy and prompt registry

- `memory.policy.yaml` is canonicalized and hashed → `policy_hash`,
  recorded in every event. Changing the policy never rewrites history:
  old events stay bound to the old hash.
- Framework prompts (`extract`, `rerank`, `reflect`, `text_form`
  templates) live in a versioned registry (`prompts/extract.v1.md`),
  hashed like policies. The quickstart ships them; devs may fork them.
- Deterministic rule formulas (`salience@v1`, `text_form@v1`) live in the
  same registry. Because rule events carry no `llm` block, their formula
  hash must be pinned elsewhere: policy canonicalization resolves every
  registry id referenced by the policy (e.g. `triage.formula`) to its
  content hash before hashing. `policy_hash` therefore changes whenever a
  referenced formula changes — forking a formula is a policy change,
  visible in every subsequent event.

---

## 9. Invariants and known constraints

1. **Append-only:** no UPDATE/DELETE on `events` (except GDPR redaction of
   payloads, which replaces content with its hash and sets
   `redacted: true`).
2. **Every derived record has sources.** `memledger why` must always reach
   `observed`, `remember` or `seeded` events.
3. **Cascade:** `deleted` with `cascade: true` marks deleted every record
   whose source set becomes empty or compromised; partially derived
   records (merges with multiple sources) are marked `tainted` and
   re-extracted at the next checkpoint from the surviving sources.
   A strict mode (full cascade) is a future policy option for compliance
   profiles.
4. **Ledger growth:** `recalled` in `full` mode dominates volume.
   Default: `sampled` (100% of selections, 1% of candidates).
5. **Concurrency (Ego profile):** one writer per file (SQLite lock).
   Shared multi-agent memory is out of scope for this version: it will
   require explicit event ordering (vector clocks or a sequencer) —
   next profile.
6. **Spec migration:** `spec_version` per event; clients must read
   versions ≤ their own and write only the current one.

---

## 10. Minimal client conformance

An SDK is conforming if it: (a) writes valid events per §3–§6, (b) never
mutates `events`, (c) passes `memledger rebuild` (reproducible
projections), (d) respects the §5.1 state machine, (e) routes every
framework LLM call through the cache.