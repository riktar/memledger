# MemLedger

```
  _______  _______  _______  _        _______  ______   _______  _______  _______ 
(       )(  ____ \(       )( \      (  ____ \(  __  \ (  ____ \(  ____ \(  ____ )
| () () || (    \/| () () || (      | (    \/| (  \  )| (    \/| (    \/| (    )|
| || || || (__    | || || || |      | (__    | |   ) || |      | (__    | (____)|
| |(_)| ||  __)   | |(_)| || |      |  __)   | |   | || | ____ |  __)   |     __)
| |   | || (      | |   | || |      | (      | |   ) || | \_  )| (      | (\ (   
| )   ( || (____/\| )   ( || (____/\| (____/\| (__/  )| (___) || (____/\| ) \ \__
|/     \|(_______/|/     \|(_______/(_______/(______/ (_______)(_______/|/   \__/
                                                                                 
```

**AI agent memory you can trust — because it can tell you *why*.**

Every memory framework solves the forgetting problem: agents now remember
things across sessions. But they create a new problem: memory becomes a
black box. The agent "knows" things, but you can't tell where a fact came
from, why it was kept, or why a user's preference from three weeks ago
just vanished. When the memory fails (and it will) you can't debug it.
You can only delete everything and start over.

**MemLedger is memory with the black box wide open.** Every fact your agent
holds has a full chain of provenance. A single command, `memledger why`,
shows you the exact sentence a memory was born from, which model extracted
it, when it was promoted to permanent knowledge, and who approved it.

It's like a bank statement for memory: you don't just see the balance, you
see every single transaction that produced it.

```bash
$ memledger why tu_01J9ZKM3
tu_01J9ZKM3  (instinct, active)  "The user prefers Python as their language"
 └─ promoted   2026-07-02  cause: impact 5.5 ≥ 5 across 4 sessions, approved by dev
    └─ extracted  2026-06-28  model: qwen3:4b  prompt: extract@v1  confidence: 0.95
       └─ observed  se_88 turn 3   "please, always Python — I don't read Go"
       └─ observed  se_91 turn 12  "again: Python examples only"
```

## The Four Pillars

1.  **Pay for intelligence only when it matters.** A zero-cost,
    deterministic filter decides which conversational turns are worth
    LLM extraction. "Ok, thanks!" will never cost you a token; "the
    deploy failed because of the env vars" will. Most chat traffic is
    phatic noise; MemLedger recognizes and skips it, drastically cutting
    memory costs.

2.  **Memory improves with models, instead of aging with them.** Other
    frameworks freeze memories the moment they're written. If today's
    model extracts poorly, that error is permanent. MemLedger always
    keeps the raw source, so when a better model comes out, you run
    `regenerate` and your agent's entire memory is re-built, *better*,
    from the original history. No competitor can do this.

3.  **Anti-poisoning by design.** New facts are quarantined until confirmed
    across multiple sessions. Nothing becomes permanent knowledge without
    your approval (`memledger review`). And if a bad fact gets through,
    provenance leads you to the source, and a cascading delete removes
    it and everything derived from it.

4.  **Truly yours.** It's a single SQLite file on your machine. You choose
    the model — even a small, free, local one via Ollama. No mandatory
    servers, no vendor lock-in, MIT licensed. It runs on a laptop.

## How it works

Three memory layers mimicking human cognition, all built on an append-only
event ledger. The ledger — not the projections — is the source of truth.

```
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │     Instinct     │   │     Episodic     │   │      Working     │
   │ (Core facts)     │   │ (Long-term)      │   │ (Current session)│
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                      │                      │
            └──────────────────┐   │   ┌──────────────────┘
                               ▼   ▼   ▼
                  ┌─────────────────────────────┐
                  │   Event Ledger (SQLite)     │
                  │ (Append-only, auditable)    │
                  └─────────────────────────────┘
```

-   **Instinct:** Core facts injected into every context. Seeded by you, or
    promoted automatically when a fact proves itself across sessions.
-   **Episodic:** Long-term tuples `(subject, relation, value)` with
    confidence, impact scores, and TTL, extracted by an LLM at checkpoint.
-   **Working:** The current session's turn-by-turn buffer.

Every extraction, promotion, merge, and deletion is an event with an
explicit cause. An LLM does the smart work, but every decision is logged,
reproducible, and reversible.

## Quickstart

```bash
pip install memledger            # add [local] for CPU embeddings
memledger init
```

```python
from memledger import Ledger, Policy

ledger = Ledger("./memory.db", policy=Policy.default(),
                memory_model="openai-compat:http://localhost:11434/v1|qwen3:4b")
                # ...or any OpenAI-compatible / Anthropic endpoint

session = ledger.session(user_id="me")

while (msg := input("> ")):
    memories = session.recall(msg, k=5)
    ctx = session.build_context(instinct=True, episodic=memories, working="tail")
    reply = your_llm(system=ctx.system, messages=ctx.messages, user=msg)
    session.observe(user=msg, assistant=reply)
    print(reply)

report = session.checkpoint()        # extract → reflect → promote
print(report.tokens_saved_in_context)
```

Runs on a laptop CPU with a local model, or with any cloud endpoint.
Storage is a single SQLite file. No server, no vendor lock-in.

## The Difference

Why not just use another memory framework?

-   **vs. LLM-based memory (Mem0, Zep, etc.):** They give you a memory
    that works until it doesn't. We give you one you can query, replay,
    and fix when it breaks.
-   **vs. Deterministic approaches:** They sacrifice semantic understanding
    for reproducibility. We keep both: rules where you need guarantees,
    LLMs where you need intelligence, and a full audit trail for everything.

**The one-liner: Others make your agent remember. MemLedger lets you know *why* it remembers.**

## CLI

`memledger why <id>` · `review` · `replay --at <ts> --cached` ·
`rebuild` · `regenerate --model <m>` · `delete <id> --cascade` · `stats`

## Configuration

Everything lives in `memory.policy.yaml` — promotion thresholds, the
impact formula, retention, retrieval and quarantine settings. It's hashed,
and the hash is recorded in every event, so config changes never rewrite
history. See the file for line-by-line docs.

## Benchmarks

MemLedger ships a benchmark harness for two long-term memory benchmarks:

-   **LoCoMo** (in progress) — question answering over very long multi-session dialogs, with
    evidence dialog ids for retrieval scoring.


## Project status & roadmap

**0.1 "Ego"** — single agent, single file, single writer. Stable spec
(`SPEC.md`), Python SDK, CLI, local + cloud model backends.

Planned: TypeScript SDK reading the same ledger format · shared
multi-agent memory (next profile) · hosted sync and audit dashboard.

## License

MIT. The `SPEC.md` ledger format is open: conforming clients in any
language are welcome — `memledger rebuild` is the conformance test.