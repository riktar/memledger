# MemLedger Usage Guide

This guide covers the practical side of MemLedger: how to install it,
what every CLI command does, and how the main memory workflows fit
together.

## Requirements

- Python 3.11 or newer
- A writable working directory for the SQLite ledger file
- An optional model backend for real extraction and reflection

MemLedger stores everything in a single SQLite database file. No server is
required.

## Installation

### Install from the repository

```bash
python -m venv .venv
. .venv/bin/activate
pip install .
```

### Install in editable mode

```bash
pip install -e .
```

### Install with development tools

```bash
pip install -e ".[dev]"
```

This adds the test, lint, and type-check dependencies used by the repo.

### Install with local retrieval extras

```bash
pip install -e ".[local]"
```

This installs the optional local embedding stack used by the vector stage
of retrieval.

### Install with both common extras

```bash
pip install -e ".[dev,local]"
```

### Initialize a workspace

```bash
memledger init
```

By default this creates `memory.db` in the current directory and copies the
default `memory.policy.yaml` plus the prompt templates into `./prompts/`.
Existing policy and prompt files are left in place.

To initialize a ledger at a custom location:

```bash
memledger init ./state/support-memory.db
```

The defaults are copied next to the target database path.

## Model configuration

If you create `Ledger(...)` without a `memory_model`, MemLedger uses the
mock backend. That is useful for tests and smoke runs, but not for real
memory extraction. MemLedger emits a runtime warning the first time a
checkpoint, rerank, or reflection call would actually use that default.

### OpenAI-compatible backends

```python
from memledger import Ledger, Policy

ledger = Ledger(
    path="./memory.db",
    policy=Policy.default(),
    memory_model="openai-compat:http://localhost:11434/v1|qwen3:4b",
)
```

Use `OPENAI_API_KEY` for most OpenAI-compatible remote providers.

If the host is `openrouter.ai`, MemLedger prefers `OPENROUTER_API_KEY` and
falls back to `OPENAI_API_KEY`.

### Anthropic backends

```python
ledger = Ledger(
    path="./memory.db",
    policy=Policy.default(),
    memory_model="anthropic:claude-sonnet-4-20250514",
)
```

Set `ANTHROPIC_API_KEY` before using an Anthropic backend.

### Shell quoting note

Quote every `openai-compat:...|...` model string in the shell because the
`|` character is a pipe operator.

```bash
memledger regenerate --db ./memory.db --model 'openai-compat:http://localhost:11434/v1|qwen3:4b'
```

## CLI overview

The CLI entry point is `memledger`.

```bash
memledger --help
```

Commands that accept `--db` use `memory.db` by default.

## CLI reference

### `memledger init [path]`

Create a new SQLite ledger file and copy the default policy and prompts.

Examples:

```bash
memledger init
memledger init ./state/memory.db
```

Behavior:

- Creates the database file if it does not exist.
- Copies `memory.policy.yaml` if it is missing.
- Copies the packaged prompts into `prompts/` if they are missing.
- Prints the target database path.

### `memledger log [--db PATH] [--type TYPE] [--session SESSION_ID] [--since TS]`

Stream ledger events as JSON lines.

Examples:

```bash
memledger log --db ./memory.db
memledger log --db ./memory.db --type extracted
memledger log --db ./memory.db --session se_01J9ZK...
memledger log --db ./memory.db --since 2026-07-07T00:00:00Z
```

Use this to inspect the raw event ledger, filter by event type, or look at
a single session timeline.

### `memledger why ID [--db PATH] [--json]`

Show the provenance graph for a record such as a tuple or instinct entry.

Example:

```bash
memledger why tu_01J9ZKM3 --db ./memory.db
memledger why tu_01J9ZKM3 --db ./memory.db --json
```

By default the command prints a human-readable provenance tree. Add
`--json` to get the raw provenance payload returned by the Python API.

It is the fastest way to answer questions such as:

- Which turns created this memory?
- Was it extracted, remembered manually, or seeded?
- Which event promoted or superseded it?

### `memledger review [--db PATH]`

List the pending `promotion_proposed` events.

Example:

```bash
memledger review --db ./memory.db
```

Current behavior in MemLedger 0.1:

- The command is an inspection queue, not an interactive approval UI.
- It prints the proposal events as formatted JSON.
- Automatic approval happens only when `instinct.autonomous: true` in
  `memory.policy.yaml`.

### `memledger replay [--db PATH] [--at TS] [--cached]`

Clear projections and replay the ledger.

Examples:

```bash
memledger replay --db ./memory.db
memledger replay --db ./memory.db --at 2026-07-07T14:03:22.117Z
memledger replay --db ./memory.db --cached
```

Use cases:

- Rebuild the current state from the ledger.
- Reconstruct the state at a historical timestamp with `--at`.
- Verify deterministic cache completeness with `--cached`.

When `--cached` is set, MemLedger checks that every historical LLM call in
the replay range is already present in the deterministic cache. If a cache
entry is missing, replay fails instead of silently calling the network.

### `memledger rebuild [--db PATH]`

Run the projection conformance check.

Example:

```bash
memledger rebuild --db ./memory.db
```

Behavior:

- Captures the current projection digest.
- Replays the ledger from scratch.
- Compares the new digest to the old digest.
- Prints `ok` on a match or `mismatch` on a failure.
- Returns exit code `0` for success and `1` for failure.

Use this when you change the implementation and want to verify that the
ledger still reconstructs to the same state.

### `memledger regenerate [--db PATH] [--model MODEL] [--prompt PROMPT]`

Re-run extraction session by session using the current policy.

Examples:

```bash
memledger regenerate --db ./memory.db
memledger regenerate --db ./memory.db --prompt extract@v1
memledger regenerate --db ./memory.db --model 'openai-compat:http://localhost:11434/v1|qwen3:4b'
```

Notes:

- `--prompt` defaults to `extract@v1`.
- `--model` overrides the ledger's configured backend for regeneration.
- The command prints the number of sessions that produced regenerated
  records.

Because regeneration re-runs triage with the current policy, it can recover
previously skipped turns after a threshold change. Turns that were marked
`ineligible` stay excluded unless the policy itself changes.

### `memledger delete ID [--db PATH] [--cascade] [--reason TEXT]`

Append a `deleted` event for a record.

Examples:

```bash
memledger delete tu_01J9ZKM3 --db ./memory.db
memledger delete tu_01J9ZKM3 --db ./memory.db --cascade --reason poisoning
```

Notes:

- This is a logical delete, not a destructive file rewrite.
- `--cascade` includes downstream records whose provenance becomes invalid
  or tainted.
- `--reason` is written into the ledger for auditability.
- The command prints the deleted record ID.

### `memledger stats [--db PATH]`

Show aggregate ledger statistics as formatted JSON.

Example:

```bash
memledger stats --db ./memory.db
```

The current output includes:

- Record counts grouped by `layer:status`
- Total tokens spent by LLM-backed memory operations
- Words skipped by triage as an approximation of context saved
- Deterministic cache stats
- Triage counts for `extract`, `skip`, and `ineligible`

## Core workflows

### 1. Bootstrap a new ledger

1. Run `memledger init` in the directory where you want to keep the
   database and policy.
2. Review `memory.policy.yaml` and adjust thresholds only if you have a
   concrete reason to change the defaults.
3. Point your Python application at the generated database path.

### 2. Run a memory-aware session

MemLedger is designed to sit inside your application loop.

```python
from memledger import Ledger, Policy

ledger = Ledger(
    path="./memory.db",
    policy=Policy.default(),
    memory_model="openai-compat:http://localhost:11434/v1|qwen3:4b",
)

session = ledger.session(user_id="demo")

user_message = "Please keep examples in Python."
memories = session.recall(user_message, k=5)
context = session.build_context(instinct=True, episodic=memories, working="tail")
reply = your_llm(system=context.system, messages=context.messages, user=user_message)
session.observe(user=user_message, assistant=reply)
```

The runtime loop usually looks like this:

1. Recall relevant episodic memory with `session.recall(...)`.
2. Build a prompt context with instinct memory, recalled episodic memory,
   and the working-memory tail.
3. Call your assistant model.
4. Persist the turn pair with `session.observe(...)`.

Optional developer signals:

- `session.feedback(+1 or -1)` attaches explicit feedback to a source.
- `session.outcome("success" or "failure", task=...)` records task-level
  outcomes.
- `session.remember((subject, relation, value))` creates an active
  episodic fact immediately.
- `ledger.instinct.seed([...])` seeds always-on instinct memory.

### 3. Checkpoint and memory formation

Call `session.checkpoint()` when you want MemLedger to turn raw turns into
maintained memory.

```python
report = session.checkpoint()
print(report)
```

The checkpoint pipeline is:

1. Triage scores every pending raw turn and emits `triaged` events with
   `extract`, `skip`, or `ineligible` verdicts.
2. Extraction runs on the `extract` subset only and emits one
   `extracted` event.
3. Exact duplicates update the existing record instead of creating a new
   semantic duplicate.
4. New facts begin in quarantine and become active after enough distinct
   sessions confirm them.
5. Impact scores are recomputed from feedback, outcomes, recall usage,
   and repetition.
6. Reflection can merge related facts, supersede contradictions, and
   propose promotions to instinct memory.
7. Expiration and tainted-record repair run at the end of the checkpoint.

The returned `CheckpointReport` summarizes how many turns were triaged,
how many memories were extracted, and how many tokens were spent or saved.

### 4. Promotion review workflow

Promotion is how stable episodic facts move into instinct memory.

The default flow is:

1. Checkpoint proposes a promotion when a record meets the eligibility
   rules and reflection agrees.
2. `memledger review --db ./memory.db` shows the current proposal queue.
3. If `instinct.autonomous: true`, approval is emitted automatically at
   checkpoint time.

In other words, `review` is currently an audit and inspection step. It is
not yet a manual approval console.

### 5. Audit and debugging workflow

Use these commands together when you need to understand memory behavior:

1. `memledger log` to inspect the event stream.
2. `memledger why <id>` to trace one memory back to its sources.
3. `memledger stats` to inspect overall memory cost and state.

This is the main workflow for debugging bad extraction, memory poisoning,
missing promotions, or unexpected recall.

### 6. Determinism and recovery workflow

MemLedger separates three maintenance operations that sound similar but do
different jobs:

1. `memledger replay` rebuilds projections from the ledger and can target
   a historical timestamp.
2. `memledger rebuild` proves that replaying the full ledger reproduces
   the current projection exactly.
3. `memledger regenerate` re-runs extraction with the current model or
   prompt while preserving the original source turns.

Use them in this order when investigating changes:

1. Run `rebuild` to check that projections are still conformant.
2. Run `replay --cached` if you want deterministic historical playback.
3. Run `regenerate` after changing prompts, thresholds, or the extraction
   model.

### 7. Deletion and correction workflow

When a memory is wrong or unsafe:

1. Inspect it with `memledger why <id>`.
2. Delete it with `memledger delete <id> --cascade --reason ...` when the
   provenance chain is compromised.
3. Run a new checkpoint or regeneration pass if you want surviving source
   material to be re-processed.

This keeps the audit trail intact while making the correction explicit in
the ledger.

## Example entry points

The repository includes three small examples that mirror the workflows
above:

- `examples/01_chat_with_memory.py` for a minimal chat loop
- `examples/02_coding_assistant.py` for cross-session preference recall
- `examples/03_support_agent.py` for outcome tracking and checkpoint stats