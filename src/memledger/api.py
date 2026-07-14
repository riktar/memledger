"""Public Ledger facade."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

from memledger.cache import CacheMissError, DeterministicCache, make_cache_key
from memledger.embeddings.base import Embedder
from memledger.events import Cause, Event, LLMCall, make_event
from memledger.ids import sha256_hex
from memledger.ledger import LedgerStore
from memledger.models.anthropic import AnthropicBackend
from memledger.models.base import ModelBackend, ModelResponse
from memledger.models.mock import MockModelBackend
from memledger.models.openai_compat import build_openai_compat_backend
from memledger.policy import Policy
from memledger.projection import Projection
from memledger.prompts import PromptRegistry
from memledger.session import Session
from memledger.triage import score_text
from memledger.tuples import make_tuple


@dataclass(slots=True)
class _InstinctFacade:
    ledger: Ledger

    def seed(self, tuples: list[tuple[str, str, str | int | float | bool]]) -> list[str]:
        records = [
            make_tuple(
                subject=subject,
                relation=relation,
                value=value,
                qualifiers={},
                confidence=1.0,
                layer="instinct",
                status="active",
                ttl=None,
                sessions_seen=[],
                sources=[],
            )
            for subject, relation, value in tuples
        ]
        event = make_event(
            type="seeded",
            actor="dev",
            cause=Cause(kind="manual", ref="seed", detail="seed instinct memory"),
            policy_hash=self.ledger.policy.hash,
            payload={"tuples": [record.to_dict() for record in records]},
            user=None,
            session=None,
        )
        self.ledger.append_event(event)
        return [record.id for record in records]


class Ledger:
    """High-level MemLedger facade implementing the BUILD public API."""

    def __init__(
        self,
        path: str,
        policy: Policy | None = None,
        memory_model: str | None = None,
        cache: str = "deterministic",
        *,
        model_backend: ModelBackend | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.policy = policy or Policy.default()
        self.store = LedgerStore(path, embedder=embedder)
        self.registry = PromptRegistry()
        self.cache_mode = cache
        self.cache = DeterministicCache(self.store.connection)
        self._using_default_mock_backend = model_backend is None and memory_model is None
        self._mock_backend_warning_emitted = False
        self.model_backend = model_backend or self._build_backend(memory_model)
        self.embedder = embedder
        self.projection = Projection(self.store, self.policy)
        self.instinct = _InstinctFacade(self)
        self.store.write_meta("spec_version", self.policy.get("spec_version", default="0.1"))
        self.store.write_meta("policy_hash", self.policy.hash)

    def close(self) -> None:
        self.store.close()

    def _build_backend(self, memory_model: str | None) -> ModelBackend:
        if memory_model is None:
            return MockModelBackend()
        if memory_model.startswith("openai-compat:"):
            return build_openai_compat_backend(memory_model)
        if memory_model.startswith("anthropic:"):
            model = memory_model.removeprefix("anthropic:")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic backends")
            return AnthropicBackend(api_key=api_key, model=model)
        raise ValueError(f"unsupported memory model spec: {memory_model}")

    def append_event(self, event: Event) -> None:
        self.append_events([event])

    def append_events(self, events: Sequence[Event]) -> None:
        with self.store.transaction():
            for event in events:
                self.store.append_event(event)
                self.projection.apply_event(event)

    def session(self, user_id: str | None = None) -> Session:
        return Session(self, user_id=user_id)

    def _warn_if_using_default_mock_backend(self) -> None:
        if not self._using_default_mock_backend or self._mock_backend_warning_emitted:
            return
        warnings.warn(
            "using mock backend: memories are deterministic test data, not real model extractions; "
            "set memory_model=... or pass model_backend=... to enable real memory formation",
            RuntimeWarning,
            stacklevel=3,
        )
        self._mock_backend_warning_emitted = True

    def call_model_json(
        self,
        *,
        prompt_id: str,
        placeholders: dict[str, Any],
        params: dict[str, Any],
        require_cache: bool = False,
        backend: ModelBackend | None = None,
    ) -> tuple[dict[str, Any], LLMCall]:
        self._warn_if_using_default_mock_backend()
        prompt = self.registry.render(prompt_id, placeholders)
        input_hash = sha256_hex({"prompt": prompt.content, "placeholders": placeholders})
        active_backend = backend or self.model_backend
        model_name = getattr(
            active_backend,
            "model",
            getattr(active_backend, "model_name", active_backend.__class__.__name__),
        )
        cache_key = make_cache_key(str(model_name), prompt.hash, input_hash, params)

        cache_hit = False
        cached_entry = None
        if self.cache_mode == "deterministic":
            if require_cache:
                cached_entry = self.cache.require(cache_key)
            else:
                cached_entry = self.cache.get(cache_key)
        if cached_entry is not None:
            response = ModelResponse(
                content=cached_entry.output,
                model=str(model_name),
                model_digest=str(model_name),
                tokens_in=cached_entry.tokens_in,
                tokens_out=cached_entry.tokens_out,
            )
            cache_hit = True
        else:
            if require_cache:
                raise CacheMissError(cache_key)
            response = active_backend.complete(prompt_id, prompt.content, params)
            if self.cache_mode == "deterministic":
                self.cache.set(
                    cache_key,
                    response.content,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                )
        llm_call = LLMCall(
            model=response.model,
            model_digest=response.model_digest,
            prompt=prompt_id,
            prompt_hash=prompt.hash,
            params=params,
            input_hash=input_hash,
            output_hash=sha256_hex(response.content),
            cache_key=cache_key,
            cache_hit=cache_hit,
            tokens={"in": response.tokens_in, "out": response.tokens_out},
        )
        return json.loads(response.content or "{}"), llm_call

    def rebuild(self) -> bool:
        before = self.store.projection_digest()
        events = self.store.iter_events()
        self.store.clear_projections()
        self.projection.replay_events(events)
        after = self.store.projection_digest()
        ok = before == after
        if ok:
            self.reindex_vectors()
        return ok

    def replay(self, *, at: str | None = None, cached: bool = False) -> bool:
        events = self.store.iter_events(at=at)
        if cached:
            for event in events:
                if event.llm is None:
                    continue
                self.cache.require(event.llm.cache_key)
        self.store.clear_projections()
        self.projection.replay_events(events)
        self.reindex_vectors()
        return True

    def regenerate(self, model: str | None = None, prompt: str = "extract@v1") -> int:
        regen_backend = self._build_backend(model) if model else self.model_backend
        sessions_reprocessed = 0
        for session_id in self.store.session_ids():
            observed = self.store.iter_events(session=session_id, type="observed")
            extract_turns = []
            for event in observed:
                result = score_text(
                    str(event.payload["text"]),
                    str(event.payload["role"]),
                    self.policy,
                )
                if result.verdict == "extract":
                    extract_turns.append(event)
            if not extract_turns:
                continue
            transcript = "\n".join(
                f"{event.payload['turn']}. {event.payload['role']}: {event.payload['text']}" for event in extract_turns
            )
            extracted_json, llm_call = self.call_model_json(
                prompt_id=prompt,
                placeholders={
                    "transcript": transcript,
                    "known_subjects": json.dumps([]),
                    "known_relations": json.dumps([]),
                    "session_id": session_id,
                    "language": "English",
                },
                params={"temperature": 0, "schema": "tuples@v1", "session": session_id},
                backend=regen_backend,
            )
            created_ids: list[str] = []
            observed_by_turn = {int(event.payload["turn"]): event.id for event in extract_turns}
            final_tuples: list[dict[str, Any]] = []
            for raw_tuple in extracted_json.get("tuples", []):
                if float(raw_tuple.get("confidence", 0.0)) < self.policy.extraction_min_confidence:
                    continue
                source_ids = [
                    observed_by_turn[int(turn)]
                    for turn in raw_tuple.get("evidence", [])
                    if int(turn) in observed_by_turn
                ]
                record = make_tuple(
                    subject=str(raw_tuple["subject"]),
                    relation=str(raw_tuple["relation"]),
                    value=raw_tuple["value"],
                    qualifiers=dict(raw_tuple.get("qualifiers", {})),
                    confidence=float(raw_tuple["confidence"]),
                    layer="episodic",
                    status="quarantined",
                    ttl=self.policy.ttl_for_layer("episodic"),
                    sessions_seen=[session_id],
                    sources=source_ids,
                    text_form=str(raw_tuple.get("text_form", "")).strip() or None,
                )
                existing = self.projection.exact_duplicate(record)
                if existing is not None:
                    existing.sessions_seen = sorted(set(existing.sessions_seen + [session_id]))
                    existing.sources = sorted(set(existing.sources + source_ids))
                    existing.confidence = max(
                        existing.confidence,
                        float(raw_tuple["confidence"]),
                    )
                    record = existing
                created_ids.append(record.id)
                final_tuples.append(record.to_dict())

            if not final_tuples:
                continue

            extracted_event = make_event(
                type="extracted",
                actor="llm",
                cause=Cause(kind="llm", ref=prompt, detail="ledger regenerate extraction"),
                policy_hash=self.policy.hash,
                payload={
                    "session": session_id,
                    "tuples": final_tuples,
                    "rejected": [],
                    "notes": list(extracted_json.get("notes", [])),
                },
                session=session_id,
                user=None,
                llm=llm_call,
                sources=[event.id for event in extract_turns],
            )
            self.append_event(extracted_event)

            event = make_event(
                type="regenerated",
                actor="llm",
                cause=Cause(kind="llm", ref=prompt, detail="ledger regenerate"),
                policy_hash=self.policy.hash,
                payload={
                    "replaces": [],
                    "with": created_ids,
                    "from_events": [event.id for event in extract_turns],
                },
                session=session_id,
                user=None,
                llm=llm_call,
                sources=[event.id for event in extract_turns],
            )
            self.append_event(event)
            if created_ids:
                sessions_reprocessed += 1
        self.reindex_vectors()
        return sessions_reprocessed

    def reindex_vectors(self) -> int:
        if self.embedder is None:
            return 0
        index_version = self.embedder.index_version
        self.store.ensure_vector_index_version(index_version)
        indexed = 0
        with self.store.transaction():
            for record in self.projection.active_or_quarantined_records():
                if self.store.has_vector(record.id, index_version):
                    continue
                if not record.text_form.strip():
                    continue
                try:
                    vector = self.embedder.embed([record.text_form])[0]
                    self.store._upsert_vector(record.id, vector, index_version)
                    indexed += 1
                except Exception as exc:
                    warnings.warn(
                        f"skipped vector index for {record.id}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        return indexed

    def delete(self, record_id: str, *, cascade: bool = False, reason: str = "manual") -> None:
        cascade_ids, tainted_ids = self.projection.plan_delete(record_id) if cascade else ([], [])
        event = make_event(
            type="deleted",
            actor="dev",
            cause=Cause(kind="manual", ref="delete", detail=reason),
            policy_hash=self.policy.hash,
            payload={
                "record": record_id,
                "cascade": cascade_ids,
                "reason": reason,
                "tainted": tainted_ids,
            },
            session=None,
            user=None,
        )
        self.append_event(event)

    def stats(self) -> dict[str, Any]:
        record_counts: dict[str, int] = {}
        for record in self.store.iter_records():
            key = f"{record.layer}:{record.status}"
            record_counts[key] = record_counts.get(key, 0) + 1
        cache_stats = self.cache.stats()
        events = self.store.iter_events()
        tokens_spent = 0
        tokens_saved = 0
        triage_counts = {"extract": 0, "skip": 0, "ineligible": 0}
        for event in events:
            if event.llm is not None:
                tokens_spent += event.llm.tokens["in"] + event.llm.tokens["out"]
            if event.type == "triaged":
                verdict = str(event.payload["verdict"])
                triage_counts[verdict] += 1
                if verdict == "skip":
                    target = self.store.get_event(str(event.payload["turn"]))
                    if target is not None:
                        tokens_saved += len(str(target.payload["text"]).split())
        return {
            "records": record_counts,
            "tokens_spent": tokens_spent,
            "tokens_saved": tokens_saved,
            "cache": cache_stats,
            "triage": triage_counts,
        }

    def why(self, record_id: str) -> dict[str, Any]:
        return self.projection.why(record_id)
