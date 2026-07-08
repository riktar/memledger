"""Tuple projection, state machine, cascade handling and repair."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from memledger.events import Event
from memledger.ids import utc_now
from memledger.ledger import LedgerStore
from memledger.policy import Policy, parse_duration
from memledger.tuples import MemoryTuple

ALLOWED_STATUSES = frozenset({"quarantined", "active", "superseded", "deleted", "expired"})


class StateTransitionError(ValueError):
    """Raised when a tuple transition violates the SPEC state machine."""


def validate_state_transition(
    current_status: str | None,
    target_status: str,
    current_layer: str | None,
    target_layer: str,
) -> None:
    if target_status not in ALLOWED_STATUSES:
        raise StateTransitionError(f"unsupported target status: {target_status}")
    if current_status is None:
        if (target_layer, target_status) not in {
            ("episodic", "quarantined"),
            ("episodic", "active"),
            ("instinct", "active"),
        }:
            raise StateTransitionError(f"illegal initial state {(target_layer, target_status)}")
        return
    if current_status == target_status and current_layer == target_layer:
        return
    allowed = {
        ("episodic", "quarantined", "episodic", "active"),
        ("episodic", "quarantined", "episodic", "superseded"),
        ("episodic", "quarantined", "episodic", "deleted"),
        ("episodic", "active", "episodic", "superseded"),
        ("episodic", "active", "episodic", "deleted"),
        ("episodic", "active", "instinct", "active"),
        ("instinct", "active", "instinct", "superseded"),
        ("instinct", "active", "instinct", "deleted"),
        ("episodic", "superseded", "episodic", "deleted"),
        ("instinct", "superseded", "instinct", "deleted"),
        ("episodic", "active", "episodic", "expired"),
        ("episodic", "quarantined", "episodic", "expired"),
    }
    if (current_layer, current_status, target_layer, target_status) not in allowed:
        raise StateTransitionError(
            f"illegal transition {(current_layer, current_status)} -> {(target_layer, target_status)}"
        )


class Projection:
    """Maintains the reconstructible tuple projection over the event ledger."""

    def __init__(self, store: LedgerStore, policy: Policy) -> None:
        self.store = store
        self.policy = policy

    def apply_event(self, event: Event) -> None:
        handler = getattr(self, f"_apply_{event.type}", None)
        if handler is None:
            return
        handler(event)

    def _apply_observed(self, event: Event) -> None:
        self.store.add_turn_to_fts(event.id, str(event.payload["text"]))

    def _apply_seeded(self, event: Event) -> None:
        for data in event.payload.get("tuples", []):
            record = MemoryTuple.from_dict(data)
            record.sources = [event.id]
            record.history = list(dict.fromkeys(record.history + [event.id]))
            self._save_record(record)

    def _apply_remember(self, event: Event) -> None:
        record = MemoryTuple.from_dict(event.payload["tuple"])
        record.sources = [event.id]
        record.history = list(dict.fromkeys(record.history + [event.id]))
        self._save_record(record)

    def _apply_extracted(self, event: Event) -> None:
        for data in event.payload.get("tuples", []):
            record = MemoryTuple.from_dict(data)
            record.history = list(dict.fromkeys(record.history + [event.id]))
            self._save_record(record)

    def _apply_quarantine_lifted(self, event: Event) -> None:
        record = self.require_record(str(event.payload["record"]))
        record.status = "active"
        record.updated_ts = event.ts
        record.history.append(event.id)
        self._save_record(record)

    def _apply_scored(self, event: Event) -> None:
        record = self.require_record(str(event.payload["record"]))
        record.impact = float(event.payload["impact_total"])
        record.updated_ts = event.ts
        record.history.append(event.id)
        self._save_record(record)

    def _apply_superseded(self, event: Event) -> None:
        record = self.require_record(str(event.payload["old"]))
        record.status = "superseded"
        record.updated_ts = event.ts
        record.history.append(event.id)
        self._save_record(record)

    def _apply_merged(self, event: Event) -> None:
        into = self.require_record(str(event.payload["into"]))
        from_ids = [
            str(value)
            for value in event.payload.get("from", [])
            if str(value) != into.id
        ]
        merged_sources = set(into.sources)
        merged_from = set(into.merged_from)
        for from_id in from_ids:
            record = self.require_record(from_id)
            merged_sources.update(record.sources)
            merged_from.add(record.id)
            record.status = "superseded"
            record.updated_ts = event.ts
            record.history.append(event.id)
            self._save_record(record)
        into.sources = sorted(merged_sources)
        into.merged_from = sorted(merged_from)
        into.updated_ts = event.ts
        into.history.append(event.id)
        self._save_record(into)

    def _apply_promotion_approved(self, event: Event) -> None:
        proposal_id = str(event.payload["proposal"])
        proposal = self.store.get_event(proposal_id)
        if proposal is None:
            raise KeyError(f"promotion proposal not found: {proposal_id}")
        record = self.require_record(str(proposal.payload["record"]))
        validate_state_transition(record.status, "active", record.layer, "instinct")
        record.layer = "instinct"
        record.status = "active"
        record.ttl = None
        record.updated_ts = event.ts
        record.history.append(event.id)
        self._save_record(record)

    def _apply_deleted(self, event: Event) -> None:
        targets = [str(event.payload["record"])] + [str(value) for value in event.payload.get("cascade", [])]
        compromised_sources: set[str] = set()
        for target in targets:
            existing = self.require_record(target)
            compromised_sources.update(existing.sources)
        for target in targets:
            record = self.require_record(target)
            validate_state_transition(record.status, "deleted", record.layer, record.layer)
            record.status = "deleted"
            record.updated_ts = event.ts
            record.history.append(event.id)
            self._save_record(record)
        for target in event.payload.get("tainted", []):
            record = self.require_record(str(target))
            record.sources = [source_id for source_id in record.sources if source_id not in compromised_sources]
            record.tainted = True
            record.updated_ts = event.ts
            record.history.append(event.id)
            self._save_record(record)

    def _apply_expired(self, event: Event) -> None:
        record = self.require_record(str(event.payload["record"]))
        validate_state_transition(record.status, "expired", record.layer, record.layer)
        record.status = "expired"
        record.updated_ts = event.ts
        record.history.append(event.id)
        self._save_record(record)

    def _apply_regenerated(self, event: Event) -> None:
        for record_id in event.payload.get("with", []):
            record = self.store.get_record(str(record_id))
            if record is None:
                continue
            record.tainted = False
            record.updated_ts = event.ts
            record.history.append(event.id)
            self._save_record(record)

    def _save_record(self, record: MemoryTuple) -> None:
        existing = self.store.get_record(record.id)
        validate_state_transition(
            None if existing is None else existing.status,
            record.status,
            None if existing is None else existing.layer,
            record.layer,
        )
        record.updated_ts = record.updated_ts or utc_now()
        self.store.upsert_record(record)

    def require_record(self, record_id: str) -> MemoryTuple:
        record = self.store.get_record(record_id)
        if record is None:
            raise KeyError(f"record not found: {record_id}")
        return record

    def active_instinct(self) -> list[MemoryTuple]:
        return [
            record
            for record in self.store.iter_records(include_deleted=False)
            if record.layer == "instinct" and record.status == "active"
        ]

    def active_or_quarantined_records(self) -> list[MemoryTuple]:
        return [
            record
            for record in self.store.iter_records(include_deleted=False)
            if record.status in {"active", "quarantined"}
        ]

    def exact_duplicate(self, candidate: MemoryTuple) -> MemoryTuple | None:
        value_json = json.dumps(candidate.value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self.store.find_record_by_key(candidate.subject, candidate.relation, value_json)

    def promotion_candidates(self) -> list[MemoryTuple]:
        candidates: list[MemoryTuple] = []
        for record in self.store.iter_records(include_deleted=False):
            if record.layer != "episodic" or record.status != "active" or record.tainted:
                continue
            if self.policy.promotion_eligible(impact=record.impact, sessions_seen=len(record.sessions_seen)):
                candidates.append(record)
        return candidates

    def pending_triage_turn_ids(self, session_id: str) -> list[str]:
        observed = {event.id for event in self.store.iter_events(session=session_id, type="observed")}
        triaged = {str(event.payload["turn"]) for event in self.store.iter_events(session=session_id, type="triaged")}
        return sorted(observed - triaged)

    def plan_delete(self, record_id: str) -> tuple[list[str], list[str]]:
        target_record = self.require_record(record_id)
        compromised_sources = set(target_record.sources)
        cascade: list[str] = []
        tainted: list[str] = []
        for record in self.store.iter_records(include_deleted=False):
            if record.id == record_id or record.status == "deleted":
                continue
            surviving_sources = [source_id for source_id in record.sources if source_id not in compromised_sources]
            if record_id not in record.merged_from and len(surviving_sources) == len(record.sources):
                continue
            if surviving_sources:
                tainted.append(record.id)
            else:
                cascade.append(record.id)
        return cascade, tainted

    def repair_tainted_records(self) -> list[MemoryTuple]:
        repaired: list[MemoryTuple] = []
        for record in self.store.iter_records(include_deleted=False):
            if not record.tainted or record.status == "deleted":
                continue
            live_sources = []
            for merged_id in record.merged_from:
                source = self.store.get_record(merged_id)
                if source is not None and source.status != "deleted":
                    live_sources.append(merged_id)
            surviving_event_sources = [
                source_id for source_id in record.sources if self.store.get_event(source_id) is not None
            ]
            if not surviving_event_sources:
                validate_state_transition(record.status, "deleted", record.layer, record.layer)
                record.status = "deleted"
                record.updated_ts = utc_now()
                record.history.append("repair")
                self._save_record(record)
                continue
            record.tainted = False
            record.merged_from = live_sources
            record.sources = surviving_event_sources
            record.history.append("repair")
            self._save_record(record)
            repaired.append(record)
        return repaired

    def replay_events(self, events: list[Event]) -> None:
        for event in events:
            self.apply_event(event)

    def related_records_for(self, records: list[MemoryTuple]) -> list[MemoryTuple]:
        seen: dict[str, MemoryTuple] = {}
        for record in records:
            for related in self.store.find_related_records(record):
                seen[related.id] = related
        return [seen[key] for key in sorted(seen)]

    def expire_due_records(self, now_ts: str) -> list[MemoryTuple]:
        current = datetime.fromisoformat(now_ts.replace("Z", "+00:00")).astimezone(UTC)
        expired: list[MemoryTuple] = []
        for record in self.store.iter_records(include_deleted=False):
            if record.layer != "episodic" or record.status not in {
                "active",
                "quarantined",
            }:
                continue
            if record.ttl is None:
                continue
            retention = parse_duration(record.ttl)
            if retention is None:
                continue
            created = datetime.fromisoformat(record.created_ts.replace("Z", "+00:00")).astimezone(UTC)
            if created + retention <= current:
                expired.append(record)
        return expired

    def update_record(self, record: MemoryTuple) -> None:
        self._save_record(record)

    def why(self, record_id: str) -> dict[str, Any]:
        record = self.require_record(record_id)
        creator_event = None
        for event_id in record.history:
            if event_id == "repair":
                continue
            event = self.store.get_event(event_id)
            if event is not None and event.type in {"remember", "seeded", "extracted"}:
                creator_event = event
                break
        source_events = [self.store.get_event(source_id) for source_id in record.sources]
        return {
            "record": record.to_dict(),
            "creator": None if creator_event is None else creator_event.to_dict(),
            "sources": [event.to_dict() for event in source_events if event is not None],
            "history": [event_id for event_id in record.history if event_id != "repair"],
        }
