"""Checkpoint orchestration: triage, extraction, lifecycle and repair."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memledger.events import Cause, Event, make_event
from memledger.ids import utc_now
from memledger.triage import score_text
from memledger.tuples import MemoryTuple, make_tuple

if TYPE_CHECKING:
    from memledger.session import Session


@dataclass(slots=True)
class CheckpointReport:
    triaged: dict[str, int]
    extracted: int
    merged: int
    superseded: int
    promotions_proposed: int
    tokens_spent_on_memory: int
    tokens_saved_in_context: int


def _transcript_from_events(events: list[Event]) -> str:
    lines = []
    for event in events:
        lines.append(f"{event.payload['turn']}. {event.payload['role']}: {event.payload['text']}")
    return "\n".join(lines)


def _build_signal_indexes(events: list[Event]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    feedback_by_source: dict[str, float] = defaultdict(float)
    outcome_by_source: dict[str, float] = defaultdict(float)
    recall_by_record: dict[str, float] = defaultdict(float)
    for event in events:
        if event.type == "feedback":
            target = str(event.payload.get("on", ""))
            if target:
                feedback_by_source[target] += float(event.payload.get("value", 0))
        elif event.type == "outcome":
            delta = 1.0 if event.payload.get("status") == "success" else -1.0
            for target in event.payload.get("on", []):
                outcome_by_source[str(target)] += delta
        elif event.type == "recalled":
            for record_id in event.payload.get("selected", []):
                recall_by_record[str(record_id)] += 1.0
    return feedback_by_source, outcome_by_source, recall_by_record


def run_checkpoint(session: Session) -> CheckpointReport:
    ledger = session.ledger
    projection = ledger.projection
    policy = ledger.policy
    pending_turn_ids = projection.pending_triage_turn_ids(session.id)
    turns_by_id = {event.id: event for event in ledger.store.iter_events(session=session.id, type="observed")}

    triaged_counts = {"extract": 0, "skip": 0, "ineligible": 0}
    extract_events = []
    skipped_word_count = 0
    for turn_id in pending_turn_ids:
        turn_event = turns_by_id[turn_id]
        result = score_text(str(turn_event.payload["text"]), str(turn_event.payload["role"]), policy)
        triage_event = make_event(
            type="triaged",
            actor="rule",
            cause=Cause(kind="rule", ref="salience@v1", detail="checkpoint triage"),
            policy_hash=policy.hash,
            payload=result.to_payload(turn_id),
            session=session.id,
            user=session.user_id,
        )
        ledger.append_event(triage_event)
        triaged_counts[result.verdict] += 1
        if result.verdict == "extract":
            extract_events.append(turn_event)
        else:
            skipped_word_count += len(str(turn_event.payload["text"]).split())

    extracted_count = 0
    merged_count = 0
    superseded_count = 0
    promotions_proposed = 0
    tokens_spent = 0
    materialized_records: list[MemoryTuple] = []

    if extract_events:
        transcript = _transcript_from_events(extract_events)
        live_records = projection.active_or_quarantined_records()
        known_subjects = sorted({record.subject for record in live_records})
        known_relations = sorted({record.relation for record in live_records})
        extracted_json, llm_call = ledger.call_model_json(
            prompt_id=str(policy.get("extraction", "prompt", default="extract@v1")),
            placeholders={
                "transcript": transcript,
                "known_subjects": json.dumps(known_subjects),
                "known_relations": json.dumps(known_relations),
                "session_id": session.id,
                "language": "English",
            },
            params={"temperature": 0, "schema": "tuples@v1", "session": session.id},
        )
        tokens_spent += llm_call.tokens["in"] + llm_call.tokens["out"]
        observed_by_turn = {int(event.payload["turn"]): event.id for event in extract_events}

        rejected: list[dict[str, Any]] = []
        final_tuples: list[dict[str, Any]] = []
        for raw_tuple in extracted_json.get("tuples", []):
            confidence = float(raw_tuple.get("confidence", 0.0))
            if confidence < policy.extraction_min_confidence:
                rejected.append({"tuple": raw_tuple, "confidence": confidence})
                continue
            evidence = [int(turn) for turn in raw_tuple.get("evidence", []) if int(turn) in observed_by_turn]
            source_ids = [observed_by_turn[turn] for turn in evidence]
            candidate = make_tuple(
                subject=str(raw_tuple["subject"]),
                relation=str(raw_tuple["relation"]),
                value=raw_tuple["value"],
                qualifiers=dict(raw_tuple.get("qualifiers", {})),
                confidence=confidence,
                layer="episodic",
                status="quarantined",
                ttl=policy.ttl_for_layer("episodic"),
                sessions_seen=[session.id],
                sources=source_ids,
                text_form=str(raw_tuple.get("text_form", "")).strip() or None,
            )
            existing = projection.exact_duplicate(candidate)
            if existing is not None:
                existing.sessions_seen = sorted(set(existing.sessions_seen + [session.id]))
                existing.sources = sorted(set(existing.sources + source_ids))
                existing.confidence = max(existing.confidence, candidate.confidence)
                existing.updated_ts = utc_now()
                existing.history = list(dict.fromkeys(existing.history))
                final_record = existing
            else:
                final_record = candidate
            final_tuples.append(final_record.to_dict())
            materialized_records.append(final_record)

        extracted_event = make_event(
            type="extracted",
            actor="llm",
            cause=Cause(kind="llm", ref="extract@v1", detail="checkpoint extraction"),
            policy_hash=policy.hash,
            payload={
                "session": session.id,
                "tuples": final_tuples,
                "rejected": rejected,
                "notes": list(extracted_json.get("notes", [])),
            },
            session=session.id,
            user=session.user_id,
            llm=llm_call,
            sources=[event.id for event in extract_events],
        )
        ledger.append_event(extracted_event)
        extracted_count = len(final_tuples)

        for record in materialized_records:
            if record.status == "quarantined" and len(record.sessions_seen) >= policy.quarantine_sessions:
                lift_event = make_event(
                    type="quarantine_lifted",
                    actor="rule",
                    cause=Cause(
                        kind="rule",
                        ref="quarantine@rule",
                        detail="confirmation threshold reached",
                    ),
                    policy_hash=policy.hash,
                    payload={
                        "record": record.id,
                        "confirmed_in_sessions": len(record.sessions_seen),
                    },
                    session=session.id,
                    user=session.user_id,
                )
                ledger.append_event(lift_event)

    touched_record_ids = {record.id for record in materialized_records}
    signal_targets: set[str] = set()
    session_events = ledger.store.iter_events(session=session.id)
    for event in session_events:
        if event.type == "feedback":
            target = str(event.payload.get("on", ""))
            if target:
                signal_targets.add(target)
        elif event.type == "outcome":
            signal_targets.update(str(item) for item in event.payload.get("on", []))
        elif event.type == "recalled":
            touched_record_ids.update(str(record_id) for record_id in event.payload.get("selected", []))

    if signal_targets:
        for record in projection.active_or_quarantined_records():
            if any(source_id in signal_targets for source_id in record.sources):
                touched_record_ids.add(record.id)

    if touched_record_ids:
        feedback_by_source, outcome_by_source, recall_by_record = _build_signal_indexes(ledger.store.iter_events())
    else:
        feedback_by_source = {}
        outcome_by_source = {}
        recall_by_record = {}

    for record_id in sorted(touched_record_ids):
        record = projection.require_record(record_id)
        feedback_total = sum(feedback_by_source.get(source_id, 0.0) for source_id in record.sources)
        outcome_total = sum(outcome_by_source.get(source_id, 0.0) for source_id in record.sources)
        recall_used = recall_by_record.get(record.id, 0.0)
        impact = policy.impact_score(
            feedback_total=feedback_total,
            outcome_total=outcome_total,
            recall_used=recall_used,
            sessions_seen=len(record.sessions_seen),
        )
        score_event = make_event(
            type="scored",
            actor="rule",
            cause=Cause(kind="rule", ref="impact@v1", detail="checkpoint impact recompute"),
            policy_hash=policy.hash,
            payload={
                "record": record.id,
                "impact_delta": impact - record.impact,
                "impact_total": impact,
                "formula": "impact@v1",
            },
            session=session.id,
            user=session.user_id,
        )
        ledger.append_event(score_event)

    reflection_enabled = bool(policy.get("reflection", "enabled", default=True))
    promotion_candidates = (
        projection.promotion_candidates() if reflection_enabled else []
    )
    if reflection_enabled and (materialized_records or promotion_candidates):
        reflection_json, llm_call = ledger.call_model_json(
            prompt_id=str(policy.get("reflection", "prompt", default="reflect@v1")),
            placeholders={
                "new_tuples": json.dumps(
                    [record.to_dict() for record in materialized_records],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "related_tuples": json.dumps(
                    [record.to_dict() for record in projection.related_records_for(materialized_records)],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "eligible_for_promotion": json.dumps(
                    [record.to_dict() for record in promotion_candidates],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            params={"temperature": 0, "schema": "reflection@v1", "session": session.id},
        )
        tokens_spent += llm_call.tokens["in"] + llm_call.tokens["out"]

        for merge in reflection_json.get("merges", []):
            event = make_event(
                type="merged",
                actor="llm",
                cause=Cause(
                    kind="llm",
                    ref="reflect@v1",
                    detail=str(merge.get("reason", "semantic dedup")),
                ),
                policy_hash=policy.hash,
                payload={
                    "into": merge["into"],
                    "from": list(merge.get("from", [])),
                    "resulting_value": projection.require_record(str(merge["into"])).value,
                },
                session=session.id,
                user=session.user_id,
                llm=llm_call,
                sources=[
                    str(merge["into"]),
                    *[str(value) for value in merge.get("from", [])],
                ],
            )
            ledger.append_event(event)
            merged_count += 1

        for supersede in reflection_json.get("supersedes", []):
            event = make_event(
                type="superseded",
                actor="llm",
                cause=Cause(
                    kind="llm",
                    ref="reflect@v1",
                    detail=str(supersede.get("reason", "contradiction resolution")),
                ),
                policy_hash=policy.hash,
                payload={
                    "old": supersede["old"],
                    "new": supersede["new"],
                    "reason": supersede.get("reason", ""),
                },
                session=session.id,
                user=session.user_id,
                llm=llm_call,
                sources=[str(supersede["old"]), str(supersede["new"])],
            )
            ledger.append_event(event)
            superseded_count += 1

        for promotion in reflection_json.get("promotions", []):
            if promotion.get("verdict") != "propose":
                continue
            proposal = make_event(
                type="promotion_proposed",
                actor="llm",
                cause=Cause(
                    kind="llm",
                    ref="reflect@v1",
                    detail=str(promotion.get("rationale", "promotion review")),
                ),
                policy_hash=policy.hash,
                payload={
                    "record": promotion["id"],
                    "to": "instinct",
                    "rationale": promotion.get("rationale", ""),
                },
                session=session.id,
                user=session.user_id,
                llm=llm_call,
                sources=[str(promotion["id"])],
            )
            ledger.append_event(proposal)
            promotions_proposed += 1
            if bool(policy.get("instinct", "autonomous", default=False)):
                approval = make_event(
                    type="promotion_approved",
                    actor="rule",
                    cause=Cause(
                        kind="policy",
                        ref="instinct.autonomous",
                        detail="automatic approval",
                    ),
                    policy_hash=policy.hash,
                    payload={"proposal": proposal.id, "by": "auto"},
                    session=session.id,
                    user=session.user_id,
                    sources=[proposal.id],
                )
                ledger.append_event(approval)

    repaired = projection.repair_tainted_records()
    if repaired:
        regen_event = make_event(
            type="regenerated",
            actor="rule",
            cause=Cause(
                kind="rule",
                ref="tainted@repair",
                detail="repaired from surviving sources",
            ),
            policy_hash=policy.hash,
            payload={
                "replaces": [record.id for record in repaired],
                "with": [record.id for record in repaired],
                "from_events": [source for record in repaired for source in record.sources],
            },
            session=session.id,
            user=session.user_id,
            sources=[source for record in repaired for source in record.sources],
        )
        ledger.append_event(regen_event)

    for record in projection.expire_due_records(utc_now()):
        expired_event = make_event(
            type="expired",
            actor="rule",
            cause=Cause(kind="rule", ref="ttl@rule", detail="ttl sweep"),
            policy_hash=policy.hash,
            payload={"record": record.id, "ttl": record.ttl},
            session=session.id,
            user=session.user_id,
        )
        ledger.append_event(expired_event)

    return CheckpointReport(
        triaged=triaged_counts,
        extracted=extracted_count,
        merged=merged_count,
        superseded=superseded_count,
        promotions_proposed=promotions_proposed,
        tokens_spent_on_memory=tokens_spent,
        tokens_saved_in_context=skipped_word_count,
    )
