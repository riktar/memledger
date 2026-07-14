"""Retrieval pipeline: FTS stage, pre-scoring, reranking and logging."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memledger.embeddings.base import Embedder
from memledger.events import Cause, make_event
from memledger.ids import sha256_hex
from memledger.ledger import LedgerStore
from memledger.policy import Policy
from memledger.tuples import MemoryTuple

if TYPE_CHECKING:
    from memledger.session import Session


@dataclass(slots=True)
class Candidate:
    record: MemoryTuple
    stage1_score: float
    pre_score: float
    age_days: float


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _age_days(record: MemoryTuple) -> float:
    now = datetime.now(UTC)
    updated = _parse_ts(record.updated_ts)
    return max((now - updated).total_seconds() / 86400.0, 0.0)


def stage1_candidates(
    store: LedgerStore,
    policy: Policy,
    query: str,
    embedder: Embedder | None = None,
) -> list[Candidate]:
    limit = int(policy.get("retrieval", "candidates", default=40))
    if embedder is not None:
        fts_limit = limit // 2
        vector_limit = limit - fts_limit
    else:
        fts_limit = limit
        vector_limit = 0
    include_quarantined = bool(policy.get("retrieval", "include_quarantined", default=True))
    hit_scores: dict[str, float] = {}
    for record_id, stage1_score in store.search_record_ids_fts(query, fts_limit):
        hit_scores[record_id] = stage1_score
    if embedder is not None and vector_limit > 0:
        try:
            query_vec = embedder.embed([query])[0]
            for record_id, stage1_score in store.search_record_ids_vector(
                query_vec,
                embedder.index_version,
                vector_limit,
            ):
                if record_id not in hit_scores:
                    hit_scores[record_id] = stage1_score
        except Exception as exc:
            warnings.warn(
                f"vector retrieval failed, falling back to FTS-only: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            for record_id, stage1_score in store.search_record_ids_fts(query, limit):
                hit_scores.setdefault(record_id, stage1_score)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    half_life_days = max(policy.recency_half_life().total_seconds() / 86400.0, 1.0)
    impact_boost = float(policy.get("retrieval", "impact_boost", default=0.05))
    for record_id, stage1_score in hit_scores.items():
        if record_id in seen:
            continue
        record = store.get_record(record_id)
        if record is None or record.status in {"deleted", "superseded", "expired"}:
            continue
        if record.status == "quarantined" and not include_quarantined:
            continue
        age_days = _age_days(record)
        pre_score = stage1_score * math.exp(-(age_days * math.log(2)) / half_life_days)
        pre_score *= 1 + (impact_boost * record.impact)
        candidates.append(
            Candidate(
                record=record,
                stage1_score=stage1_score,
                pre_score=pre_score,
                age_days=age_days,
            )
        )
        seen.add(record_id)
    candidates.sort(key=lambda candidate: candidate.pre_score, reverse=True)
    return candidates


def _should_log_candidates(query_hash: str, log_mode: str) -> bool:
    if log_mode == "full":
        return True
    if log_mode == "off":
        return False
    return int(query_hash[:2], 16) < 3


def retrieve(
    session: Session,
    query: str,
    k: int | None = None,
) -> list[MemoryTuple]:
    policy = session.ledger.policy
    requested_k = k or policy.retrieval_k
    candidates = stage1_candidates(session.ledger.store, policy, query, session.ledger.embedder)
    selected = candidates[:requested_k]
    reasons = {candidate.record.id: "top pre-score candidate" for candidate in selected}

    if policy.get("retrieval", "rerank", default=True) and candidates:
        payload = {
            "query": query,
            "working_summary": session.working_summary(),
            "candidates": json.dumps(
                [
                    {
                        "id": candidate.record.id,
                        "text_form": candidate.record.text_form,
                        "layer": candidate.record.layer,
                        "status": candidate.record.status,
                        "impact": candidate.record.impact,
                        "age_days": round(candidate.age_days, 3),
                        "confidence": candidate.record.confidence,
                    }
                    for candidate in candidates
                ],
                ensure_ascii=False,
                sort_keys=True,
            ),
            "k": requested_k,
        }
        reranked, llm_call = session.ledger.call_model_json(
            prompt_id=str(policy.get("retrieval", "rerank_prompt", default="rerank@v1")),
            placeholders=payload,
            params={"temperature": 0, "schema": "selection@v1"},
        )
        chosen_ids = [str(item["id"]) for item in reranked.get("selected", [])][:requested_k]
        selected = [candidate for candidate in candidates if candidate.record.id in chosen_ids]
        selected.sort(key=lambda candidate: chosen_ids.index(candidate.record.id))
        reasons = {
            str(item["id"]): str(item.get("reason", "reranked selection")) for item in reranked.get("selected", [])
        }
        actor = "llm"
        cause = Cause(kind="llm", ref="rerank@v1", detail="retrieval selection")
        llm_block = llm_call
    else:
        actor = "rule"
        cause = Cause(kind="rule", ref="retrieval@rule", detail="pre-score selection")
        llm_block = None

    log_mode = str(policy.get("retrieval", "log", default="sampled"))
    if log_mode != "off":
        query_hash = sha256_hex(query)
        selected_ids = [candidate.record.id for candidate in selected]
        logged_candidates = (
            [candidate.record.id for candidate in candidates]
            if _should_log_candidates(query_hash, log_mode)
            else selected_ids
        )
        event = make_event(
            type="recalled",
            actor=actor,
            cause=cause,
            policy_hash=session.ledger.policy.hash,
            payload={
                "query_hash": query_hash,
                "candidates": logged_candidates,
                "selected": selected_ids,
                "reasons": reasons,
                "index_version": f"hybrid:{session.ledger.embedder.index_version}" if session.ledger.embedder is not None else "fts5",
            },
            session=session.id,
            user=session.user_id,
            llm=llm_block,
        )
        session.ledger.append_event(event)
    return [candidate.record for candidate in selected]
