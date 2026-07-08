"""Event envelope types and validation rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from memledger.ids import SPEC_VERSION, new_id, utc_now

DERIVED_EVENT_TYPES = frozenset(
    {
        "extracted",
        "merged",
        "promoted",
        "promotion_approved",
        "compacted",
        "superseded",
        "regenerated",
    }
)
CAUSE_KINDS = frozenset({"policy", "rule", "signal", "llm", "manual"})
ACTORS = frozenset({"dev", "rule", "llm", "system"})


@dataclass(slots=True)
class Cause:
    kind: str
    ref: str
    detail: str = ""

    def validate(self) -> None:
        if self.kind not in CAUSE_KINDS:
            raise ValueError(f"invalid cause kind: {self.kind}")
        if not self.ref:
            raise ValueError("cause.ref is required")


@dataclass(slots=True)
class LLMCall:
    model: str
    model_digest: str
    prompt: str
    prompt_hash: str
    params: dict[str, Any]
    input_hash: str
    output_hash: str
    cache_key: str
    cache_hit: bool
    tokens: dict[str, int]


@dataclass(slots=True)
class Event:
    id: str
    ts: str
    session: str | None
    user: str | None
    type: str
    actor: str
    cause: Cause
    policy_hash: str
    spec_version: str
    payload: dict[str, Any]
    llm: LLMCall | None = None
    sources: list[str] | None = None

    def validate(self) -> None:
        if self.actor not in ACTORS:
            raise ValueError(f"invalid actor: {self.actor}")
        self.cause.validate()
        if self.actor == "llm" and self.llm is None:
            raise ValueError("llm block is required when actor is llm")
        if self.actor != "llm" and self.llm is not None:
            raise ValueError("llm block is only allowed when actor is llm")
        if self.type in DERIVED_EVENT_TYPES and not self.sources:
            raise ValueError(f"sources are required for derived event type {self.type}")
        if not self.policy_hash:
            raise ValueError("policy_hash is required")
        if self.spec_version != SPEC_VERSION:
            raise ValueError(f"unsupported spec version: {self.spec_version}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["llm"] is None:
            data.pop("llm")
        if data["sources"] is None:
            data["sources"] = []
        return data


def make_event(
    *,
    type: str,
    actor: str,
    cause: Cause,
    policy_hash: str,
    payload: dict[str, Any],
    session: str | None = None,
    user: str | None = None,
    llm: LLMCall | None = None,
    sources: list[str] | None = None,
) -> Event:
    event = Event(
        id=new_id("ev"),
        ts=utc_now(),
        session=session,
        user=user,
        type=type,
        actor=actor,
        cause=cause,
        policy_hash=policy_hash,
        spec_version=SPEC_VERSION,
        payload=payload,
        llm=llm,
        sources=sources,
    )
    event.validate()
    return event


def event_from_dict(data: dict[str, Any]) -> Event:
    llm = data.get("llm")
    event = Event(
        id=data["id"],
        ts=data["ts"],
        session=data.get("session"),
        user=data.get("user"),
        type=data["type"],
        actor=data["actor"],
        cause=Cause(**data["cause"]),
        policy_hash=data["policy_hash"],
        spec_version=data["spec_version"],
        payload=data["payload"],
        llm=LLMCall(**llm) if llm else None,
        sources=data.get("sources", []),
    )
    event.validate()
    return event
