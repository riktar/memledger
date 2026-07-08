"""Tuple model and deterministic text-form rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memledger.ids import new_id, utc_now
from memledger.prompts import TextFormTemplates, load_text_form_templates

_SNAKE_CASE_RE = re.compile(r"[^a-z0-9]+")


def normalize_identifier(value: str) -> str:
    lowered = value.strip().lower().replace("-", "_")
    normalized = _SNAKE_CASE_RE.sub("_", lowered).strip("_")
    return normalized or "unknown"


def canonical_value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class MemoryTuple:
    id: str
    subject: str
    relation: str
    value: str | int | float | bool
    qualifiers: dict[str, Any]
    confidence: float
    layer: str
    status: str
    impact: float
    ttl: str | None
    sessions_seen: list[str]
    sources: list[str]
    text_form: str
    generator: str = "llm"
    created_ts: str = field(default_factory=utc_now)
    updated_ts: str = field(default_factory=utc_now)
    tainted: bool = False
    merged_from: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, canonical_value_key(self.value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "value": self.value,
            "qualifiers": self.qualifiers,
            "confidence": self.confidence,
            "layer": self.layer,
            "status": self.status,
            "impact": self.impact,
            "ttl": self.ttl,
            "sessions_seen": self.sessions_seen,
            "sources": self.sources,
            "text_form": self.text_form,
            "generator": self.generator,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "tainted": self.tainted,
            "merged_from": self.merged_from,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryTuple:
        return cls(
            id=data["id"],
            subject=data["subject"],
            relation=data["relation"],
            value=data["value"],
            qualifiers=data.get("qualifiers", {}),
            confidence=float(data["confidence"]),
            layer=data["layer"],
            status=data["status"],
            impact=float(data.get("impact", 0.0)),
            ttl=data.get("ttl"),
            sessions_seen=list(data.get("sessions_seen", [])),
            sources=list(data.get("sources", [])),
            text_form=data["text_form"],
            generator=data.get("generator", "llm"),
            created_ts=data.get("created_ts", utc_now()),
            updated_ts=data.get("updated_ts", utc_now()),
            tainted=bool(data.get("tainted", False)),
            merged_from=list(data.get("merged_from", [])),
            history=list(data.get("history", [])),
        )


def render_text_form(
    subject: str,
    relation: str,
    value: Any,
    qualifiers: dict[str, Any] | None = None,
    templates: TextFormTemplates | None = None,
) -> tuple[str, str]:
    templates = templates or load_text_form_templates()
    qualifiers = qualifiers or {}
    return templates.render(
        relation,
        {
            "subject": subject,
            "relation": relation,
            "value": value,
            "when": qualifiers.get("when", ""),
            "context": qualifiers.get("context", ""),
        },
    )


def make_tuple(
    *,
    subject: str,
    relation: str,
    value: str | int | float | bool,
    qualifiers: dict[str, Any] | None = None,
    confidence: float,
    layer: str,
    status: str,
    ttl: str | None,
    sessions_seen: list[str],
    sources: list[str],
    text_form: str | None = None,
    templates: TextFormTemplates | None = None,
) -> MemoryTuple:
    subject_norm = normalize_identifier(subject)
    relation_norm = normalize_identifier(relation)
    qualifier_map = qualifiers or {}
    if text_form is None:
        text_form, generator = render_text_form(
            subject_norm,
            relation_norm,
            value,
            qualifier_map,
            templates,
        )
    else:
        rendered, generator = render_text_form(
            subject_norm,
            relation_norm,
            value,
            qualifier_map,
            templates,
        )
        text_form = rendered if generator != "fallback:default" else text_form.rstrip(".") + "."
    return MemoryTuple(
        id=new_id("tu"),
        subject=subject_norm,
        relation=relation_norm,
        value=value,
        qualifiers=qualifier_map,
        confidence=confidence,
        layer=layer,
        status=status,
        impact=0.0,
        ttl=ttl,
        sessions_seen=sessions_seen,
        sources=sources,
        text_form=text_form,
        generator=generator,
    )
