"""Policy loading, canonicalization and impact evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from memledger.ids import canonical_json, sha256_hex
from memledger.prompts import PromptRegistry, find_project_root

_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[dhm])$")


def parse_duration(raw: str | None) -> timedelta | None:
    if raw is None:
        return None
    match = _DURATION_RE.match(raw)
    if not match:
        raise ValueError(f"invalid duration: {raw}")
    value = int(match.group("value"))
    unit = match.group("unit")
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    return timedelta(minutes=value)


@dataclass(slots=True)
class Policy:
    """Runtime policy plus canonical hash."""

    raw: dict[str, Any]
    hash: str
    registry_hashes: dict[str, str]
    path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, registry: PromptRegistry | None = None) -> Policy:
        target = Path(path)
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        return cls.from_dict(raw, path=target, registry=registry)

    @classmethod
    def default(cls) -> Policy:
        root = find_project_root()
        return cls.from_yaml(root / "memory.policy.yaml")

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        path: Path | None = None,
        registry: PromptRegistry | None = None,
    ) -> Policy:
        registry = registry or PromptRegistry(path.parent if path else None)
        refs = cls._collect_registry_refs(raw)
        registry_hashes = {ref: registry.load(ref).hash for ref in refs if ref in registry_ref_allowlist()}
        canonical = cls._canonical_payload(raw, registry_hashes)
        return cls(
            raw=raw,
            hash=sha256_hex(canonical_json(canonical)),
            registry_hashes=registry_hashes,
            path=path,
        )

    @staticmethod
    def _collect_registry_refs(raw: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        for section, key in (
            ("triage", "formula"),
            ("extraction", "prompt"),
            ("retrieval", "rerank_prompt"),
            ("reflection", "prompt"),
        ):
            section_value = raw.get(section, {})
            ref = section_value.get(key)
            if isinstance(ref, str):
                refs.add(ref)
        return refs

    @staticmethod
    def _canonical_payload(raw: dict[str, Any], registry_hashes: dict[str, str]) -> dict[str, Any]:
        payload = dict(raw)
        payload["registry_hashes"] = dict(sorted(registry_hashes.items()))
        return payload

    def get(self, *path: str, default: Any = None) -> Any:
        current: Any = self.raw
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    @property
    def working_token_budget(self) -> int:
        return int(self.get("working", "token_budget", default=4000))

    @property
    def extraction_min_confidence(self) -> float:
        return float(self.get("extraction", "min_confidence", default=0.7))

    @property
    def quarantine_sessions(self) -> int:
        return int(self.get("quarantine", "new_facts", default=2))

    @property
    def retrieval_k(self) -> int:
        return int(self.get("retrieval", "k", default=5))

    def impact_score(
        self,
        *,
        feedback_total: float,
        outcome_total: float,
        recall_used: float,
        sessions_seen: int,
    ) -> float:
        weights = self.get("impact", "weights", default={})
        score = (
            float(weights.get("feedback", 0.0)) * feedback_total
            + float(weights.get("outcome", 0.0)) * outcome_total
            + float(weights.get("recall_used", 0.0)) * recall_used
            + float(weights.get("repeated", 0.0)) * max(sessions_seen - 1, 0)
        )
        clamp = self.get("impact", "clamp", default=[0, 20])
        return max(float(clamp[0]), min(float(clamp[1]), score))

    def promotion_eligible(self, *, impact: float, sessions_seen: int) -> bool:
        expression = str(self.get("instinct", "promote_when", default="impact >= 5 AND sessions_seen >= 3"))
        safe_expr = expression.replace("AND", "and").replace("OR", "or")
        if not re.fullmatch(r"[\w\s><=!.()&|+-]+", safe_expr):
            raise ValueError(f"unsafe promotion expression: {expression}")
        namespace = {"impact": impact, "sessions_seen": sessions_seen, "math": math}
        return bool(eval(safe_expr, {"__builtins__": {}}, namespace))

    def ttl_for_layer(self, layer: str) -> str | None:
        if layer == "instinct":
            return None
        return str(self.get("episodic", "retention", default="90d"))

    def raw_ttl(self) -> str:
        return str(self.get("episodic", "raw_retention", default="30d"))

    def recency_half_life(self) -> timedelta:
        half_life = self.get("retrieval", "recency_half_life", default="45d")
        parsed = parse_duration(str(half_life))
        if parsed is None:
            raise ValueError("retrieval.recency_half_life must be set")
        return parsed

    def copy_with_updates(self, updates: dict[str, Any]) -> Policy:
        merged = merge_dicts(self.raw, updates)
        return Policy.from_dict(merged, path=self.path)


def merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def registry_ref_allowlist() -> set[str]:
    return {"extract@v1", "rerank@v1", "reflect@v1", "salience@v1"}
