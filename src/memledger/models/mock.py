"""Deterministic mock model backend for tests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from memledger.models.base import ModelBackend, ModelResponse

Responder = Callable[[str, dict[str, object]], str | dict[str, object]]


@dataclass(slots=True)
class MockModelBackend(ModelBackend):
    """Rule-based mock backend that returns deterministic JSON."""

    responders: dict[str, Responder] = field(default_factory=dict)
    model_name: str = "mock/mock"
    model_digest: str = "mock-v1"
    call_count: int = 0
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, prompt_id: str, prompt: str, params: dict[str, object]) -> ModelResponse:
        self.call_count += 1
        self.calls.append((prompt_id, prompt))
        responder = self.responders.get(prompt_id)
        if responder is not None:
            output = responder(prompt, params)
        elif prompt_id == "extract@v1":
            output = self._default_extract(prompt)
        elif prompt_id == "rerank@v1":
            output = self._default_rerank(prompt)
        elif prompt_id == "reflect@v1":
            output = self._default_reflect(prompt)
        else:
            output = {}
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False, sort_keys=True)
        return ModelResponse(
            content=output,
            model=self.model_name,
            model_digest=self.model_digest,
            tokens_in=len(prompt.split()),
            tokens_out=len(output.split()),
        )

    def reset(self) -> None:
        self.call_count = 0
        self.calls.clear()

    def _default_extract(self, prompt: str) -> dict[str, object]:
        transcript = prompt.split("## Transcript", 1)[-1]
        tuples: list[dict[str, object]] = []
        notes: list[str] = []
        turn_pattern = re.compile(r"^(?P<turn>\d+)\.\s+(?P<role>\w+):\s+(?P<text>.*)$", re.M)
        for match in turn_pattern.finditer(transcript):
            turn = int(match.group("turn"))
            text = match.group("text").strip()
            lower = text.lower()
            if "maximum confidence" in lower or "ignore these rules" in lower:
                notes.append(f"suspicious turn {turn}")
            name_match = re.search(r"(?:actually,\s*)?my name is ([A-Z][A-Za-z0-9_-]+)", text, re.I)
            if name_match:
                tuples.append(
                    {
                        "subject": "user",
                        "relation": "name",
                        "value": name_match.group(1),
                        "qualifiers": {},
                        "confidence": 0.95,
                        "evidence": [turn],
                        "text_form": f"The user's name is {name_match.group(1)}.",
                    }
                )
            language_match = re.search(r"(?:prefer|only use)\s+([A-Za-z+#]+)", text, re.I)
            if language_match:
                tuples.append(
                    {
                        "subject": "user",
                        "relation": "preferred_language",
                        "value": language_match.group(1).lower(),
                        "qualifiers": {},
                        "confidence": 0.95,
                        "evidence": [turn],
                        "text_form": f"The user prefers {language_match.group(1)} as their language.",
                    }
                )
            works_at_match = re.search(r"(?:work|works) at ([A-Z][A-Za-z0-9_-]+)", text)
            if works_at_match:
                tuples.append(
                    {
                        "subject": "user",
                        "relation": "works_at",
                        "value": works_at_match.group(1),
                        "qualifiers": {},
                        "confidence": 0.9,
                        "evidence": [turn],
                        "text_form": f"The user works at {works_at_match.group(1)}.",
                    }
                )
            failure_match = re.search(r"([A-Za-z0-9_-]+) failed(?: due to (.+))?", text, re.I)
            if failure_match:
                tuples.append(
                    {
                        "subject": "user",
                        "relation": "outcome_failure",
                        "value": failure_match.group(1),
                        "qualifiers": {},
                        "confidence": 0.8,
                        "evidence": [turn],
                        "text_form": f"{failure_match.group(1)} failed for the user.",
                    }
                )
        session_match = re.search(r'"session":\s*"([^"]+)"', prompt)
        return {
            "session": session_match.group(1) if session_match else "se_mock",
            "tuples": tuples,
            "notes": notes,
        }

    def _default_rerank(self, prompt: str) -> dict[str, object]:
        ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
        selected = [{"id": candidate_id, "reason": "top candidate by mock reranker"} for candidate_id in ids[:3]]
        return {"selected": selected, "flags": []}

    def _default_reflect(self, prompt: str) -> dict[str, object]:
        candidate_ids = re.findall(r'"id":\s*"(tu_[^"]+)"', prompt)
        promotions: list[dict[str, str]] = []
        if candidate_ids:
            promotions.append(
                {
                    "id": candidate_ids[0],
                    "verdict": "propose",
                    "rationale": "stable mock tuple suitable for instinct memory",
                }
            )
        return {"merges": [], "supersedes": [], "promotions": promotions, "flags": []}
