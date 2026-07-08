"""Model backend protocol and response type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ModelResponse:
    content: str
    model: str
    model_digest: str
    tokens_in: int
    tokens_out: int


class ModelBackend(Protocol):
    """Minimal backend contract for framework model calls."""

    def complete(self, prompt_id: str, prompt: str, params: dict[str, object]) -> ModelResponse: ...
