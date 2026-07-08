"""Anthropic backend wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from memledger.models.base import ModelBackend, ModelResponse


@dataclass(slots=True)
class AnthropicBackend(ModelBackend):
    api_key: str
    model: str
    base_url: str = "https://api.anthropic.com/v1"
    timeout: float = 30.0

    def complete(self, prompt_id: str, prompt: str, params: dict[str, object]) -> ModelResponse:
        max_tokens = params.get("max_tokens", 2000)
        if not isinstance(max_tokens, int):
            max_tokens = 2000
        response = httpx.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": params.get("temperature", 0),
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        return ModelResponse(
            content=content,
            model=self.model,
            model_digest=self.model,
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
        )
