"""OpenAI-compatible backend wrapper."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from memledger.models.base import ModelBackend, ModelResponse


def parse_openai_compat_model_spec(model_spec: str) -> tuple[str, str]:
    if not model_spec.startswith("openai-compat:"):
        raise ValueError(f"unsupported model spec: {model_spec}")
    config = model_spec.removeprefix("openai-compat:")
    base_url, separator, model = config.partition("|")
    base_url = base_url.strip()
    model = model.strip()
    if not separator or not base_url or not model:
        raise ValueError(
            "openai-compat specs must use the format openai-compat:<base_url>|<model>"
        )
    return base_url, model


def resolve_openai_compat_api_key(base_url: str) -> str | None:
    host = urlparse(base_url).netloc.lower()
    if host.endswith("openrouter.ai"):
        return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return os.environ.get("OPENAI_API_KEY")


def build_openai_compat_backend(model_spec: str) -> "OpenAICompatBackend":
    base_url, model = parse_openai_compat_model_spec(model_spec)
    return OpenAICompatBackend(
        base_url=base_url,
        model=model,
        api_key=resolve_openai_compat_api_key(base_url),
    )


@dataclass(slots=True)
class OpenAICompatBackend(ModelBackend):
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 30.0

    def complete(self, prompt_id: str, prompt: str, params: dict[str, object]) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0),
            "response_format": {"type": "json_object"},
        }
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:4000]
            raise RuntimeError(
                f"OpenAI-compatible request failed for {self.model}: "
                f"{response.status_code} {detail}"
            ) from exc
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return ModelResponse(
            content=content if isinstance(content, str) else json.dumps(content),
            model=self.model,
            model_digest=data.get("system_fingerprint", self.model),
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
        )
