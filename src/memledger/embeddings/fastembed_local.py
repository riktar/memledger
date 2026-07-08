"""Optional fastembed-backed local embeddings."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from memledger.embeddings.base import Embedder


@dataclass(slots=True)
class FastEmbedLocal(Embedder):
    model_name: str = "BAAI/bge-small-en-v1.5"
    index_version: str = "fastembed-local-v1"
    _model: Any = None

    def __post_init__(self) -> None:
        try:
            module = importlib.import_module("fastembed")
        except ImportError as exc:
            raise RuntimeError("install memledger[local] to use local embeddings") from exc
        text_embedding = module.TextEmbedding
        self._model = text_embedding(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._model.embed(texts)]
