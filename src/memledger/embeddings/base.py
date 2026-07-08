"""Embedding backend protocol."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    index_version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
