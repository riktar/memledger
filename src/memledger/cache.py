"""Deterministic cache helpers for framework LLM calls."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from memledger.ids import canonical_json, sha256_hex, utc_now


class CacheMissError(KeyError):
    """Raised when replay requires a missing cache entry."""


@dataclass(slots=True)
class CacheEntry:
    cache_key: str
    output: str
    tokens_in: int
    tokens_out: int
    ts: str


def make_cache_key(model: str, prompt_hash: str, input_hash: str, params: dict[str, Any]) -> str:
    payload = f"{model}\n{prompt_hash}\n{input_hash}\n{canonical_json(params)}"
    return sha256_hex(payload)


class DeterministicCache:
    """SQLite-backed cache keyed by model, prompt hash, input hash and params."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def get(self, cache_key: str) -> CacheEntry | None:
        row = self.connection.execute(
            "SELECT cache_key, output, tokens_in, tokens_out, ts FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return CacheEntry(
            cache_key=row[0],
            output=row[1],
            tokens_in=int(row[2]),
            tokens_out=int(row[3]),
            ts=row[4],
        )

    def require(self, cache_key: str) -> CacheEntry:
        entry = self.get(cache_key)
        if entry is None:
            raise CacheMissError(cache_key)
        return entry

    def set(self, cache_key: str, output: str, *, tokens_in: int, tokens_out: int) -> None:
        self.connection.execute(
            """
            INSERT INTO llm_cache (cache_key, output, tokens_in, tokens_out, ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO NOTHING
            """,
            (cache_key, output, tokens_in, tokens_out, utc_now()),
        )
        self.connection.commit()

    def stats(self) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0) FROM llm_cache"
        ).fetchone()
        assert row is not None
        return {
            "entries": int(row[0]),
            "tokens_in": int(row[1]),
            "tokens_out": int(row[2]),
        }
