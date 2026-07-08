"""Deterministic identifiers, canonical JSON and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

SPEC_VERSION = "0.1"
ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
PREFIXES = frozenset({"ev", "tu", "ep", "in", "se", "pr", "po"})

_ulid_lock = threading.Lock()
_last_timestamp_ms = -1
_last_random = 0


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_hex(data: Any) -> str:
    if isinstance(data, bytes):
        payload = data
    elif isinstance(data, str):
        payload = data.encode("utf-8")
    else:
        payload = canonical_json(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _encode_ulid(value: int) -> str:
    chars: list[str] = []
    for _ in range(26):
        value, remainder = divmod(value, 32)
        chars.append(ULID_ALPHABET[remainder])
    return "".join(reversed(chars))


def _next_ulid_body() -> str:
    global _last_random, _last_timestamp_ms
    timestamp_ms = int(time.time() * 1000)
    with _ulid_lock:
        if timestamp_ms > _last_timestamp_ms:
            _last_timestamp_ms = timestamp_ms
            _last_random = int.from_bytes(os.urandom(10), "big")
        else:
            _last_random = (_last_random + 1) & ((1 << 80) - 1)
        value = (_last_timestamp_ms << 80) | _last_random
        return _encode_ulid(value)


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise ValueError(f"unsupported prefix: {prefix}")
    return f"{prefix}_{_next_ulid_body()}"


def truncate_hash(value: str, length: int = 12) -> str:
    return value[:length]
