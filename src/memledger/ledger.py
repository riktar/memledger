"""SQLite-backed append-only ledger store."""

from __future__ import annotations

import array
import heapq
import json
import math
import os
import re
import sqlite3
import warnings
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from memledger.embeddings.base import Embedder
from memledger.events import Event, event_from_dict
from memledger.ids import canonical_json, sha256_hex
from memledger.tuples import MemoryTuple

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
VECTOR_INDEX_VERSION_META = "vector_index_version"
_NON_INDEXED_STATUSES = frozenset({"deleted", "superseded", "expired"})


def _pack_vector(values: list[float]) -> bytes:
    return array.array("f", values).tobytes()


def _unpack_vector(blob: bytes) -> list[float]:
    packed = array.array("f")
    packed.frombytes(blob)
    return list(packed)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


class LedgerLockError(RuntimeError):
    """Raised when another writer already holds the database lock."""


class LedgerStore:
    """Low-level SQLite event store and projection persistence."""

    def __init__(self, path: str | Path, *, embedder: Embedder | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._lock_fd = self._acquire_lock(self.lock_path)
        self._lock_finalizer = weakref.finalize(self, self._release_lock, self.lock_path, self._lock_fd)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._transaction_depth = 0
        self.embedder = embedder
        self._init_schema()
        if embedder is not None:
            self.ensure_vector_index_version(embedder.index_version)

    @contextmanager
    def transaction(self) -> Iterable[None]:
        savepoint = f"memledger_txn_{self._transaction_depth}"
        self._transaction_depth += 1
        self.connection.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
        except Exception:
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        finally:
            self._transaction_depth -= 1

    def _maybe_commit(self) -> None:
        if self._transaction_depth == 0:
            self.connection.commit()

    @staticmethod
    def _acquire_lock(lock_path: Path) -> int:
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError as exc:
            raise LedgerLockError(f"ledger is already locked: {lock_path}") from exc

    @staticmethod
    def _release_lock(lock_path: Path, lock_fd: int) -> None:
        try:
            os.close(lock_fd)
        except OSError:
            pass
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def close(self) -> None:
        self.connection.close()
        self._lock_finalizer()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                session TEXT,
                type TEXT NOT NULL,
                actor TEXT NOT NULL,
                envelope_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session, ts);
            CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(type, ts);

            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                layer TEXT NOT NULL,
                status TEXT NOT NULL,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                value_json TEXT NOT NULL,
                impact REAL NOT NULL,
                sessions_seen_count INTEGER NOT NULL,
                text_form TEXT NOT NULL,
                tainted INTEGER NOT NULL DEFAULT 0,
                updated_ts TEXT NOT NULL,
                data_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_layer_status ON records(layer, status);
            CREATE INDEX IF NOT EXISTS idx_records_subject_relation ON records(subject, relation);

            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(target_id UNINDEXED, kind UNINDEXED, text);

            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                index_version TEXT NOT NULL,
                vector BLOB
            );

            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                output TEXT NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                ts TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._maybe_commit()

    def append_event(self, event: Event) -> None:
        self.connection.execute(
            "INSERT INTO events (id, ts, session, type, actor, envelope_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.ts,
                event.session,
                event.type,
                event.actor,
                canonical_json(event.to_dict()),
            ),
        )
        self._maybe_commit()

    def get_event(self, event_id: str) -> Event | None:
        row = self.connection.execute("SELECT envelope_json FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return event_from_dict(json.loads(row[0]))

    def iter_events(
        self,
        *,
        session: str | None = None,
        type: str | None = None,
        since: str | None = None,
        at: str | None = None,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[str] = []
        if session is not None:
            clauses.append("session = ?")
            params.append(session)
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if at is not None:
            clauses.append("ts <= ?")
            params.append(at)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT envelope_json FROM events {where} ORDER BY ts, id",
            params,
        ).fetchall()
        return [event_from_dict(json.loads(row[0])) for row in rows]

    def session_ids(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT session FROM events WHERE session IS NOT NULL ORDER BY session"
        ).fetchall()
        return [str(row[0]) for row in rows]

    def next_turn(self, session_id: str) -> int:
        try:
            row = self.connection.execute(
                """
                SELECT MAX(CAST(json_extract(envelope_json, '$.payload.turn') AS INTEGER))
                FROM events
                WHERE session = ? AND type = 'observed'
                """,
                (session_id,),
            ).fetchone()
            if row is not None and row[0] is not None:
                return int(row[0]) + 1
            return 1
        except sqlite3.OperationalError:
            max_turn = 0
            for event in self.iter_events(session=session_id, type="observed"):
                max_turn = max(max_turn, int(event.payload.get("turn", 0)))
            return max_turn + 1

    def upsert_record(self, record: MemoryTuple) -> None:
        self.connection.execute(
            """
            INSERT INTO records (
                id, layer, status, subject, relation, value_json,
                impact, sessions_seen_count, text_form, tainted, updated_ts, data_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                layer = excluded.layer,
                status = excluded.status,
                subject = excluded.subject,
                relation = excluded.relation,
                value_json = excluded.value_json,
                impact = excluded.impact,
                sessions_seen_count = excluded.sessions_seen_count,
                text_form = excluded.text_form,
                tainted = excluded.tainted,
                updated_ts = excluded.updated_ts,
                data_json = excluded.data_json
            """,
            (
                record.id,
                record.layer,
                record.status,
                record.subject,
                record.relation,
                canonical_json(record.value),
                record.impact,
                len(record.sessions_seen),
                record.text_form,
                1 if record.tainted else 0,
                record.updated_ts,
                canonical_json(record.to_dict()),
            ),
        )
        self.connection.execute("DELETE FROM fts WHERE target_id = ? AND kind = 'record'", (record.id,))
        self.connection.execute(
            "INSERT INTO fts (target_id, kind, text) VALUES (?, 'record', ?)",
            (record.id, record.text_form),
        )
        if record.status in _NON_INDEXED_STATUSES:
            self._delete_vector(record.id)
        elif self.embedder is not None and record.text_form.strip():
            try:
                self.ensure_vector_index_version(self.embedder.index_version)
                vector = self.embedder.embed([record.text_form])[0]
                self._upsert_vector(record.id, vector, self.embedder.index_version)
            except Exception as exc:
                warnings.warn(
                    f"skipped vector index for {record.id}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._maybe_commit()

    def ensure_vector_index_version(self, index_version: str) -> None:
        stored = self.read_meta(VECTOR_INDEX_VERSION_META)
        if stored is not None and stored != index_version:
            self.connection.execute("DELETE FROM vectors")
        if stored != index_version:
            self.write_meta(VECTOR_INDEX_VERSION_META, index_version)

    def has_vector(self, record_id: str, index_version: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM vectors WHERE id = ? AND index_version = ?",
            (record_id, index_version),
        ).fetchone()
        return row is not None

    def _upsert_vector(self, record_id: str, vector: list[float], index_version: str) -> None:
        self.connection.execute(
            """
            INSERT INTO vectors (id, index_version, vector) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                index_version = excluded.index_version,
                vector = excluded.vector
            """,
            (record_id, index_version, _pack_vector(vector)),
        )

    def _delete_vector(self, record_id: str) -> None:
        self.connection.execute("DELETE FROM vectors WHERE id = ?", (record_id,))

    def delete_record_index(self, record_id: str) -> None:
        self.connection.execute("DELETE FROM fts WHERE target_id = ? AND kind = 'record'", (record_id,))
        self._delete_vector(record_id)
        self._maybe_commit()

    def get_record(self, record_id: str) -> MemoryTuple | None:
        row = self.connection.execute("SELECT data_json FROM records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        return MemoryTuple.from_dict(json.loads(row[0]))

    def iter_records(self, *, include_deleted: bool = True) -> list[MemoryTuple]:
        if include_deleted:
            rows = self.connection.execute("SELECT data_json FROM records ORDER BY id").fetchall()
        else:
            rows = self.connection.execute(
                "SELECT data_json FROM records WHERE status != 'deleted' ORDER BY id"
            ).fetchall()
        return [MemoryTuple.from_dict(json.loads(row[0])) for row in rows]

    def find_record_by_key(self, subject: str, relation: str, value_json: str) -> MemoryTuple | None:
        row = self.connection.execute(
            """
            SELECT data_json FROM records
            WHERE subject = ? AND relation = ? AND value_json = ? AND status != 'deleted'
            ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'quarantined' THEN 1 ELSE 2 END, id
            LIMIT 1
            """,
            (subject, relation, value_json),
        ).fetchone()
        if row is None:
            return None
        return MemoryTuple.from_dict(json.loads(row[0]))

    def find_related_records(self, record: MemoryTuple) -> list[MemoryTuple]:
        rows = self.connection.execute(
            """
            SELECT data_json FROM records
            WHERE id != ? AND status != 'deleted' AND (subject = ? OR relation = ?)
            ORDER BY id
            """,
            (record.id, record.subject, record.relation),
        ).fetchall()
        return [MemoryTuple.from_dict(json.loads(row[0])) for row in rows]

    def add_turn_to_fts(self, event_id: str, text: str) -> None:
        self.connection.execute("DELETE FROM fts WHERE target_id = ? AND kind = 'turn'", (event_id,))
        self.connection.execute(
            "INSERT INTO fts (target_id, kind, text) VALUES (?, 'turn', ?)",
            (event_id, text),
        )
        self._maybe_commit()

    def search_record_ids_fts(self, query: str, limit: int) -> list[tuple[str, float]]:
        tokens = [token.lower() for token in _QUERY_TOKEN_RE.findall(query) if len(token) >= 3]
        try:
            rows = self.connection.execute(
                """
                SELECT target_id, bm25(fts) AS score
                FROM fts
                WHERE fts MATCH ? AND kind = 'record'
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
            if rows:
                return [(str(row[0]), float(-row[1])) for row in rows]
        except sqlite3.OperationalError:
            pass
        if tokens:
            try:
                fts_query = " OR ".join(f'"{token}"' for token in tokens)
                rows = self.connection.execute(
                    """
                    SELECT target_id, bm25(fts) AS score
                    FROM fts
                    WHERE fts MATCH ? AND kind = 'record'
                    ORDER BY score
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
                if rows:
                    return [(str(row[0]), float(-row[1])) for row in rows]
            except sqlite3.OperationalError:
                pass
        like_query = f"%{query}%"
        rows = self.connection.execute(
            "SELECT id, 1.0 AS score FROM records WHERE text_form LIKE ? AND status != 'deleted' LIMIT ?",
            (like_query, limit),
        ).fetchall()
        if rows:
            return [(str(row[0]), float(row[1])) for row in rows]
        if not tokens:
            return []
        rows = self.connection.execute("SELECT id, text_form FROM records WHERE status != 'deleted'").fetchall()
        scored_rows: list[tuple[str, float]] = []
        for row in rows:
            text = str(row[1]).lower()
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored_rows.append((str(row[0]), float(score)))
        scored_rows.sort(key=lambda item: (-item[1], item[0]))
        return scored_rows[:limit]

    def search_record_ids_vector(
        self,
        query_vec: list[float],
        index_version: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        cursor = self.connection.execute(
            """
            SELECT v.id, v.vector
            FROM vectors v
            INNER JOIN records r ON r.id = v.id
            WHERE v.index_version = ? AND r.status NOT IN ('deleted', 'superseded', 'expired')
            """,
            (index_version,),
        )
        scored_rows: list[tuple[str, float]] = []
        for row in cursor:
            score = _cosine_similarity(query_vec, _unpack_vector(bytes(row[1])))
            if score > 0.0:
                scored_rows.append((str(row[0]), score))
        return heapq.nsmallest(limit, scored_rows, key=lambda item: (-item[1], item[0]))

    def clear_projections(self) -> None:
        self.connection.executescript(
            """
            DELETE FROM records;
            DELETE FROM fts;
            DELETE FROM vectors;
            """
        )
        self._maybe_commit()

    def projection_digest(self) -> str:
        rows = self.connection.execute("SELECT data_json FROM records ORDER BY id").fetchall()
        payload = [json.loads(row[0]) for row in rows]
        return sha256_hex(payload)

    def write_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._maybe_commit()

    def read_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def delete_cache_entry(self, cache_key: str) -> None:
        self.connection.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
        self._maybe_commit()

    def count_events(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM events").fetchone()
        assert row is not None
        return int(row[0])

    def count_records(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM records").fetchone()
        assert row is not None
        return int(row[0])

    def iter_rows(self, query: str, params: Iterable[object] = ()) -> list[sqlite3.Row]:
        return self.connection.execute(query, tuple(params)).fetchall()
