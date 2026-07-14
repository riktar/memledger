from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from memledger import Ledger, Policy
from memledger.events import Cause, make_event
from memledger.ledger import _unpack_vector
from memledger.models.mock import MockModelBackend
from memledger.retrieval import stage1_candidates
from memledger.tuples import MemoryTuple, make_tuple


@dataclass
class MockEmbedder:
  mappings: dict[str, list[float]]
  default: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
  index_version: str = "mock-v1"

  def embed(self, texts: list[str]) -> list[list[float]]:
    return [self.mappings.get(text, self.default) for text in texts]


def _recall_policy() -> Policy:
  return Policy.default().copy_with_updates(
    {
      "retrieval": {
        "rerank": False,
        "log": "off",
        "candidates": 10,
      }
    }
  )


def _save_record(ledger: Ledger, record: MemoryTuple) -> None:
  event = make_event(
    type="seeded",
    actor="dev",
    cause=Cause(kind="manual", ref="test", detail="hybrid recall test"),
    policy_hash=ledger.policy.hash,
    payload={"tuples": [record.to_dict()]},
    user=None,
    session=None,
  )
  ledger.append_event(event)


def _active_record(
  *,
  subject: str,
  relation: str,
  value: str,
  text_form: str,
) -> MemoryTuple:
  return make_tuple(
    subject=subject,
    relation=relation,
    value=value,
    qualifiers={},
    confidence=1.0,
    layer="episodic",
    status="active",
    ttl=None,
    sessions_seen=[],
    sources=[],
    text_form=text_form,
  )


def test_upsert_writes_vector(tmp_path: Path) -> None:
  record = _active_record(
    subject="user",
    relation="prefers_language",
    value="Python",
    text_form="The user prefers Python for all projects",
  )
  vector = [1.0, 0.0, 0.0]
  embedder = MockEmbedder({record.text_form: vector})
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=_recall_policy(),
    model_backend=MockModelBackend(),
    embedder=embedder,
  )
  _save_record(ledger, record)

  row = ledger.store.connection.execute(
    "SELECT index_version, vector FROM vectors WHERE id = ?",
    (record.id,),
  ).fetchone()
  assert row is not None
  assert row[0] == "mock-v1"
  assert _unpack_vector(bytes(row[1])) == vector
  ledger.close()


def test_vector_search_orders_by_cosine(tmp_path: Path) -> None:
  near_record = _active_record(
    subject="a",
    relation="fact",
    value="near",
    text_form="near match text",
  )
  far_record = _active_record(
    subject="b",
    relation="fact",
    value="far",
    text_form="far match text",
  )
  near = [1.0, 0.1]
  far = [0.1, 1.0]
  query = [1.0, 0.0]
  embedder = MockEmbedder(
    {
      near_record.text_form: near,
      far_record.text_form: far,
    }
  )
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=_recall_policy(),
    model_backend=MockModelBackend(),
    embedder=embedder,
  )
  _save_record(ledger, near_record)
  _save_record(ledger, far_record)

  hits = ledger.store.search_record_ids_vector(query, embedder.index_version, limit=2)
  assert [record_id for record_id, _score in hits] == [near_record.id, far_record.id]
  assert hits[0][1] > hits[1][1]
  ledger.close()


def test_hybrid_surfaces_lexically_disjoint_fact(tmp_path: Path) -> None:
  record_a = _active_record(
    subject="user",
    relation="prefers",
    value="python",
    text_form="User prefers python keyword alpha",
  )
  record_b = _active_record(
    subject="user",
    relation="enjoys",
    value="scripting",
    text_form="User enjoys beta scripting hobby unrelated",
  )
  query = "python keyword question"
  vec_a = [1.0, 0.0]
  vec_b = [0.98, 0.02]
  vec_query = [1.0, 0.0]
  embedder = MockEmbedder(
    {
      record_a.text_form: vec_a,
      record_b.text_form: vec_b,
      query: vec_query,
    }
  )
  policy = _recall_policy()
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=policy,
    model_backend=MockModelBackend(),
    embedder=embedder,
  )
  _save_record(ledger, record_a)
  _save_record(ledger, record_b)

  fts_only = stage1_candidates(ledger.store, policy, query, embedder=None)
  hybrid = stage1_candidates(ledger.store, policy, query, embedder=embedder)

  fts_ids = {candidate.record.id for candidate in fts_only}
  hybrid_ids = {candidate.record.id for candidate in hybrid}
  assert record_a.id in fts_ids
  assert record_b.id not in fts_ids
  assert record_a.id in hybrid_ids
  assert record_b.id in hybrid_ids
  ledger.close()


def test_embedder_none_matches_fts_baseline(tmp_path: Path) -> None:
  record = _active_record(
    subject="user",
    relation="prefers",
    value="python",
    text_form="User prefers python keyword alpha",
  )
  embedder = MockEmbedder({record.text_form: [1.0, 0.0, 0.0]})
  policy = _recall_policy()
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=policy,
    model_backend=MockModelBackend(),
    embedder=embedder,
  )
  _save_record(ledger, record)

  without_embedder = stage1_candidates(ledger.store, policy, "python keyword", embedder=None)
  ledger_no_embedder = Ledger(
    path=str(tmp_path / "memory-none.db"),
    policy=policy,
    model_backend=MockModelBackend(),
    embedder=None,
  )
  _save_record(ledger_no_embedder, record)
  baseline = stage1_candidates(ledger_no_embedder.store, policy, "python keyword", embedder=None)

  assert [(candidate.record.id, candidate.stage1_score) for candidate in without_embedder] == [
    (candidate.record.id, candidate.stage1_score) for candidate in baseline
  ]
  ledger.close()
  ledger_no_embedder.close()


def test_reindex_populates_existing(tmp_path: Path) -> None:
  record = _active_record(
    subject="user",
    relation="note",
    value="existing",
    text_form="Existing record without initial vector index",
  )
  policy = _recall_policy()
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=policy,
    model_backend=MockModelBackend(),
    embedder=None,
  )
  _save_record(ledger, record)
  assert ledger.store.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0

  ledger.embedder = MockEmbedder({record.text_form: [0.5, 0.5, 0.0]})
  ledger.store.embedder = ledger.embedder
  assert ledger.reindex_vectors() == 1
  assert ledger.store.connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 1
  assert ledger.reindex_vectors() == 0
  ledger.close()


def test_deleted_record_drops_vector(tmp_path: Path) -> None:
  record = _active_record(
    subject="user",
    relation="temp",
    value="delete-me",
    text_form="Temporary fact to delete",
  )
  embedder = MockEmbedder({record.text_form: [1.0, 0.0]})
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=_recall_policy(),
    model_backend=MockModelBackend(),
    embedder=embedder,
  )
  _save_record(ledger, record)
  assert ledger.store.has_vector(record.id, embedder.index_version)

  record.status = "deleted"
  ledger.store.upsert_record(record)
  assert not ledger.store.has_vector(record.id, embedder.index_version)
  ledger.close()


def test_rebuild_reindexes_vectors(tmp_path: Path) -> None:
  record = _active_record(
    subject="user",
    relation="note",
    value="rebuild",
    text_form="Rebuild should restore vector index",
  )
  embedder = MockEmbedder({record.text_form: [0.2, 0.8, 0.0]})
  ledger = Ledger(
    path=str(tmp_path / "memory.db"),
    policy=_recall_policy(),
    model_backend=MockModelBackend(),
    embedder=embedder,
  )
  _save_record(ledger, record)
  session = ledger.session(user_id="user")
  session.observe(user="hello", assistant="world")
  assert ledger.rebuild() is True
  assert ledger.store.has_vector(record.id, embedder.index_version)
  ledger.close()
