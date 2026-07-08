from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.run_locomo import NO_INFORMATION
from evals.run_locomo import _token_f1
from evals.run_locomo import load_samples, run_benchmark
from memledger.models.mock import MockModelBackend


def test_locomo_runner_scores_small_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture_path = Path("tests/fixtures/locomo-mini.json")
    samples = load_samples(fixture_path)
    assert samples[0].sample_id == "conv-mini"

    def extract_responder(prompt: str, params: dict[str, object]) -> dict[str, object]:
        del params
        transcript = prompt.split("## Transcript", 1)[-1]
        tuples: list[dict[str, object]] = []
        if "7 May 2023" in transcript:
            tuples.append(
                {
                    "subject": "alice",
                    "relation": "started_guitar_lessons_on",
                    "value": "7 May 2023",
                    "qualifiers": {},
                    "confidence": 0.95,
                    "evidence": [1],
                    "text_form": "Alice started guitar lessons on 7 May 2023.",
                }
            )
            tuples.append(
                {
                    "subject": "alice",
                    "relation": "learning_instrument",
                    "value": "guitar",
                    "qualifiers": {},
                    "confidence": 0.95,
                    "evidence": [1],
                    "text_form": "Alice is learning guitar.",
                }
            )
        elif "guitar practice" in transcript:
            tuples.append(
                {
                    "subject": "alice",
                    "relation": "learning_instrument",
                    "value": "guitar",
                    "qualifiers": {},
                    "confidence": 0.95,
                    "evidence": [1],
                    "text_form": "Alice is learning guitar.",
                }
            )
        return {"session": "se_mock", "tuples": tuples, "notes": []}

    def answer_responder(prompt: str, params: dict[str, object]) -> dict[str, object]:
        del params
        if "When did Alice start guitar lessons?" in prompt:
            return {"answer": "7 May 2023"}
        if "What instrument is Alice learning?" in prompt:
            return {"answer": "guitar"}
        return {"answer": "No information available"}

    memory_backend = MockModelBackend(
        responders={
            "extract@v1": extract_responder,
            "reflect@v1": lambda prompt, params: {"merges": [], "supersedes": [], "promotions": [], "flags": []},
        }
    )
    answer_backend = MockModelBackend(responders={"locomo_answer@v1": answer_responder})

    summary = run_benchmark(
        data_file=fixture_path,
        output_file=tmp_path / "locomo.json",
        stats_file=tmp_path / "locomo_stats.json",
        ledger_dir=tmp_path / "ledgers",
        policy_path=Path("memory.policy.yaml"),
        memory_backend=memory_backend,
        answer_backend=answer_backend,
        sample_limit=None,
        question_limit=None,
        categories=None,
        sample_ids=None,
        retrieval_k=5,
        checkpoint_every=1,
        reingest=True,
        log_every_questions=1,
    )

    assert summary["sample_count"] == 1
    assert summary["question_count"] == 2
    assert summary["f1"] == 1.0
    assert summary["retrieval_recall"] == 1.0
    assert summary["official_metrics"]["accuracy_by_category"]["2"] == 1.0
    assert summary["official_metrics"]["accuracy_by_category"]["4"] == 1.0

    persisted = json.loads((tmp_path / "locomo.json").read_text(encoding="utf-8"))
    assert persisted["samples"][0]["qa"][0]["prediction"] == "7 May 2023"
    stats = json.loads((tmp_path / "locomo_stats.json").read_text(encoding="utf-8"))
    assert stats["overall_accuracy"] == 1.0
    assert stats["overall_recall_accuracy"] == 1.0

    stdout = capsys.readouterr().out
    assert "[plan]" in stdout
    assert "eta=" in stdout


def test_locomo_loader_accepts_category_5_adversarial_shape(tmp_path: Path) -> None:
    data = [
        {
            "sample_id": "conv-adv",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "7 May 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "We talked about music."}
                ],
            },
            "observation": {},
            "session_summary": {},
            "event_summary": [],
            "qa": [
                {
                    "question": "What did Alice realize after her charity race?",
                    "evidence": ["D1:1"],
                    "category": 5,
                    "adversarial_answer": "self-care is important",
                }
            ],
        }
    ]
    fixture = tmp_path / "locomo-adv.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")

    samples = load_samples(fixture)

    assert samples[0].qa[0].answer == NO_INFORMATION
    assert samples[0].qa[0].adversarial_answer == "self-care is important"


def test_official_locomo_token_f1_uses_stemming() -> None:
    assert _token_f1("guitars", "guitar") == 1.0


def test_checkpoint_every_groups_multiple_source_sessions(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/locomo-mini.json")

    def extract_responder(prompt: str, params: dict[str, object]) -> dict[str, object]:
        del params
        tuples: list[dict[str, object]] = []
        if "7 May 2023" in prompt:
            tuples.append(
                {
                    "subject": "alice",
                    "relation": "started_guitar_lessons_on",
                    "value": "7 May 2023",
                    "qualifiers": {},
                    "confidence": 0.95,
                    "evidence": [1],
                    "text_form": "Alice started guitar lessons on 7 May 2023.",
                }
            )
        if "guitar practice" in prompt:
            tuples.append(
                {
                    "subject": "alice",
                    "relation": "learning_instrument",
                    "value": "guitar",
                    "qualifiers": {},
                    "confidence": 0.95,
                    "evidence": [3],
                    "text_form": "Alice is learning guitar.",
                }
            )
        return {"session": "se_mock", "tuples": tuples, "notes": []}

    memory_backend = MockModelBackend(
        responders={
            "extract@v1": extract_responder,
            "reflect@v1": lambda prompt, params: {"merges": [], "supersedes": [], "promotions": [], "flags": []},
        }
    )
    answer_backend = MockModelBackend(responders={"locomo_answer@v1": lambda prompt, params: {"answer": "guitar"}})

    run_benchmark(
        data_file=fixture_path,
        output_file=tmp_path / "locomo.json",
        stats_file=tmp_path / "locomo_stats.json",
        ledger_dir=tmp_path / "ledgers",
        policy_path=Path("memory.policy.yaml"),
        memory_backend=memory_backend,
        answer_backend=answer_backend,
        sample_limit=None,
        question_limit=0,
        categories=None,
        sample_ids=None,
        retrieval_k=5,
        checkpoint_every=2,
        reingest=True,
        log_every_questions=10,
    )

    extract_calls = [prompt_id for prompt_id, _prompt in memory_backend.calls if prompt_id == "extract@v1"]
    assert len(extract_calls) == 1


def test_disable_reflection_skips_reflect_calls(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/locomo-mini.json")

    memory_backend = MockModelBackend(
        responders={
            "extract@v1": lambda prompt, params: {"session": "se_mock", "tuples": [], "notes": []},
            "reflect@v1": lambda prompt, params: {"merges": [], "supersedes": [], "promotions": [], "flags": []},
        }
    )
    answer_backend = MockModelBackend(responders={"locomo_answer@v1": lambda prompt, params: {"answer": NO_INFORMATION}})

    summary = run_benchmark(
        data_file=fixture_path,
        output_file=tmp_path / "locomo.json",
        stats_file=tmp_path / "locomo_stats.json",
        ledger_dir=tmp_path / "ledgers",
        policy_path=Path("memory.policy.yaml"),
        memory_backend=memory_backend,
        answer_backend=answer_backend,
        sample_limit=None,
        question_limit=0,
        categories=None,
        sample_ids=None,
        retrieval_k=5,
        checkpoint_every=1,
        disable_reflection=True,
        reingest=True,
        log_every_questions=10,
    )

    assert summary["reflection_enabled"] is False
    reflect_calls = [prompt_id for prompt_id, _prompt in memory_backend.calls if prompt_id == "reflect@v1"]
    assert reflect_calls == []


def test_disable_retrieval_rerank_and_log_skip_extra_recall_work(tmp_path: Path) -> None:
    fixture_path = Path("tests/fixtures/locomo-mini.json")

    def extract_responder(prompt: str, params: dict[str, object]) -> dict[str, object]:
        del params
        return {
            "session": "se_mock",
            "tuples": [
                {
                    "subject": "alice",
                    "relation": "learning_instrument",
                    "value": "guitar",
                    "qualifiers": {},
                    "confidence": 0.95,
                    "evidence": [1],
                    "text_form": "Alice is learning guitar.",
                }
            ],
            "notes": [],
        }

    memory_backend = MockModelBackend(
        responders={
            "extract@v1": extract_responder,
            "reflect@v1": lambda prompt, params: {"merges": [], "supersedes": [], "promotions": [], "flags": []},
            "rerank@v1": lambda prompt, params: {"selected": []},
        }
    )
    answer_backend = MockModelBackend(responders={"locomo_answer@v1": lambda prompt, params: {"answer": "guitar"}})

    summary = run_benchmark(
        data_file=fixture_path,
        output_file=tmp_path / "locomo.json",
        stats_file=tmp_path / "locomo_stats.json",
        ledger_dir=tmp_path / "ledgers",
        policy_path=Path("memory.policy.yaml"),
        memory_backend=memory_backend,
        answer_backend=answer_backend,
        sample_limit=None,
        question_limit=1,
        categories=None,
        sample_ids=None,
        retrieval_k=5,
        checkpoint_every=1,
        disable_reflection=True,
        disable_retrieval_rerank=True,
        disable_retrieval_log=True,
        reingest=True,
        log_every_questions=10,
    )

    assert summary["retrieval_rerank_enabled"] is False
    assert summary["retrieval_log_mode"] == "off"
    rerank_calls = [prompt_id for prompt_id, _prompt in memory_backend.calls if prompt_id == "rerank@v1"]
    assert rerank_calls == []