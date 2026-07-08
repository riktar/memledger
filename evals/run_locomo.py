"""LoCoMo benchmark harness for MemLedger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from nltk.stem import PorterStemmer

from memledger import Ledger, Policy
from memledger.events import Cause, Event, make_event
from memledger.models.anthropic import AnthropicBackend
from memledger.models.base import ModelBackend
from memledger.models.mock import MockModelBackend
from memledger.models.openai_compat import build_openai_compat_backend
from memledger.prompts import find_project_root
from memledger.session import Session

DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
NO_INFORMATION = "No information available"
CATEGORY_ORDER = (4, 1, 2, 3, 5)
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "single_hop",
    4: "open_domain",
    5: "adversarial",
}


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    speaker: str
    dia_id: str
    text: str
    date_time: str


@dataclass(frozen=True, slots=True)
class ConversationSession:
    number: int
    date_time: str
    turns: tuple[ConversationTurn, ...]


@dataclass(frozen=True, slots=True)
class QaItem:
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]
    adversarial_answer: str | None = None


@dataclass(frozen=True, slots=True)
class LocomoSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[ConversationSession, ...]
    qa: tuple[QaItem, ...]
    raw_hash: str


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    dia_id: str
    role: str
    text: str
    date_time: str


@dataclass(frozen=True, slots=True)
class QaResult:
    question: str
    answer: str
    prediction: str
    category: int
    evidence: tuple[str, ...]
    retrieved_record_ids: tuple[str, ...]
    retrieved_dia_ids: tuple[str, ...]
    score: float
    retrieval_recall: float
    answer_tokens: int


@dataclass(frozen=True, slots=True)
class IngestStats:
    sessions: int
    turns: int
    memory_tokens: int


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    sample_count: int
    source_session_count: int
    checkpoint_batch_count: int
    question_count: int


@dataclass(slots=True)
class ProgressTracker:
    plan: BenchmarkPlan
    log_every_questions: int
    start_time: float
    completed_samples: int = 0
    completed_source_sessions: int = 0
    completed_checkpoint_batches: int = 0
    completed_questions: int = 0

    @classmethod
    def start(cls, plan: BenchmarkPlan, log_every_questions: int) -> "ProgressTracker":
        return cls(plan=plan, log_every_questions=max(log_every_questions, 1), start_time=time.monotonic())

    def log_run_start(self, *, retrieval_k: int) -> None:
        print(
            "[plan] "
            f"samples={self.plan.sample_count} source_sessions={self.plan.source_session_count} "
            f"checkpoint_batches={self.plan.checkpoint_batch_count} "
            f"questions={self.plan.question_count} retrieval_k={retrieval_k}"
        )

    def mark_cached_ingest(self, sample: LocomoSample, *, batch_count: int) -> None:
        self.completed_source_sessions += len(sample.sessions)
        self.completed_checkpoint_batches += batch_count
        print(
            f"[ingest] {self._progress_prefix()} sample={sample.sample_id} cached=true "
            f"source_sessions={len(sample.sessions)} checkpoint_batches={batch_count}"
        )

    def mark_batch_complete(
        self,
        *,
        sample: LocomoSample,
        batch_index: int,
        batch_count: int,
        source_sessions_completed: int,
        source_session_count: int,
        source_session_range: tuple[int, int],
        turns: int,
        tokens_spent: int,
    ) -> None:
        self.completed_source_sessions = source_sessions_completed
        self.completed_checkpoint_batches += 1
        print(
            f"[ingest] {self._progress_prefix()} sample={sample.sample_id} "
            f"batch={batch_index}/{batch_count} source_sessions={source_session_range[0]}-{source_session_range[1]}/{source_session_count} "
            f"turns={turns} memory_tokens={tokens_spent}"
        )

    def mark_question_complete(
        self,
        *,
        sample: LocomoSample,
        sample_question_index: int,
        sample_question_count: int,
        score: float,
    ) -> None:
        self.completed_questions += 1
        if sample_question_index % self.log_every_questions != 0 and sample_question_index != sample_question_count:
            return
        print(
            f"[qa] {self._progress_prefix()} sample={sample.sample_id} "
            f"question={sample_question_index}/{sample_question_count} score={score:.3f}"
        )

    def mark_sample_complete(self, *, sample_id: str, question_count: int, f1: float, recall: float) -> None:
        self.completed_samples += 1
        print(
            f"[sample] {self._progress_prefix()} sample={sample_id} "
            f"questions={question_count} f1={f1:.4f} retrieval_recall={recall:.4f}"
        )

    def _progress_prefix(self) -> str:
        total_units = self.plan.checkpoint_batch_count + self.plan.question_count
        completed_units = self.completed_checkpoint_batches + self.completed_questions
        percent = 100.0 if total_units == 0 else (completed_units / total_units) * 100.0
        elapsed = time.monotonic() - self.start_time
        if completed_units == 0 or total_units == 0:
            eta_text = "unknown"
        else:
            remaining_units = max(total_units - completed_units, 0)
            rate = completed_units / max(elapsed, 1e-9)
            eta_text = _format_seconds(remaining_units / rate) if rate > 0 else "unknown"
        return (
            f"overall={completed_units}/{total_units} ({percent:.1f}%) "
            f"questions={self.completed_questions}/{self.plan.question_count} "
            f"batches={self.completed_checkpoint_batches}/{self.plan.checkpoint_batch_count} "
            f"source_sessions={self.completed_source_sessions}/{self.plan.source_session_count} "
            f"elapsed={_format_seconds(elapsed)} eta={eta_text}"
        )


def _project_root() -> Path:
    return find_project_root()


def _default_data_file() -> Path:
    return _project_root() / "evals" / "locomo" / "data" / "locomo10.json"


def _default_output_file() -> Path:
    return _project_root() / "evals" / "locomo" / "results" / "latest.json"


def _default_stats_file(output_file: Path) -> Path:
    return output_file.with_name(f"{output_file.stem}_stats.json")


def _default_ledger_dir() -> Path:
    return _project_root() / "evals" / "locomo" / "work" / "ledgers"


def _default_policy_path() -> Path:
    return _project_root() / "memory.policy.yaml"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _format_seconds(value: float) -> str:
    seconds = max(int(round(value)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _normalize_turn_text(raw_turn: dict[str, Any]) -> str:
    text = str(raw_turn.get("text", "")).strip()
    caption = str(raw_turn.get("blip_caption", "")).strip()
    if caption:
        return f"{text} [shared image: {caption}]"
    return text


def _parse_sample(raw_sample: object) -> LocomoSample:
    if not isinstance(raw_sample, dict):
        raise ValueError("LoCoMo samples must be mappings")
    conversation = raw_sample.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError("sample is missing a conversation mapping")
    sample_id = str(raw_sample["sample_id"])
    speaker_a = str(conversation["speaker_a"])
    speaker_b = str(conversation["speaker_b"])
    session_numbers = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    sessions: list[ConversationSession] = []
    for number in session_numbers:
        date_time = str(conversation[f"session_{number}_date_time"])
        raw_turns = conversation[f"session_{number}"]
        if not isinstance(raw_turns, list):
            raise ValueError(f"session_{number} must be a list")
        turns = tuple(
            ConversationTurn(
                speaker=str(raw_turn["speaker"]),
                dia_id=str(raw_turn["dia_id"]),
                text=_normalize_turn_text(raw_turn),
                date_time=date_time,
            )
            for raw_turn in raw_turns
            if isinstance(raw_turn, dict)
        )
        sessions.append(ConversationSession(number=number, date_time=date_time, turns=turns))

    raw_qa = raw_sample.get("qa")
    if not isinstance(raw_qa, list):
        raise ValueError("sample is missing qa annotations")
    qa_items = tuple(
        QaItem(
            question=str(item["question"]),
            answer=_qa_expected_answer(item),
            category=int(item["category"]),
            evidence=tuple(str(evidence_id) for evidence_id in item.get("evidence", [])),
            adversarial_answer=(
                str(item["adversarial_answer"])
                if "adversarial_answer" in item and item["adversarial_answer"] is not None
                else None
            ),
        )
        for item in raw_qa
        if isinstance(item, dict)
    )
    raw_hash = _sha256_text(json.dumps(raw_sample, ensure_ascii=False, sort_keys=True))
    return LocomoSample(
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=tuple(sessions),
        qa=qa_items,
        raw_hash=raw_hash,
    )


def _qa_expected_answer(item: dict[str, Any]) -> str:
    if "answer" in item and item["answer"] is not None:
        return str(item["answer"])
    if int(item.get("category", -1)) == 5:
        return NO_INFORMATION
    raise ValueError(f"QA item is missing an answer field: {item}")


def filter_qa_items(
    sample: LocomoSample,
    *,
    question_limit: int | None,
    categories: set[int] | None,
) -> list[QaItem]:
    qa_items = list(sample.qa)
    if categories is not None:
        qa_items = [item for item in qa_items if item.category in categories]
    if question_limit is not None:
        qa_items = qa_items[:question_limit]
    return qa_items


def _session_batches(
    sessions: Sequence[ConversationSession],
    checkpoint_every: int,
) -> Iterable[tuple[ConversationSession, ...]]:
    current: list[ConversationSession] = []
    for conversation_session in sessions:
        current.append(conversation_session)
        if len(current) == checkpoint_every:
            yield tuple(current)
            current.clear()
    if current:
        yield tuple(current)


def count_checkpoint_batches(session_count: int, checkpoint_every: int) -> int:
    if session_count == 0:
        return 0
    return (session_count + checkpoint_every - 1) // checkpoint_every


def load_samples(data_file: Path) -> list[LocomoSample]:
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{data_file} must contain a top-level list")
    return [_parse_sample(item) for item in raw]


def download_dataset(target: Path, *, force: bool = False, dataset_url: str = DATASET_URL) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(dataset_url) as response:
        payload = response.read()
    target.write_bytes(payload)


def build_backend(model_spec: str | None) -> ModelBackend:
    if model_spec in {None, "", "mock"}:
        return MockModelBackend()
    if model_spec.startswith("openai-compat:"):
        return build_openai_compat_backend(model_spec)
    if model_spec.startswith("anthropic:"):
        model = model_spec.removeprefix("anthropic:")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic backends")
        return AnthropicBackend(api_key=api_key, model=model)
    raise ValueError(f"unsupported model spec: {model_spec}")


def _build_observed_turn_event(
    session: Session,
    turn: ConversationTurn,
    *,
    turn_number: int,
    source_session_number: int,
) -> Event:
    return make_event(
        type="observed",
        actor="dev",
        cause=Cause(kind="signal", ref="locomo_ingest", detail="LoCoMo session ingest"),
        policy_hash=session.ledger.policy.hash,
        payload={
            "role": turn.speaker,
            "text": turn.text,
            "turn": turn_number,
            "dia_id": turn.dia_id,
            "date_time": turn.date_time,
            "source_session": source_session_number,
        },
        session=session.id,
        user=session.user_id,
    )


def ingest_sample(
    ledger: Ledger,
    sample: LocomoSample,
    *,
    checkpoint_every: int,
    progress: ProgressTracker | None = None,
) -> IngestStats:
    memory_tokens = 0
    sessions = 0
    turns = 0
    batch_count = count_checkpoint_batches(len(sample.sessions), checkpoint_every)
    for batch_index, session_batch in enumerate(_session_batches(sample.sessions, checkpoint_every), start=1):
        session = Session(
            ledger,
            session_id=f"{sample.sample_id}_batch_{batch_index}",
            user_id=sample.sample_id,
        )
        observed_events: list[Event] = []
        session_turns = 0
        turn_number = 1
        source_session_start = sessions + 1
        for conversation_session in session_batch:
            for turn in conversation_session.turns:
                observed_events.append(
                    _build_observed_turn_event(
                        session,
                        turn,
                        turn_number=turn_number,
                        source_session_number=conversation_session.number,
                    )
                )
                turn_number += 1
                turns += 1
                session_turns += 1
            sessions += 1
        ledger.append_events(observed_events)
        report = session.checkpoint()
        memory_tokens += report.tokens_spent_on_memory
        if progress is not None:
            progress.mark_batch_complete(
                sample=sample,
                batch_index=batch_index,
                batch_count=batch_count,
                source_sessions_completed=sessions,
                source_session_count=len(sample.sessions),
                source_session_range=(source_session_start, sessions),
                turns=session_turns,
                tokens_spent=report.tokens_spent_on_memory,
            )
    ledger.store.write_meta("locomo_sample_id", sample.sample_id)
    ledger.store.write_meta(
        "locomo_ingest_signature",
        ingest_signature(sample, ledger.policy, ledger.model_backend, checkpoint_every=checkpoint_every),
    )
    return IngestStats(sessions=sessions, turns=turns, memory_tokens=memory_tokens)


def ingest_signature(
    sample: LocomoSample,
    policy: Policy,
    memory_backend: ModelBackend,
    *,
    checkpoint_every: int,
) -> str:
    backend_label = getattr(memory_backend, "model", getattr(memory_backend, "model_name", memory_backend.__class__.__name__))
    payload = {
        "sample": sample.sample_id,
        "sample_hash": sample.raw_hash,
        "policy_hash": policy.hash,
        "memory_backend": str(backend_label),
        "checkpoint_every": checkpoint_every,
    }
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _reset_ledger_path(path: Path) -> None:
    if path.exists():
        path.unlink()
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    if lock_path.exists():
        lock_path.unlink()


def open_sample_ledger(
    *,
    db_path: Path,
    sample: LocomoSample,
    policy: Policy,
    memory_backend: ModelBackend,
    checkpoint_every: int,
    reingest: bool,
    progress: ProgressTracker | None = None,
) -> tuple[Ledger, bool, IngestStats]:
    def create_ledger() -> Ledger:
        return Ledger(path=str(db_path), policy=policy, model_backend=memory_backend)

    expected_signature = ingest_signature(sample, policy, memory_backend, checkpoint_every=checkpoint_every)
    if db_path.exists() and not reingest:
        ledger = create_ledger()
        try:
            stored_signature = ledger.store.read_meta("locomo_ingest_signature")
            if stored_signature == expected_signature:
                return ledger, False, IngestStats(sessions=0, turns=0, memory_tokens=0)
        except Exception:
            ledger.close()
            raise
        ledger.close()

    _reset_ledger_path(db_path)
    ledger = create_ledger()
    stats = ingest_sample(ledger, sample, checkpoint_every=checkpoint_every, progress=progress)
    return ledger, True, stats


def _normalize_answer(text: str) -> str:
    text = text.replace(",", "")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stemmed_tokens(text: str) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in _normalize_answer(text).split()]


def _token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _stemmed_tokens(prediction)
    ground_truth_tokens = _stemmed_tokens(ground_truth)
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [item.strip() for item in prediction.split(",") if item.strip()]
    ground_truths = [item.strip() for item in ground_truth.split(",") if item.strip()]
    if not predictions or not ground_truths:
        return 0.0
    return mean(max(_token_f1(candidate, truth) for candidate in predictions) for truth in ground_truths)


def score_prediction(qa: QaItem, prediction: str) -> float:
    if qa.category in {2, 4}:
        return _token_f1(prediction, qa.answer)
    if qa.category == 3:
        return _token_f1(prediction, qa.answer.split(";", 1)[0].strip())
    if qa.category == 1:
        return _multi_answer_f1(prediction, qa.answer)
    if qa.category == 5:
        lowered = prediction.lower()
        return 1.0 if NO_INFORMATION.lower() in lowered or "not mentioned" in lowered else 0.0
    raise ValueError(f"unsupported LoCoMo QA category: {qa.category}")


def evidence_recall(expected: Sequence[str], retrieved: Sequence[str]) -> float:
    if not expected:
        return 1.0
    retrieved_set = set(retrieved)
    matched = sum(1 for item in expected if item in retrieved_set)
    return matched / len(expected)


def _question_hint(category: int) -> str:
    if category == 2:
        return "If this is a date question, answer with a short date phrase grounded in the evidence."
    if category == 5:
        return "If the memory does not support the answer, return exactly 'No information available'."
    return "Answer in a few words."


def collect_record_evidence(ledger: Ledger, record_id: str) -> list[EvidenceSnippet]:
    record = ledger.store.get_record(record_id)
    if record is None:
        return []
    evidence: list[EvidenceSnippet] = []
    for source_id in record.sources:
        event = ledger.store.get_event(source_id)
        if event is None or event.type != "observed":
            continue
        evidence.append(
            EvidenceSnippet(
                dia_id=str(event.payload.get("dia_id", "")),
                role=str(event.payload.get("role", "")),
                text=str(event.payload.get("text", "")).strip(),
                date_time=str(event.payload.get("date_time", "")),
            )
        )
    return evidence


def format_memory_context(ledger: Ledger, record_ids: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    lines: list[str] = []
    retrieved_dia_ids: list[str] = []
    for index, record_id in enumerate(record_ids, start=1):
        record = ledger.store.get_record(record_id)
        if record is None:
            continue
        lines.append(f"Memory {index}: {record.text_form}")
        for snippet in collect_record_evidence(ledger, record_id)[:3]:
            details = snippet.dia_id or "unknown"
            if snippet.date_time:
                details = f"{details} | {snippet.date_time}"
            lines.append(f"- Evidence {details}: {snippet.role}: {snippet.text}")
            if snippet.dia_id:
                retrieved_dia_ids.append(snippet.dia_id)
    if not lines:
        return "No retrieved memories.", tuple()
    ordered_dia_ids = tuple(dict.fromkeys(retrieved_dia_ids))
    return "\n".join(lines), ordered_dia_ids


def answer_question(
    *,
    ledger: Ledger,
    qa_session: Session,
    sample: LocomoSample,
    qa: QaItem,
    retrieval_k: int,
    answer_backend: ModelBackend,
) -> QaResult:
    memories = qa_session.recall(qa.question, k=retrieval_k)
    record_ids = tuple(record.id for record in memories)
    memory_context, retrieved_dia_ids = format_memory_context(ledger, record_ids)
    answer_tokens = 0
    if not record_ids:
        prediction = NO_INFORMATION
    else:
        try:
            answer_json, llm_call = ledger.call_model_json(
                prompt_id="locomo_answer@v1",
                placeholders={
                    "speakers": f"{sample.speaker_a} and {sample.speaker_b}",
                    "question_hint": _question_hint(qa.category),
                    "question": qa.question,
                    "memory_context": memory_context,
                },
                params={"temperature": 0, "schema": "locomo_answer@v1", "sample": sample.sample_id},
                backend=answer_backend,
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "QA model returned invalid JSON for locomo_answer@v1; prefer an openai-compatible backend for LoCoMo runs."
            ) from exc
        prediction = str(answer_json.get("answer", "")).strip() or NO_INFORMATION
        answer_tokens = llm_call.tokens["in"] + llm_call.tokens["out"]
    score = score_prediction(qa, prediction)
    recall = evidence_recall(qa.evidence, retrieved_dia_ids)
    return QaResult(
        question=qa.question,
        answer=qa.answer,
        prediction=prediction,
        category=qa.category,
        evidence=qa.evidence,
        retrieved_record_ids=record_ids,
        retrieved_dia_ids=retrieved_dia_ids,
        score=score,
        retrieval_recall=recall,
        answer_tokens=answer_tokens,
    )


def evaluate_sample(
    *,
    ledger: Ledger,
    sample: LocomoSample,
    answer_backend: ModelBackend,
    retrieval_k: int,
    question_limit: int | None,
    categories: set[int] | None,
    progress: ProgressTracker | None = None,
) -> dict[str, Any]:
    qa_items = filter_qa_items(sample, question_limit=question_limit, categories=categories)

    qa_session = ledger.session(user_id=sample.sample_id)
    results: list[QaResult] = []
    for index, item in enumerate(qa_items, start=1):
        result = answer_question(
            ledger=ledger,
            qa_session=qa_session,
            sample=sample,
            qa=item,
            retrieval_k=retrieval_k,
            answer_backend=answer_backend,
        )
        results.append(result)
        if progress is not None:
            progress.mark_question_complete(
                sample=sample,
                sample_question_index=index,
                sample_question_count=len(qa_items),
                score=result.score,
            )
    scores = [result.score for result in results]
    recalls = [result.retrieval_recall for result in results]
    answer_tokens = sum(result.answer_tokens for result in results)
    return {
        "sample_id": sample.sample_id,
        "speakers": [sample.speaker_a, sample.speaker_b],
        "question_count": len(results),
        "f1": round(mean(scores), 4) if scores else 0.0,
        "retrieval_recall": round(mean(recalls), 4) if recalls else 0.0,
        "answer_tokens": answer_tokens,
        "qa": [
            {
                "question": result.question,
                "answer": result.answer,
                "prediction": result.prediction,
                "category": result.category,
                "evidence": list(result.evidence),
                "adversarial_answer": item.adversarial_answer,
                "retrieved_record_ids": list(result.retrieved_record_ids),
                "retrieved_dia_ids": list(result.retrieved_dia_ids),
                "f1": round(result.score, 4),
                "retrieval_recall": round(result.retrieval_recall, 4),
            }
            for result, item in zip(results, qa_items, strict=True)
        ],
    }


def parse_category_filter(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def parse_sample_filter(raw: str) -> set[str] | None:
    if not raw.strip():
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def build_plan(
    samples: Sequence[LocomoSample],
    *,
    checkpoint_every: int,
    question_limit: int | None,
    categories: set[int] | None,
) -> BenchmarkPlan:
    return BenchmarkPlan(
        sample_count=len(samples),
        source_session_count=sum(len(sample.sessions) for sample in samples),
        checkpoint_batch_count=sum(count_checkpoint_batches(len(sample.sessions), checkpoint_every) for sample in samples),
        question_count=sum(len(filter_qa_items(sample, question_limit=question_limit, categories=categories)) for sample in samples),
    )


def _conversation_lengths(raw_conversation: dict[str, Any]) -> dict[str, int]:
    total_conv_length = 0
    id_to_length: dict[str, int] = {}
    session_numbers = sorted(
        int(key.split("_")[-1])
        for key in raw_conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    for session_number in session_numbers:
        raw_session = raw_conversation.get(f"session_{session_number}")
        if not isinstance(raw_session, list) or not raw_session:
            continue
        for dialog in raw_session:
            if not isinstance(dialog, dict):
                continue
            dialog_tokens = f"{dialog['speaker']}: {dialog['text']}\n"
            if "img_file" in dialog and dialog["img_file"]:
                dialog_tokens += f"[shares {dialog['blip_caption']}]\n"
            dialog_length = len(dialog_tokens)
            id_to_length[str(dialog["dia_id"])] = total_conv_length + dialog_length
            total_conv_length += dialog_length
    return id_to_length


def _sanitize_evidence_id(evidence_id: str) -> str:
    return evidence_id.replace("(", "").replace(")", "")


def _parse_dialog_ref(evidence_id: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"D(\d+):(\d+)", _sanitize_evidence_id(evidence_id))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _serialize_number_map(values: dict[int, float | int]) -> dict[str, float | int]:
    return {str(key): values.get(key, 0) for key in CATEGORY_ORDER}


def _serialize_nested_number_map(values: dict[int, dict[int, float | int]]) -> dict[str, dict[str, float | int]]:
    serialized: dict[str, dict[str, float | int]] = {}
    ordered_categories = list(CATEGORY_ORDER) + [key for key in sorted(values) if key not in CATEGORY_ORDER]
    for category in ordered_categories:
        inner = values.get(category, {})
        serialized[str(category)] = {str(key): inner[key] for key in sorted(inner)}
    return serialized


def _serialize_bucket_map(values: dict[int, float | int]) -> dict[str, float | int]:
    return {str(key): values[key] for key in sorted(values)}


def _normalized_by_category(
    total_counts: dict[int, int],
    cumulative_values: dict[int, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for category in CATEGORY_ORDER:
        total = total_counts.get(category, 0)
        result[str(category)] = round(float(cumulative_values.get(category, 0.0)) / total, 4) if total else 0.0
    return result


def _normalized_nested_by_category(
    totals: dict[int, dict[int, int]],
    cumulative_values: dict[int, dict[int, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    ordered_categories = list(CATEGORY_ORDER) + [key for key in sorted(totals) if key not in CATEGORY_ORDER]
    for category in ordered_categories:
        category_totals = totals.get(category, {})
        category_scores = cumulative_values.get(category, {})
        result[str(category)] = {
            str(bucket): round(float(category_scores.get(bucket, 0.0)) / count, 4) if count else 0.0
            for bucket, count in sorted(category_totals.items())
        }
    return result


def _normalized_bucket_map(totals: dict[int, int], cumulative_values: dict[int, float]) -> dict[str, float]:
    return {
        str(bucket): round(float(cumulative_values.get(bucket, 0.0)) / count, 4) if count else 0.0
        for bucket, count in sorted(totals.items())
    }


def compute_official_metrics(
    *,
    raw_by_sample: dict[str, dict[str, Any]],
    sample_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_counts: dict[int, int] = defaultdict(int)
    acc_counts: dict[int, float] = defaultdict(float)
    memory_counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    memory_counts_og: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    context_len_counts: dict[int, float] = defaultdict(float)
    context_len_og: dict[int, int] = defaultdict(int)
    recall_by_category: dict[int, float] = defaultdict(float)

    for sample_result in sample_results:
        sample_id = str(sample_result["sample_id"])
        raw_sample = raw_by_sample[sample_id]
        id_to_length = _conversation_lengths(raw_sample["conversation"])
        for qa in sample_result["qa"]:
            category = int(qa["category"])
            total_counts[category] += 1
            score = float(qa["f1"])
            acc_counts[category] += score
            evidence_ids = [_sanitize_evidence_id(str(item)) for item in qa.get("evidence", [])]
            if evidence_ids:
                recall_by_category[category] += float(qa.get("retrieval_recall", 1.0))
                refs = [ref for evidence_id in evidence_ids if (ref := _parse_dialog_ref(evidence_id)) is not None]
                if not refs:
                    continue
                farthest_session, farthest_dialog = min(refs)
                farthest_key = f"D{farthest_session}:{farthest_dialog}"
                farthest_length = id_to_length.get(farthest_key)
                if farthest_length is None:
                    continue
                memory_bucket = math.ceil(farthest_length / 1000)
                memory_counts_og[category][memory_bucket] += 1
                memory_counts[category][memory_bucket] += score
                if category == 1:
                    latest_session, latest_dialog = max(refs)
                    latest_key = f"D{latest_session}:{latest_dialog}"
                    latest_length = id_to_length.get(latest_key)
                    if latest_length is None:
                        continue
                    context_bucket = math.ceil((latest_length - farthest_length) / 1000)
                    context_len_og[context_bucket] += 1
                    context_len_counts[context_bucket] += score

    total_questions = sum(total_counts.values())
    total_accuracy = sum(acc_counts.values())
    total_recall = sum(recall_by_category.values())
    return {
        "category_order": list(CATEGORY_ORDER),
        "category_names": {str(key): value for key, value in CATEGORY_NAMES.items()},
        "category_counts": _serialize_number_map(total_counts),
        "cum_accuracy_by_category": _serialize_number_map(acc_counts),
        "accuracy_by_category": _normalized_by_category(total_counts, acc_counts),
        "overall_accuracy": round(total_accuracy / total_questions, 4) if total_questions else 0.0,
        "recall_by_category": _normalized_by_category(total_counts, recall_by_category),
        "overall_recall_accuracy": round(total_recall / total_questions, 4) if total_questions else 0.0,
        "category_counts_by_memory": _serialize_nested_number_map(memory_counts_og),
        "cum_accuracy_by_category_by_memory": _serialize_nested_number_map(memory_counts),
        "accuracy_by_category_by_memory": _normalized_nested_by_category(memory_counts_og, memory_counts),
        "context_length_counts": _serialize_bucket_map(context_len_og),
        "cum_accuracy_by_context_length": _serialize_bucket_map(context_len_counts),
        "accuracy_by_context_length": _normalized_bucket_map(context_len_og, context_len_counts),
    }


def run_benchmark(
    *,
    data_file: Path,
    output_file: Path,
    stats_file: Path,
    ledger_dir: Path,
    policy_path: Path,
    memory_backend: ModelBackend,
    answer_backend: ModelBackend,
    sample_limit: int | None,
    question_limit: int | None,
    categories: set[int] | None,
    sample_ids: set[str] | None,
    retrieval_k: int,
    checkpoint_every: int,
    disable_reflection: bool = False,
    disable_retrieval_rerank: bool = False,
    disable_retrieval_log: bool = False,
    reingest: bool,
    log_every_questions: int,
) -> dict[str, Any]:
    policy = Policy.from_yaml(policy_path)
    if disable_reflection:
        policy = policy.copy_with_updates({"reflection": {"enabled": False}})
    if disable_retrieval_rerank:
        policy = policy.copy_with_updates({"retrieval": {"rerank": False}})
    if disable_retrieval_log:
        policy = policy.copy_with_updates({"retrieval": {"log": "off"}})
    raw_samples = json.loads(data_file.read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raise ValueError(f"{data_file} must contain a top-level list")
    raw_by_sample = {str(item["sample_id"]): item for item in raw_samples if isinstance(item, dict)}
    samples = [_parse_sample(item) for item in raw_samples if isinstance(item, dict)]
    if sample_ids is not None:
        samples = [sample for sample in samples if sample.sample_id in sample_ids]
    if sample_limit is not None:
        samples = samples[:sample_limit]

    ledger_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)

    plan = build_plan(samples, checkpoint_every=checkpoint_every, question_limit=question_limit, categories=categories)
    progress = ProgressTracker.start(plan, log_every_questions=log_every_questions)
    progress.log_run_start(retrieval_k=retrieval_k)

    sample_results: list[dict[str, Any]] = []
    total_memory_tokens = 0
    total_answer_tokens = 0
    ingested_samples = 0

    for sample in samples:
        db_path = ledger_dir / f"{sample.sample_id}.sqlite"
        ledger, did_ingest, ingest_stats = open_sample_ledger(
            db_path=db_path,
            sample=sample,
            policy=policy,
            memory_backend=memory_backend,
            checkpoint_every=checkpoint_every,
            reingest=reingest,
            progress=progress,
        )
        try:
            if did_ingest:
                ingested_samples += 1
                total_memory_tokens += ingest_stats.memory_tokens
            else:
                progress.mark_cached_ingest(
                    sample,
                    batch_count=count_checkpoint_batches(len(sample.sessions), checkpoint_every),
                )
            result = evaluate_sample(
                ledger=ledger,
                sample=sample,
                answer_backend=answer_backend,
                retrieval_k=retrieval_k,
                question_limit=question_limit,
                categories=categories,
                progress=progress,
            )
            sample_results.append(result)
            total_answer_tokens += int(result["answer_tokens"])
            progress.mark_sample_complete(
                sample_id=sample.sample_id,
                question_count=int(result["question_count"]),
                f1=float(result["f1"]),
                recall=float(result["retrieval_recall"]),
            )
        finally:
            ledger.close()

    flat_results = [qa for sample in sample_results for qa in sample["qa"]]
    official_metrics = compute_official_metrics(raw_by_sample=raw_by_sample, sample_results=sample_results)
    summary = {
        "data_file": str(data_file),
        "stats_file": str(stats_file),
        "sample_count": len(sample_results),
        "question_count": len(flat_results),
        "retrieval_k": retrieval_k,
        "checkpoint_every": checkpoint_every,
        "reflection_enabled": bool(policy.get("reflection", "enabled", default=True)),
        "retrieval_rerank_enabled": bool(policy.get("retrieval", "rerank", default=True)),
        "retrieval_log_mode": str(policy.get("retrieval", "log", default="sampled")),
        "checkpoint_batch_count": plan.checkpoint_batch_count,
        "ingested_samples": ingested_samples,
        "memory_tokens": total_memory_tokens,
        "answer_tokens": total_answer_tokens,
        "f1": float(official_metrics["overall_accuracy"]),
        "retrieval_recall": float(official_metrics["overall_recall_accuracy"]),
        "official_metrics": official_metrics,
        "samples": sample_results,
    }
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_file.write_text(json.dumps(official_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LoCoMo benchmark against MemLedger.")
    parser.add_argument("--data-file", default=str(_default_data_file()), help="Path to locomo10.json.")
    parser.add_argument("--output-file", default=str(_default_output_file()), help="Where to write benchmark JSON.")
    parser.add_argument("--stats-file", default=None, help="Where to write LoCoMo official metrics JSON.")
    parser.add_argument("--ledger-dir", default=str(_default_ledger_dir()), help="Directory for per-sample SQLite ledgers.")
    parser.add_argument("--policy", default=str(_default_policy_path()), help="Policy file used during ingest and retrieval.")
    parser.add_argument(
        "--memory-model",
        default=os.environ.get("MEMORY_MODEL"),
        help="Model used for MemLedger extraction and reflection. Quote openai-compat values.",
    )
    parser.add_argument(
        "--qa-model",
        default=None,
        help="Optional model used for final QA answers. Defaults to --memory-model.",
    )
    parser.add_argument("--sample-limit", type=int, default=None, help="Evaluate only the first N conversations.")
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions per conversation.",
    )
    parser.add_argument(
        "--categories",
        default="",
        help="Optional comma-separated LoCoMo categories to evaluate, for example 1,2,5.",
    )
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated sample ids, for example conv-26,conv-31.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Group this many LoCoMo source sessions into one MemLedger checkpoint for faster ingest.",
    )
    parser.add_argument(
        "--disable-reflection",
        action="store_true",
        help="Skip MemLedger reflection during ingest: no merge, supersede, or promotion proposal pass.",
    )
    parser.add_argument(
        "--disable-retrieval-rerank",
        action="store_true",
        help="Skip the LLM rerank pass inside retrieval and use only rule-based pre-scoring for QA recall.",
    )
    parser.add_argument(
        "--disable-retrieval-log",
        action="store_true",
        help="Disable `recalled` event logging during QA to reduce SQLite write overhead.",
    )
    parser.add_argument("--retrieval-k", type=int, default=5, help="Number of memories retrieved per question.")
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Re-download locomo10.json even if it is already present locally.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading the dataset when --data-file is missing.",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Rebuild per-sample ledgers even if a compatible cached ingest already exists.",
    )
    parser.add_argument(
        "--log-every-questions",
        type=int,
        default=10,
        help="Emit QA progress and ETA every N answered questions within each sample.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.memory_model is None:
        parser.error("set --memory-model or MEMORY_MODEL; use --memory-model mock only for smoke tests")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be >= 1")

    data_file = Path(args.data_file)
    if not data_file.exists():
        if args.no_download:
            parser.error(f"dataset not found: {data_file}")
        download_dataset(data_file, force=args.refresh_dataset)
    elif args.refresh_dataset:
        download_dataset(data_file, force=True)

    memory_backend = build_backend(args.memory_model)
    answer_backend = build_backend(args.qa_model or args.memory_model)
    output_file = Path(args.output_file)
    stats_file = Path(args.stats_file) if args.stats_file is not None else _default_stats_file(output_file)
    summary = run_benchmark(
        data_file=data_file,
        output_file=output_file,
        stats_file=stats_file,
        ledger_dir=Path(args.ledger_dir),
        policy_path=Path(args.policy),
        memory_backend=memory_backend,
        answer_backend=answer_backend,
        sample_limit=args.sample_limit,
        question_limit=args.question_limit,
        categories=parse_category_filter(args.categories),
        sample_ids=parse_sample_filter(args.sample_ids),
        retrieval_k=args.retrieval_k,
        checkpoint_every=args.checkpoint_every,
        disable_reflection=args.disable_reflection,
        disable_retrieval_rerank=args.disable_retrieval_rerank,
        disable_retrieval_log=args.disable_retrieval_log,
        reingest=args.reingest,
        log_every_questions=args.log_every_questions,
    )
    print(
        f"summary: samples={summary['sample_count']} questions={summary['question_count']} "
        f"f1={summary['f1']:.4f} retrieval_recall={summary['retrieval_recall']:.4f} "
        f"reflection_enabled={summary['reflection_enabled']} "
        f"retrieval_rerank_enabled={summary['retrieval_rerank_enabled']} "
        f"retrieval_log_mode={summary['retrieval_log_mode']} "
        f"memory_tokens={summary['memory_tokens']} answer_tokens={summary['answer_tokens']} "
        f"stats={summary['stats_file']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())