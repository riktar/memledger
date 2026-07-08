"""Golden-case extraction regression harness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from memledger.ids import canonical_json
from memledger.models.mock import MockModelBackend
from memledger.policy import Policy
from memledger.prompts import PromptRegistry, find_project_root


@dataclass(frozen=True, slots=True)
class TupleKey:
    subject: str
    relation: str
    value: str


@dataclass(frozen=True, slots=True)
class RegressionTurn:
    role: str
    text: str


@dataclass(frozen=True, slots=True)
class RegressionCase:
    name: str
    session_id: str
    turns: tuple[RegressionTurn, ...]
    expected_tuples: frozenset[TupleKey]
    path: Path


def _default_cases_dir() -> Path:
    return Path(__file__).resolve().parent / "regression" / "cases"


def _default_policy_path() -> Path:
    return find_project_root() / "memory.policy.yaml"


def _tuple_key(subject: object, relation: object, value: object) -> TupleKey:
    return TupleKey(
        subject=str(subject),
        relation=str(relation),
        value=canonical_json(value),
    )


def _coerce_turns(raw_turns: object, *, path: Path) -> tuple[RegressionTurn, ...]:
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"{path}: turns must be a non-empty list")
    turns: list[RegressionTurn] = []
    for index, item in enumerate(raw_turns, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: turn {index} must be a mapping")
        role = item.get("role")
        text = item.get("text")
        if not isinstance(role, str) or not isinstance(text, str):
            raise ValueError(f"{path}: turn {index} must include string role/text")
        turns.append(RegressionTurn(role=role, text=text))
    return tuple(turns)


def _coerce_expected_tuples(raw_expected: object, *, path: Path) -> frozenset[TupleKey]:
    if not isinstance(raw_expected, list):
        raise ValueError(f"{path}: expected_tuples must be a list")
    expected: set[TupleKey] = set()
    for index, item in enumerate(raw_expected, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: expected_tuples[{index}] must be a mapping")
        try:
            subject = item["subject"]
            relation = item["relation"]
            value = item["value"]
        except KeyError as exc:
            raise ValueError(f"{path}: expected_tuples[{index}] is missing {exc.args[0]!r}") from exc
        expected.add(_tuple_key(subject, relation, value))
    return frozenset(expected)


def load_case(path: Path) -> RegressionCase:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: case file must contain a mapping")
    name = raw.get("name", path.stem)
    session_id = raw.get("session_id", f"se_{path.stem}")
    if not isinstance(name, str) or not isinstance(session_id, str):
        raise ValueError(f"{path}: name and session_id must be strings")
    return RegressionCase(
        name=name,
        session_id=session_id,
        turns=_coerce_turns(raw.get("turns"), path=path),
        expected_tuples=_coerce_expected_tuples(raw.get("expected_tuples"), path=path),
        path=path,
    )


def discover_cases(cases_dir: Path) -> list[RegressionCase]:
    case_paths = sorted(cases_dir.glob("*.yaml"))
    return [load_case(path) for path in case_paths]


def render_transcript(case: RegressionCase) -> str:
    return "\n".join(f"{index}. {turn.role}: {turn.text}" for index, turn in enumerate(case.turns, start=1))


def run_case(case: RegressionCase, registry: PromptRegistry, backend: MockModelBackend) -> tuple[set[TupleKey], set[TupleKey]]:
    prompt = registry.render(
        "extract@v1",
        {
            "transcript": render_transcript(case),
            "known_subjects": json.dumps([], ensure_ascii=False),
            "known_relations": json.dumps([], ensure_ascii=False),
            "session_id": case.session_id,
            "language": "English",
        },
    )
    response = backend.complete(
        "extract@v1",
        prompt.content,
        {"temperature": 0, "schema": "tuples@v1", "session": case.session_id},
    )
    payload = json.loads(response.content or "{}")
    predicted = {
        _tuple_key(item.get("subject"), item.get("relation"), item.get("value"))
        for item in payload.get("tuples", [])
        if isinstance(item, dict)
    }
    return predicted, set(case.expected_tuples)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MemLedger extraction regression cases.")
    parser.add_argument(
        "--cases-dir",
        default=str(_default_cases_dir()),
        help="Directory containing regression case YAML files.",
    )
    parser.add_argument(
        "--policy",
        default=str(_default_policy_path()),
        help="Policy YAML used for eval thresholds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case tuple sets in addition to the summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_dir():
        print(f"cases directory not found: {cases_dir}")
        return 2

    cases = discover_cases(cases_dir)
    if not cases:
        print(f"no regression cases found in {cases_dir}")
        return 2

    policy = Policy.from_yaml(args.policy)
    precision_min = float(policy.get("evals", "extraction_precision_min", default=1.0))
    recall_min = float(policy.get("evals", "extraction_recall_min", default=1.0))

    registry = PromptRegistry(find_project_root())
    backend = MockModelBackend()

    total_expected = 0
    total_predicted = 0
    total_matched = 0

    for case in cases:
        predicted, expected = run_case(case, registry, backend)
        matched = predicted & expected
        total_expected += len(expected)
        total_predicted += len(predicted)
        total_matched += len(matched)
        case_precision = _safe_ratio(len(matched), len(predicted))
        case_recall = _safe_ratio(len(matched), len(expected))
        print(
            f"{case.name}: precision={case_precision:.3f} recall={case_recall:.3f} "
            f"matched={len(matched)}/{len(expected)} predicted={len(predicted)}"
        )
        if args.verbose:
            print(f"  expected={sorted(expected, key=lambda item: (item.subject, item.relation, item.value))}")
            print(f"  predicted={sorted(predicted, key=lambda item: (item.subject, item.relation, item.value))}")

    precision = _safe_ratio(total_matched, total_predicted)
    recall = _safe_ratio(total_matched, total_expected)
    print(
        f"summary: precision={precision:.3f} recall={recall:.3f} "
        f"thresholds=({precision_min:.3f}, {recall_min:.3f})"
    )

    if precision < precision_min or recall < recall_min:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())