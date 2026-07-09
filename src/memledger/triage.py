"""Deterministic lexical salience scoring and triage."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from memledger.policy import Policy

STOPWORDS = frozenset(
    """
    a about above after again all am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers
    him his how i if in into is it its just me more most my no nor not of
    off on once only or other our ours out over own same she should so some
    such than that the their theirs them then there these they this those
    through to too under until up very was we were what when where which
    while who whom why will with would you your yours
    yeah yes ok okay hi hello hey thanks thank please sure got right cool
    great awesome nice fine well hmm oh ah wow bye goodbye
    """.split()
)

CUE_PATTERNS = {
    "preference": re.compile(
        r"\b(i|we)\s+(really\s+)?(prefer|like|love|hate|dislike)\b|\bmy\s+favou?rite\b",
        re.I,
    ),
    "constraint": re.compile(
        r"\b(never|always|must(\s+not)?|do\s+not|don'?t|avoid|only\s+use|make\s+sure)\b",
        re.I,
    ),
    "correction": re.compile(
        r"\bactually\b|\bi\s+meant\b|\bthat'?s\s+(not\s+right|wrong)\b|\bcorrection\b|^no[,.]",
        re.I,
    ),
    "replacement": re.compile(
        r"\bnot\s+\w+\s+but\b|\binstead\s+of\b|\bswitch(ed)?\s+to\b|\bfrom\s+now\s+on\b",
        re.I,
    ),
    "current_state": re.compile(
        r"\bcurrently\b|\bright\s+now\b|\bas\s+of\s+(now|today)\b|\bthese\s+days\b",
        re.I,
    ),
}

TRIAGE_FORMULA_V1 = "salience@v1"
TRIAGE_FORMULA_V2 = "salience@v2"
DEFAULT_TRIAGE_FORMULA = TRIAGE_FORMULA_V2


@dataclass(slots=True)
class TriageResult:
    salience: float
    lexical_density: float
    token_count: int | None
    length_norm: float | None
    entity_norm: float
    cue_classes: list[str]
    verdict: str
    formula: str = DEFAULT_TRIAGE_FORMULA

    def to_payload(self, turn_id: str) -> dict[str, object]:
        signals: dict[str, object] = {
            "lexical_density": self.lexical_density,
            "entity_norm": self.entity_norm,
            "cue_classes": self.cue_classes,
            "salience": self.salience,
        }
        if self.token_count is not None:
            signals["token_count"] = self.token_count
        if self.length_norm is not None:
            signals["length_norm"] = self.length_norm
        return {
            "turn": turn_id,
            "salience": self.salience,
            "signals": signals,
            "verdict": self.verdict,
            "formula": self.formula,
        }


def _is_token_start(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N"}


def _is_token_continue(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N"} or char in "'’-"


def tokenize(text: str) -> tuple[list[str], list[bool]]:
    normalized = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    sentence_initial: list[bool] = []
    next_initial = True
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if _is_token_start(char):
            start = index
            index += 1
            while index < len(normalized) and _is_token_continue(normalized[index]):
                index += 1
            tokens.append(normalized[start:index])
            sentence_initial.append(next_initial)
            next_initial = False
            continue
        if char in ".!?":
            next_initial = True
        index += 1
    return tokens, sentence_initial


def _length_norm(total_tokens: int, *, cap: int) -> float | None:
    if total_tokens == 0:
        return 0.0
    if cap <= 0:
        return None
    return min(total_tokens, cap) / cap


def score_text(text: str, role: str, policy: Policy) -> TriageResult:
    tokens, sentence_initial = tokenize(text)
    lowered = [token.lower() for token in tokens]
    total_tokens = len(tokens)
    stopword_tokens = sum(1 for token in lowered if token in STOPWORDS)
    lexical_density = 0.0 if total_tokens == 0 else 1 - (stopword_tokens / total_tokens)

    formula = str(policy.get("triage", "formula", default=DEFAULT_TRIAGE_FORMULA))
    token_count: int | None = None
    length_norm: float | None = None
    density_signal = lexical_density
    if formula == TRIAGE_FORMULA_V2:
        token_count = total_tokens
        length_cap = int(policy.get("triage", "density_length_cap", default=6))
        length_power = float(policy.get("triage", "density_length_power", default=2.0))
        length_norm = _length_norm(total_tokens, cap=length_cap)
        if length_norm is not None:
            density_signal = lexical_density * math.pow(length_norm, length_power)
    elif formula != TRIAGE_FORMULA_V1:
        raise ValueError(f"unsupported triage formula: {formula}")

    entity_count = 0
    for token, is_sentence_initial in zip(tokens, sentence_initial, strict=True):
        if token.isupper() and 2 <= len(token) <= 6:
            entity_count += 1
            continue
        if any(char.isdigit() for char in token):
            entity_count += 1
            continue
        if token[:1].isupper() and not is_sentence_initial:
            entity_count += 1

    entity_cap = int(policy.get("triage", "entity_cap", default=5))
    entity_norm = min(entity_count, entity_cap) / entity_cap if entity_cap else 0.0

    cue_classes = [name for name, pattern in CUE_PATTERNS.items() if pattern.search(text)]
    cue_cap = int(policy.get("triage", "cue_cap", default=2))
    cues_norm = min(len(cue_classes), cue_cap) / cue_cap if cue_cap else 0.0

    z = (
        float(policy.get("triage", "weights", "density", default=3.0)) * density_signal
        + float(policy.get("triage", "weights", "entities", default=2.0)) * entity_norm
        + float(policy.get("triage", "weights", "cues", default=1.5)) * cues_norm
    )
    x0 = float(policy.get("triage", "x0", default=1.5))
    salience = 1 / (1 + math.exp(-(z - x0)))

    verdict = "skip"
    ineligible_roles = set(policy.get("triage", "ineligible_roles", default=[]))
    always_extract = set(policy.get("triage", "always_extract_cues", default=[]))
    if role in ineligible_roles:
        verdict = "ineligible"
    elif any(cue in always_extract for cue in cue_classes):
        verdict = "extract"
    elif salience >= float(policy.get("triage", "threshold", default=0.35)):
        verdict = "extract"

    return TriageResult(
        salience=round(salience, 6),
        lexical_density=round(lexical_density, 6),
        token_count=token_count,
        length_norm=None if length_norm is None else round(length_norm, 6),
        entity_norm=round(entity_norm, 6),
        cue_classes=cue_classes,
        verdict=verdict,
        formula=formula,
    )
