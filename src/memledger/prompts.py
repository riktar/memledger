"""Prompt registry and deterministic text template loading."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memledger.ids import sha256_hex

PROMPT_FILE_MAP = {
    "extract@v1": "extract.v1.md",
    "locomo_answer@v1": "locomo_answer.v1.md",
    "rerank@v1": "rerank.v1.md",
    "reflect@v1": "reflect.v1.md",
    "salience@v1": "salience.v1.md",
    "salience@v2": "salience.v2.md",
    "text_form@v1": "text_form.v1.md",
}


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "prompts").is_dir() and (parent / "memory.policy.yaml").exists():
            return parent
    return package_assets_root()


def package_assets_root() -> Path:
    assets = importlib.resources.files("memledger").joinpath("assets")
    return Path(str(assets))


@dataclass(slots=True)
class Prompt:
    id: str
    path: Path
    content: str
    hash: str


class PromptRegistry:
    """Loads prompt and formula assets from the registry."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root()

    def get_path(self, prompt_id: str) -> Path:
        try:
            filename = PROMPT_FILE_MAP[prompt_id]
        except KeyError as exc:
            raise KeyError(f"unknown prompt id: {prompt_id}") from exc
        return self.root / "prompts" / filename

    def load(self, prompt_id: str) -> Prompt:
        path = self.get_path(prompt_id)
        content = path.read_text(encoding="utf-8")
        return Prompt(id=prompt_id, path=path, content=content, hash=sha256_hex(content))

    def render(self, prompt_id: str, placeholders: dict[str, Any]) -> Prompt:
        prompt = self.load(prompt_id)
        content = prompt.content
        for key, value in placeholders.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
        return Prompt(id=prompt.id, path=prompt.path, content=content, hash=prompt.hash)


@dataclass(slots=True)
class TextFormTemplates:
    exact: dict[str, str]
    category: dict[str, str]
    fallback: dict[str, str]

    def render(self, relation: str, values: dict[str, Any]) -> tuple[str, str]:
        qualifiers: list[str] = []
        if values.get("when"):
            qualifiers.append(f" (as of {values['when']})")
        if values.get("context"):
            qualifiers.append(f" in the context of {values['context']}")

        if relation in self.exact:
            rendered = self._apply(self.exact[relation], relation, values)
            return f"{rendered}{''.join(qualifiers)}.", f"exact:{relation}"

        for pattern, template in self.category.items():
            if not pattern.endswith("*"):
                continue
            prefix = pattern[:-1]
            if relation.startswith(prefix):
                rendered = self._apply(template, relation, values, strip_prefix=prefix)
                return f"{rendered}{''.join(qualifiers)}.", f"category:{pattern}"

        template = self.fallback.get("default", "{subject}: {relation_words} is {value}")
        rendered = self._apply(template, relation, values)
        return f"{rendered}{''.join(qualifiers)}.", "fallback:default"

    def _apply(
        self,
        template: str,
        relation: str,
        values: dict[str, Any],
        strip_prefix: str = "",
    ) -> str:
        relation_words = relation.replace("_", " ")
        if strip_prefix:
            stripped = relation[len(strip_prefix) :]
            relation_words = stripped.replace("_", " ")
        rendered = template.replace("{relation_words}", relation_words)
        rendered = rendered.replace(
            "{relation_words#preferred_}",
            relation.removeprefix("preferred_").replace("_", " "),
        )
        rendered = rendered.replace("{relation_words#uses_}", relation.removeprefix("uses_").replace("_", " "))
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered.strip().rstrip(".")


def load_text_form_templates(
    registry: PromptRegistry | None = None,
) -> TextFormTemplates:
    registry = registry or PromptRegistry()
    prompt = registry.load("text_form@v1")
    current: str | None = None
    sections: dict[str, dict[str, str]] = {"exact": {}, "category": {}, "fallback": {}}
    for raw_line in prompt.content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]")
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        sections[current][key.strip()] = value.strip().strip('"')
    return TextFormTemplates(
        exact=sections["exact"],
        category=sections["category"],
        fallback=sections["fallback"],
    )
