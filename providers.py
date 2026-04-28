"""LLM-agnostic provider layer for the MQM benchmark.

Every provider implements BaseProvider.evaluate(unit) -> dict and validates
its output against the PaperMQMEvaluation Pydantic schema. The rest of the
pipeline (caching, scoring, analysis) never touches an SDK.

To add a new provider:
    1. Subclass BaseProvider, implement evaluate().
    2. Register it in PROVIDER_REGISTRY below.
That's it — the CLI picks it up via --provider <name>:<model>.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from mqm_paper_core import (
    MQMError,
    PaperMQMEvaluation,
    SegmentUnit,
    build_prompt,
    score_mqm_errors,
)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-sonnet-4-6",
    "mock": "no-error",
}


class BaseProvider:
    """Abstract LLM provider. Subclass and implement evaluate()."""

    def __init__(self, provider_name: str, model: str, prompt_builder: Callable[[SegmentUnit], str] | None = None):
        self.provider_name = provider_name
        self.model = model
        # prompt_builder lets the caller swap in prompt variants without
        # touching provider code. Defaults to the paper-faithful build_prompt.
        self.prompt_builder = prompt_builder or build_prompt

    @property
    def identifier(self) -> str:
        return f"{self.provider_name}:{self.model}"

    def evaluate(self, unit: SegmentUnit) -> dict:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions with structured-output enforcement.

    Uses beta.chat.completions.parse() so the response must match the
    PaperMQMEvaluation Pydantic schema server-side.
    """

    def __init__(self, model: str, prompt_builder: Callable[[SegmentUnit], str] | None = None):
        super().__init__("openai", model, prompt_builder)
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI provider.")
        self.client = OpenAI(api_key=api_key)

    def evaluate(self, unit: SegmentUnit) -> dict:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an MQM expert. Respond only with valid JSON."},
                {"role": "user", "content": self.prompt_builder(unit)},
            ],
            temperature=0.0,
            response_format=PaperMQMEvaluation,
        )
        return response.choices[0].message.parsed.model_dump()


class GeminiProvider(BaseProvider):
    """Google Gemini with JSON response mode + Pydantic validation."""

    def __init__(self, model: str, prompt_builder: Callable[[SegmentUnit], str] | None = None):
        super().__init__("gemini", model, prompt_builder)
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini provider.")
        genai.configure(api_key=api_key)
        self.genai = genai
        self.client = genai.GenerativeModel(model)

    def evaluate(self, unit: SegmentUnit) -> dict:
        response = self.client.generate_content(
            self.prompt_builder(unit),
            generation_config=self.genai.GenerationConfig(response_mime_type="application/json"),
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        payload = json.loads(text)
        return PaperMQMEvaluation.model_validate(payload).model_dump()


class AnthropicProvider(BaseProvider):
    """Anthropic Claude with JSON-only response and Pydantic validation.

    Claude doesn't have a server-enforced schema mode, so we prompt for JSON
    and validate locally. The system prompt forces a bare-JSON response.
    """

    def __init__(self, model: str, prompt_builder: Callable[[SegmentUnit], str] | None = None):
        super().__init__("anthropic", model, prompt_builder)
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic provider.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def evaluate(self, unit: SegmentUnit) -> dict:
        # claude-opus-4-x and later flagship models deprecate temperature
        supports_temperature = "claude-opus-4" not in self.model
        create_kwargs: dict = dict(
            model=self.model,
            max_tokens=2048,
            system=(
                "You are an MQM expert. Respond with a single JSON object matching "
                "the requested schema. Output JSON only — no markdown fences, no prose."
            ),
            messages=[{"role": "user", "content": self.prompt_builder(unit)}],
        )
        if supports_temperature:
            create_kwargs["temperature"] = 0.0
        response = self.client.messages.create(**create_kwargs)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        payload = json.loads(text)
        return PaperMQMEvaluation.model_validate(payload).model_dump()


class MockProvider(BaseProvider):
    """Offline stand-in for pipeline tests.

    model="no-error":    zero errors per segment.
    model="style-hunter": one Minor Style/Awkward error per segment.
    """

    def __init__(self, model: str, prompt_builder: Callable[[SegmentUnit], str] | None = None):
        super().__init__("mock", model, prompt_builder)

    def evaluate(self, unit: SegmentUnit) -> dict:
        if self.model == "style-hunter":
            return {
                "errors": [
                    MQMError(
                        error_span=unit.target_segment[: min(len(unit.target_segment), 40)],
                        category="Style",
                        subcategory="Awkward",
                        severity="Minor",
                        explanation="Mock provider emits one stylistic error for testing.",
                    ).model_dump()
                ],
                "overall_comment": "Mock provider result for pipeline testing.",
            }
        return {
            "errors": [],
            "overall_comment": "Mock provider returns no errors so the pipeline can be tested without API calls.",
        }


PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


def parse_provider_spec(spec: str) -> tuple[str, str]:
    """Parse 'provider:model' (or bare 'provider') into (provider_name, model)."""
    if ":" in spec:
        provider_name, model = spec.split(":", 1)
        provider_name = provider_name.strip().lower()
        model = model.strip()
        if not model:
            model = DEFAULT_MODELS[provider_name]
        return provider_name, model
    provider_name = spec.strip().lower()
    return provider_name, DEFAULT_MODELS[provider_name]


def create_provider(
    spec: str,
    prompt_builder: Callable[[SegmentUnit], str] | None = None,
) -> BaseProvider:
    """Instantiate the provider named in spec, optionally with a custom prompt builder."""
    provider_name, model = parse_provider_spec(spec)
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Known providers: {sorted(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[provider_name](model, prompt_builder=prompt_builder)


def default_provider_specs() -> list[str]:
    """Auto-detect which providers to use based on available API key env vars."""
    specs = []
    if os.environ.get("OPENAI_API_KEY"):
        specs.append(f"openai:{DEFAULT_MODELS['openai']}")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        specs.append(f"gemini:{DEFAULT_MODELS['gemini']}")
    if os.environ.get("ANTHROPIC_API_KEY"):
        specs.append(f"anthropic:{DEFAULT_MODELS['anthropic']}")
    return specs


def validate_and_score(raw_result: dict) -> dict:
    """Validate LLM output against PaperMQMEvaluation, then score the errors."""
    validated = PaperMQMEvaluation.model_validate(raw_result).model_dump()
    validated.update(score_mqm_errors(validated["errors"]))
    return validated


def evaluate_with_retries(provider: BaseProvider, unit: SegmentUnit, retries: int) -> dict:
    """Call provider.evaluate() with exponential backoff (1s, 2s, 4s, ...)."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return validate_and_score(provider.evaluate(unit))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"{provider.identifier} failed after {retries} attempts: {last_error}") from last_error


def load_jsonl_map(path: Path) -> dict[str, dict]:
    """Load a JSONL cache file keyed by sample_key. Empty dict if file missing."""
    if not path.exists():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["sample_key"]] = record
    return records


def append_jsonl_record(path: Path, record: dict) -> None:
    """Append one JSON record as a new line to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
