"""LLM access.

The whole pipeline must run end to end with no API key and no network, so that
the evaluation harness is reproducible and CI is possible. ``ScriptedLLM`` and
``NullLLM`` satisfy that; ``AnthropicLLM`` is the real path.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str: ...


@dataclass
class NullLLM:
    """Always fails. Forces the deterministic fallback path.

    Used as the default so that nothing silently depends on a model being
    reachable.
    """

    calls: int = 0

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        self.calls += 1
        raise RuntimeError("no LLM configured")


@dataclass
class ScriptedLLM:
    """Returns canned responses in order. For tests.

    Once exhausted it repeats the final response, so a loop of unknown length
    does not need an exactly-sized script.
    """

    responses: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("ScriptedLLM has no responses")
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    @staticmethod
    def ranking(pairs: list[tuple[str, float]]) -> str:
        """Helper to build a well-formed ranking response."""
        return json.dumps(
            {"ranking": [{"label": l, "probability": p, "why": ""} for l, p in pairs]}
        )


@dataclass
class AnthropicLLM:
    """Thin wrapper over the Messages API.

    Kept minimal on purpose: retries, rate limiting, and cost accounting belong
    in one place, and that place should be chosen once the evaluation loop's
    call volume is known rather than guessed at now.
    """

    model: str = "claude-sonnet-4-6"
    api_key: str | None = None
    temperature: float = 0.0  # determinism matters more than variety here

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system or "You are a careful clinical reasoning assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )


@dataclass
class GeminiLLM:
    """Thin wrapper over the Gemini API.

    Not the brief's mandated model -- a free-tier substitute for exercising
    the synthesis pipeline without a paid Anthropic key. Using this instead of
    ``AnthropicLLM`` is a deviation from the spec and belongs in the write-up
    as one, the same way the UMLS and clinician deviations are recorded rather
    than left implicit.
    """

    # Pinned rather than an alias like ``gemini-flash-lite-latest``: a corpus
    # is worth only as much as the record of what generated it, and an alias
    # silently changes model under a stored batch. ``gemini-2.0-flash`` was
    # the first default and is no longer usable -- it now returns 429 with
    # ``limit: 0`` on the free tier, which reads like exhausted usage and is
    # actually no quota at all.
    model: str = "gemini-3.5-flash-lite"
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")

    # The free tier allows 15 requests per minute per model. A synthesis run
    # is one call per vignette plus one per critique -- comfortably past that
    # -- so without backoff most of a batch fails and the run reports a
    # near-empty corpus for a reason that has nothing to do with the model's
    # ability to write vignettes.
    max_retries: int = 5

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install google-genai") from exc

        client = genai.Client(api_key=self.api_key)
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=0.0,
        )

        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                return response.text or ""
            except Exception as exc:
                if "429" not in str(exc) or attempt == self.max_retries - 1:
                    raise
                # The API states how long to wait; prefer that to a guess,
                # since a guessed backoff either wastes minutes or retries
                # into the same limit.
                match = re.search(r"'retryDelay':\s*'(\d+)s'", str(exc))
                delay = int(match.group(1)) + 1 if match else 2 ** (attempt + 3)
                time.sleep(delay)

        raise RuntimeError("unreachable")  # pragma: no cover


__all__ = ["AnthropicLLM", "GeminiLLM", "LLMClient", "NullLLM", "ScriptedLLM"]
