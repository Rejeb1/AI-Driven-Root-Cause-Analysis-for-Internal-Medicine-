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

    # The brief mandates "Claude Opus 4.x class". That generation is no longer
    # current; claude-opus-5 is the present-day Opus-tier model and the
    # closest honest match to what was specified. Not verified against a live
    # call in this repo -- no ANTHROPIC_API_KEY was available when this was
    # set. Confirm the id resolves before relying on --llm or
    # scripts/synthesize.py --provider anthropic.
    model: str = "claude-opus-5"
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


@dataclass
class OpenAICompatibleLLM:
    """Any server that speaks the OpenAI chat-completions format.

    Written for Ollama -- a local model, no key, nothing leaves the machine,
    which is the only configuration under which the privacy claims in
    RESPONSIBLE_AI.md survive contact with a language model -- and it also
    covers Groq, OpenRouter, Mistral and the rest, which use the same wire
    format behind a key. Standard library only: the whole project runs with
    nothing installed, and an adapter that needs a client package would be
    the first thing to break that.

    Still a deviation from the mandated model, recorded as one. Determinism
    is requested (temperature 0) and not guaranteed: a local model's output
    can vary with the build and the hardware, and the write-up must not
    describe a run with this as reproducible without saying so.
    """

    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:7b"
    api_key: str | None = None
    temperature: float = 0.0
    timeout: float = 300.0  # a 7B model on a laptop CPU is slow, not broken
    # Ask the server to constrain the output to JSON. Ollama honours this;
    # servers that do not simply ignore it, and the proposer's parser copes.
    json_mode: bool = True
    # Temperature 0 was only half of determinism: without a seed, Ollama's
    # sampler still varied between runs, and the single-pass baseline moved
    # from 42.9% to 28.6% on identical input an hour apart. A fixed seed
    # makes a local model repeat itself on the same build and hardware;
    # across builds and machines it still may not, and that caveat stands.
    seed: int | None = 0

    def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        import urllib.error
        import urllib.request

        body: dict = {
            "model": self.model,
            "messages": (
                [{"role": "system", "content": system}] if system else []
            ) + [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            body["seed"] = self.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"no OpenAI-compatible server at {self.base_url}: {exc.reason}. "
                "For a local model: install Ollama, `ollama pull "
                f"{self.model}`, and make sure `ollama serve` is running."
            ) from exc
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response shape: {payload!r:.200}") from exc

    def available(self) -> bool:
        """Whether the server answers at all. Cheap; used to skip, not to fail."""
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.base_url.rstrip('/')}/models", timeout=3
            ) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False


def ollama(model: str = "qwen2.5:7b") -> OpenAICompatibleLLM:
    """A local Ollama model: free, offline, no key."""
    return OpenAICompatibleLLM(model=model)


def from_provider(provider: str, model: str | None = None) -> LLMClient:
    """The client the scripts' ``--provider`` flag names.

    ``ollama`` is the default everywhere a flag exists, because it is the one
    option that needs neither a key nor a network and keeps the project's
    "runs from a clean checkout" property.
    """
    if provider == "ollama":
        return ollama(model or "qwen2.5:7b")
    if provider == "anthropic":
        return AnthropicLLM(model=model) if model else AnthropicLLM()
    if provider == "gemini":
        return GeminiLLM(model=model) if model else GeminiLLM()
    if provider == "openai-compatible":
        # Any hosted OpenAI-format API: base URL and key from the environment,
        # so neither ever appears on a command line or in a shell history.
        base = os.environ.get("LLM_BASE_URL")
        if not base:
            raise RuntimeError("LLM_BASE_URL is not set")
        return OpenAICompatibleLLM(
            base_url=base, model=model or os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY"),
        )
    raise ValueError(f"unknown provider {provider!r}")


PROVIDERS = ("ollama", "anthropic", "gemini", "openai-compatible")

__all__ = [
    "AnthropicLLM", "GeminiLLM", "LLMClient", "NullLLM", "OpenAICompatibleLLM",
    "PROVIDERS", "ScriptedLLM", "from_provider", "ollama",
]
