"""Findings from case-report text, proposed by a model and checked by rule.

The one job in this project a language model is suited to. Nineteen real
cases were extracted by hand under the rules in ``datasets/real_cases.py``;
this module puts those rules in front of a model and then enforces the part
of them that can be enforced mechanically: every finding must carry a quote
that appears verbatim in the source, or it is dropped. What comes out is a
*proposal* for a human to check against the report -- never a case that
goes into the set on the model's word.

Why the quote requirement is the whole design. The failure mode of model
extraction is not a wrong polarity; it is a plausible finding the text does
not contain. A quote that must be found in the source turns that from a
judgement call into a string search. The model can still misread a quote it
found -- "no history of fever" recorded as fever present -- and that is what
the human check is for; but it cannot invent evidence and have it survive.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .schemas import Polarity

# The rules, as the model sees them. Kept close to the wording in
# real_cases.py so a reader can check the two say the same thing.
EXTRACTION_SYSTEM = """You extract clinical findings from a published case report \
into a fixed vocabulary. You are an extraction component, not a diagnostician: \
you do not infer, you transcribe.

Rules:
- Record a finding ONLY when the text states it explicitly. "present" needs an \
explicit statement that it was observed or reported; "absent" needs an explicit \
denial or a measured normal value. Silence is not absence: a finding the report \
never mentions is NOT recorded at all.
- Every recorded finding must carry a "quote": a short passage copied VERBATIM \
from the text that states it. Do not paraphrase the quote. If you cannot quote \
it, do not record it.
- A close-but-not-quite match is left out. A positional chest pain is not \
pleuritic pain. A prolonged PR interval is not PR depression. A CT scan with no \
infiltrate is not a chest-radiograph negative. "Examination otherwise \
unremarkable" is not a negative for any specific sign; "lungs were clear" is.
- Thresholds: tachycardia means heart rate above 100; hypoxia means oxygen \
saturation below 95% on room air; fever means 38C or above, measured or \
reported; raised white cell count means above about 11 x10^9/L; raised \
troponin, D-dimer and BNP mean above the reporting laboratory's stated cutoff.
- Findings that arose later in hospital as complications of treatment are not \
part of the presentation; record the presentation.
- Respond with JSON only, no prose and no markdown fences, in the form:
  {"findings": [{"concept": "<vocabulary key>", "polarity": "present" | "absent", \
"quote": "<verbatim passage>"}]}
- Use ONLY concept keys from the provided vocabulary. Omit any concept the text \
does not explicitly settle."""


@dataclass(frozen=True)
class Extracted:
    concept: str
    polarity: Polarity
    quote: str


@dataclass(frozen=True)
class Rejected:
    concept: str
    polarity: str
    quote: str
    reason: str


@dataclass
class ExtractionResult:
    accepted: list[Extracted] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    chunks: int = 0
    raw: list[str] = field(default_factory=list)

    @property
    def features(self) -> dict[str, bool]:
        """The proposal in the shape ``Case.features`` takes."""
        return {e.concept: e.polarity is Polarity.PRESENT for e in self.accepted}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """Whether the quote appears verbatim in the text.

    Whitespace and case are normalised, because XML-to-text conversion
    changes both and a model that copied the passage faithfully should not
    lose it to a line break. Nothing else is forgiven: a paraphrase fails.
    """
    q = _normalise(quote)
    return len(q) >= 8 and q in _normalise(text)


def case_section(text: str) -> str:
    """The case-presentation part of a report, when a heading marks it.

    Discussion sections quote other patients and general frequencies; a model
    reading "hemoptysis (50%, 6/12)" from a literature review will happily
    record haemoptysis. Cut at the discussion when one can be found.
    """
    start = re.search(r"\b(case (presentation|report|description|summary)|case\b)", text, re.I)
    begin = start.start() if start else 0
    end = re.search(r"\n\s*(discussion|conclusion)s?\b", text[begin:], re.I)
    stop = begin + end.start() if end else len(text)
    # Only trust a cut that leaves something; a heading regex that matched
    # the word "case" in a running sentence would otherwise return a stub.
    return text[begin:stop] if (start or end) and stop - begin > 80 else text


def chunk(text: str, size: int = 5000, overlap: int = 400) -> list[str]:
    """Overlapping windows, because a local model's context is small and a
    report silently truncated at the context limit loses its labs."""
    if len(text) <= size:
        return [text]
    # An overlap at or above the window size would step backwards forever;
    # cap it at a quarter of the window rather than trust the caller.
    overlap = min(overlap, size // 4)
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def vocabulary_prompt(concepts: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {label}" for key, label in concepts.items())


def extract_findings(
    text: str, llm, concepts: dict[str, str], chunk_size: int = 5000
) -> ExtractionResult:
    """Propose findings for ``text`` over ``concepts`` (key -> human label).

    Runs the model once per chunk of the case section and merges. A finding
    the model reports twice with different polarities is rejected as a
    conflict rather than resolved; the human resolves it.
    """
    result = ExtractionResult()
    section = case_section(text)
    seen: dict[str, Extracted] = {}
    vocab = vocabulary_prompt(concepts)
    for piece in chunk(section, chunk_size):
        result.chunks += 1
        prompt = f"Vocabulary:\n{vocab}\n\nCase report text:\n\"\"\"\n{piece}\n\"\"\""
        try:
            raw = llm.complete(prompt, system=EXTRACTION_SYSTEM, max_tokens=2048)
        except Exception as exc:  # the model failing is a rejection, not a crash
            result.rejected.append(Rejected("*", "*", "", f"model call failed: {exc}"))
            continue
        result.raw.append(raw)
        try:
            payload = json.loads(raw)
            items = payload.get("findings", []) if isinstance(payload, dict) else []
        except (json.JSONDecodeError, AttributeError):
            result.rejected.append(Rejected("*", "*", raw[:120], "response was not JSON"))
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            concept = str(item.get("concept", "")).strip()
            polarity = str(item.get("polarity", "")).strip().lower()
            quote = str(item.get("quote", "")).strip()
            if concept not in concepts:
                result.rejected.append(Rejected(concept, polarity, quote, "not a vocabulary concept"))
                continue
            if polarity not in ("present", "absent"):
                result.rejected.append(Rejected(concept, polarity, quote, "polarity must be present or absent"))
                continue
            if not quote_in_text(quote, text):
                result.rejected.append(Rejected(concept, polarity, quote, "quote not found verbatim in the source"))
                continue
            pol = Polarity.PRESENT if polarity == "present" else Polarity.ABSENT
            if concept in seen and seen[concept].polarity is not pol:
                result.rejected.append(Rejected(concept, polarity, quote, "conflicts with an earlier chunk; human resolves"))
                result.accepted = [e for e in result.accepted if e.concept != concept]
                continue
            if concept not in seen:
                seen[concept] = Extracted(concept, pol, quote)
                result.accepted.append(seen[concept])
    return result


__all__ = ["EXTRACTION_SYSTEM", "Extracted", "ExtractionResult", "Rejected",
           "case_section", "chunk", "extract_findings", "quote_in_text"]
