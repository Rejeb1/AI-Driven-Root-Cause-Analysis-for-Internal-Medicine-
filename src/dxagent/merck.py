"""Curated excerpts from the Merck Manual (19th ed.), read by hand.

The single source of truth for two consumers that must never be allowed to
say two different things about the same sentence: the narrative-tier
likelihoods in ``datasets.fixtures`` (a quote becomes a number, via the
fixed rubric in ``provenance``) and the retrieval corpus in ``retrieval``
(the same quote, embedded and made searchable, so a hypothesis can be
grounded in the actual sentence rather than just the number it produced).
Keeping them as two independently-maintained lists is how they drift; this
file exists so there is exactly one place to add or correct a quote.

Every entry is one short sentence, individually attributed to a chapter.
The manual itself is not reproduced -- nothing here is more than what the
project's own likelihood tables already cite, and none of it approaches
chunk-and-index-the-whole-chapter, which would mean redistributing pages of
a copyrighted, purchased textbook rather than citing specific claims from
it. That is a deliberate scope decision, not an oversight: see
``dxagent.retrieval`` for what it means for the "searchable KB" deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Citation


@dataclass(frozen=True)
class NarrativeQuote:
    """One MSD sentence, the disease/finding it sources, and how to convert it."""

    disease: str
    concept: str
    phrase: str  # a key in provenance.NARRATIVE_RUBRIC
    citation: Citation


# Read from the full 19th-edition text (via the PDF's own bookmarks, not a
# web search) for six chapters: Coronary Artery Disease/ACS, Heart Failure,
# Pneumonia, Pulmonary Embolism, Pericarditis, Anxiety Disorders/Panic. Most
# of that text does not convert -- reference prose enumerates findings
# without attaching a frequency word to each one -- and these nine sentences
# are what did. See ``datasets.fixtures`` for the two specific targets that
# were checked and came up empty on purpose (ACS/palpitations, panic
# disorder's cardiac workup), which stay invented rather than being
# force-mapped from a paraphrase.
QUOTES: tuple[NarrativeQuote, ...] = (
    NarrativeQuote(
        "acute_pulmonary_oedema",
        "dyspnoea_at_rest",
        "common",
        Citation(
            "MSD-19E",
            "Ch. 211 Heart Failure, Symptoms and Signs",
            "the most common symptoms are dyspnea, reflecting pulmonary "
            "congestion, and fatigue",
        ),
    ),
    NarrativeQuote(
        "acute_pulmonary_oedema",
        "exam:tachycardia",
        "common",
        Citation(
            "MSD-19E",
            "Ch. 211 Heart Failure",
            "sinus tachycardia, a common compensatory change in HF",
        ),
    ),
    NarrativeQuote(
        "community_acquired_pneumonia",
        "dyspnoea_at_rest",
        "rare",
        Citation(
            "MSD-19E",
            "Ch. 196 Pneumonia, Symptoms and Signs",
            "dyspnea usually is mild and exertional and is rarely present "
            "at rest",
        ),
    ),
    NarrativeQuote(
        "community_acquired_pneumonia",
        "imaging:cxr_consolidation",
        "always",
        Citation(
            "MSD-19E",
            "Ch. 196 Pneumonia, Diagnosis",
            "chest x-ray almost always shows some degree of infiltrate; "
            "rarely, an infiltrate is absent in the first 24 to 48 h",
        ),
    ),
    NarrativeQuote(
        "pulmonary_embolism",
        "exam:tachycardia",
        "common",
        Citation(
            "MSD-19E",
            "Ch. 194 Pulmonary Embolism, Symptoms and Signs",
            "the most common signs of PE are tachycardia and tachypnea",
        ),
    ),
    NarrativeQuote(
        "pericarditis",
        "fever",
        "common",
        Citation(
            "MSD-19E",
            "Ch. 216 Pericarditis, Symptoms and Signs",
            "fever, chills, and weakness are common",
        ),
    ),
    NarrativeQuote(
        "pericarditis",
        "dyspnoea_at_rest",
        "sometimes",
        Citation(
            "MSD-19E",
            "Ch. 216 Pericarditis, Symptoms and Signs",
            "acute pericarditis tends to cause chest pain and a pericardial "
            "rub, sometimes with dyspnea",
        ),
    ),
    NarrativeQuote(
        "pericarditis",
        "lab:raised_troponin",
        "always",
        Citation(
            "MSD-19E",
            "Ch. 216 Pericarditis, Diagnosis",
            "troponin is almost always elevated in acute pericarditis due "
            "to epicardial involvement",
        ),
    ),
    # The DSM definition itself, not a frequency claim about a population --
    # "hallmark" fits a defining criterion better than "common" would.
    NarrativeQuote(
        "panic_attack",
        "sudden_onset",
        "hallmark",
        Citation(
            "MSD-19E",
            "Ch. 158 Anxiety Disorders, Panic Disorder",
            "a panic attack is the sudden onset of a discrete, brief period "
            "of intense discomfort, anxiety, or fear",
        ),
    ),
)


__all__ = ["NarrativeQuote", "QUOTES"]
