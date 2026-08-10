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
# web search) across two passes over eight chapters: Asthma, COPD, Pulmonary
# Embolism, Pneumonia, Coronary Artery Disease/ACS, Heart Failure,
# Pericarditis, Anxiety Disorders/Panic.
#
# Why fourteen sentences and not a hundred. The second pass scanned every
# chapter against all 107 unsourced cells and produced about thirty
# candidates; five survived reading them in context. The rejections are worth
# recording because they are the same few failure modes, and anyone extending
# this file will hit them again:
#
#   - a drug's adverse effect read as a presenting sign ("tachycardia and
#     tremor are the most common acute adverse effects of inhaled
#     beta-2 agonists" is about salbutamol, not about COPD)
#   - a different entity inside the same chapter (pulmonary infarction and
#     fat embolism inside Pulmonary Embolism; PJP and hospital-acquired
#     pneumonia inside Pneumonia; ABPA inside Asthma)
#   - a neighbouring chapter bleeding across a page boundary (acute
#     bronchitis' "the most common symptom is cough" landing in the PE range,
#     abdominal aortic aneurysm landing in Pericarditis)
#   - a frequency word that attaches to something other than the finding
#     ("usually have combinations of [six signs]" does not say any one of the
#     six is usual; "usually in most leads" describes lead distribution, not
#     whether ST changes occur)
#   - a statement about a test rather than a finding ("sputum Gram stain and
#     culture usually have no role")
#
# Two candidates were rejected as judgement calls rather than errors, and
# both could reasonably be argued the other way. Pericarditis' "the most
# important physical finding is a ... friction rub" is the strongest textual
# support for any single finding in this knowledge base, and "most important"
# is not a frequency -- mapping it to "characteristic" would be choosing the
# number that looks right. Pericarditis' "nonproductive cough may be present"
# implies productive cough is atypical, but implication is not statement.
# Both stay invented.
#
# See ``datasets.fixtures`` for two further targets checked and confirmed
# absent: the ACS chapter never mentions palpitations, and the panic-disorder
# chapter discusses cardiac workup only as exclusion, never as frequency.
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
    # --- second pass ------------------------------------------------------
    # A scan of all eight relevant chapters against the 107 unsourced cells
    # returned about thirty candidate sentences. Read in context, most fail
    # for a nameable reason and are listed in the module docstring. These five
    # survive.
    NarrativeQuote(
        "copd_exacerbation",
        "exam:reduced_breath_sounds",
        "common",
        Citation(
            "MSD-19E",
            "Ch. 192 COPD, Symptoms and Signs",
            "common signs include decreased breath sounds, prolonged "
            "expiratory phase of respiration, and wheezing",
        ),
    ),
    NarrativeQuote(
        "pulmonary_embolism",
        "productive_cough",
        "less common",
        Citation(
            "MSD-19E",
            "Ch. 194 Pulmonary Embolism, Symptoms and Signs",
            "less common symptoms include cough and hemoptysis",
        ),
    ),
    NarrativeQuote(
        "pulmonary_embolism",
        "exam:crackles",
        "less common",
        Citation(
            "MSD-19E",
            "Ch. 194 Pulmonary Embolism, Symptoms and Signs",
            "less commonly, patients have hypotension, a loud second heart "
            "sound, and crackles or wheezing",
        ),
    ),
    NarrativeQuote(
        "pulmonary_embolism",
        "fever",
        "can occur",
        Citation(
            "MSD-19E",
            "Ch. 194 Pulmonary Embolism, Symptoms and Signs",
            "fever can occur; DVT and PE are often overlooked causes of fever",
        ),
    ),
    # The chapter is not self-consistent here: this sentence says "sometimes"
    # while a nearby one calls cough with blood-tinged sputum "common". Taking
    # the weaker of the two, and recording that the source disagrees with
    # itself rather than quietly choosing the reading that suits. The value
    # happens to equal the invented one it replaces, so nothing moves -- what
    # changes is that the number now has a citation instead of an author.
    NarrativeQuote(
        "acute_pulmonary_oedema",
        "productive_cough",
        "sometimes",
        Citation(
            "MSD-19E",
            "Ch. 211 Heart Failure, acute pulmonary edema",
            "findings are severe dyspnea, diaphoresis, wheezing, and "
            "sometimes blood-tinged frothy sputum",
        ),
    ),
)


__all__ = ["NarrativeQuote", "QUOTES"]
