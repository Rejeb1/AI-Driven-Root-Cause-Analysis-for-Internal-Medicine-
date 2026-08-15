"""Synthetic development fixtures: a small dyspnoea/chest-pain knowledge base.

!! NOT CLINICALLY VALIDATED !!

Every number below is invented by the author of this module for the purpose of
exercising the code path. The likelihoods are plausible-looking, not sourced;
the prevalences are not from any population; the feature sets are incomplete in
ways a physician would notice immediately. Nothing here should be read as
clinical information, and no result computed from it says anything about real
diagnostic performance.

The fixtures exist so that the loop, the gate, the calibrator and the metrics
can be developed and tested without credentialed data. They are the first thing
to delete once real cases are available, and the numbers should be replaced
with sourced prevalences and likelihood ratios under clinician review rather
than incrementally patched. That replacement has started: see ``_SOURCED``
below and ``dxagent.provenance`` for which entries now carry a citation.

The eight conditions cover a differential that is genuinely hard for the right
reason: several of them share their presenting features and separate only on
specific tests, which is what makes the information-gain selector and the
abstention gate do observable work.
"""

from __future__ import annotations

from ..environment import Case
import dataclasses

from ..knowledge import Citation, DiseaseEntry, InMemoryKnowledgeBase, register_costs
from ..merck import QUOTES as _MERCK_QUOTES
from ..provenance import LikelihoodSource, from_narrative, measured

# Acquisition costs in arbitrary consistent units, ordered by tier: history is
# near-free, bedside exam cheap, bloods moderate, cross-sectional imaging dear.
COSTS: dict[str, float] = {
    # history
    "fever": 0.2,
    "productive_cough": 0.2,
    "pleuritic_pain": 0.2,
    "dyspnoea_at_rest": 0.2,
    "orthopnoea": 0.2,
    "leg_swelling": 0.2,
    "calf_tenderness": 0.2,
    "smoking_history": 0.2,
    "exertional_chest_pain": 0.2,
    "sudden_onset": 0.2,
    "recent_immobility": 0.2,
    "palpitations": 0.2,
    "wheeze_subjective": 0.2,
    # bedside
    "exam:crackles": 1.0,
    "exam:reduced_breath_sounds": 1.0,
    "exam:raised_jvp": 1.0,
    "exam:tachycardia": 0.5,
    "exam:friction_rub": 1.0,
    "exam:hypoxia": 0.5,
    "exam:ecg_st_changes": 2.0,
    # bloods
    "lab:raised_wcc": 4.0,
    "lab:raised_d_dimer": 4.0,
    "lab:raised_troponin": 5.0,
    "lab:raised_bnp": 5.0,
    # imaging
    "imaging:cxr_consolidation": 8.0,
    "imaging:cxr_pulmonary_oedema": 8.0,
    "imaging:ctpa_filling_defect": 20.0,
}


def _cite(source: str, locator: str, snippet: str = "") -> tuple[Citation, ...]:
    return (Citation(source_id=source, locator=locator, snippet=snippet),)



# Findings that are facets of one clinical picture rather than independent
# observations. Invented like every other number in this module, and labelled
# as such -- but the *structure* is not arbitrary: each group below is a set of
# findings a clinician would read as one story, and reading them as several is
# precisely the error the naive-Bayes product makes.
#
# The pulmonary-embolism group is the one that matters. In fx-009 all four are
# observed absent, and multiplied independently they bury the true diagnosis
# deep enough that a positive CTPA afterwards cannot retrieve it. They are not
# four independent reassurances; they are one absent clinical picture.
_CORRELATION_GROUPS: tuple[tuple[float, tuple[str, ...]], ...] = (
    # venous thromboembolism: one picture, four ways of asking about it
    (0.85, ("sudden_onset", "leg_swelling", "calf_tenderness", "recent_immobility")),
    # consolidation
    (0.75, ("fever", "productive_cough", "exam:crackles", "imaging:cxr_consolidation")),
    # congestion
    (0.80, ("orthopnoea", "exam:raised_jvp", "lab:raised_bnp",
            "imaging:cxr_pulmonary_oedema")),
    # obstructive airway
    (0.70, ("wheeze_subjective", "smoking_history", "exam:reduced_breath_sounds")),
    # ischaemia
    (0.75, ("exertional_chest_pain", "exam:ecg_st_changes", "lab:raised_troponin")),
)


def correlation_pairs() -> dict[frozenset[str], float]:
    """Pairwise correlations implied by the groups above."""
    pairs: dict[frozenset[str], float] = {}
    for strength, members in _CORRELATION_GROUPS:
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                pairs[frozenset((first, second))] = strength
    return pairs


# ---------------------------------------------------------------------------
# Sourced likelihoods.
#
# Every number in the tables below is invented. This is where they get replaced
# one at a time, as each is looked up and cited. Add an entry here and it
# overrides the invented value *and* records where the replacement came from,
# so `scripts/sensitivity.py` can report coverage and sweep the uncertainty.
#
# Do not source all 135. `scripts/sensitivity.py --correlated` currently finds
# only 2-4 that decide anything on this case set, and the count moves as the
# rest of the model changes (it was 30 before correlation weighting and the
# workup-preemption fix) -- rerun it before picking a target rather than
# trusting a number written down here.
#
# Two ways to add one:
#
#   ("acute_pulmonary_oedema", "lab:raised_bnp"): measured(
#       0.00,                                  # <- the frequency from the paper
#       Citation("AUTHOR-YEAR", "table 2", "quote the sentence you took it from"),
#       low=0.00, high=0.00,                   # <- the reported CI, if given
#   ),
#
#   ("pericarditis", "exam:friction_rub"): from_narrative(
#       "common",                              # <- the phrase the reference uses
#       Citation("MSD", "pericarditis", "a friction rub is common"),
#   ),
#
# `from_narrative` returns the value and the source together, so it is written
# as the whole entry; `measured` returns only the source, so it is paired with
# the value. The rubric behind `from_narrative` is fixed in advance in
# `provenance.py` -- look the phrase up rather than picking a number that seems
# right, because choosing values after seeing which ones score well is how a
# knowledge base gets fitted to its own benchmark.
#
# Leave this empty rather than filling it with plausible guesses. An invented
# number labelled invented is honest; an invented number labelled "measured" is
# not, and it is the only thing here that would actually be misconduct.
_FROM_DDXPLUS: dict[tuple[str, str], tuple[float, LikelihoodSource]] = {
    # generated by scripts/map_concepts.py -- measured from DDXPlus,
    # which is a simulator, so these describe its generating model.
    ("pulmonary_embolism", "pleuritic_pain"): measured(
        0.714,
        Citation("DDXPLUS", "E_220", "3768 of 5277 pulmonary_embolism patients"),
    ),
    ("pulmonary_embolism", "recent_immobility"): measured(
        0.614,
        Citation("DDXPLUS", "E_110", "3239 of 5277 pulmonary_embolism patients"),
    ),
    ("pulmonary_embolism", "leg_swelling"): measured(
        0.899,
        Citation("DDXPLUS", "E_152_@_V_119 +15", "4746 of 5277 pulmonary_embolism patients"),
    ),
    ("community_acquired_pneumonia", "fever"): measured(
        0.694,
        Citation("DDXPLUS", "E_91", "3321 of 4786 community_acquired_pneumonia patients"),
    ),
    ("community_acquired_pneumonia", "productive_cough"): measured(
        0.853,
        Citation("DDXPLUS", "E_77", "4084 of 4786 community_acquired_pneumonia patients"),
    ),
    ("community_acquired_pneumonia", "pleuritic_pain"): measured(
        0.532,
        Citation("DDXPLUS", "E_220", "2546 of 4786 community_acquired_pneumonia patients"),
    ),
    ("acute_coronary_syndrome", "smoking_history"): measured(
        0.723,
        Citation("DDXPLUS", "E_79", "5774 of 7982 acute_coronary_syndrome patients"),
    ),
    ("acute_coronary_syndrome", "exertional_chest_pain"): measured(
        0.360,
        Citation("DDXPLUS", "E_218", "2873 of 7982 acute_coronary_syndrome patients"),
    ),
    ("acute_pulmonary_oedema", "orthopnoea"): measured(
        0.759,
        Citation("DDXPLUS", "E_217", "2709 of 3569 acute_pulmonary_oedema patients"),
    ),
    ("acute_pulmonary_oedema", "exertional_chest_pain"): measured(
        0.767,
        Citation("DDXPLUS", "E_218", "2739 of 3569 acute_pulmonary_oedema patients"),
    ),
    ("acute_pulmonary_oedema", "leg_swelling"): measured(
        0.999,
        Citation("DDXPLUS", "E_152_@_V_119 +15", "3567 of 3569 acute_pulmonary_oedema patients"),
    ),
    ("copd_exacerbation", "productive_cough"): measured(
        0.738,
        Citation("DDXPLUS", "E_77", "2781 of 3769 copd_exacerbation patients"),
    ),
    ("copd_exacerbation", "smoking_history"): measured(
        0.819,
        Citation("DDXPLUS", "E_79", "3086 of 3769 copd_exacerbation patients"),
    ),
    ("copd_exacerbation", "wheeze_subjective"): measured(
        0.910,
        Citation("DDXPLUS", "E_214", "3428 of 3769 copd_exacerbation patients"),
    ),
    ("asthma_exacerbation", "wheeze_subjective"): measured(
        0.868,
        Citation("DDXPLUS", "E_214", "3539 of 4079 asthma_exacerbation patients"),
    ),
    ("pericarditis", "pleuritic_pain"): measured(
        0.690,
        Citation("DDXPLUS", "E_220", "2965 of 4300 pericarditis patients"),
    ),
    ("panic_attack", "palpitations"): measured(
        0.726,
        Citation("DDXPLUS", "E_155", "3428 of 4724 panic_attack patients"),
    ),
}


# The brief's Step 2 route: MSD Manual narrative text, converted through the
# fixed rubric in provenance.py. The first pass read four web pages and found
# one convertible sentence; this pass read the full 19th-edition text (the
# actual chapters: Coronary Artery Disease/ACS, Heart Failure, Pneumonia,
# Pulmonary Embolism, Pericarditis, Anxiety Disorders/Panic) and found nine.
# The gap between "read the whole chapter" and "nine sentences" is the same
# finding as before, just measured on more text: reference prose mostly
# *enumerates* findings ("symptoms include chest discomfort, dyspnoea,
# nausea...") without attaching a frequency word to each one. A sentence has
# to actually carry a rubric term to convert; paraphrasing "the most
# important physical finding" into "characteristic" would be picking the
# number that looks right, which is exactly what the fixed-rubric rule
# exists to prevent -- so pericarditis's friction rub (the textbook's own
# words: "the most important physical finding") stays invented despite being
# the single most textually well-supported finding in this whole KB.
#
# Two targets from the first pass were re-checked against the full chapter
# rather than a search-engine excerpt, and confirmed empty: the ACS chapter
# never mentions palpitations at all, and the panic-disorder chapter (read in
# full, including its "Symptoms and Signs" section) discusses cardiac workup
# only as "general medical evaluation to exclude" other causes -- never a
# stated frequency for troponin or ECG. Both stay invented on purpose.
#
# The quotes themselves live in ``dxagent.merck``, not here -- the same nine
# sentences also back the retrieval corpus (``retrieval.merck_passages``),
# and a number and a retrieved citation for the same finding must never be
# free to say different things about what the source actually said.
_FROM_NARRATIVE: dict[tuple[str, str], tuple[float, LikelihoodSource]] = {
    (q.disease, q.concept): from_narrative(q.phrase, q.citation)
    for q in _MERCK_QUOTES
}

# Frequencies from a published cohort of real patients. These take precedence
# over the DDXPlus figures below, and where the two disagree the disagreement
# is itself worth reporting: DDXPlus puts pleuritic pain in 71% of its
# pulmonary embolism patients, Miniati's 360 real ones show 33%. The simulator
# overstates it twofold, which is the clearest available evidence that
# DDXPLUS-cited numbers describe a generating model rather than patients.
_FROM_LITERATURE: dict[tuple[str, str], tuple[float, LikelihoodSource]] = {
    ("pulmonary_embolism", "pleuritic_pain"): measured(
        0.328,
        Citation(
            "MINIATI-2012",
            "PLoS ONE 7(2):e30891, Firenze sample",
            "pleuritic chest pain in 118 of 360 confirmed acute PE patients",
        ),
    ),
    ("pulmonary_embolism", "leg_swelling"): measured(
        0.175,
        Citation(
            "MINIATI-2012",
            "PLoS ONE 7(2):e30891, Firenze sample",
            "unilateral limb swelling in 63 of 360 confirmed acute PE patients",
        ),
    ),
    ("pulmonary_embolism", "orthopnoea"): measured(
        0.008,
        Citation(
            "MINIATI-2012",
            "PLoS ONE 7(2):e30891, Firenze sample",
            "orthopnoea in 3 of 360 confirmed acute PE patients",
        ),
    ),
    # Reported as *sudden-onset dyspnoea*, 281 of 360. Read here as the fixture's
    # `sudden_onset`, which is a slightly wider claim than the paper makes --
    # the paper counts sudden onset of one symptom, the fixture means the
    # presentation began abruptly. Flagged rather than silently equated.
    ("pulmonary_embolism", "sudden_onset"): measured(
        0.781,
        Citation(
            "MINIATI-2012",
            "PLoS ONE 7(2):e30891, Firenze sample",
            "sudden onset dyspnoea in 281 of 360 confirmed acute PE patients",
        ),
    ),
    # Found by targeted search after both bulk routes were exhausted: DDXPlus
    # records zero ACS patients with palpitations (a rule-base artefact, not a
    # frequency) and the Merck ACS chapter never mentions the symptom at all.
    # Aggregated across the three ACS subtypes the paper reports separately --
    # STEMI 15/117, NSTEMI 52/251, unstable angina 31/105 -- because this
    # knowledge base carries one `acute_coronary_syndrome` entry spanning all
    # three. The subtype denominators sum to 473 against the paper's stated
    # 474 confirmed-ACS patients, which is the arithmetic check that the
    # aggregation is reading the table correctly.
    #
    # The rate differs threefold across subtypes (12.8% STEMI to 29.5%
    # unstable angina), so a single number for "ACS" is a real simplification
    # of this source, not a faithful transcription of it. It is still a
    # measured frequency in real patients where the previous value was a
    # guess.
    ("acute_coronary_syndrome", "palpitations"): measured(
        0.207,
        Citation(
            "ZEGRE-HEMSEY-2018",
            "Research in Nursing & Health 41(5):459-468",
            "palpitations at presentation in 98 of 474 confirmed ACS patients "
            "across five emergency departments",
        ),
        low=0.128,
        high=0.295,
    ),
    # StatPearls, suggested as a source after DDXPlus and Merck were
    # exhausted. It reports frequencies where Merck reports adjectives, and
    # this is the number Merck could not supply: its pericarditis chapter
    # calls the friction rub "the most important physical finding", which is
    # emphasis rather than a frequency, so it stayed invented through two
    # passes.
    #
    # The band is the honest part. StatPearls reports 35-85% and says so
    # explicitly -- "a wide variation in the range of prevalence reported in
    # the literature" -- so the point estimate carries a factor-of-two
    # uncertainty that `sensitivity.py --bands` can sweep. Recorded as
    # measured rather than narrative because it is a reported frequency
    # range, not a qualitative phrase run through the rubric, though it is
    # secondhand: StatPearls is summarising primary literature it cites.
    #
    # One mismatch, flagged rather than equated. The source counts a rub
    # heard "at some point during the illness"; this knowledge base asks
    # whether one is heard at examination. The rub is famously evanescent --
    # the Merck chapter calls it "often intermittent and evanescent" -- so
    # the presentation-time figure is probably lower than the illness-course
    # one, and the band's lower bound is the safer read.
    ("pericarditis", "exam:friction_rub"): measured(
        0.60,
        Citation(
            "STATPEARLS-PERICARDITIS",
            "NCBI Bookshelf NBK431080, History and Physical",
            "friction rub present in 35% to 85% of cases at some point during "
            "the illness, with wide variation across the literature",
        ),
        low=0.35,
        high=0.85,
    ),
}

# Load-bearing numbers that no source can supply, and the reason is structural
# ---------------------------------------------------------------------------
# `scripts/sensitivity.py` names four likelihoods that currently change a
# diagnosis. Two of them are on panic attack and both were searched for
# directly, after DDXPlus was exhausted and the Merck panic-disorder chapter
# came up empty:
#
#   panic_attack / lab:raised_troponin    (0.03, invented)
#   panic_attack / exertional_chest_pain  (0.20, invented)
#
# Neither is measurable, and not because nobody has looked. Panic attack is a
# diagnosis of exclusion: a normal troponin is part of how the diagnosis is
# reached, and exertional chest pain is one of the features that argues the
# patient *out* of it and toward a cardiac cause. A study reporting "troponin
# was raised in X% of panic attack patients" would be reporting on a cohort
# assembled by a rule that partly excludes raised troponin. The frequency is
# not unpublished; it is close to undefined.
#
# This is a different failure from the Merck ceiling. There the text simply
# does not state a frequency for a finding it does describe. Here the finding
# is entangled with the diagnostic criterion, so the quantity a citation would
# have to measure barely exists. Both numbers stay invented, deliberately, and
# the honest description of them is that they encode a definitional relation
# rather than an observed rate.

# Precedence, weakest first: a converted phrase from a reference text, then a
# frequency counted in the simulator, then a frequency counted in real
# patients. Each later dict overwrites the earlier ones on a shared key, so
# the strongest available source always wins regardless of dict-literal order.
_SOURCED: dict[tuple[str, str], tuple[float, LikelihoodSource]] = {
    **_FROM_NARRATIVE,
    **_FROM_DDXPLUS,
    **_FROM_LITERATURE,
}


def _apply_sources(kb: InMemoryKnowledgeBase) -> None:
    """Overwrite invented likelihoods with sourced ones, recording provenance."""
    by_disease: dict[str, dict[str, tuple[float, LikelihoodSource]]] = {}
    for (label, concept), pair in _SOURCED.items():
        by_disease.setdefault(label, {})[concept] = pair

    for label, replacements in by_disease.items():
        entry = kb.entries.get(label)
        if entry is None:
            raise KeyError(
                f"_SOURCED names {label!r}, which is not in the knowledge base. "
                "A typo here would silently source nothing."
            )
        features = dict(entry.features)
        sources = dict(entry.sources)
        for concept, (value, source) in replacements.items():
            if concept not in features:
                raise KeyError(
                    f"_SOURCED names {label}/{concept}, which that entry does "
                    "not characterise."
                )
            features[concept] = value
            sources[concept] = source
        kb.entries[label] = dataclasses.replace(
            entry, features=features, sources=sources
        )
    kb._marginals.clear()


def build_knowledge_base(correlated: bool = False) -> InMemoryKnowledgeBase:
    """Return the synthetic KB. See the module docstring: numbers are invented.

    ``correlated=True`` additionally supplies the dependence structure above.
    Off by default so that every measurement taken before it existed still
    describes the default knowledge base.
    """
    register_costs(COSTS)
    kb = InMemoryKnowledgeBase()

    kb.add(
        DiseaseEntry(
            label="community_acquired_pneumonia",
            prevalence=0.20,
            features={
                "fever": 0.85,
                "productive_cough": 0.80,
                "pleuritic_pain": 0.45,
                "dyspnoea_at_rest": 0.55,
                "sudden_onset": 0.20,
                "orthopnoea": 0.10,
                "leg_swelling": 0.05,
                "exam:crackles": 0.80,
                "exam:tachycardia": 0.60,
                "exam:hypoxia": 0.45,
                "exam:raised_jvp": 0.05,
                "lab:raised_wcc": 0.80,
                "lab:raised_d_dimer": 0.35,
                "lab:raised_troponin": 0.08,
                "lab:raised_bnp": 0.15,
                "imaging:cxr_consolidation": 0.90,
                "imaging:cxr_pulmonary_oedema": 0.05,
                "imaging:ctpa_filling_defect": 0.02,
            },
            citations=_cite("FIXTURE-KB", "cap", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="pulmonary_embolism",
            prevalence=0.10,
            features={
                "fever": 0.15,
                "productive_cough": 0.10,
                "pleuritic_pain": 0.65,
                "dyspnoea_at_rest": 0.80,
                "sudden_onset": 0.75,
                "calf_tenderness": 0.40,
                "leg_swelling": 0.35,
                "recent_immobility": 0.50,
                "orthopnoea": 0.10,
                "exam:tachycardia": 0.75,
                "exam:hypoxia": 0.60,
                "exam:crackles": 0.15,
                "exam:raised_jvp": 0.20,
                "lab:raised_d_dimer": 0.95,
                "lab:raised_wcc": 0.25,
                "lab:raised_troponin": 0.25,
                "lab:raised_bnp": 0.20,
                "imaging:ctpa_filling_defect": 0.95,
                "imaging:cxr_consolidation": 0.10,
                "imaging:cxr_pulmonary_oedema": 0.03,
            },
            red_flag=True,
            citations=_cite("FIXTURE-KB", "pe", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="acute_coronary_syndrome",
            prevalence=0.12,
            features={
                "exertional_chest_pain": 0.80,
                "sudden_onset": 0.60,
                "dyspnoea_at_rest": 0.45,
                "pleuritic_pain": 0.10,
                "fever": 0.08,
                "productive_cough": 0.05,
                "palpitations": 0.35,
                "smoking_history": 0.55,
                "exam:tachycardia": 0.45,
                "exam:ecg_st_changes": 0.75,
                "exam:crackles": 0.20,
                "lab:raised_troponin": 0.90,
                "lab:raised_d_dimer": 0.20,
                "lab:raised_wcc": 0.25,
                "lab:raised_bnp": 0.30,
                "imaging:cxr_consolidation": 0.05,
                "imaging:ctpa_filling_defect": 0.02,
            },
            red_flag=True,
            citations=_cite("FIXTURE-KB", "acs", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="acute_pulmonary_oedema",
            prevalence=0.18,
            features={
                "orthopnoea": 0.85,
                "leg_swelling": 0.75,
                "dyspnoea_at_rest": 0.80,
                "fever": 0.10,
                "productive_cough": 0.30,
                "sudden_onset": 0.25,
                "exertional_chest_pain": 0.20,
                "exam:crackles": 0.75,
                "exam:raised_jvp": 0.80,
                "exam:tachycardia": 0.50,
                "exam:hypoxia": 0.40,
                "lab:raised_bnp": 0.92,
                "lab:raised_troponin": 0.30,
                "lab:raised_d_dimer": 0.30,
                "lab:raised_wcc": 0.20,
                "imaging:cxr_pulmonary_oedema": 0.85,
                "imaging:cxr_consolidation": 0.15,
                "imaging:ctpa_filling_defect": 0.03,
            },
            citations=_cite("FIXTURE-KB", "hf", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="copd_exacerbation",
            prevalence=0.16,
            features={
                "smoking_history": 0.90,
                "productive_cough": 0.75,
                "wheeze_subjective": 0.70,
                "dyspnoea_at_rest": 0.70,
                "fever": 0.30,
                "orthopnoea": 0.25,
                "leg_swelling": 0.20,
                "sudden_onset": 0.20,
                "exam:reduced_breath_sounds": 0.70,
                "exam:crackles": 0.30,
                "exam:hypoxia": 0.55,
                "exam:tachycardia": 0.45,
                "exam:raised_jvp": 0.15,
                "lab:raised_wcc": 0.40,
                "lab:raised_bnp": 0.20,
                "lab:raised_d_dimer": 0.25,
                "imaging:cxr_consolidation": 0.20,
                "imaging:cxr_pulmonary_oedema": 0.08,
            },
            citations=_cite("FIXTURE-KB", "copd", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="asthma_exacerbation",
            prevalence=0.10,
            features={
                "wheeze_subjective": 0.85,
                "dyspnoea_at_rest": 0.65,
                "productive_cough": 0.40,
                "sudden_onset": 0.55,
                "smoking_history": 0.20,
                "fever": 0.15,
                "orthopnoea": 0.10,
                "exam:reduced_breath_sounds": 0.50,
                "exam:tachycardia": 0.45,
                "exam:hypoxia": 0.35,
                "exam:crackles": 0.10,
                "lab:raised_wcc": 0.20,
                "lab:raised_bnp": 0.05,
                "imaging:cxr_consolidation": 0.05,
                "imaging:cxr_pulmonary_oedema": 0.03,
            },
            citations=_cite("FIXTURE-KB", "asthma", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="pericarditis",
            prevalence=0.06,
            features={
                "pleuritic_pain": 0.80,
                "fever": 0.45,
                "sudden_onset": 0.50,
                "exertional_chest_pain": 0.25,
                "dyspnoea_at_rest": 0.30,
                "productive_cough": 0.10,
                "exam:friction_rub": 0.55,
                "exam:ecg_st_changes": 0.60,
                "exam:tachycardia": 0.40,
                "exam:crackles": 0.08,
                "lab:raised_troponin": 0.35,
                "lab:raised_wcc": 0.45,
                "lab:raised_d_dimer": 0.15,
                "imaging:cxr_consolidation": 0.05,
            },
            citations=_cite("FIXTURE-KB", "pericarditis", "synthetic entry, not sourced"),
        )
    )
    kb.add(
        DiseaseEntry(
            label="panic_attack",
            prevalence=0.08,
            features={
                "palpitations": 0.80,
                "sudden_onset": 0.85,
                "dyspnoea_at_rest": 0.60,
                "exertional_chest_pain": 0.20,
                "pleuritic_pain": 0.15,
                "fever": 0.03,
                "productive_cough": 0.03,
                "exam:tachycardia": 0.70,
                "exam:hypoxia": 0.03,
                "exam:crackles": 0.03,
                "lab:raised_troponin": 0.03,
                "lab:raised_d_dimer": 0.10,
                "lab:raised_wcc": 0.10,
                "imaging:cxr_consolidation": 0.02,
                "imaging:ctpa_filling_defect": 0.01,
            },
            citations=_cite("FIXTURE-KB", "panic", "synthetic entry, not sourced"),
        )
    )
    _apply_sources(kb)
    if correlated:
        kb.set_correlations(correlation_pairs())
    return kb


def build_cases() -> list[Case]:
    """A small labelled case set, deliberately mixed in difficulty.

    Includes clean presentations, two genuinely ambiguous cases that a competent
    system *should* escalate rather than guess on, and one case whose true
    diagnosis is under-supported by the volunteered history so that the loop has
    to go and get evidence.
    """
    return [
        Case(
            case_id="fx-001",
            presenting_complaint="fever and productive cough for three days",
            diagnosis="community_acquired_pneumonia",
            initial_findings=("fever", "productive_cough"),
            features={
                "fever": True,
                "productive_cough": True,
                "pleuritic_pain": True,
                "dyspnoea_at_rest": True,
                "sudden_onset": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": False,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": True,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": True,
                "lab:raised_d_dimer": False,
                "lab:raised_troponin": False,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": True,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        Case(
            case_id="fx-002",
            presenting_complaint="sudden breathlessness after a long flight",
            diagnosis="pulmonary_embolism",
            initial_findings=("dyspnoea_at_rest", "sudden_onset"),
            features={
                "fever": False,
                "productive_cough": False,
                "pleuritic_pain": True,
                "dyspnoea_at_rest": True,
                "sudden_onset": True,
                "orthopnoea": False,
                "leg_swelling": True,
                "calf_tenderness": True,
                "smoking_history": False,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": False,
                "recent_immobility": True,
                "exam:crackles": False,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": True,
                "lab:raised_troponin": False,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": True,
            },
        ),
        Case(
            case_id="fx-003",
            presenting_complaint="orthopnoea and ankle swelling worsening over a week",
            diagnosis="acute_pulmonary_oedema",
            initial_findings=("orthopnoea", "leg_swelling"),
            features={
                "fever": False,
                "productive_cough": True,
                "pleuritic_pain": False,
                "dyspnoea_at_rest": True,
                "sudden_onset": False,
                "orthopnoea": True,
                "leg_swelling": True,
                "calf_tenderness": False,
                "smoking_history": True,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": True,
                "exam:tachycardia": False,
                "exam:hypoxia": True,
                "exam:raised_jvp": True,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": True,
                "lab:raised_troponin": False,
                "lab:raised_bnp": True,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": True,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        Case(
            case_id="fx-004",
            presenting_complaint="chest tightness on exertion with palpitations",
            diagnosis="acute_coronary_syndrome",
            initial_findings=("exertional_chest_pain", "palpitations"),
            features={
                "fever": False,
                "productive_cough": False,
                "pleuritic_pain": False,
                "dyspnoea_at_rest": True,
                "sudden_onset": True,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": True,
                "exertional_chest_pain": True,
                "palpitations": True,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": False,
                "exam:tachycardia": True,
                "exam:hypoxia": False,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": True,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": False,
                "lab:raised_troponin": True,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        Case(
            case_id="fx-005",
            presenting_complaint="breathless smoker with wheeze and cough",
            diagnosis="copd_exacerbation",
            initial_findings=("smoking_history", "wheeze_subjective", "productive_cough"),
            features={
                "fever": False,
                "productive_cough": True,
                "pleuritic_pain": False,
                "dyspnoea_at_rest": True,
                "sudden_onset": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": True,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": True,
                "recent_immobility": False,
                "exam:crackles": False,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": True,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": False,
                "lab:raised_troponin": False,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        Case(
            case_id="fx-006",
            presenting_complaint="sharp chest pain worse on breathing, low-grade fever",
            diagnosis="pericarditis",
            initial_findings=("pleuritic_pain", "fever"),
            features={
                "fever": True,
                "productive_cough": False,
                "pleuritic_pain": True,
                "dyspnoea_at_rest": False,
                "sudden_onset": True,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": False,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": False,
                "exam:tachycardia": True,
                "exam:hypoxia": False,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": True,
                "exam:ecg_st_changes": True,
                "lab:raised_wcc": True,
                "lab:raised_d_dimer": False,
                "lab:raised_troponin": True,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        Case(
            case_id="fx-007",
            presenting_complaint="sudden breathlessness with palpitations, no other symptoms",
            diagnosis="panic_attack",
            initial_findings=("sudden_onset", "palpitations", "dyspnoea_at_rest"),
            features={
                "fever": False,
                "productive_cough": False,
                "pleuritic_pain": False,
                "dyspnoea_at_rest": True,
                "sudden_onset": True,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": False,
                "exertional_chest_pain": False,
                "palpitations": True,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": False,
                "exam:tachycardia": True,
                "exam:hypoxia": False,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": False,
                "lab:raised_troponin": False,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        # Deliberately ambiguous: overlapping COPD / heart-failure features in a
        # smoker with both. The correct behaviour here is escalation, not a coin
        # flip, and this case exists to check that the gate does that.
        Case(
            case_id="fx-008",
            presenting_complaint="breathless smoker with swollen ankles and wheeze",
            diagnosis="acute_pulmonary_oedema",
            initial_findings=("smoking_history", "wheeze_subjective", "leg_swelling"),
            features={
                "fever": False,
                "productive_cough": True,
                "pleuritic_pain": False,
                "dyspnoea_at_rest": True,
                "sudden_onset": False,
                "orthopnoea": True,
                "leg_swelling": True,
                "calf_tenderness": False,
                "smoking_history": True,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": True,
                "recent_immobility": False,
                "exam:crackles": True,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "exam:raised_jvp": True,
                "exam:reduced_breath_sounds": True,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": True,
                "lab:raised_troponin": False,
                "lab:raised_bnp": True,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": True,
                "imaging:ctpa_filling_defect": False,
            },
        ),
        # Atypical presentations. In both of these the true diagnosis is NOT the
        # argmax under the KB's modal picture of it, because the patient does not
        # present the way the textbook says. This is the only part of the fixture
        # set that can exercise the abstention gate honestly: without cases the
        # system gets wrong, selective-prediction metrics are undefined, and a
        # gate that never fires scores identically to no gate at all.
        #
        # Note what is *not* being done here: the cases are not perturbed at
        # random to manufacture errors. Atypical presentation is a real
        # phenomenon with real clinical stakes, and a system that abstains on
        # exactly these is behaving correctly.
        Case(
            case_id="fx-009",
            # PE masquerading as pneumonia: infarction gives fever and pleuritic
            # pain, and the raised white count points the wrong way.
            presenting_complaint="fever, pleuritic chest pain and cough",
            diagnosis="pulmonary_embolism",
            initial_findings=("fever", "pleuritic_pain", "productive_cough"),
            features={
                "fever": True,
                "productive_cough": True,
                "pleuritic_pain": True,
                "dyspnoea_at_rest": True,
                "sudden_onset": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": False,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": True,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": False,
                "lab:raised_wcc": True,
                "lab:raised_d_dimer": True,
                "lab:raised_troponin": False,
                "lab:raised_bnp": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": True,
            },
        ),
        Case(
            case_id="fx-010",
            # Silent ischaemia: breathlessness as the sole complaint, no chest
            # pain at all. The history alone points at acute pulmonary oedema or COPD.
            presenting_complaint="breathlessness at rest, no chest pain",
            diagnosis="acute_coronary_syndrome",
            initial_findings=("dyspnoea_at_rest", "smoking_history"),
            features={
                "fever": False,
                "productive_cough": False,
                "pleuritic_pain": False,
                "dyspnoea_at_rest": True,
                "sudden_onset": False,
                "orthopnoea": True,
                "leg_swelling": False,
                "calf_tenderness": False,
                "smoking_history": True,
                "exertional_chest_pain": False,
                "palpitations": False,
                "wheeze_subjective": False,
                "recent_immobility": False,
                "exam:crackles": True,
                "exam:tachycardia": False,
                "exam:hypoxia": True,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "exam:friction_rub": False,
                "exam:ecg_st_changes": True,
                "lab:raised_wcc": False,
                "lab:raised_d_dimer": False,
                "lab:raised_troponin": True,
                "lab:raised_bnp": True,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "imaging:ctpa_filling_defect": False,
            },
        ),
    ]


__all__ = ["COSTS", "build_cases", "build_knowledge_base"]
