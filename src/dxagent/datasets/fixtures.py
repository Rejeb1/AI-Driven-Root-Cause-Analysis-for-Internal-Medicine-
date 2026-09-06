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
from ..provenance import (
    NARRATIVE_RUBRIC,
    LikelihoodSource,
    Provenance,
    from_narrative,
    measured,
)

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
    # Same source, same chapter, a different section (Evaluation, not History
    # and Physical). Found by searching for this one specific unsourced cell
    # after real-case testing showed the selector never picked it -- not a
    # broad re-search of ground already covered.
    #
    # The source gives a floor, not a range: "more than half", not "35% to
    # 85%" the way the friction rub sentence does. There is no honest way to
    # invent an upper bound the source does not state, so none is given --
    # `high=None` here means exactly that, not "unbounded above 1.0".
    #
    # This point estimate (0.50) is *lower* than the invented value it
    # replaces (0.60). Sourcing it does not necessarily help the selector
    # prioritise this test on the real case that prompted the search, and it
    # was added anyway: the question was whether the number is real, not
    # whether the real number is convenient.
    # Sourced because the guess would have been badly wrong in the direction
    # that matters. Filling this cell was prompted by a confirmed NSTEMI
    # ranking fifth, and the tempting move was a low value -- COPD is not a
    # cardiac diagnosis, so a raised troponin "should" argue against it. The
    # literature says the opposite: troponin elevation is common in acute
    # exacerbations, and inventing 0.05 here would have made the model
    # discriminate COPD from ACS far more confidently than the evidence
    # allows, in a direction nobody would have checked.
    #
    # The band is wide because the reported range genuinely is: 18-27% on
    # conventional assays, up to 74% with high-sensitivity troponin T, which
    # is an assay difference rather than a disagreement about patients. The
    # point estimate is this study's own headline figure; note its abstract
    # says 32% where its results table gives 19 of 50, which is 38%. The
    # discrepancy is the source's, recorded rather than silently resolved.
    ("copd_exacerbation", "lab:raised_troponin"): measured(
        0.32,
        Citation(
            "NOORAIN-2016",
            "Noorain S, Lung India 2016;33(1):53-57",
            "cardiac troponin I was positive in 32% of patients admitted "
            "with acute exacerbation of COPD (n=50, cutoff 0.017 ug/L)",
        ),
        low=0.18,
        high=0.74,
    ),
    # The first sourcing pass in this project that *confirmed* an invented
    # value instead of overturning it. The cell held 0.20 by judgement; the
    # three largest studies in a systematic review pool to 0.18. Recorded
    # because the entries above all tell the opposite story, and a file that
    # only preserves the corrections which embarrassed the author is keeping
    # a flattering record of its own honesty.
    #
    # It also confirms a prediction about *which* cells are sourceable at
    # all. Consolidation is a binary radiological observation, so studies
    # using "consolidation", "new infiltrate" and "pneumonic infiltrate" are
    # answering one question and can be pooled. The natriuretic-peptide cells
    # were attempted first and abandoned for the opposite reason: BNP against
    # NT-proBNP, cutoffs of 100 against 300 against age-specific, so the
    # published figures are not on a common scale and a column assembled from
    # them would manufacture comparisons the sources do not support.
    #
    # The band is the literature's own spread and it is wide: 13% to 54%
    # across seven studies. The point estimate takes the three largest
    # (Emerman 88/685, Myint 1505/9338, Saleh 2714/14111 -> 4307/24134),
    # which are ED and national-audit populations; the high end comes from
    # small selected inpatient cohorts of 63 and 113 patients.
    ("copd_exacerbation", "imaging:cxr_consolidation"): measured(
        0.18,
        Citation(
            "AECOPD-IMAGING-SR-2020",
            "Thoracic Imaging at Exacerbation of COPD: A Systematic Review, "
            "PMC7385406, chest radiograph studies",
            "consolidation or new infiltrate in 4307 of 24134 exacerbations "
            "across the three largest included studies (Emerman 1993, "
            "Myint 2011, Saleh 2015); reported range 13-54%",
        ),
        low=0.13,
        high=0.54,
    ),
    # Recovered from pooled likelihood ratios rather than counted, which needs
    # stating because it is a different act from the entries above.
    #
    # The JAMA Rational Clinical Examination review of dyspnoea in the
    # emergency department reports LR+ and LR- for each finding and no raw
    # frequencies, which is why an earlier pass through this project recorded
    # the series as unusable. That was too quick. The two ratios are both
    # functions of the same pair of unknowns and the system inverts:
    #
    #     LR+ = sens / (1 - spec)      spec = (LR+ - 1) / (LR+ - LR-)
    #     LR- = (1 - sens) / spec      sens = 1 - LR- * spec
    #
    # and sensitivity in a cohort of dyspnoeic emergency patients is exactly
    # P(finding | heart failure) over this project's own population. Only the
    # sensitivity half is taken. The complement, 1 - spec, is P(finding | not
    # heart failure) pooled over the other seven, and crackles and a raised
    # JVP are differentially caused by pneumonia and by cor pulmonale, so
    # pooling them would smear one disease's rate across the rest -- the same
    # objection that kept PIOPED II's crackles row out of this file.
    #
    # Two caveats travel with the method. A meta-analysis may pool LR+ and
    # LR- over different subsets of studies, so the recovered pair is close
    # rather than exact; and the bands here are the reported confidence
    # intervals pushed through the same inversion, which spans the reported
    # uncertainty without being a confidence interval in its own right.
    #
    # It was worth doing. A raised JVP was invented at 0.80 against a
    # recovered 0.39, so the guess was more than twice the measurement, in a
    # finding this differential leans on to separate cardiac from pulmonary
    # causes.
    ("acute_pulmonary_oedema", "exam:raised_jvp"): measured(
        0.39,
        Citation(
            "WANG-2005",
            "Wang CS et al., JAMA 2005;294(15):1944-56, jugular venous distension",
            "positive LR 5.1 (95% CI 3.2-7.9) and negative LR 0.66 (0.57-0.77) "
            "for heart failure in dyspnoeic emergency patients, which invert "
            "to a sensitivity of 0.39",
        ),
        low=0.30,
        high=0.46,
    ),
    ("acute_pulmonary_oedema", "exam:crackles"): measured(
        0.60,
        Citation(
            "WANG-2005",
            "Wang CS et al., JAMA 2005;294(15):1944-56, pulmonary rales",
            "positive LR 2.8 (95% CI 1.9-4.1) and negative LR 0.51 (0.37-0.70) "
            "for heart failure in dyspnoeic emergency patients, which invert "
            "to a sensitivity of 0.60",
        ),
        low=0.48,
        high=0.69,
    ),
    # Overwrites the DDXPlus value of 0.755 by the precedence rule below: a
    # frequency counted in real emergency patients outranks one counted in a
    # simulator. The two disagree by half again, which is the same
    # simulator-against-cohort gap already recorded for pleuritic pain in
    # pulmonary embolism, and it is reported rather than reconciled.
    ("acute_pulmonary_oedema", "orthopnoea"): measured(
        0.50,
        Citation(
            "WANG-2005",
            "Wang CS et al., JAMA 2005;294(15):1944-56, orthopnoea",
            "positive LR 2.2 (95% CI 1.2-3.9) and negative LR 0.65 (0.45-0.92) "
            "for heart failure in dyspnoeic emergency patients, which invert "
            "to a sensitivity of 0.50; DDXPlus's simulated rate is 0.755",
        ),
        low=0.34,
        high=0.62,
    ),
    # Two rivals attempted from quartiles and NOT taken. The attempt is
    # recorded because the reasoning is the useful part.
    #
    # Two studies report D-dimer by final diagnosis as medians with
    # interquartile ranges rather than as fractions above a threshold. Where
    # the 0.5 cutoff falls among the quartiles is a distribution-free fact,
    # and by that alone both invented values are wrong:
    #
    #     acute coronary syndrome, chest-symptom ED (n=1039)
    #         median 0.40, IQR 0.27-0.80  ->  between 0.25 and 0.50 above
    #     acute coronary syndrome, all-ED D-dimer ordered (n=552)
    #         median 0.57, IQR 0.32-1.22  ->  between 0.50 and 0.75 above
    #     heart failure, chest-symptom ED (n=451)
    #         median 1.70, IQR 0.90-3.10  ->  at least 0.75 above
    #
    # The invented cells hold 0.20 and 0.30, outside every one of those
    # ranges. That is a third independent confirmation that this column
    # understates the rivals, and it is why the defect is recorded below
    # rather than dismissed.
    #
    # They are still not written, for three reasons that hold regardless of
    # what they would do to any score.
    #
    # The two acute coronary syndrome sources give bounds that **do not
    # overlap**: 0.25-0.50 against 0.50-0.75. That is not a wide band, it is
    # two studies disagreeing about the quantity, most likely because the
    # second population is patients in whom someone ordered a D-dimer and is
    # therefore selected for suspected thrombosis. A value cannot be called
    # measured when the sources contradict each other.
    #
    # A point estimate inside the bounds needs a log-normal assumption for
    # the analyte and a sigma fitted from the reported log IQR. Every other
    # value in this file is a counted proportion or exact algebra on reported
    # statistics; this would be the first to depend on a distributional
    # model, and it is not labelled differently in the provenance tiers.
    #
    # And a half-sourced column is not obviously better than a uniformly
    # invented one. Writing these two would leave pulmonary embolism at 0.93,
    # heart failure at 0.91 and acute coronary syndrome near 0.4 beside
    # pneumonia, pericarditis and panic attack still at 0.35, 0.15 and 0.10 --
    # invented values the same evidence says are too low. The contrast
    # between sourced and unsourced cells in one column is an artefact of the
    # sourcing effort rather than a fact about patients, and here it is a
    # large one.
    #
    # Measured before reverting, so the cost of not doing it is on the record:
    # writing them took the fixture set from 9/10 to 10/10 and cost a real
    # patient its correct answer, a hard case its ranking, and calibration
    # 0.324 to 0.358 ECE. The instruments this project trusts moved the wrong
    # way, which is corroboration rather than the reason.
    #
    # One of the six understated rivals, measured directly.
    #
    # The entry below records that a D-dimer meta-analysis puts specificity at
    # 51%, so P(raised D-dimer | not pulmonary embolism) is 0.49 pooled, while
    # this knowledge base gave the seven rivals values averaging 0.22. That
    # defect was left unfixed because 0.49 is pooled across a mix of diseases
    # and writing it into all seven would assert a panic attack raises a
    # D-dimer as often as a pneumonia does.
    #
    # A per-disease figure has no such problem. In 148 patients admitted with
    # an acute COPD exacerbation and investigated for pulmonary embolism, 92
    # had it excluded, and 53 of those still had a D-dimer above the
    # conventional 0.5 threshold: 0.58 against an invented 0.25.
    #
    # Read carefully, because the paper's own phrasing invites an error. It
    # says "fifty-three patients (36%) who did not have PE had higher than
    # normal D-dimer levels", and 36% is 53 of the full 148 rather than of the
    # 92 without embolism. The quantity this cell needs is the latter. The
    # paper also writes its units as pg/mL where the values are plainly mg/L,
    # which is recorded rather than silently corrected.
    #
    # This is the second, independent confirmation that the rivals were
    # understated about twofold, and it arrives from a completely different
    # study design than the pooled specificity did.
    ("copd_exacerbation", "lab:raised_d_dimer"): measured(
        0.58,
        Citation(
            "DDIMER-AECOPD-2013",
            "D-dimer cut-off in AECOPD, PMC3755691, 148 patients",
            "53 of the 92 patients with acute COPD exacerbation in whom "
            "pulmonary embolism was excluded still had a D-dimer above the "
            "conventional 0.5 threshold (57.6%)",
        ),
        low=0.47,
        high=0.68,
    ),
    # A confirming source, and a defect it exposes that is deliberately left
    # unfixed.
    #
    # Sensitivity is P(finding | disease) directly, so a diagnostic-accuracy
    # meta-analysis needs no inversion. This one puts D-dimer at 93% sensitive
    # for pulmonary embolism against an invented 0.95, which is the second
    # time sourcing has confirmed a guess rather than overturned it.
    #
    # The specificity is the uncomfortable half. At 51%, P(raised D-dimer |
    # not pulmonary embolism) is 0.49 across the arm of suspected patients who
    # turned out not to have one -- and this knowledge base gives those same
    # seven diseases 0.10 to 0.35, averaging about 0.22. The rivals are
    # collectively understated by roughly a factor of two, which makes
    # D-dimer a stronger discriminator here than it is in a real emergency
    # department.
    #
    # That is not fixed, and the reason is the same one that rejected the
    # blanket backoff and the completed grid: 0.49 is a pooled figure, and
    # writing it into all seven would assert that a panic attack raises a
    # D-dimer as often as a pneumonia does. Both policies of the form "write
    # one value everywhere" have been measured in this project and both were
    # rejected. The defect is recorded in DESIGN.md as measured rather than
    # papered over with a number nobody chose.
    #
    # The pooled studies use cutoffs from 190 to 500 microg/L on six
    # different turbidimetric assays. That heterogeneity is the same objection
    # that blocked the natriuretic-peptide column, and it is tolerable here
    # only because a single figure is being taken from a single pooled
    # analysis rather than a column assembled across incompatible sources.
    ("pulmonary_embolism", "lab:raised_d_dimer"): measured(
        0.93,
        Citation(
            "DDIMER-TURBIDIMETRIC-META",
            "Turbidimetric D-dimer meta-analysis, DARE NBK70277, 9 studies n=1901",
            "pooled sensitivity 93% (95% CI 89-96) for pulmonary embolism in "
            "the emergency department; pooled specificity 51% (42-59)",
        ),
        low=0.89,
        high=0.96,
    ),
    # A faithfully transcribed number that was wrong anyway, because the
    # source describes a different population from the one this model reasons
    # over. Found by reading a demo transcript, not by a failing test.
    #
    # The Merck chapter on pneumonia says dyspnoea "usually is mild and
    # exertional and is rarely present at rest", the rubric maps "rare" to
    # 0.05, and the transcription is correct. But that chapter describes
    # pneumonia at every severity, most of it managed at home, while this
    # knowledge base's population is people who came to an emergency
    # department *because* they were breathless. Inside that population a
    # pneumonia patient is breathless at rest most of the time.
    #
    # The cost was not hypothetical. On pmc-4775775, a real 85-year-old with
    # a confirmed pneumonia, the single strongest argument the model made
    # *against* the correct diagnosis was that she was breathless at rest --
    # and it ranked pneumonia fifth of eight on a patient with a productive
    # cough and a basal infiltrate. A sourced number attracts less scrutiny
    # than an invented one, which is how a value thirteen times too small
    # survived several audits with a citation attached.
    #
    # Replaced from a population-matched cohort: 954 acutely admitted
    # patients, 265 with expert-panel-confirmed CAP. One definitional gap
    # remains and is stated rather than hidden -- the study records
    # "dyspnoea", not "dyspnoea at rest" specifically, so this value may run
    # slightly high. That gap is far smaller than the thirteenfold one it
    # replaces, and the direction of the previous error was to argue against
    # the right answer.
    ("community_acquired_pneumonia", "dyspnoea_at_rest"): measured(
        0.67,
        Citation(
            "CAP-DIAGNOSTIC-MODEL-2024",
            "Community-acquired pneumonia diagnostic model, PMC11141191, Table 1",
            "dyspnoea recorded in 171 of 265 patients with expert-panel "
            "confirmed community-acquired pneumonia among 954 acutely "
            "admitted patients (67.3%)",
        ),
        low=0.61,
        high=0.73,
    ),
    # Two cells the PIOPED II pass should have written and did not.
    #
    # That pass took the *no-PE* arm of Tables 3 and 6 and used it for the
    # seven rival diseases, which was the whole point of it: a study of
    # patients investigated for embolism and found not to have one describes
    # exactly this differential's other candidates. The same two tables also
    # report the PE arm, and those columns were read, quoted in this file's
    # own citation snippet, and then not applied to pulmonary embolism
    # itself. The disease the study is about was left on a simulator value
    # and an invented one.
    #
    #     finding             was                    PIOPED II, PE arm
    #     recent_immobility   0.61  DDXPlus          48 of 192   (0.25)
    #     calf_tenderness     0.40  invented         90 of 192   (0.47)
    #
    # The immobility correction is the larger one and follows a precedent
    # this project already set twice: a frequency counted in real patients
    # outranks one counted in a simulator. DDXPlus generates immobility in
    # 61% of its embolism patients; PIOPED II observed 25% in 192 real ones.
    # That is the same simulator-against-cohort gap recorded for pleuritic
    # pain, 71% against 33%, and resolved the same way.
    #
    # Worth noticing how it survived. The pass that introduced these numbers
    # was careful about the arm it was reading, wrote a long comment
    # justifying the pooling across rivals, and never asked whether the
    # source it had open said anything about the disease in the middle of the
    # differential. A sourcing pass aimed at one part of a table can leave
    # the rest of that table unread.
    ("pulmonary_embolism", "recent_immobility"): measured(
        0.25,
        Citation(
            "PIOPED-II-2007",
            "Stein PD et al., Am J Med 2007;120(10):871-9, Table 3",
            "immobilization in 48 of 192 patients with confirmed pulmonary "
            "embolism (25%), against 121 of 632 in whom it was excluded",
        ),
        low=0.20,
        high=0.25,
    ),
    ("pulmonary_embolism", "calf_tenderness"): measured(
        0.47,
        Citation(
            "PIOPED-II-2007",
            "Stein PD et al., Am J Med 2007;120(10):871-9, Table 6",
            "signs of deep venous thrombosis in the calf or thigh in 90 of "
            "192 patients with confirmed pulmonary embolism (47%), against "
            "146 of 632 in whom it was excluded",
        ),
        low=0.47,
        high=0.47,
    ),
    # The second cell in the column this project declared blocked, and the
    # one that explains why an earlier correction backfired.
    #
    # Acute pulmonary embolism obstructs the pulmonary circulation and strains
    # the right ventricle, and a strained ventricle releases natriuretic
    # peptide. The knowledge base had P(raised BNP | pulmonary embolism) at an
    # invented 0.20, which asserts the opposite of the mechanism. In 63
    # consecutive emergency patients with acute pulmonary embolism -- with the
    # haemodynamically unstable deliberately excluded, so if anything this
    # understates -- 39 had NT-proBNP at or above 350 ng/l. That is 0.62.
    #
    # **The threshold equivalence is an assumption and is stated here rather
    # than buried.** The acute pulmonary oedema cell in this column comes from
    # a BNP > 100 pg/mL study; this one is NT-proBNP >= 350 ng/l. Those are
    # different analytes. They are taken as answering the same question --
    # "did the natriuretic peptide come back raised" -- because each is the
    # standard clinical decision threshold for its own assay in acute
    # dyspnoea. That is weaker than a same-assay comparison and a reader
    # should know it.
    #
    # It is also narrower than the case this file refused earlier, and the
    # distinction is the point rather than an excuse. The rejected COPD
    # figure rested on an "age-specific NT-proBNP" threshold that varies per
    # patient and was never stated numerically in its source; there was no
    # scale to compare against. Here both thresholds are single fixed
    # published numbers. That is the second time the blanket claim "this
    # column cannot be assembled" has had to be narrowed, which suggests the
    # original claim was reached too quickly.
    #
    # Why it matters beyond one cell. A correction to the D-dimer column --
    # raising acute pulmonary oedema from an indefensible 0.30 -- was tried,
    # measured, and reverted because it cost a real patient its answer:
    # pmc-4565285, a pulmonary embolism with a raised pro-BNP of 11,479 and a
    # positive CTPA, which flipped to acute pulmonary oedema. The reason was
    # this cell. With BNP at 0.20 for embolism and 0.93 for oedema, a raised
    # natriuretic peptide argued almost five to one for the wrong diagnosis,
    # and the understated D-dimer values had been quietly cancelling that
    # error. Two invented numbers wrong in opposite directions can look like
    # a working model, and correcting either one alone makes it worse.
    ("pulmonary_embolism", "lab:raised_bnp"): measured(
        0.62,
        Citation(
            "NTPROBNP-APE-2012",
            "NT-proBNP and RV overload in acute pulmonary embolism, PMC3422178",
            "NT-proBNP was at or above 350 ng/l in 39 of 63 consecutive "
            "emergency patients with acute pulmonary embolism (32 of 37 with "
            "right ventricular dysfunction, 7 of 26 without); high-risk "
            "haemodynamically unstable patients were excluded",
        ),
        low=0.50,
        high=0.75,
    ),
    # A column this project declared blocked, and the one cell in it that
    # turns out not to be. The distinction is worth keeping straight because
    # the earlier claim was too broad.
    #
    # The objection to natriuretic peptide was never that no figure exists.
    # It was that assembling a *column* means pairing a heart-failure value
    # measured on BNP > 100 pg/mL with a COPD value measured on an
    # age-specific NT-proBNP threshold, which are different analytes on
    # different scales, and the resulting column would state a comparison its
    # sources do not support. That objection still holds for the other five
    # cells and they stay invented.
    #
    # It does not apply to a single cell taken from a single study at a
    # single stated cutoff. Wang's review reports BNP at a 100 pg/mL
    # threshold with sensitivity 93.5% and specificity 52.9%, and those two
    # cross-check against the negative likelihood ratio of 0.11 quoted in the
    # same review: (1 - 0.935) / 0.529 = 0.123, which recovers it. An
    # independent figure from the Breathing Not Properly cohort at the same
    # cutoff gives 90% and 76%, so the band spans the two.
    ("acute_pulmonary_oedema", "lab:raised_bnp"): measured(
        0.93,
        Citation(
            "WANG-2005",
            "Wang CS et al., JAMA 2005;294(15):1944-56, BNP at 100 pg/mL",
            "sensitivity 93.5% and specificity 52.9% for heart failure in "
            "dyspnoeic emergency patients at a BNP cutoff of 100 pg/mL, "
            "consistent with the same review's negative LR of 0.11",
        ),
        low=0.90,
        high=0.94,
    ),
    # Same review, same inversion, two more findings that carry matched
    # ratios. The chest radiograph one is worth noticing on its own account:
    # a sensitivity of 0.54 says the film is normal in nearly half of acute
    # heart failure, which is the well-known insensitivity of radiography in
    # this setting and precisely the sort of thing an invented 0.85 erases.
    ("acute_pulmonary_oedema", "imaging:cxr_pulmonary_oedema"): measured(
        0.54,
        Citation(
            "WANG-2005",
            "Wang CS et al., JAMA 2005;294(15):1944-56, pulmonary venous congestion",
            "positive LR 12 (95% CI 6.8-21.0) and negative LR 0.48 (0.28-0.83) "
            "for heart failure in dyspnoeic emergency patients, which invert "
            "to a sensitivity of 0.54",
        ),
        low=0.19,
        high=0.73,
    ),
    # DDXPlus records leg swelling in 7202 of 7205 simulated acute pulmonary
    # oedema patients. A likelihood of 0.999 for any clinical finding should
    # have been suspicious on its face; the real figure is half that. This is
    # the clearest instance yet of the caution this file keeps repeating
    # about simulator-derived numbers, and it survived several passes because
    # a *sourced* number attracts less scrutiny than an invented one.
    ("acute_pulmonary_oedema", "leg_swelling"): measured(
        0.50,
        Citation(
            "WANG-2005",
            "Wang CS et al., JAMA 2005;294(15):1944-56, leg oedema",
            "positive LR 2.3 (95% CI 1.5-3.7) and negative LR 0.64 (0.47-0.87) "
            "for heart failure in dyspnoeic emergency patients, which invert "
            "to a sensitivity of 0.50; DDXPlus's simulated rate is 0.999",
        ),
        low=0.31,
        high=0.61,
    ),
    # Deliberately not taken from the same table, and recorded so the decision
    # is visible rather than looking like an oversight: "any abnormal ECG"
    # carries LR+ 2.2 and LR- 0.64, which invert to 0.51. The concept here is
    # exam:ecg_st_changes, which is ST-segment change specifically, not any
    # abnormality at all. Writing 0.51 into it would be the mis-mapping the
    # DDXPlus mapping file warns about -- a confidently sourced wrong number,
    # which is worse than the invented one it replaced.
    ("pericarditis", "exam:ecg_st_changes"): measured(
        0.50,
        Citation(
            "STATPEARLS-PERICARDITIS",
            "NCBI Bookshelf NBK431080, Evaluation",
            "more than half of patients with acute pericarditis exhibit "
            "characteristic electrocardiogram changes that evolve through "
            "4 stages over several weeks",
        ),
        low=0.50,
        high=None,
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

# ---------------------------------------------------------------------------
# PIOPED II's "No PE" arm, for findings no disease here causes.
#
# A route that only became visible when someone asked why a clinician was
# needed for numbers the internet might hold. Most cells in this grid really
# are unpublished -- nobody measures P(raised BNP | panic attack). But two of
# them are Wells criteria, and PE rule-out studies necessarily report how
# often each criterion appears in the patients who turned out *not* to have
# a pulmonary embolism. That negative arm is exactly this knowledge base's
# other seven diseases: people who presented with acute dyspnoea or chest
# pain and had something else.
#
# Pooling across those seven is legitimate here and would not be for most
# findings. Immobility and the signs of a DVT are not *caused* by pneumonia
# or by a panic attack, so their rate barely depends on which of the seven a
# patient turns out to have. Wheeze, breath sounds and crackles are also in
# this paper's tables and are deliberately not taken from it: those vary by
# disease, and the pooled figure would smear asthma's rate across
# pericarditis.
#
# Both replace an inherited marginal that was indefensible -- 0.61 for
# immobility and 0.40 for DVT signs, in every disease that did not describe
# them, because pulmonary embolism was the only entry that did.
#
# Worth recording that the guess would have gone the wrong way again. The
# invented values written for these cells earlier ranged 0.03-0.10 for DVT
# signs; the measured figure is 0.23, two to eight times higher. That is the
# third time in this session that sourcing has overturned an intuition, and
# every time the intuition was too tidy.
_PIOPED_NO_PE_CITATION = Citation(
    "PIOPED-II-2007",
    "Stein PD et al., Am J Med 2007;120(10):871-9, Tables 3 and 6",
    "among patients investigated for suspected pulmonary embolism in whom it "
    "was excluded, immobilization was present in 121 of 632 (19%) and signs "
    "of deep venous thrombosis in the calf or thigh in 146 of 632 (23%)",
)

# Every disease here except pulmonary embolism itself, which describes both
# findings already and from a stronger source.
_NON_PE_DISEASES: tuple[str, ...] = (
    "community_acquired_pneumonia",
    "acute_coronary_syndrome",
    "acute_pulmonary_oedema",
    "copd_exacerbation",
    "asthma_exacerbation",
    "pericarditis",
    "panic_attack",
)

# Bands span the paper's own two columns -- patients with no prior
# cardiopulmonary disease against all patients -- rather than being widened
# by judgement.
_PIOPED_NO_PE: dict[str, tuple[float, float, float]] = {
    "recent_immobility": (0.19, 0.16, 0.19),
    "calf_tenderness": (0.23, 0.21, 0.23),
}

for _concept, (_value, _low, _high) in _PIOPED_NO_PE.items():
    for _label in _NON_PE_DISEASES:
        _FROM_LITERATURE[(_label, _concept)] = measured(
            _value, _PIOPED_NO_PE_CITATION, low=_low, high=_high
        )


def _seed_pioped_cells(kb: InMemoryKnowledgeBase) -> None:
    """Put the cells in place so ``_apply_sources`` can source them.

    ``_apply_sources`` refuses to write a concept an entry does not already
    carry, which is the check that stops a typo silently sourcing nothing.
    These fourteen cells are new rather than corrections, so they are seeded
    with the value that is about to overwrite them.
    """
    for label in _NON_PE_DISEASES:
        entry = kb.entries[label]
        features = dict(entry.features)
        for concept, (value, _, _) in _PIOPED_NO_PE.items():
            features.setdefault(concept, value)
        kb.entries[label] = dataclasses.replace(entry, features=features)
    kb._marginals.clear()


# Precedence, weakest first: a converted phrase from a reference text, then a
# frequency counted in the simulator, then a frequency counted in real
# patients. Each later dict overwrites the earlier ones on a shared key, so
# the strongest available source always wins regardless of dict-literal order.
_SOURCED: dict[tuple[str, str], tuple[float, LikelihoodSource]] = {
    **_FROM_NARRATIVE,
    **_FROM_DDXPLUS,
    **_FROM_LITERATURE,
}


# Disease priors, derived from presentation-conditional aetiology
# ---------------------------------------------------------------------------
# For most of this project the priors were invented and no source for them
# was known. DDXPlus cannot supply them -- its paper says generation rates
# were capped into a 10-100% band, so its per-pathology counts are a
# rebalancing artefact -- and the Merck chapters give disease epidemiology in
# the population, which is a different quantity from "of patients presenting
# this way, how many have each cause".
#
# StatPearls has that quantity, in two symptom-side articles rather than the
# disease-side ones searched first:
#
#   Dyspnea (NBK499965): pneumonia/LRTI 20-26%, heart failure 15-28%,
#     COPD 13-18%, asthma 13-15%, and PE, ACS and psychogenic dyspnoea each
#     "fewer than 5%".
#   Chest Pain (NBK470557, citing Fruergaard et al.): ACS 31%, GERD 30%,
#     musculoskeletal 28%, pericarditis 4%, PE 2%, pneumonia/pleuritis 2%.
#
# **These are derived numbers, not measured ones, and the derivation rests on
# three assumptions that a reader is entitled to reject.** They are stated
# here rather than buried, and each value's band is wide enough to survive
# them being wrong:
#
#   1. *Renormalisation.* Both sources list causes outside this differential
#      -- GERD and musculoskeletal pain are 58% of chest pain between them --
#      and the shares below are rescaled so the eight sum to one. That is
#      consistent with the knowledge base's own closed-world assumption, and
#      it does inflate every one of these eight.
#   2. *Below-threshold values.* "Fewer than 5%" and "not named at all" are
#      both read as 2.5%. The first is an upper bound; the second is not a
#      statement about frequency at all.
#   3. *The presentation mix.* This project's presentation is dyspnoea *and*
#      chest pain, and nothing says how the two are mixed. The point estimate
#      assumes 50/50.
#
# The band is the honest part, and for acute coronary syndrome it is
# enormous: 3% if the presentation is dyspnoea-dominated, 63% if chest-pain
# dominated. That twenty-fold spread is a real property of the differential,
# not a defect in the sourcing -- which of the two complaints a patient leads
# with genuinely does change the prior that much. A single confident number
# there would be the dishonest option.
_PRIOR_CITATION = Citation(
    "STATPEARLS-PRESENTATION",
    "NBK499965 Dyspnea, Epidemiology; NBK470557 Chest Pain, Etiology "
    "(Fruergaard et al.)",
    "aetiology of dyspnoea and of chest pain presentations to the emergency "
    "department, renormalised over this differential's eight causes",
)

_PRIOR_NOTE = (
    "derived: 50/50 blend of the dyspnoea and chest-pain aetiologies, "
    "renormalised over eight causes; band spans the two source presentations"
)

# value, low (dyspnoea-only), high (chest-pain-only)
_PRIORS: dict[str, tuple[float, float, float]] = {
    "acute_coronary_syndrome": (0.252, 0.030, 0.633),
    "community_acquired_pneumonia": (0.188, 0.041, 0.274),
    "acute_pulmonary_oedema": (0.180, 0.051, 0.256),
    "copd_exacerbation": (0.135, 0.051, 0.185),
    "asthma_exacerbation": (0.124, 0.051, 0.167),
    "pericarditis": (0.049, 0.030, 0.082),
    "panic_attack": (0.038, 0.030, 0.051),
    "pulmonary_embolism": (0.034, 0.030, 0.041),
}


# ---------------------------------------------------------------------------
# Completing the grid.
#
# Every cell below was previously absent from its disease's feature dict and
# therefore answered by the KB-wide marginal. That was never "no claim": the
# reasoner used a number either way, so the only question was whether anyone
# chose it. Nobody did, and the marginal chose badly, because it is computed
# only over the diseases that *do* describe a finding -- so the more specific
# a sign is, the higher the value it hands to every disease that lacks it.
# A pericardial friction rub, listed only for pericarditis at 0.60, was
# therefore asserted at 0.60 for panic attack. Recent immobility, a Wells
# criterion listed only for pulmonary embolism at 0.61, was asserted at 0.61
# for pneumonia. These are worse than invented numbers; they are invented
# numbers nobody can be held to.
#
# So they are chosen here, deliberately and visibly, and they are INVENTED --
# the provenance report counts them as such and the sourced fraction drops
# accordingly. That drop is not a regression. It is the count finally
# including claims the model was already making.
#
# The values are not one rule applied 77 times. Blanket-low was measured and
# rejected (see DESIGN.md), and the reason it failed is visible in the split
# below: a finding can be absent from a disease's profile for two entirely
# different reasons.
#
#   Disease-generated signs -- produced by the pathology itself. A friction
#   rub needs an inflamed pericardium; a CTPA filling defect needs clot.
#   Absent from another disease's profile, these really are near-absent.
#
#   Risk factors and comorbid findings -- NOT produced by the disease, so
#   their rate in a disease's population is a base rate, not zero. A patient
#   with pneumonia can perfectly well have been immobile for three days or
#   have smoked for forty years. Setting these low would be exactly as wrong
#   as the marginal is, in the opposite direction, and this is the same trap
#   that nearly produced an invented 0.05 for troponin in COPD before the
#   literature said 32%.
#
# None of this is sourced and none of it is clinician-reviewed. It is one
# engineer's judgement, recorded so it can be argued with.
_COMPLETIONS: dict[tuple[str, str], float] = {}


def _complete(concept: str, values: dict[str, float]) -> None:
    for label, value in values.items():
        _COMPLETIONS[(label, concept)] = value


# -- disease-generated signs ------------------------------------------------
# Pericardial rub: needs inflamed pericardium. Slightly higher for the two
# diseases that can produce a *pleural* rub an examiner might record here.
_complete("exam:friction_rub", {
    "pulmonary_embolism": 0.05, "community_acquired_pneumonia": 0.05,
    "acute_coronary_syndrome": 0.03, "acute_pulmonary_oedema": 0.02,
    "copd_exacerbation": 0.02, "asthma_exacerbation": 0.02,
    "panic_attack": 0.02,
})
# Clot on CTPA. Near-absent without pulmonary embolism.
_complete("imaging:ctpa_filling_defect", {
    "copd_exacerbation": 0.02, "asthma_exacerbation": 0.02, "pericarditis": 0.02,
})
# ST changes. Not near-zero everywhere: PE genuinely mimics ACS on the ECG
# (the project's own PE case report is titled exactly that), and oedema is
# often ischaemia-driven.
_complete("exam:ecg_st_changes", {
    "pulmonary_embolism": 0.25, "acute_pulmonary_oedema": 0.20,
    "community_acquired_pneumonia": 0.05, "panic_attack": 0.03,
})
# Oedema on CXR. ACS causes cardiogenic oedema often enough that a low value
# would be wrong; pericarditis gives effusion rather than oedema.
_complete("imaging:cxr_pulmonary_oedema", {
    "acute_coronary_syndrome": 0.20, "pericarditis": 0.05, "panic_attack": 0.02,
})
# Raised JVP. Not a respiratory sign, but tamponade physiology makes it a
# real pericarditis finding, and RV infarction makes it a real ACS one.
_complete("exam:raised_jvp", {
    "pericarditis": 0.35, "acute_coronary_syndrome": 0.15,
    "asthma_exacerbation": 0.05, "panic_attack": 0.02,
})
_complete("exam:hypoxia", {
    "acute_coronary_syndrome": 0.20, "pericarditis": 0.08,
})
_complete("lab:raised_bnp", {"pericarditis": 0.15, "panic_attack": 0.03})
_complete("lab:raised_d_dimer", {"asthma_exacerbation": 0.12})

# -- shared symptoms, rate varying by disease -------------------------------
# Wheeze: the marginal handed 0.89 to six diseases because only the two
# airway diseases describe it. "Cardiac asthma" makes oedema the highest of
# the rest.
_complete("wheeze_subjective", {
    "acute_pulmonary_oedema": 0.30, "community_acquired_pneumonia": 0.20,
    "pulmonary_embolism": 0.10, "panic_attack": 0.10,
    "acute_coronary_syndrome": 0.08, "pericarditis": 0.03,
})
_complete("exam:reduced_breath_sounds", {
    "community_acquired_pneumonia": 0.35, "acute_pulmonary_oedema": 0.15,
    "pulmonary_embolism": 0.10, "acute_coronary_syndrome": 0.05,
    "pericarditis": 0.05, "panic_attack": 0.02,
})
_complete("palpitations", {
    "pulmonary_embolism": 0.25, "acute_pulmonary_oedema": 0.20,
    "pericarditis": 0.20, "copd_exacerbation": 0.12,
    "asthma_exacerbation": 0.12, "community_acquired_pneumonia": 0.08,
})
_complete("pleuritic_pain", {
    "copd_exacerbation": 0.10, "acute_pulmonary_oedema": 0.08,
    "asthma_exacerbation": 0.08,
})
_complete("exertional_chest_pain", {
    "pulmonary_embolism": 0.15, "copd_exacerbation": 0.10,
    "community_acquired_pneumonia": 0.08, "asthma_exacerbation": 0.08,
})
_complete("orthopnoea", {
    "acute_coronary_syndrome": 0.20, "pericarditis": 0.15, "panic_attack": 0.05,
})
_complete("leg_swelling", {
    "acute_coronary_syndrome": 0.10, "pericarditis": 0.10,
    "asthma_exacerbation": 0.05, "panic_attack": 0.03,
})

# -- risk factors: base rates, deliberately NOT low -------------------------
# The trap this whole block exists to avoid. Smoking is not caused by any of
# these diseases, so its rate in each disease's population is roughly the
# population's own.
#
# Immobility and the signs of a DVT used to be chosen here too, on the same
# argument. They are now measured instead, from PIOPED II's arm of patients
# investigated for pulmonary embolism who turned out not to have one -- see
# _PIOPED_NO_PE above. The values written here by judgement were 0.03-0.10
# for DVT signs against a measured 0.23, so the guess was wrong by two to
# eight times, which is the argument for sourcing over completing wherever a
# source exists.
_complete("smoking_history", {
    "acute_pulmonary_oedema": 0.45, "community_acquired_pneumonia": 0.35,
    "pulmonary_embolism": 0.30, "pericarditis": 0.25, "panic_attack": 0.25,
})


def _complete_uncharacterised(kb: InMemoryKnowledgeBase) -> None:
    """Write the chosen values in, and refuse to paper over a stale entry.

    Raising rather than skipping when a cell is already characterised keeps
    this block from quietly shadowing a sourced number later: if sourcing
    reaches one of these cells, the completion for it must be deleted
    deliberately, and the failure says so.
    """
    by_disease: dict[str, dict[str, float]] = {}
    for (label, concept), value in _COMPLETIONS.items():
        by_disease.setdefault(label, {})[concept] = value

    for label, additions in by_disease.items():
        entry = kb.entries.get(label)
        if entry is None:
            raise KeyError(f"_COMPLETIONS names {label!r}, not in the KB")
        features = dict(entry.features)
        for concept, value in additions.items():
            if concept in features:
                raise KeyError(
                    f"_COMPLETIONS sets {label}/{concept}, which the entry "
                    "already characterises. Delete the completion rather than "
                    "letting it shadow the existing value."
                )
            features[concept] = value
        kb.entries[label] = dataclasses.replace(entry, features=features)
    kb._marginals.clear()


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

    for label, (value, low, high) in _PRIORS.items():
        entry = kb.entries.get(label)
        if entry is None:
            raise KeyError(
                f"_PRIORS names {label!r}, which is not in the knowledge base."
            )
        kb.entries[label] = dataclasses.replace(
            entry,
            prevalence=value,
            prior_source=LikelihoodSource(
                provenance=Provenance.MEASURED,
                citation=_PRIOR_CITATION,
                low=low,
                high=high,
                note=_PRIOR_NOTE,
            ),
        )

    kb._marginals.clear()


# What a disease that does not list a finding is taken to say about it, when
# ``unlisted_as_atypical`` is on. Not a new invented constant: it is the point
# estimate the project's own pre-committed narrative rubric already assigns to
# "not typical", which is exactly the claim being made -- a reference text that
# enumerates a disease's features and omits this one has said, weakly, that it
# is not a feature of that disease. Taking the number from the rubric rather
# than choosing one keeps the assumption inspectable and stops it drifting to
# whatever value happens to score best.
UNLISTED_IS_ATYPICAL = NARRATIVE_RUBRIC["not typical"][0]


def build_knowledge_base(
    correlated: bool = False,
    unlisted_as_atypical: bool = False,
    complete_grid: bool = False,
) -> InMemoryKnowledgeBase:
    """Return the synthetic KB. See the module docstring: numbers are invented.

    ``correlated=True`` additionally supplies the dependence structure above.
    Off by default so that every measurement taken before it existed still
    describes the default knowledge base.

    ``unlisted_as_atypical=True`` switches the backoff for uncharacterised
    cells from the KB-wide marginal to the rubric's "not typical" value. See
    ``InMemoryKnowledgeBase.backoff`` for the argument and what it costs; it
    is off by default because it changes every uncharacterised cell at once
    and that is a decision to take on measurement, not by default.

    ``complete_grid=True`` writes the 77 chosen values in ``_COMPLETIONS``
    into the entries, so no cell falls back to anything. Off by default for
    a reason recorded rather than assumed: it is 77 invented numbers from one
    non-clinician, it takes the sourced fraction from 27% to 17%, and it
    revises four findings this project had already measured. See DESIGN.md.
    """
    register_costs(COSTS)
    kb = InMemoryKnowledgeBase(
        unlisted_likelihood=UNLISTED_IS_ATYPICAL if unlisted_as_atypical else None
    )

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
                # Placeholder; sourced in _FROM_LITERATURE below, and the
                # sourced figure is far higher than this file would have
                # guessed. See that entry.
                "lab:raised_troponin": 0.32,
                # Invented, and stated as a claim rather than left to the
                # marginal: COPD is an airway disease, and ischaemic-pattern
                # ST change is not part of how an exacerbation presents.
                # Left uncharacterised it inherited the KB marginal of 0.71,
                # because the only two entries describing this finding are
                # both cardiac -- so the model held that ST changes were
                # likelier than not in a COPD exacerbation. Any stated value
                # beats that, and a low one is the honest reading.
                "exam:ecg_st_changes": 0.08,
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
                # Both invented, both stated for the same reason as the COPD
                # pair above: uncharacterised, these inherited marginals of
                # 0.45 and 0.71, so a raised troponin and ischaemic ECG
                # changes barely argued against asthma at all -- which is how
                # a confirmed NSTEMI came to rank behind it.
                #
                # Set lower than the COPD figures deliberately. The sourced
                # COPD number (0.32) reflects a population with epicardial
                # disease, right-heart strain and decades of smoking; an
                # asthma exacerbation population is younger and largely
                # without those, so demand ischaemia is rarer. That ordering
                # is the defensible part; the exact values are not, and both
                # are tagged invented.
                "lab:raised_troponin": 0.08,
                "exam:ecg_st_changes": 0.05,
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
    _seed_pioped_cells(kb)
    if complete_grid:
        _complete_uncharacterised(kb)
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


def build_hard_cases() -> list[Case]:
    """Five diagnostic traps, kept separate from the ten development fixtures.

    The ten cases in ``build_cases`` are saturated. The shipped configuration
    ranks all ten correctly and the held-out split scores 7 of 7, so the set
    can no longer separate a good change from a bad one -- and the abstention
    gate, whose entire purpose is declining the cases the model gets wrong,
    measures +0.0% accuracy gained on it, because there is nothing left to
    decline. A test set the system aces is not evidence that the system is
    finished; it is evidence that the test set is spent.

    These five are an instrument to replace it, and they are **not** added to
    ``build_cases``. Every measurement in DESIGN.md is stated against those
    ten, and silently changing the denominator would invalidate the record
    this project keeps deliberately.

    **How they were chosen, because that is the part that could be abused.**
    Each is a named diagnostic trap in this exact differential, written from
    the clinical picture before the model was run on any of them, and none was
    kept or discarded on the basis of whether the model got it right. The
    obvious way to manufacture a discriminating test set is to generate many
    cases, keep the failures, and report a set the system fails; that would
    measure nothing but a willingness to search. This is the reverse: the
    traps were fixed first and the score is whatever it turns out to be.

    - **fx-h01, cardiac asthma.** Left ventricular failure produces bronchial
      oedema and wheeze, and an ex-smoker who is wheezing reads as COPD. The
      knowledge base states wheeze at 0.91 for COPD, so this aims at a value
      the model holds strongly.
    - **fx-h02, myopericarditis.** Pleuritic pain with a raised troponin and
      ST changes is the acute-coronary signature; the friction rub is the only
      thing separating them. Also probes the red-flag rule, which should
      resist committing while acute coronary syndrome is still live.
    - **fx-h03, silent ischaemia.** An elderly diabetic with breathlessness
      and no chest pain at all. Removes the finding the model's ACS row leans
      on, and exercises the presentation-triggered troponin and ECG.
    - **fx-h04, pneumonia complicating COPD.** Both diagnoses are true and the
      label is the acute actionable one. Every COPD feature is present, so
      consolidation is the only discriminator -- which puts the weight on the
      cell measured from the AECOPD imaging review at 0.18.
    - **fx-h05, asthma without a smoking history.** SCOPE.md records that at
      the default threshold asthma exacerbation has *no supporting edges at
      all*: every finding it carries is claimed at least as hard by COPD.
      This tests the model's structurally weakest row directly.
    """
    return [
        Case(
            case_id="fx-h01",
            presenting_complaint=(
                "woke at 3am fighting for breath, wheezing, has to sit upright"
            ),
            diagnosis="acute_pulmonary_oedema",
            initial_findings=("dyspnoea_at_rest", "wheeze_subjective", "orthopnoea"),
            features={
                "dyspnoea_at_rest": True,
                "orthopnoea": True,
                "wheeze_subjective": True,
                "exam:raised_jvp": True,
                "lab:raised_bnp": True,
                "imaging:cxr_pulmonary_oedema": True,
                "leg_swelling": True,
                "exam:crackles": True,
                "smoking_history": True,
                "sudden_onset": True,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "fever": False,
                "productive_cough": False,
                "imaging:cxr_consolidation": False,
                "lab:raised_wcc": False,
                "pleuritic_pain": False,
                "exertional_chest_pain": False,
                "lab:raised_troponin": False,
                "exam:ecg_st_changes": False,
                "exam:reduced_breath_sounds": False,
                "calf_tenderness": False,
                "recent_immobility": False,
                "lab:raised_d_dimer": False,
                "imaging:ctpa_filling_defect": False,
                "palpitations": False,
                "exam:friction_rub": False,
            },
        ),
        Case(
            case_id="fx-h02",
            presenting_complaint=(
                "sharp chest pain after a week of flu, worse flat, eased sitting forward"
            ),
            diagnosis="pericarditis",
            initial_findings=("pleuritic_pain", "fever"),
            features={
                "pleuritic_pain": True,
                "fever": True,
                "exam:friction_rub": True,
                "lab:raised_troponin": True,
                "exam:ecg_st_changes": True,
                "sudden_onset": True,
                "exam:tachycardia": True,
                "lab:raised_wcc": True,
                "dyspnoea_at_rest": False,
                "exertional_chest_pain": False,
                "productive_cough": False,
                "exam:crackles": False,
                "imaging:cxr_consolidation": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "recent_immobility": False,
                "lab:raised_d_dimer": False,
                "imaging:ctpa_filling_defect": False,
                "lab:raised_bnp": False,
                "imaging:cxr_pulmonary_oedema": False,
                "smoking_history": False,
                "exam:hypoxia": False,
                "wheeze_subjective": False,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
                "palpitations": False,
            },
        ),
        Case(
            case_id="fx-h03",
            presenting_complaint=(
                "77-year-old diabetic, sudden breathlessness this morning, no chest pain"
            ),
            diagnosis="acute_coronary_syndrome",
            initial_findings=("dyspnoea_at_rest", "sudden_onset"),
            features={
                "dyspnoea_at_rest": True,
                "sudden_onset": True,
                "lab:raised_troponin": True,
                "exam:ecg_st_changes": True,
                "exam:tachycardia": True,
                "smoking_history": True,
                "exertional_chest_pain": False,
                "pleuritic_pain": False,
                "fever": False,
                "productive_cough": False,
                "exam:crackles": False,
                "imaging:cxr_consolidation": False,
                "imaging:cxr_pulmonary_oedema": False,
                "lab:raised_bnp": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "recent_immobility": False,
                "lab:raised_d_dimer": False,
                "imaging:ctpa_filling_defect": False,
                "lab:raised_wcc": False,
                "wheeze_subjective": False,
                "exam:hypoxia": False,
                "palpitations": False,
                "exam:friction_rub": False,
                "exam:raised_jvp": False,
                "exam:reduced_breath_sounds": False,
            },
        ),
        Case(
            case_id="fx-h04",
            presenting_complaint=(
                "lifelong smoker, four days of fever, green sputum and worsening wheeze"
            ),
            diagnosis="community_acquired_pneumonia",
            initial_findings=("fever", "productive_cough", "wheeze_subjective"),
            features={
                "fever": True,
                "productive_cough": True,
                "imaging:cxr_consolidation": True,
                "lab:raised_wcc": True,
                "exam:crackles": True,
                "smoking_history": True,
                "wheeze_subjective": True,
                "exam:reduced_breath_sounds": True,
                "dyspnoea_at_rest": True,
                "exam:hypoxia": True,
                "pleuritic_pain": True,
                "exam:tachycardia": True,
                "sudden_onset": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "calf_tenderness": False,
                "recent_immobility": False,
                "lab:raised_d_dimer": False,
                "imaging:ctpa_filling_defect": False,
                "lab:raised_bnp": False,
                "imaging:cxr_pulmonary_oedema": False,
                "lab:raised_troponin": False,
                "exam:ecg_st_changes": False,
                "exertional_chest_pain": False,
                "palpitations": False,
                "exam:friction_rub": False,
                "exam:raised_jvp": False,
            },
        ),
        Case(
            case_id="fx-h05",
            presenting_complaint=(
                "24-year-old, never smoked, sudden wheeze and breathlessness this evening"
            ),
            diagnosis="asthma_exacerbation",
            initial_findings=("wheeze_subjective", "dyspnoea_at_rest"),
            features={
                "wheeze_subjective": True,
                "dyspnoea_at_rest": True,
                "sudden_onset": True,
                "exam:reduced_breath_sounds": True,
                "exam:tachycardia": True,
                "exam:hypoxia": True,
                "smoking_history": False,
                "fever": False,
                "productive_cough": False,
                "exam:crackles": False,
                "imaging:cxr_consolidation": False,
                "lab:raised_wcc": False,
                "orthopnoea": False,
                "leg_swelling": False,
                "lab:raised_bnp": False,
                "imaging:cxr_pulmonary_oedema": False,
                "exam:raised_jvp": False,
                "lab:raised_troponin": False,
                "exam:ecg_st_changes": False,
                "pleuritic_pain": False,
                "exertional_chest_pain": False,
                "calf_tenderness": False,
                "recent_immobility": False,
                "lab:raised_d_dimer": False,
                "imaging:ctpa_filling_defect": False,
                "palpitations": False,
                "exam:friction_rub": False,
            },
        ),
    ]


__all__ = ["COSTS", "build_cases", "build_hard_cases", "build_knowledge_base"]
