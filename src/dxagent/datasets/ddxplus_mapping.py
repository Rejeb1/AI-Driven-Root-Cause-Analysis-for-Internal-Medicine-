"""Mapping between the fixture vocabulary and DDXPlus evidence codes.

Filling this in lets `scripts/map_concepts.py` read P(finding | disease) off
1.3M patients instead of leaving it invented.

How far it gets, measured rather than hoped: 25 of 135 likelihoods, 19%. Not
"most of the knowledge base", which is what this file claimed before the
mapping was done and the counting was run. Three things cap it -- DDXPlus holds
no examination signs, laboratory results or imaging; a finding its rule base
omits is generated for nobody, so the zero is the simulator's and not a
frequency; and a pairing the fixture entry does not already carry cannot be
sourced without extending the model rather than grounding it.

A mapping is a claim
--------------------
Writing ``"fever": "E_91"`` asserts that the fixture's ``fever`` and the
question "Do you have a fever?" mean the same thing. Get that wrong and the
result is a *confidently sourced wrong number*, which is worse than the
invented one it replaced -- an invented number is labelled invented and nobody
trusts it, while a mis-mapped one arrives with 1.3M patients behind it.

So this file is deliberately not auto-populated. Keyword matching gets about
eight of twenty-seven and several of those are wrong: it pairs
``pleuritic_pain`` with "How fast did the pain appear?" and
``dyspnoea_at_rest`` with "Do you have chest pain even at rest?". Both look
plausible in a diff and neither is right. Use ``--search`` to read the actual
question text before writing an entry.

Leave a concept out rather than guessing. Unmapped concepts keep their invented
value and are reported as such, which is the honest outcome.

The circularity, which has to be stated wherever these numbers are used
----------------------------------------------------------------------
Likelihoods measured from DDXPlus cannot then be validated on DDXPlus. Doing
so measures whether the data agrees with itself. Use them for the fixture
knowledge base, and when reporting DDXPlus results say which knowledge base
produced them.
"""

from __future__ import annotations

# Fixture disease label -> the DDXPlus pathologies it corresponds to.
# Verified against the release by scripts/ddxplus_verify.py, which resolves all
# eight. Acute coronary syndrome spans two DDXPlus pathologies and they are
# pooled, since the fixture treats it as one condition.
DISEASE_MAP: dict[str, tuple[str, ...]] = {
    "pulmonary_embolism": ("pulmonary_embolism",),
    "community_acquired_pneumonia": ("pneumonia",),
    "acute_coronary_syndrome": ("possible_nstemi_/_stemi", "unstable_angina"),
    "acute_pulmonary_oedema": ("acute_pulmonary_edema",),
    "copd_exacerbation": ("acute_copd_exacerbation_/_infection",),
    "asthma_exacerbation": ("bronchospasm_/_acute_asthma_exacerbation",),
    "pericarditis": ("pericarditis",),
    "panic_attack": ("panic_attack",),
}

# Fixture concept -> DDXPlus evidence code, or several codes counted as a
# disjunction (the patient has the finding if any of them is recorded).
#
#   binary evidence          "fever": "E_91"
#   one categorical value    "x": "E_54_@_V_12"
#   several values           "leg_swelling": ("E_152_@_V_119", "E_152_@_V_120", ...)
#
# The disjunction is not a convenience. DDXPlus records swelling by anatomical
# site across sixteen leg-ish values, so mapping `leg_swelling` to any single
# one of them would count a fraction of the patients who have it and report the
# result as measured.
#
# Confidence is noted per entry. Anything marked uncertain is a judgement about
# whether two phrasings mean the same clinical thing, and should be checked
# before the numbers derived from it are quoted.
CONCEPT_MAP: dict[str, str | tuple[str, ...]] = {
    # --- confident: the two phrasings ask the same question -------------
    "fever": "E_91",                     # Do you have a fever?
    "productive_cough": "E_77",          # cough producing coloured/abundant sputum
    "pleuritic_pain": "E_220",           # pain increased when breathing in deeply
    "orthopnoea": "E_217",               # worse lying down, better sitting up
    "smoking_history": "E_79",           # Do you smoke cigarettes?
    "exertional_chest_pain": "E_218",    # worse on exertion, relieved by rest
    "palpitations": "E_155",             # heart racing / irregular / pounding
    "wheeze_subjective": "E_214",        # wheezing sound when you exhale
    "recent_immobility": "E_110",        # immobile 3+ days in the last 4 weeks

    # leg swelling is recorded by site, so it is the union of the leg-ish ones
    "leg_swelling": (
        "E_152_@_V_119", "E_152_@_V_120",   # calf R / L
        "E_152_@_V_34", "E_152_@_V_35",     # ankle R / L
        "E_152_@_V_23", "E_152_@_V_24",     # posterior ankle R / L
        "E_152_@_V_51", "E_152_@_V_52",     # thigh R / L
        "E_152_@_V_92", "E_152_@_V_93",     # knee R / L
        "E_152_@_V_172", "E_152_@_V_173",   # tibia R / L
        "E_152_@_V_72", "E_152_@_V_73",     # dorsum of foot R / L
        "E_152_@_V_43", "E_152_@_V_44",     # lateral foot R / L
    ),

    # --- uncertain: check before quoting anything derived from these ----
    # E_66 is "shortness of breath in a significant way", which is dyspnoea but
    # not specifically *at rest*. E_64 ("out of breath with minimal effort") is
    # nearer to exertional. Neither is exact; E_66 is the closer of the two.
    "dyspnoea_at_rest": "E_66",
}

# Concepts DDXPlus is not expected to cover, recorded so that an empty mapping
# is distinguishable from an unexamined one. DDXPlus is built from a
# telemedicine questionnaire: it asks what a patient can report, so laboratory
# and imaging results are largely outside it. Those keep their invented values
# and must be sourced from the literature instead -- which is also where the
# numbers deciding fx-010 sit, so this is not a small remainder.
EXPECTED_ABSENT: frozenset[str] = frozenset({
    # DDXPlus is a telemedicine questionnaire: it records what a patient can
    # report. Examination signs, laboratory results and imaging are outside it
    # entirely, and no amount of searching will find them.
    "exam:crackles",
    "exam:tachycardia",
    "exam:hypoxia",
    "exam:raised_jvp",
    "exam:reduced_breath_sounds",
    "exam:friction_rub",
    "exam:ecg_st_changes",
    "lab:raised_wcc",
    "lab:raised_d_dimer",
    "lab:raised_troponin",
    "lab:raised_bnp",
    "imaging:cxr_consolidation",
    "imaging:cxr_pulmonary_oedema",
    "imaging:ctpa_filling_defect",
    # Recorded as E_59, "how fast did the pain appear", on a 0-10 scale. Which
    # point on that scale counts as "sudden" is a clinical threshold, not a
    # lookup, so it is left out rather than guessed.
    "sudden_onset",
    # E_152 records calf *swelling*, not tenderness. Nothing in the release
    # asks about tenderness on palpation, which is an examination sign.
    "calf_tenderness",
})


__all__ = ["CONCEPT_MAP", "DISEASE_MAP", "EXPECTED_ABSENT"]
