"""Mapping between the fixture vocabulary and DDXPlus evidence codes.

Filling this in lets `scripts/map_concepts.py` read P(finding | disease) off
1.3M patients instead of leaving it invented. One afternoon of work replaces
most of the knowledge base's numbers with measured ones.

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

# Fixture concept -> DDXPlus evidence code.
#
# For a binary evidence, the code alone: "fever": "E_91"
# For one value of a categorical, the full form: "pleuritic_pain": "E_54_@_V_12"
#
# Run `python scripts/map_concepts.py --search cough` to find candidates, and
# read the question text before committing to one.
CONCEPT_MAP: dict[str, str] = {
    # --- history -------------------------------------------------------
    # "fever": "",
    # "productive_cough": "",
    # "pleuritic_pain": "",
    # "dyspnoea_at_rest": "",
    # "sudden_onset": "",
    # "orthopnoea": "",
    # "leg_swelling": "",
    # "calf_tenderness": "",
    # "smoking_history": "",
    # "exertional_chest_pain": "",
    # "palpitations": "",
    # "wheeze_subjective": "",
    # "recent_immobility": "",
    # --- examination ---------------------------------------------------
    # "exam:crackles": "",
    # "exam:tachycardia": "",
    # "exam:hypoxia": "",
    # "exam:raised_jvp": "",
    # "exam:reduced_breath_sounds": "",
    # "exam:friction_rub": "",
    # "exam:ecg_st_changes": "",
    # --- laboratory ----------------------------------------------------
    # "lab:raised_wcc": "",
    # "lab:raised_d_dimer": "",
    # "lab:raised_troponin": "",
    # "lab:raised_bnp": "",
    # --- imaging -------------------------------------------------------
    # "imaging:cxr_consolidation": "",
    # "imaging:cxr_pulmonary_oedema": "",
    # "imaging:ctpa_filling_defect": "",
}

# Concepts DDXPlus is not expected to cover, recorded so that an empty mapping
# is distinguishable from an unexamined one. DDXPlus is built from a
# telemedicine questionnaire: it asks what a patient can report, so laboratory
# and imaging results are largely outside it. Those keep their invented values
# and must be sourced from the literature instead -- which is also where the
# numbers deciding fx-010 sit, so this is not a small remainder.
EXPECTED_ABSENT: frozenset[str] = frozenset()


__all__ = ["CONCEPT_MAP", "DISEASE_MAP", "EXPECTED_ABSENT"]
