"""Textbook-level claims about the knowledge base, written so a clinician
can accept or reject each in one sentence.

These lived in scripts/plausibility_check.py, where the audit that checks
them still runs. They moved here because the loop now reads them too: the
claims say which findings *define* each disease, and a commit can use that to
disclose which rivals' defining findings it never examined
(``guidelines.unexamined_signatures``). One list, two readers.
"""

from __future__ import annotations

# Ordering claims: which disease must lead a column, and over which rivals
# ---------------------------------------------------------------------------
# The CLAIMS above test direction against a marginal, and they only cover the
# findings SCOPE.md happened to name. Nothing checked the *ordering within a
# column* until five errors were found by doing it by hand:
#
#     P(raised BNP | PE)            0.20, below every rival that mattered
#     P(recent immobility | PE)     0.61, a simulator value 2.4x the truth
#     P(raised troponin | PE)       0.25, below the sourced 0.32 for COPD
#     P(defect | PE)                0.95, overstating a scan that misses 1 in 6
#     P(exertional pain | oedema)   0.77, above ACS, from a mis-mapped question
#
# None was caught by the test suite, because tests check outputs and these are
# inputs. Each was obvious once the column was sorted and read against what the
# diseases do, so that reading is encoded here.
#
# Each entry says: for this finding, this disease must score strictly above
# each of these rivals, for this reason. The reasons are deliberately
# textbook-level, so a clinician can agree or disagree with each in one
# sentence -- which is the closest this project gets to the review it cannot
# have. They are claims about *ordering* rather than magnitude, because
# ordering is what a non-clinician can assert honestly.
ORDERING_CLAIMS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("lab:raised_troponin", "acute_coronary_syndrome",
     ("community_acquired_pneumonia", "copd_exacerbation", "asthma_exacerbation",
      "panic_attack", "acute_pulmonary_oedema", "pulmonary_embolism"),
     "myocardial injury is what defines the diagnosis"),
    ("lab:raised_troponin", "pulmonary_embolism",
     ("community_acquired_pneumonia", "asthma_exacerbation", "panic_attack"),
     "right ventricular strain releases troponin; a pneumonia does not"),
    ("imaging:ctpa_filling_defect", "pulmonary_embolism",
     ("community_acquired_pneumonia", "acute_coronary_syndrome",
      "acute_pulmonary_oedema", "panic_attack"),
     "the finding names the diagnosis"),
    ("imaging:cxr_consolidation", "community_acquired_pneumonia",
     ("pulmonary_embolism", "acute_coronary_syndrome", "acute_pulmonary_oedema",
      "copd_exacerbation", "asthma_exacerbation", "pericarditis", "panic_attack"),
     "consolidation is the radiological definition of pneumonia"),
    ("imaging:cxr_pulmonary_oedema", "acute_pulmonary_oedema",
     ("community_acquired_pneumonia", "pulmonary_embolism", "copd_exacerbation",
      "asthma_exacerbation"),
     "the finding names the diagnosis"),
    ("lab:raised_bnp", "acute_pulmonary_oedema",
     ("community_acquired_pneumonia", "pulmonary_embolism",
      "acute_coronary_syndrome", "copd_exacerbation", "asthma_exacerbation"),
     "ventricular stretch is the mechanism that releases it"),
    ("lab:raised_d_dimer", "pulmonary_embolism",
     ("community_acquired_pneumonia", "acute_coronary_syndrome",
      "acute_pulmonary_oedema", "copd_exacerbation", "pericarditis",
      "panic_attack"),
     "the one thrombotic diagnosis in the differential"),
    ("fever", "community_acquired_pneumonia",
     ("acute_coronary_syndrome", "acute_pulmonary_oedema", "panic_attack",
      "asthma_exacerbation"),
     "an infection, against causes that are not infective"),
    ("orthopnoea", "acute_pulmonary_oedema",
     ("community_acquired_pneumonia", "pulmonary_embolism", "copd_exacerbation",
      "asthma_exacerbation"),
     "pulmonary congestion is what worsens lying flat"),
    ("leg_swelling", "acute_pulmonary_oedema", ("community_acquired_pneumonia",),
     "fluid overload"),
    ("smoking_history", "copd_exacerbation", ("asthma_exacerbation",),
     "COPD is overwhelmingly smoking-caused; asthma is not"),
    ("sudden_onset", "pulmonary_embolism",
     ("community_acquired_pneumonia", "copd_exacerbation"),
     "abrupt vascular occlusion, against days of infection or decline"),
    ("sudden_onset", "panic_attack",
     ("community_acquired_pneumonia", "copd_exacerbation"),
     "abrupt onset is part of the diagnostic criterion"),
    ("exam:crackles", "community_acquired_pneumonia",
     ("asthma_exacerbation", "panic_attack"),
     "alveolar filling, against bronchospasm and against nothing"),
    ("exam:crackles", "acute_pulmonary_oedema",
     ("asthma_exacerbation", "panic_attack"),
     "alveolar fluid, against bronchospasm and against nothing"),
    ("exam:ecg_st_changes", "acute_coronary_syndrome",
     ("copd_exacerbation", "asthma_exacerbation"),
     "ischaemia, against airway disease"),
    ("exertional_chest_pain", "acute_coronary_syndrome",
     ("acute_pulmonary_oedema", "pericarditis", "panic_attack"),
     "exertional ischaemic pain is the cardinal symptom of the diagnosis"),
    ("pleuritic_pain", "pericarditis", ("acute_coronary_syndrome",),
     "pleuritic pain, against ischaemic pain"),
    ("lab:raised_wcc", "community_acquired_pneumonia",
     ("panic_attack", "acute_pulmonary_oedema", "asthma_exacerbation"),
     "bacterial infection raises the white count"),
    # This one is stated in the form believed to be true, and currently
    # fails against two of the three rivals. Stating only the half that
    # passes would have hidden the defect inside the check meant to find it.
    ("exam:hypoxia", "acute_pulmonary_oedema",
     ("panic_attack", "copd_exacerbation", "pulmonary_embolism"),
     "alveolar flooding impairing gas exchange defines the diagnosis"),
    # The three pericarditis-defining findings. All three were inert until
    # the rivals were given cells -- a single-describer concept backs every
    # other disease off to its own value -- so these claims exist to keep
    # the rival cells below the leader, whatever anyone later sets them to.
    ("exam:friction_rub", "pericarditis",
     ("community_acquired_pneumonia", "pulmonary_embolism",
      "acute_coronary_syndrome", "acute_pulmonary_oedema", "copd_exacerbation",
      "asthma_exacerbation", "panic_attack"),
     "the rub is the pericardium itself, inflamed"),
    ("imaging:pericardial_effusion", "pericarditis",
     ("community_acquired_pneumonia", "pulmonary_embolism",
      "acute_coronary_syndrome", "acute_pulmonary_oedema", "copd_exacerbation",
      "asthma_exacerbation", "panic_attack"),
     "one of the four diagnostic criteria for the disease"),
    ("exam:ecg_pr_depression", "pericarditis",
     ("community_acquired_pneumonia", "pulmonary_embolism",
      "acute_coronary_syndrome", "acute_pulmonary_oedema", "copd_exacerbation",
      "asthma_exacerbation", "panic_attack"),
     "PR depression is the ECG sign that separates pericarditis from infarction"),
)


__all__ = ["ORDERING_CLAIMS"]
