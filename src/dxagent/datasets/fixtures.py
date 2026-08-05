"""Synthetic development fixtures: a small dyspnoea/chest-pain knowledge base.

!! NOT CLINICALLY VALIDATED !!

Every number below is invented by the author of this module for the purpose of
exercising the code path. The likelihoods are plausible-looking, not sourced;
the prevalences are not from any population; the feature sets are incomplete in
ways a physician would notice immediately. Nothing here should be read as
clinical information, and no result computed from it says anything about real
diagnostic performance.

The fixtures exist so that the loop, the gate, the calibrator, and the metrics
can be developed and tested before MIMIC credentialing and the UMLS licence
clear. They are the first thing to delete once real data is available, and the
KB numbers should be replaced with sourced prevalences and likelihood ratios
under clinician review rather than incrementally patched.

The eight conditions cover a differential that is genuinely hard for the right
reason: several of them share their presenting features and separate only on
specific tests, which is what makes the information-gain selector and the
abstention gate do observable work.
"""

from __future__ import annotations

from ..environment import Case
from ..knowledge import Citation, DiseaseEntry, InMemoryKnowledgeBase, register_costs

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
            label="heart_failure_exacerbation",
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
            diagnosis="heart_failure_exacerbation",
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
            diagnosis="heart_failure_exacerbation",
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
            # pain at all. The history alone points at heart failure or COPD.
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
