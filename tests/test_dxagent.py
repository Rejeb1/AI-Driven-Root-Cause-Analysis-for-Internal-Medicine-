"""Tests.

Emphasis is on the properties that would be dangerous to get wrong silently:
that probabilities stay normalised, that the gate refuses for the reason it
claims, that the loop always terminates, and that the metrics do not flatter the
system. Correctness of the diagnoses themselves is not testable against
synthetic fixtures and is not attempted here.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from dxagent import (
    AbstentionGate,
    Action,
    ActionKind,
    Case,
    CaseOracle,
    DiagnosticAgent,
    Differential,
    Finding,
    InformationGainSelector,
    LoopLimits,
    NoisyOracle,
    Polarity,
    ConformalPredictor,
    ScriptedLLM,
    TemperatureScaler,
    Verdict,
)
from dxagent.baselines import RetrievalOnlyBaseline, SinglePassBaseline
from dxagent.belief import BayesianProposer, ConsensusProposer, LLMProposer
from dxagent.datasets import build_cases, build_knowledge_base
from dxagent.evaluation import evaluate, run_agent, split_cases
from dxagent.evaluation.metrics import (
    abstention_metrics,
    calibration_metrics,
    differential_metrics,
    ranking_metrics,
    risk_coverage_curve,
    selective_metrics,
)
from dxagent.schemas import CaseOutcome, CaseState, Citation, Hypothesis


@pytest.fixture
def kb():
    return build_knowledge_base()


def grounded(kb, scores: dict[str, float]) -> Differential:
    """Build a differential carrying real KB citations.

    Needed because the gate checks grounding before confidence, so an
    uncited differential escalates for that reason and never reaches the
    threshold logic under test.
    """
    return Differential.from_scores(
        scores, support={label: kb.citations_for(label) for label in scores}
    )


@pytest.fixture
def cases():
    return build_cases()


# --------------------------------------------------------------------------
# schemas


def test_differential_rejects_unnormalised():
    with pytest.raises(ValueError):
        Differential((Hypothesis("a", 0.4), Hypothesis("b", 0.4)))


def test_differential_rejects_empty():
    with pytest.raises(ValueError):
        Differential(())


def test_from_scores_normalises_and_sorts():
    d = Differential.from_scores({"a": 2.0, "b": 6.0, "c": 2.0})
    assert d.top.label == "b"
    assert math.isclose(sum(h.probability for h in d.hypotheses), 1.0)
    assert math.isclose(d.top.probability, 0.6)


def test_from_scores_degenerate_falls_back_to_uniform():
    """All-zero scores must not crash; the gate should see the low confidence."""
    d = Differential.from_scores({"a": 0.0, "b": 0.0})
    assert math.isclose(d.top.probability, 0.5)


def test_margin_and_entropy():
    d = Differential.from_scores({"a": 0.5, "b": 0.3, "c": 0.2})
    assert math.isclose(d.margin, 0.2)
    assert d.entropy > 0
    peaked = Differential.from_scores({"a": 0.999, "b": 0.001})
    assert peaked.entropy < d.entropy


def test_absent_findings_are_distinct_from_unknown(kb):
    """A negative result must inform the posterior; 'not asked' must not."""
    proposer = BayesianProposer(kb)
    absent = proposer.propose([Finding("fever", Polarity.ABSENT)])
    unknown = proposer.propose([Finding("fever", Polarity.UNKNOWN)])
    baseline = proposer.propose([])
    assert absent.probability_of("community_acquired_pneumonia") < baseline.probability_of(
        "community_acquired_pneumonia"
    )
    assert math.isclose(
        unknown.probability_of("community_acquired_pneumonia"),
        baseline.probability_of("community_acquired_pneumonia"),
    )


# --------------------------------------------------------------------------
# knowledge base


def test_sparse_feature_table_does_not_win_by_default(kb):
    """Regression: uncharacterised features must back off to the marginal.

    Returning 1.0 for an uncharacterised feature made sparsely-described
    diseases artificially easy to fit, which produced a wrong top-1 on the
    fixture set.
    """
    finding = Finding("smoking_history", Polarity.PRESENT)
    # CAP does not characterise smoking history; COPD does, strongly.
    cap = kb.likelihood("community_acquired_pneumonia", finding)
    copd = kb.likelihood("copd_exacerbation", finding)
    assert cap < 1.0, "uncharacterised feature must not be a free pass"
    assert copd > cap


def test_likelihood_is_bounded(kb):
    for label in (e.label for e in kb.diseases()):
        for polarity in (Polarity.PRESENT, Polarity.ABSENT):
            value = kb.likelihood(label, Finding("fever", polarity))
            assert 0.0 < value < 1.0


def test_background_marginal_in_range(kb):
    assert 0.0 <= kb.background("fever") <= 1.0
    assert kb.background("nonexistent_concept") == 0.5


# --------------------------------------------------------------------------
# action selection


def test_expected_gain_is_non_negative(kb):
    selector = InformationGainSelector(kb)
    d = BayesianProposer(kb).propose([Finding("fever")])
    for concept in ("exam:crackles", "lab:raised_d_dimer", "orthopnoea"):
        assert selector.expected_gain(d, concept) >= 0.0


def test_selector_does_not_repeat_questions(kb, cases):
    """Regression: asking the same thing twice wastes the turn budget."""
    agent = DiagnosticAgent(kb=kb)
    outcome = agent.run(cases[0])
    targets = [s.action.target for s in outcome.steps]
    assert len(targets) == len(set(targets))


def test_expensive_low_value_test_loses_to_cheap_high_value(kb):
    """Regression: bare gain/cost made the cheapest action always win."""
    selector = InformationGainSelector(kb)
    cheap_useful = Action(ActionKind.EXAMINE, "exam:crackles", cost=1.0,
                          expected_information_gain=0.5)
    dear_useless = Action(ActionKind.IMAGING, "imaging:ctpa_filling_defect",
                          cost=20.0, expected_information_gain=0.05)
    cheap_useless = Action(ActionKind.ASK, "fever", cost=0.2,
                           expected_information_gain=0.005)
    labels = ("community_acquired_pneumonia",)
    assert selector.value(cheap_useful, labels) > selector.value(dear_useless, labels)
    assert selector.value(cheap_useful, labels) > selector.value(cheap_useless, labels)


def test_ruleout_fires_for_live_red_flag(kb):
    """A live PE must trigger an exclusion test even at modest probability."""
    selector = InformationGainSelector(kb)
    from dxagent.schemas import CaseState

    state = CaseState(case_id="t", presenting_complaint="dyspnoea")
    d = Differential.from_scores(
        {
            "community_acquired_pneumonia": 0.70,
            "pulmonary_embolism": 0.20,
            "panic_attack": 0.10,
        }
    )
    action = selector.ruleout_action(state, d)
    assert action is not None
    assert "rule out pulmonary_embolism" in action.rationale


def test_ruleout_silent_when_no_red_flag_is_live(kb):
    from dxagent.schemas import CaseState

    selector = InformationGainSelector(kb)
    state = CaseState(case_id="t", presenting_complaint="cough")
    d = Differential.from_scores(
        {"community_acquired_pneumonia": 0.98, "panic_attack": 0.02}
    )
    assert selector.ruleout_action(state, d) is None


# --------------------------------------------------------------------------
# gate


def test_gate_blocks_low_confidence(kb):
    gate = AbstentionGate(kb, min_confidence=0.8, min_margin=0.0)
    d = grounded(
        kb, {"community_acquired_pneumonia": 0.6, "copd_exacerbation": 0.4}
    )
    decision = gate.evaluate(d)
    assert decision.should_escalate
    assert "below threshold" in decision.reason


def test_gate_blocks_contested_posterior(kb):
    """Peaked but contested: 46 vs 44 should not be reported as an answer."""
    gate = AbstentionGate(kb, min_confidence=0.4, min_margin=0.15)
    d = grounded(
        kb, {"community_acquired_pneumonia": 0.46, "copd_exacerbation": 0.44,
         "panic_attack": 0.10}
    )
    decision = gate.evaluate(d)
    assert decision.should_escalate
    assert "contested" in decision.reason


def test_gate_blocks_ungrounded_top_hypothesis(kb):
    gate = AbstentionGate(kb, min_confidence=0.1, min_margin=0.0)
    d = Differential((Hypothesis("mystery_illness", 1.0, support=()),))
    decision = gate.evaluate(d)
    assert decision.should_escalate
    assert "citation" in decision.reason


def test_gate_blocks_competing_red_flag(kb):
    gate = AbstentionGate(kb, min_confidence=0.5, min_margin=0.0,
                          red_flag_tolerance=0.10)
    d = grounded(
        kb, {"community_acquired_pneumonia": 0.75, "pulmonary_embolism": 0.25}
    )
    decision = gate.evaluate(d)
    assert decision.should_escalate
    assert "time-critical" in decision.reason


def test_gate_permits_committing_to_the_red_flag_itself(kb):
    """Regression: the gate once refused to report a PE it had correctly found."""
    gate = AbstentionGate(kb, min_confidence=0.5, min_margin=0.1)
    d = grounded(
        kb, {"pulmonary_embolism": 0.90, "community_acquired_pneumonia": 0.10}
    )
    decision = gate.evaluate(d)
    assert decision.should_commit, decision.reason


def test_gate_blocks_on_proposer_disagreement(kb):
    gate = AbstentionGate(kb, min_confidence=0.5, min_margin=0.0,
                          max_disagreement=0.3)
    d = grounded(
        kb, {"community_acquired_pneumonia": 0.9, "copd_exacerbation": 0.1}
    )
    assert gate.evaluate(d, disagreement=0.8).should_escalate
    assert gate.evaluate(d, disagreement=0.1).should_commit


# --------------------------------------------------------------------------
# calibration


def test_evidence_fit_is_high_for_a_textbook_presentation(kb, cases):
    """A case the KB describes well should be well explained by it."""
    proposer = BayesianProposer(kb)
    case = cases[0]
    findings = [
        Finding(concept=c, polarity=Polarity.PRESENT if present else Polarity.ABSENT)
        for c, present in case.features.items()
    ]
    proposer.propose(findings)
    assert proposer.last_evidence_fit > 0.0


def test_evidence_fit_falls_when_no_diagnosis_explains_the_findings(kb):
    """A combination no single disease accounts for scores below the marginal.

    Built from the most characteristic finding of four *different* diseases.
    Each is individually typical, so this is not a test of implausible
    findings; it is a test of an implausible combination, which is what an
    out-of-coverage presentation looks like.
    """
    proposer = BayesianProposer(kb)
    entries = kb.diseases()

    coherent = [
        Finding(concept=c, polarity=Polarity.PRESENT)
        for c, p in entries[0].features.items()
        if p >= 0.7
    ]
    incoherent = [
        Finding(
            concept=max(entry.features.items(), key=lambda kv: kv[1])[0],
            polarity=Polarity.PRESENT,
        )
        for entry in entries[:4]
    ]

    proposer.propose(coherent)
    fit_coherent = proposer.last_evidence_fit
    proposer.propose(incoherent)
    fit_incoherent = proposer.last_evidence_fit

    assert fit_incoherent < 0.0 < fit_coherent


def _superseded_test_evidence_fit_does_not_separate_masquerade_errors(kb, cases):
    """Kept unrun as a record of what was true before sourcing.

    The cases the loop got wrong were ones where the *wrong* diagnosis
    explained the evidence well, not ones where nothing did -- so evidence
    fit sat inside the range of the cases it got right, and no threshold on
    it separated them. Sourcing changed that; see
    ``test_evidence_fit_now_separates_the_remaining_masquerade_failure``.
    """
    agent = DiagnosticAgent(kb=kb)
    proposer = BayesianProposer(kb)
    correct, wrong = [], []

    for case in cases:
        outcome = agent.run(case)
        findings = case.initial() + [f for s in outcome.steps for f in s.findings]
        proposer.propose(findings)
        fit = proposer.last_evidence_fit
        hit = outcome.differential.top.label == case.diagnosis
        (correct if hit else wrong).append(fit)

    assert wrong, "fixture set no longer contains a failing case"
    assert min(correct) < max(wrong)


def test_evidence_fit_does_not_separate_the_masquerade_failure():
    """Three reversals, and then a test bed that can actually hold the claim.

    The original claim was that evidence fit cannot tell the loop's failures
    from its successes, because a masquerade failure is one where the *wrong*
    diagnosis explains the evidence well. Nine sourced likelihoods appeared to
    repair that; five more from the same source undid the repair; the Wang
    inversions repaired it again. Three reversals on the same ten cases.

    Every one of those measurements was taken on a deliberately weakened arm
    carrying a single failure, and a separation demonstrated over one failure
    is a property of that failure rather than of evidence fit. The previous
    version of this test said so in its own docstring and then went on
    asserting the separation anyway, which is why it kept flipping.

    It is now measured where a real failure exists under the *shipped*
    configuration: fx-h04 in the hard case set, a pneumonia in a COPD patient
    committed wrongly at 70%. And the result is stronger than the original
    claim rather than merely consistent with it. The failure does not just
    score inside the range of the successes; it scores **above all of them**
    -- 0.28 against a best correct case of 0.25. The wrong diagnosis explains
    this evidence better than the right diagnosis explains any of the others,
    which is exactly what a masquerade is and exactly why no threshold on
    evidence fit can be used as a safety check.
    """
    kb = build_knowledge_base(correlated=True)
    agent = DiagnosticAgent(kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb))
    proposer = BayesianProposer(kb)

    from dxagent.datasets.fixtures import build_hard_cases

    correct, wrong = [], []
    for case in build_hard_cases():
        outcome = agent.run(case)
        findings = case.initial() + [f for s in outcome.steps for f in s.findings]
        proposer.propose(findings)
        fit = proposer.last_evidence_fit
        hit = outcome.differential.top.label == case.diagnosis
        (correct if hit else wrong).append(fit)

    assert wrong, "hard case set no longer contains a failing case"
    # Not merely unseparated: the masquerade is the best-explained case in
    # the set. A threshold placed anywhere would reject successes first.
    assert max(wrong) > max(correct), (
        f"masquerade fit {max(wrong):.4f} no longer tops correct {max(correct):.4f}"
    )


def test_gate_escalates_unexplained_evidence_however_confident(kb):
    """The whole point: peaked posterior, still escalated."""
    gate = AbstentionGate(kb, min_evidence_fit=0.0)
    labels = [e.label for e in kb.diseases()]
    confident = grounded(kb, {labels[0]: 0.97, labels[1]: 0.03})

    assert gate.evaluate(confident, evidence_fit=1.5).should_commit
    unexplained = gate.evaluate(confident, evidence_fit=-0.4)
    assert unexplained.should_escalate
    assert "not explained" in unexplained.reason


def test_evidence_fit_check_is_inert_without_a_reported_fit(kb):
    """Callers that do not supply a fit keep the previous behaviour."""
    gate = AbstentionGate(kb)
    labels = [e.label for e in kb.diseases()]
    assert gate.evaluate(grounded(kb, {labels[0]: 0.97, labels[1]: 0.03})).should_commit


def test_evidence_fit_is_infinite_before_any_evidence(kb):
    """No findings means nothing unexplained, not everything unexplained."""
    proposer = BayesianProposer(kb)
    proposer.propose([])
    assert proposer.last_evidence_fit == float("inf")

    unknown_only = [Finding(concept="fever", polarity=Polarity.UNKNOWN)]
    proposer.propose(unknown_only)
    assert proposer.last_evidence_fit == float("inf")


def test_hypotheses_carry_both_supporting_and_contradicting_citations(kb, cases):
    """The project's headline claim: every hypothesis tied to evidence both ways.

    ``against`` existed on the schema from the start but nothing ever populated
    it, so every hypothesis reported an empty case against itself -- which
    reads as "nothing argues against this" rather than "this was never
    assessed".
    """
    case = cases[0]
    findings = [
        Finding(concept=c, polarity=Polarity.PRESENT if present else Polarity.ABSENT)
        for c, present in case.features.items()
    ]
    differential = BayesianProposer(kb).propose(findings, case.presenting_complaint)

    assert any(h.against for h in differential.hypotheses)
    assert all(h.support for h in differential.hypotheses)

    cited = [c for h in differential.hypotheses for c in h.support + h.against]
    assert all(c.source_id and c.locator for c in cited)


def test_an_expected_finding_that_is_absent_counts_against(kb):
    """Contradiction is not only about unexpected findings being present.

    A finding a disease strongly predicts, observed absent, argues against it.
    Scoring only positive findings drops that whole class of reasoning.
    """
    entry = max(
        kb.diseases(), key=lambda e: max(e.features.values(), default=0.0)
    )
    concept, probability = max(entry.features.items(), key=lambda kv: kv[1])
    assert probability >= 0.7, "fixture KB lacks a strongly-predicted feature"

    present = Finding(concept=concept, polarity=Polarity.PRESENT)
    absent = Finding(concept=concept, polarity=Polarity.ABSENT)

    supporting, _ = kb.evidence_split(entry.label, [present])
    _, contradicting = kb.evidence_split(entry.label, [absent])

    assert supporting, "a strongly-predicted finding present should support"
    assert contradicting, "the same finding absent should count against"


def test_near_neutral_findings_are_cited_on_neither_side(kb):
    """Findings that barely move the ratio should be omitted, not padded in."""
    entry = kb.diseases()[0]
    concept = next(iter(entry.features))
    finding = Finding(concept=concept, polarity=Polarity.PRESENT)

    # A threshold beyond any ratio in the fixture KB admits nothing.
    supporting, against = kb.evidence_split(entry.label, [finding], min_ratio=1e6)
    assert not supporting and not against


def test_grounding_survives_a_hypothesis_with_no_supporting_findings(kb, cases):
    """A hypothesis nothing supports must still be grounded, or the gate
    escalates for missing provenance rather than for weak evidence."""
    case = cases[0]
    findings = [
        Finding(concept=c, polarity=Polarity.PRESENT if present else Polarity.ABSENT)
        for c, present in case.features.items()
    ]
    differential = BayesianProposer(kb).propose(findings, case.presenting_complaint)
    weakest = differential.hypotheses[-1]
    assert weakest.is_grounded


def test_temperature_softens_and_sharpens():
    d = Differential.from_scores({"a": 0.9, "b": 0.1})
    assert TemperatureScaler(3.0).apply(d).top.probability < 0.9
    assert TemperatureScaler(0.5).apply(d).top.probability > 0.9
    assert math.isclose(TemperatureScaler(1.0).apply(d).top.probability, 0.9)


def test_temperature_preserves_normalisation():
    d = Differential.from_scores({"a": 0.5, "b": 0.3, "c": 0.2})
    for t in (0.5, 1.0, 2.0, 4.0):
        scaled = TemperatureScaler(t).apply(d)
        assert math.isclose(sum(h.probability for h in scaled.hypotheses), 1.0)


def test_temperature_refuses_tiny_fit_sets():
    """Regression: fitting on 2 cases produced T=0.5 and worsened calibration."""
    scaler = TemperatureScaler(min_fit_samples=30)
    samples = [(Differential.from_scores({"a": 0.9, "b": 0.1}), "b")] * 3
    scaler.fit(samples)
    assert scaler.temperature == 1.0
    assert not scaler.fitted


def test_temperature_fit_softens_overconfident_samples():
    """Given enough consistently-wrong confident predictions, T must rise."""
    scaler = TemperatureScaler(min_fit_samples=5)
    samples = [(Differential.from_scores({"a": 0.95, "b": 0.05}), "b")] * 40
    scaler.fit(samples)
    assert scaler.temperature > 1.0
    assert scaler.fitted


# --------------------------------------------------------------------------
# loop


def test_loop_terminates_on_every_fixture(kb, cases):
    agent = DiagnosticAgent(kb=kb, limits=LoopLimits(max_turns=8, max_cost=30.0))
    for case in cases:
        outcome = agent.run(case)
        assert len(outcome.steps) <= 8
        assert outcome.budget_spent <= 30.0
        assert outcome.verdict in (Verdict.COMMITTED, Verdict.ESCALATED)


def test_loop_terminates_with_zero_turn_budget(kb, cases):
    agent = DiagnosticAgent(kb=kb, limits=LoopLimits(max_turns=0))
    outcome = agent.run(cases[0])
    assert outcome.steps == ()


def test_loop_terminates_with_zero_cost_budget(kb, cases):
    """With no budget and an unmet threshold, escalate rather than spin."""
    agent = DiagnosticAgent(
        kb=kb,
        gate=AbstentionGate(kb, min_confidence=0.99),
        limits=LoopLimits(max_cost=0.0),
    )
    outcome = agent.run(cases[0])
    assert outcome.abstained
    assert outcome.budget_spent == 0.0


def test_escalation_packet_is_actionable(kb, cases):
    """An abstention that carries no evidence transfers no work off the clinician."""
    agent = DiagnosticAgent(
        kb=kb, gate=AbstentionGate(kb, min_confidence=0.99, min_margin=0.98)
    )
    outcome = agent.run(cases[0])
    assert outcome.abstained
    packet = outcome.escalation
    assert packet.reason
    assert packet.differential.hypotheses
    assert packet.turns_used == len(outcome.steps)


def test_due_diligence_prevents_turn_zero_commit(kb, cases):
    """Regression: the loop committed on volunteered symptoms alone.

    Also regression on the flag's semantics: 0.0 must *disable* the rule, not
    make every action count as outstanding and drive the loop to its limit.
    """
    # require_decisive_tests is switched off in both arms so this isolates the
    # gain flag. Left on, it keeps the loop running in the eager arm too and
    # the comparison stops measuring the thing it names.
    #
    # require_workup is switched off for the same reason, which took a
    # reversal to notice. ``still_outstanding`` keeps the loop alive while
    # ``mandated is not None``, and the floor that clears the mandate only
    # fires when the best action's gain falls *below* the threshold -- so at
    # 0.0 the floor never fires, the mandate never clears, and the eager arm
    # runs longer than the careful one. That is the workup's behaviour, not
    # the flag's, and with the sourced likelihoods it was enough to invert
    # the comparison.
    limits = dict(require_decisive_tests=False, require_workup=False)
    eager = DiagnosticAgent(
        kb=kb,
        gate=AbstentionGate(kb, min_confidence=0.3, min_margin=0.0),
        limits=LoopLimits(due_diligence_gain=0.0, **limits),
    )
    careful = DiagnosticAgent(
        kb=kb,
        gate=AbstentionGate(kb, min_confidence=0.3, min_margin=0.0),
        limits=LoopLimits(due_diligence_gain=0.10, **limits),
    )
    case = cases[0]
    assert len(careful.run(case).steps) > len(eager.run(case).steps)


def test_flip_action_is_silent_once_nothing_can_change_the_answer(kb, cases):
    """A settled differential must not keep ordering tests forever."""
    selector = InformationGainSelector(kb)
    agent = DiagnosticAgent(kb=kb)
    case = cases[1]  # the decisive imaging case
    outcome = agent.run(case)

    state = CaseState(case_id=case.case_id, presenting_complaint=case.presenting_complaint)
    state.asked.update(s.action.target for s in outcome.steps)
    state.asked.update(case.initial_findings)

    assert selector.flip_action(state, outcome.differential) is None


def test_flip_action_ignores_outcomes_it_believes_will_not_happen(kb, cases):
    """The probability floor is what stops it chasing impossible reversals."""
    selector = InformationGainSelector(kb, min_flip_outcome=0.0)
    strict = InformationGainSelector(kb, min_flip_outcome=0.95)
    proposer = BayesianProposer(kb)
    case = cases[0]
    differential = proposer.propose(case.initial(), case.presenting_complaint)
    state = CaseState(case_id=case.case_id, presenting_complaint=case.presenting_complaint)
    state.asked.update(case.initial_findings)

    permissive_hit = selector.flip_action(state, differential)
    strict_hit = strict.flip_action(state, differential)

    assert permissive_hit is not None
    # Demanding a near-certain flipping outcome must not find more than
    # accepting any outcome does.
    assert strict_hit is None or strict_hit.target == permissive_hit.target


def _superseded_test_decisive_test_signal_flags_the_masquerade_failures(kb, cases):
    """Kept unrun as a record of what was true before sourcing.

    Under these deliberately weakened limits, every case the loop got wrong
    used to have an unasked test that would change the answer. Sourcing
    added a second, differently-shaped failure that breaks this; see
    ``test_decisive_test_signal_no_longer_flags_every_masquerade_failure``.
    """
    plain = DiagnosticAgent(
        kb=kb,
        limits=LoopLimits(require_decisive_tests=False, require_workup=False),
    )
    selector = InformationGainSelector(kb)

    flagged_when_wrong = 0
    wrong = 0
    for case in cases:
        outcome = plain.run(case)
        state = CaseState(
            case_id=case.case_id, presenting_complaint=case.presenting_complaint
        )
        state.asked.update(case.initial_findings)
        state.asked.update(s.action.target for s in outcome.steps)
        has_flip = selector.flip_action(state, outcome.differential) is not None
        if outcome.differential.top.label != case.diagnosis:
            wrong += 1
            flagged_when_wrong += has_flip

    assert wrong, "fixture set no longer contains a failing case"
    assert flagged_when_wrong == wrong


def test_the_weakened_loop_still_has_exactly_one_masquerade_failure(kb, cases):
    """Where the remaining failures live, once every safety rule is removed.

    This arm switches off the workup floor, the decisive-test rule and
    correlation weighting -- everything that exists to catch a masquerade --
    and one case fails: fx-009, a pulmonary embolism read as pneumonia. **The
    shipped configuration gets all ten**, which is the number that describes
    the system; this one describes what the safety machinery is carrying.

    The count here has moved every time the knowledge base has: two failures
    on invented numbers, none after the Merck likelihoods, one once the
    priors were sourced, two again after PIOPED II, and one now that five of
    acute pulmonary oedema's likelihoods are recovered from the Wang review
    instead of invented. fx-001, the pneumonia read as COPD, went with them:
    it had been losing partly because pulmonary oedema claimed crackles at an
    invented 0.75 against a measured 0.60.

    That last move is worth stating plainly, because it made a number go the
    wrong way for the right reason. fx-009 used to pass on this arm partly
    because its rivals were penalised for *absent* venous-thromboembolism
    findings at invented rates of 0.61 and 0.40 -- a pneumonia patient was
    treated as unlikely to be free of calf tenderness. PIOPED II measured
    those rates at 0.19 and 0.23 in patients investigated for embolism who
    did not have one, so the penalty was largely an artefact and it is now
    gone. The case did not get harder; the model stopped being wrong in a
    direction that happened to help.
    """
    plain = DiagnosticAgent(
        kb=kb,
        limits=LoopLimits(require_decisive_tests=False, require_workup=False),
    )
    selector = InformationGainSelector(kb)

    wrong = [
        case.case_id
        for case in cases
        if plain.run(case).differential.top.label != case.diagnosis
    ]
    # Sourcing brought these back on this deliberately weakened arm -- no
    # workup floor, no decisive-test rule, no correlation weighting. The
    # shipped configuration gets all ten; see the docstring.
    assert wrong == ["fx-009"], f"unexpected failures: {wrong}"

    # Correlation weighting alone -- still no workup floor, still no
    # decisive-test rule -- carries both of them. That is the measurement
    # behind calling it the load-bearing piece of this arm.
    aware = DiagnosticAgent(
        kb=build_knowledge_base(correlated=True),
        limits=LoopLimits(require_decisive_tests=False, require_workup=False),
    )
    assert not [
        case.case_id
        for case in cases
        if aware.run(case).differential.top.label != case.diagnosis
    ]


def test_correlation_pays_and_decisive_tests_still_do_not(kb, cases):
    """Which mechanism helps, and the reversal that produced this answer.

    The history matters more than the assertion. On the invented knowledge base
    the loop scored 8/10 and neither mechanism helped. Sourcing 17 likelihoods
    from DDXPlus lifted it to 9/10 and appeared to make the decisive-test rule
    pay, reaching 10/10 alongside the correlation structure. Replacing four of
    those with frequencies from a real cohort reversed it: correlation alone
    reaches 10/10 and adding decisive tests drops back to 9/10 at half again
    the cost.

    The apparent value of the decisive-test rule was an artefact of the
    simulator. DDXPlus records pleuritic pain in 71% of its pulmonary embolism
    patients; Miniati's 360 real ones show 33%. Conclusions about mechanism
    drawn on simulator-derived numbers did not survive contact with measured
    ones, which is the caution to carry into any result sourced this way.
    """
    from dxagent.datasets.fixtures import build_knowledge_base as build

    def run(correlated: bool, decisive: bool):
        agent = DiagnosticAgent(
            kb=build(correlated=correlated),
            limits=LoopLimits(require_decisive_tests=decisive, require_workup=False),
        )
        outcomes = [agent.run(c) for c in cases]
        correct = sum(
            o.differential.top.label == c.diagnosis
            for o, c in zip(outcomes, cases)
        )
        return correct, sum(o.budget_spent for o in outcomes) / len(outcomes)

    plain, plain_cost = run(False, False)
    correlated, correlated_cost = run(True, False)
    both, both_cost = run(True, True)

    # Fourth reversal on this measurement. Correlation helped on invented
    # numbers, stopped helping when Merck sourcing removed the extremity it
    # was compensating for, and helps again now that the priors are sourced:
    # pulmonary embolism turns out to be genuinely rare in this presentation
    # (StatPearls puts it under 5% of dyspnoea, 2% of chest pain), so its
    # prior fell threefold and the diagnosis needs the correlated reading of
    # its four absent findings to survive at all.
    #
    # Four measurements, four answers, none of them a mistake -- each was
    # correct about a different knowledge base. The durable lesson is that a
    # mechanism's value is a property of the numbers underneath it, so
    # "does correlation weighting help" has no answer independent of them.
    assert correlated == len(cases), "correlation now gets every fixture case"
    assert plain < correlated, "and the plain loop gives one back"
    assert both <= correlated, "decisive tests still do not improve on the best arm"


def _superseded_test_decisive_tests_do_not_repair(kb, cases):
    """Kept unrun as a record of what was true before sourcing.

    Acting on the signal cost more and bought no accuracy: the tests it orders
    are the ones the current posterior rates as relevant, and the posterior is
    what is wrong.
    """
    def run(require: bool):
        agent = DiagnosticAgent(
            kb=kb,
            limits=LoopLimits(
                require_decisive_tests=require, require_workup=False
            ),
        )
        outcomes = [agent.run(c) for c in cases]
        correct = sum(
            o.differential.top.label == c.diagnosis
            for o, c in zip(outcomes, cases)
        )
        return correct, sum(o.budget_spent for o in outcomes) / len(outcomes)

    off_correct, off_cost = run(False)
    on_correct, on_cost = run(True)

    assert on_correct <= off_correct
    assert on_cost > off_cost


def test_baselines_are_evaluable_by_the_same_harness(kb, cases):
    """A baseline scored by a different code path is not a comparison."""
    truths = {c.case_id: c.diagnosis for c in cases}
    for baseline in (SinglePassBaseline(kb), RetrievalOnlyBaseline(kb)):
        outcomes = run_agent(baseline, cases)
        assert len(outcomes) == len(cases)
        metrics = ranking_metrics(outcomes, truths)
        assert 0.0 <= metrics.top1 <= 1.0
        # Neither gathers evidence, and neither abstains -- that is the
        # property under test, not an omission.
        assert all(o.steps == () for o in outcomes)
        assert all(not o.abstained for o in outcomes)


def test_single_pass_baseline_sees_negative_findings_too(kb, cases):
    """Handed only positives, the baseline would face an easier task than the
    agent, which can ask about anything and get an answer either way."""
    case = cases[0]
    assert any(present is False for present in case.features.values())
    outcome = SinglePassBaseline(kb).run(case)
    assert outcome.differential.hypotheses


def test_retrieval_baseline_does_not_reward_verbose_kb_entries(kb):
    """Normalisation guards a coverage artefact, not a retrieval result."""
    from dxagent.knowledge import DiseaseEntry, InMemoryKnowledgeBase

    lopsided = InMemoryKnowledgeBase()
    lopsided.add(
        DiseaseEntry(
            label="sparse_but_matching",
            prevalence=0.5,
            features={"a": 0.9},
            citations=(Citation("T", "sparse"),),
        )
    )
    lopsided.add(
        DiseaseEntry(
            label="verbose_but_not_matching",
            prevalence=0.5,
            features={"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9},
            citations=(Citation("T", "verbose"),),
        )
    )
    case = Case(
        case_id="c",
        presenting_complaint="a",
        diagnosis="sparse_but_matching",
        features={"a": True},
    )
    outcome = RetrievalOnlyBaseline(lopsided).run(case)
    assert outcome.differential.top.label == "sparse_but_matching"


def test_conformal_coverage_guarantee_holds_on_held_out_cases(kb, cases):
    """The whole point of conformal prediction is that this is checkable."""
    import random

    proposer = BayesianProposer(kb)
    rng = random.Random(0)
    samples = []
    for _ in range(400):
        case = rng.choice(cases)
        observed = [
            Finding(
                concept=c,
                polarity=Polarity.PRESENT if present else Polarity.ABSENT,
            )
            for c, present in case.features.items()
            if rng.random() < 0.4
        ]
        samples.append((proposer.propose(observed), case.diagnosis))

    calibration, held_out = samples[:200], samples[200:]
    for alpha in (0.1, 0.2):
        predictor = ConformalPredictor(alpha=alpha).fit(calibration)
        assert predictor.fitted
        coverage, mean_size = predictor.coverage(held_out)
        # Finite-sample slack: the guarantee is 1-alpha in expectation, so a
        # single split can land slightly under. Anything far below means the
        # quantile is wrong, not that the sample was unlucky.
        assert coverage >= (1 - alpha) - 0.1
        assert 1 <= mean_size <= len(kb.diseases())


def test_conformal_refuses_to_certify_from_too_few_samples(kb, cases):
    proposer = BayesianProposer(kb)
    samples = [(proposer.propose(c.initial()), c.diagnosis) for c in cases[:5]]
    predictor = ConformalPredictor().fit(samples)
    assert not predictor.fitted
    assert predictor.fit_n == 5


def test_conformal_set_is_never_empty(kb, cases):
    """An empty prediction set is not a usable answer for a clinician."""
    proposer = BayesianProposer(kb)
    differential = proposer.propose(cases[0].initial())
    predictor = ConformalPredictor()
    predictor.fitted = True
    predictor.threshold = 1.5  # unreachable: no probability can clear it
    assert predictor.predict_set(differential) == (differential.top.label,)


def test_gate_ignores_conformal_until_it_is_fitted(kb):
    """An unfitted predictor must leave prior gate behaviour untouched."""
    gate = AbstentionGate(kb)
    labels = [e.label for e in kb.diseases()]
    assert not gate.conformal.fitted
    assert gate.evaluate(grounded(kb, {labels[0]: 0.97, labels[1]: 0.03})).should_commit


def test_gate_escalates_when_the_conformal_set_stays_wide(kb):
    gate = AbstentionGate(kb, max_conformal_set=1)
    # Red flags are checked before the conformal set, and correctly so -- an
    # unexcluded time-critical diagnosis is a more specific reason to hand over
    # than a wide set. Use benign labels so this isolates the conformal rule.
    benign = [e.label for e in kb.diseases() if not e.red_flag]
    assert len(benign) >= 2, "fixture KB needs two non-red-flag entries"
    gate.conformal.fitted = True
    gate.conformal.threshold = 0.01  # admits almost everything
    decision = gate.evaluate(grounded(kb, {benign[0]: 0.6, benign[1]: 0.4}))
    assert decision.should_escalate
    assert "coverage" in decision.reason


def test_rule_kinds_are_distinguished(kb):
    """A severity score must not be usable as diagnostic support."""
    from dxagent.guidelines import CURB65, PERC, WELLS_PE, RuleKind

    assert PERC.kind is RuleKind.RULE_OUT
    assert WELLS_PE.kind is RuleKind.RISK_STRATIFY
    assert CURB65.kind is RuleKind.SEVERITY
    assert "category error" in CURB65.note


def test_unmappable_criteria_are_reported_not_dropped():
    """A rule scored on part of itself is a different, weaker rule."""
    from dxagent.guidelines import PERC, vocabulary_gaps

    assert PERC.unmappable, "PERC has criteria the fixture vocabulary lacks"
    assert 0.0 < PERC.coverage < 1.0
    gaps = vocabulary_gaps()
    assert any("Haemoptysis" in c for c in gaps[PERC.name])
    assert PERC.interpret([]).endswith("not expressible in the current vocabulary")


def test_unasked_criteria_do_not_count_as_negative():
    """The way rule-out rules are most often misused in practice."""
    from dxagent.guidelines import PERC

    nothing_asked = PERC.score([])
    assert nothing_asked == 0.0
    # ...but the rule must report itself as incomplete rather than negative.
    assert PERC.unobserved([]) == PERC.mappable
    assert "unanswered" in PERC.interpret([])

    answered_negative = [
        Finding(concept=c.concept, polarity=Polarity.ABSENT) for c in PERC.mappable
    ]
    assert PERC.unobserved(answered_negative) == ()


def test_mandatory_workup_triggers_on_presentation_not_posterior(kb):
    """The property that makes this work where four other mechanisms failed."""
    from dxagent.guidelines import PE_WORKUP

    presenting = [Finding("pleuritic_pain", Polarity.PRESENT)]
    assert PE_WORKUP.is_triggered(presenting)
    assert "lab:raised_d_dimer" in PE_WORKUP.outstanding(presenting)

    # Nothing about the differential is consulted: the same findings trigger it
    # whether or not the model rates PE as plausible.
    answered = presenting + [Finding("lab:raised_d_dimer", Polarity.ABSENT)]
    assert PE_WORKUP.outstanding(answered) == ()

    assert not PE_WORKUP.is_triggered([Finding("fever", Polarity.PRESENT)])


def _superseded_test_workup_orders_the_test_the_posterior_would_not(kb, cases):
    """Kept unrun. True while pulmonary embolism was buried on fx-009.

    PE sat at 0.5%, so no posterior-driven rule requested a D-dimer and only
    the guideline rule did -- the argument for having a rule that never asks
    what the model believes. Sourcing raised PE's fever, cough and crackles
    likelihoods, the diagnosis is no longer buried, and the selector now asks
    for the D-dimer on its own. The rule has not become pointless; it has
    stopped being the only thing that would order the test on this case.
    """
    case = {c.case_id: c for c in cases}["fx-009"]
    without = DiagnosticAgent(kb=kb, limits=LoopLimits(require_workup=False))
    asked_without = {s.action.target for s in without.run(case).steps}
    assert "lab:raised_d_dimer" not in asked_without


def test_workup_guarantees_the_test_the_posterior_now_also_wants(kb, cases):
    """The rule's guarantee, stated without leaning on the old failure.

    A presentation-triggered workup earns its place by guaranteeing the item
    is gathered whatever the model believes. Demonstrating that by finding a
    case where the model would otherwise skip the test was always the weaker
    argument, because it depends on the model being wrong in a particular way
    -- and on fx-009 it no longer is.
    """
    case = {c.case_id: c for c in cases}["fx-009"]
    with_workup = DiagnosticAgent(kb=kb, limits=LoopLimits(require_workup=True))
    asked = {s.action.target for s in with_workup.run(case).steps}
    assert "lab:raised_d_dimer" in asked


def _superseded_test_confirmatory_evidence_cannot_rescue_a_buried_diagnosis(kb):
    """Kept unrun. This was the project's headline negative result.

    Two findings typical of pneumonia and atypical of PE were enough, under
    the independence assumption, that a *positive confirmatory test* for PE
    still left it far from the top. The conclusion drawn was that no
    test-selection policy could fix it and the remaining work was the
    likelihood model.

    That conclusion was right, and sourcing the likelihood model repaired it.
    See ``test_sourcing_repaired_the_buried_diagnosis``.
    """
    proposer = BayesianProposer(kb)
    confirmatory = [
        Finding("fever", Polarity.PRESENT),
        Finding("productive_cough", Polarity.PRESENT),
        Finding("imaging:ctpa_filling_defect", Polarity.PRESENT),
    ]
    after = proposer.propose(confirmatory)
    assert after.probability_of("pulmonary_embolism") < 0.5
    assert after.top.label != "pulmonary_embolism"


def test_sourcing_repaired_the_buried_diagnosis(kb):
    """The headline negative result, reversed by five sentences of textbook.

    The superseded version above recorded that a positive CTPA could not
    rescue pulmonary embolism from two pneumonia-typical findings, and
    concluded the fault lay in the likelihood values rather than in evidence
    gathering. Three of those values then acquired citations:

        P(fever            | PE)  0.15 invented -> 0.30  "fever can occur"
        P(productive_cough | PE)  0.10 invented -> 0.20  "less common symptoms
                                                          include cough"
        P(exam:crackles    | PE)  0.15 invented -> 0.20  "less commonly ...
                                                          crackles or wheezing"

    The invented numbers said pulmonary embolism essentially never presents
    with fever or a cough. The textbook says both occur, uncommonly. Under a
    product of independent likelihoods that difference compounds, and the
    diagnosis is no longer buried deep enough that confirmatory imaging
    cannot retrieve it.

    Worth being precise about what this does and does not show. It is not
    evidence that the model is now correct -- 102 of 135 likelihoods remain
    invented, and the same compounding will be distorting them too. It is
    evidence that the diagnosis of *why* the old result happened was right,
    and that the cost of an invented number is not spread evenly: three of
    them, on one disease, were holding the whole failure in place.
    """
    from dxagent.datasets.fixtures import build_knowledge_base as build

    confirmatory = [
        Finding("fever", Polarity.PRESENT),
        Finding("productive_cough", Polarity.PRESENT),
        Finding("imaging:ctpa_filling_defect", Polarity.PRESENT),
    ]
    assert kb.get("pulmonary_embolism").features["imaging:ctpa_filling_defect"] >= 0.9

    # Sourcing the priors moved this again. Pulmonary embolism turns out to be
    # rare among these presentations -- StatPearls puts it under 5% of dyspnoea
    # and at 2% of chest pain -- so its prior fell from an invented 0.10 to a
    # derived 0.034, and the diagnosis now starts three times further back.
    # The likelihood repair alone no longer carries it: on the plain knowledge
    # base pneumonia wins again.
    plain = BayesianProposer(build(correlated=False)).propose(confirmatory)
    assert plain.top.label == "community_acquired_pneumonia"

    # With correlation weighting it still is retrieved, which is the point.
    # Four correlated negatives about PE are not four independent penalties,
    # and once they stop being counted as such a positive CTPA is enough even
    # against the lower prior.
    aware = BayesianProposer(build(correlated=True)).propose(confirmatory)
    assert aware.top.label == "pulmonary_embolism"
    assert aware.probability_of("pulmonary_embolism") < 0.5


def test_state_records_realised_information_gain(kb, cases):
    agent = DiagnosticAgent(kb=kb)
    outcome = agent.run(cases[1])
    for step in outcome.steps:
        assert step.entropy_after >= 0.0


def test_unknown_findings_from_oracle_when_feature_absent_from_case(kb):
    case = Case(
        case_id="sparse",
        presenting_complaint="cough",
        diagnosis="community_acquired_pneumonia",
        features={"fever": True},
        initial_findings=("fever",),
    )
    oracle = CaseOracle(case)
    findings = oracle.respond(Action(ActionKind.LAB, "lab:raised_bnp"))
    assert findings[0].polarity is Polarity.UNKNOWN


def _gold_case(differential, diagnosis="pulmonary_embolism"):
    return Case(
        case_id="c",
        presenting_complaint="dyspnoea",
        diagnosis=diagnosis,
        features={},
        differential=differential,
    )


def test_ambiguity_tracks_gold_confidence_in_the_true_diagnosis():
    """The abstention proxy has to order cases the way a clinician would."""
    certain = _gold_case((("pulmonary_embolism", 0.95), ("pneumonia", 0.05)))
    tied = _gold_case((("pulmonary_embolism", 0.5), ("pneumonia", 0.5)))
    doubtful = _gold_case((("pneumonia", 0.75), ("pulmonary_embolism", 0.25)))

    assert certain.ambiguity == pytest.approx(0.05)
    assert tied.ambiguity == pytest.approx(0.5)
    assert doubtful.ambiguity == pytest.approx(0.75)
    assert certain.ambiguity < tied.ambiguity < doubtful.ambiguity

    # A source with no gold differential must not read as maximally uncertain.
    assert _gold_case(()).ambiguity == 0.0


def test_ambiguity_separates_cases_that_entropy_cannot():
    """Why the proxy is gold confidence and not gold entropy.

    Both cases below spread mass over five causes, so their entropy is close.
    They are not equally hard: one puts most of the mass on the truth, the
    other buries it. On DDXPlus, where nearly every differential is long,
    entropy saturates and stops separating cases at all.
    """
    easy = _gold_case(
        (("pulmonary_embolism", 0.8), ("a", 0.05), ("b", 0.05), ("c", 0.05), ("d", 0.05))
    )
    hard = _gold_case(
        (("a", 0.8), ("b", 0.05), ("c", 0.05), ("d", 0.05), ("pulmonary_embolism", 0.05))
    )

    assert easy.differential_entropy == pytest.approx(hard.differential_entropy)
    assert easy.ambiguity == pytest.approx(0.2)
    assert hard.ambiguity == pytest.approx(0.95)


def test_differential_entropy_is_comparable_across_lengths():
    """Kept as a secondary signal, so its normalisation still has to hold."""
    two_way_tie = _gold_case((("a", 0.5), ("b", 0.5)))
    five_way_tie = _gold_case(tuple((chr(97 + i), 0.2) for i in range(5)))
    broad_but_peaked = _gold_case(
        (("a", 0.8), ("b", 0.05), ("c", 0.05), ("d", 0.05), ("e", 0.05))
    )

    assert two_way_tie.differential_entropy == pytest.approx(1.0)
    assert five_way_tie.differential_entropy == pytest.approx(1.0)
    assert broad_but_peaked.differential_entropy < two_way_tie.differential_entropy


def test_ddx_recall_rewards_covering_the_gold_differential(kb):
    labels = [e.label for e in kb.diseases()][:3]
    outcome = CaseOutcome(
        case_id="c1",
        verdict=Verdict.COMMITTED,
        differential=Differential.from_scores(
            {labels[0]: 0.6, labels[1]: 0.3, labels[2]: 0.1}
        ),
        confidence=0.6,
        escalation=None,
        steps=(),
        budget_spent=0.0,
    )

    full = differential_metrics(
        [outcome], {"c1": ((labels[0], 0.7), (labels[1], 0.3))}
    )
    assert full.recall_at_gold_length == pytest.approx(1.0)

    missed = differential_metrics(
        [outcome], {"c1": ((labels[0], 0.7), ("something_absent", 0.3))}
    )
    assert missed.recall_at_gold_length == pytest.approx(0.5)

    # Cases without a gold differential are skipped, not scored zero.
    assert differential_metrics([outcome], {}).n == 0


def test_abstention_proxy_scores_deferral_on_ambiguous_cases(kb):
    label = kb.diseases()[0].label

    def outcome(case_id, verdict):
        return CaseOutcome(
            case_id=case_id,
            verdict=verdict,
            differential=Differential.from_scores({label: 1.0}),
            confidence=0.9,
            escalation=None,
            steps=(),
            budget_spent=0.0,
        )

    outcomes = [
        outcome("ambiguous_deferred", Verdict.ESCALATED),
        outcome("ambiguous_answered", Verdict.COMMITTED),
        outcome("clear_deferred", Verdict.ESCALATED),
        outcome("clear_answered", Verdict.COMMITTED),
    ]
    ambiguity = {
        "ambiguous_deferred": 0.9,
        "ambiguous_answered": 0.9,
        "clear_deferred": 0.1,
        "clear_answered": 0.1,
    }

    metrics = abstention_metrics(outcomes, ambiguity, threshold=0.5)
    assert metrics.n_ambiguous == 2
    assert metrics.abstention_rate == pytest.approx(0.5)
    assert metrics.proxy_precision == pytest.approx(0.5)  # 1 of 2 deferrals
    assert metrics.proxy_recall == pytest.approx(0.5)  # caught 1 of 2 ambiguous


def test_closed_world_vocabulary_turns_silence_into_a_negative():
    """A sparse record that declares its vocabulary answers ABSENT, not UNKNOWN.

    DDXPlus stores positive evidences only. Read open-world, every question
    about a symptom the patient does not have returns UNKNOWN -- likelihood
    1.0, no information -- so the agent gathers nothing and times out.
    """
    case = Case(
        case_id="closed",
        presenting_complaint="cough",
        diagnosis="community_acquired_pneumonia",
        features={"fever": True},
        initial_findings=("fever",),
        vocabulary=frozenset({"fever", "pleuritic_pain"}),
    )
    oracle = CaseOracle(case)

    declared = oracle.respond(Action(ActionKind.ASK, "pleuritic_pain"))
    assert declared[0].polarity is Polarity.ABSENT

    # Outside the declared vocabulary the record says nothing, so the honest
    # answer is still UNKNOWN.
    undeclared = oracle.respond(Action(ActionKind.LAB, "lab:raised_bnp"))
    assert undeclared[0].polarity is Polarity.UNKNOWN


def test_open_world_case_is_unchanged_by_the_vocabulary_field():
    """Cases that enumerate their features keep the old behaviour."""
    case = Case(
        case_id="open",
        presenting_complaint="cough",
        diagnosis="community_acquired_pneumonia",
        features={"fever": True},
        initial_findings=("fever",),
    )
    findings = CaseOracle(case).respond(Action(ActionKind.ASK, "pleuritic_pain"))
    assert findings[0].polarity is Polarity.UNKNOWN


def test_closed_world_negative_moves_the_posterior(kb):
    """The point of the fix: an ABSENT answer has to change the ranking.

    Guards the actual failure mode rather than the polarity enum -- an UNKNOWN
    reply leaves the posterior untouched, which is what made the bug silent.
    """
    proposer = BayesianProposer(kb)
    concept = kb.discriminating_features([e.label for e in kb.diseases()])[0]

    before = proposer.propose([])
    after_absent = proposer.propose(
        [Finding(concept=concept, polarity=Polarity.ABSENT)]
    )
    after_unknown = proposer.propose(
        [Finding(concept=concept, polarity=Polarity.UNKNOWN)]
    )

    assert after_unknown.probability_of(before.top.label) == pytest.approx(
        before.probability_of(before.top.label)
    )
    assert after_absent.probability_of(before.top.label) != pytest.approx(
        before.probability_of(before.top.label)
    )


def test_noisy_oracle_only_degrades_history(kb, cases):
    case = cases[0]
    oracle = NoisyOracle(case, recall_failure=1.0, seed=1)
    asked = oracle.respond(Action(ActionKind.ASK, "pleuritic_pain"))
    tested = oracle.respond(Action(ActionKind.LAB, "lab:raised_wcc"))
    assert asked[0].polarity is Polarity.ABSENT  # recall failure
    assert tested[0].polarity is Polarity.PRESENT  # tests are reliable


# --------------------------------------------------------------------------
# LLM proposer


def test_llm_proposer_uses_valid_ranking(kb):
    llm = ScriptedLLM([
        ScriptedLLM.ranking([("pulmonary_embolism", 0.7),
                             ("community_acquired_pneumonia", 0.3)])
    ])
    d = LLMProposer(kb, llm).propose([Finding("dyspnoea_at_rest")])
    assert d.top.label == "pulmonary_embolism"


def test_llm_proposer_drops_hallucinated_labels(kb):
    llm = ScriptedLLM([
        ScriptedLLM.ranking([("dragon_pox", 0.9), ("pulmonary_embolism", 0.1)])
    ])
    d = LLMProposer(kb, llm).propose([Finding("dyspnoea_at_rest")])
    assert "dragon_pox" not in [h.label for h in d.hypotheses]


def test_llm_proposer_falls_back_on_garbage(kb):
    """A malformed generation must not take the case down."""
    d = LLMProposer(kb, ScriptedLLM(["not json at all"])).propose([Finding("fever")])
    assert d.top.probability > 0


def test_llm_proposer_falls_back_when_llm_raises(kb):
    from dxagent import NullLLM

    d = LLMProposer(kb, NullLLM()).propose([Finding("fever")])
    assert d.top.probability > 0


def test_consensus_reports_disagreement(kb):
    agree = ScriptedLLM([ScriptedLLM.ranking([("community_acquired_pneumonia", 1.0)])])
    consensus = ConsensusProposer(
        primary=BayesianProposer(kb), secondary=LLMProposer(kb, agree)
    )
    consensus.propose([Finding("fever"), Finding("productive_cough")])
    assert 0.0 <= consensus.last_disagreement <= 1.0


# --------------------------------------------------------------------------
# metrics


def _outcomes(kb, cases):
    return run_agent(DiagnosticAgent(kb=kb), cases)


def test_metrics_are_consistent(kb, cases):
    outcomes = _outcomes(kb, cases)
    truths = {c.case_id: c.diagnosis for c in cases}
    ranking = ranking_metrics(outcomes, truths)
    assert ranking.top1 <= ranking.top3 <= ranking.top5
    assert 0.0 <= ranking.mrr <= 1.0
    calibration = calibration_metrics(outcomes, truths)
    assert 0.0 <= calibration.ece <= 1.0
    assert 0.0 <= calibration.brier <= 1.0
    selective = selective_metrics(outcomes, truths)
    assert 0.0 <= selective.coverage <= 1.0


def test_full_coverage_accuracy_matches_top1(kb, cases):
    """The gate must not be able to change the ranking numbers."""
    outcomes = _outcomes(kb, cases)
    truths = {c.case_id: c.diagnosis for c in cases}
    assert math.isclose(
        selective_metrics(outcomes, truths).full_coverage_accuracy,
        ranking_metrics(outcomes, truths).top1,
    )


def test_risk_coverage_curve_is_monotone_in_coverage(kb, cases):
    outcomes = _outcomes(kb, cases)
    truths = {c.case_id: c.diagnosis for c in cases}
    curve = risk_coverage_curve(outcomes, truths)
    coverages = [c for c, _ in curve]
    assert coverages == sorted(coverages)
    assert all(0.0 <= r <= 1.0 for _, r in curve)


def test_metrics_handle_empty_input():
    assert ranking_metrics([], {}).n == 0
    assert calibration_metrics([], {}).n == 0
    assert selective_metrics([], {}).n == 0
    assert risk_coverage_curve([], {}) == []


def test_evaluate_end_to_end(kb, cases):
    calibration_cases, test_cases = split_cases(cases, calibration_fraction=0.3)
    result = evaluate(DiagnosticAgent(kb=kb), test_cases,
                      calibration_cases=calibration_cases)
    assert result.ranking.n == len(test_cases)
    assert result.report()


def test_split_is_deterministic_and_disjoint(cases):
    a1, b1 = split_cases(cases)
    a2, b2 = split_cases(list(reversed(cases)))
    assert [c.case_id for c in a1] == [c.case_id for c in a2]
    assert not ({c.case_id for c in b1} & {c.case_id for c in a1})


# --------------------------------------------------------------------------
# vocabulary (HPO concept layer)


@pytest.fixture
def vocab():
    from dxagent.vocabulary import Vocabulary
    from pathlib import Path

    obo = Path(__file__).resolve().parent.parent / "data" / "hp.obo"
    if not obo.exists():
        pytest.skip("hp.obo not present; run scripts/build_vocabulary.py")
    return Vocabulary.build(obo)


def test_every_curated_id_resolves(vocab):
    """A mapping table that rots between HPO releases is worse than none."""
    from dxagent.vocabulary import CURATED

    for key, (hpo_id, _, _) in CURATED.items():
        if hpo_id is None:
            continue
        assert vocab.term(hpo_id) is not None, f"{key} -> {hpo_id} missing"


def test_no_mapping_warnings(vocab):
    assert not [c for c in vocab.concepts.values() if "WARNING" in c.note]


def test_productive_cough_is_not_its_own_antonym(vocab):
    """Regression: HP:0031246 is 'Nonproductive cough', the adjacent id.

    Polarity inversions are the most dangerous mapping error -- they corrupt
    every posterior silently and look completely normal in a coverage report.
    """
    concept = vocab.concept("productive_cough")
    assert concept.hpo_name == "Productive cough"
    assert "Nonproductive" not in concept.hpo_name


def test_mapped_names_are_not_negations(vocab):
    """Cheap guard against the whole antonym class, not just the one instance."""
    for concept in vocab.concepts.values():
        if not concept.hpo_name:
            continue
        lowered = concept.hpo_name.lower()
        for prefix in ("non", "absent ", "decreased ", "reduced "):
            if lowered.startswith(prefix):
                assert "decreased" in concept.key or "reduced" in concept.key or \
                       "absent" in concept.key, (
                    f"{concept.key} maps to negated term {concept.hpo_name}"
                )


def test_exact_lookup_rejects_substring_matches(vocab):
    """'rale' must not resolve to 'P mitrale'."""
    assert vocab.lookup_exact("rale") is None
    assert vocab.lookup_exact("Crackles").hpo_id == "HP:0030830"


def test_unmapped_concepts_are_explicit_not_approximated(vocab):
    """Risk factors and exposures are not phenotypes; forcing them is wrong."""
    for key in ("smoking_history", "recent_immobility", "sudden_onset"):
        concept = vocab.concept(key)
        assert not concept.is_grounded
        assert concept.note, "unmapped concepts must record why"


def test_generalisation_walks_up_the_hierarchy(vocab):
    parent = vocab.generalise("pleuritic_pain")
    assert vocab.term(parent).name == "Chest pain"


def test_generalisation_skips_structural_roots(vocab):
    for key in vocab.concepts:
        result = vocab.generalise(key, levels=10)
        assert result not in ("HP:0000001", "HP:0000118")


def test_freeze_and_reload_roundtrip(vocab, tmp_path):
    """Runs must be reproducible without re-downloading a monthly release."""
    from dxagent.vocabulary import Vocabulary

    path = tmp_path / "vocab.json"
    vocab.to_json(path)
    reloaded = Vocabulary.from_json(path)
    assert reloaded.release == vocab.release
    assert len(reloaded.concepts) == len(vocab.concepts)
    assert reloaded.concept("pleuritic_pain").hpo_id == "HP:0033771"


def test_fixture_kb_concepts_are_covered(vocab):
    """Every finding the KB reasons over should have a vocabulary entry."""
    from dxagent.datasets.fixtures import COSTS

    missing = [c for c in COSTS if vocab.concept(c) is None]
    assert not missing, f"no vocabulary entry for: {missing}"


# --------------------------------------------------------------------------
# correlation structure


def test_no_correlations_leaves_inference_unchanged(kb, cases):
    """Adding the mechanism must not silently restate every earlier result."""
    from dxagent.datasets.fixtures import build_knowledge_base

    plain = build_knowledge_base()
    assert plain.correlations == {}
    findings = cases[0].initial()
    before = BayesianProposer(plain).propose(findings)
    after = BayesianProposer(build_knowledge_base(correlated=False)).propose(findings)
    assert before.top.probability == pytest.approx(after.top.probability)


def test_redundancy_weights_count_correlated_evidence_once():
    """Two findings correlated at 1.0 must together weigh what one does."""
    from dxagent.knowledge import InMemoryKnowledgeBase

    kb = InMemoryKnowledgeBase()
    kb.set_correlations({frozenset(("a", "b")): 1.0})

    weights = kb.redundancy_weights(["a", "b"])
    assert weights["a"] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)

    # Uncorrelated findings are untouched.
    assert kb.redundancy_weights(["a", "z"])["z"] == pytest.approx(1.0)

    # Four findings of one picture count as a little over one, not four.
    kb.set_correlations(
        {
            frozenset(pair): 1.0
            for pair in (("p", "q"), ("p", "r"), ("p", "s"), ("q", "r"), ("q", "s"), ("r", "s"))
        }
    )
    assert sum(kb.redundancy_weights(["p", "q", "r", "s"]).values()) == pytest.approx(1.0)


def test_correlation_rescues_a_diagnosis_buried_by_repeated_evidence():
    """The defect this exists for, and the size of the effect.

    Four absent findings that are facets of one clinical picture must not
    multiply into four independent penalties. Measured on fx-009's evidence,
    correcting for that lifts the true diagnosis by an order of magnitude.
    """
    from dxagent.datasets.fixtures import build_knowledge_base

    evidence = [
        Finding("fever", Polarity.PRESENT),
        Finding("productive_cough", Polarity.PRESENT),
        Finding("sudden_onset", Polarity.ABSENT),
        Finding("leg_swelling", Polarity.ABSENT),
        Finding("calf_tenderness", Polarity.ABSENT),
        Finding("recent_immobility", Polarity.ABSENT),
    ]
    naive = BayesianProposer(build_knowledge_base()).propose(evidence)
    aware = BayesianProposer(
        build_knowledge_base(correlated=True)
    ).propose(evidence)

    p_naive = naive.probability_of("pulmonary_embolism")
    p_aware = aware.probability_of("pulmonary_embolism")
    assert p_aware > 5 * p_naive


def test_correlation_survives_the_sourcing_that_removed_its_prop():
    """A repair that was resting on a bad number, and what was left after.

    This test used to assert that correlation weighting plus Merck sourcing
    put pulmonary embolism top of this evidence set. That was true, and part
    of the reason was indefensible. Pneumonia and the rest were penalised for
    the *absence* of calf tenderness and recent immobility at invented rates
    of 0.40 and 0.61 -- as though a pneumonia patient would usually have a
    swollen calf. PIOPED II measured those rates at 0.23 and 0.19 among
    patients investigated for embolism who turned out not to have one, and
    with real numbers the penalty is small and pulmonary embolism no longer
    reaches the top of *this frozen evidence set* from the proposer alone.

    So the claim is narrowed to what survives, which is still the mechanism
    the correlation work was for: four correlated negatives are not four
    independent penalties. Weighting them lifts pulmonary embolism from
    fourth at 3.9% to second at 27.6% -- seven-fold -- on identical evidence
    and an identical knowledge base.

    The case itself is not lost. fx-009 is a case, not an evidence set, and
    the shipped loop goes on to gather the evidence this snapshot freezes out
    and commits to pulmonary embolism correctly. That assertion is the last
    one here, and it is the one that describes the system.
    """
    evidence = [
        Finding("fever", Polarity.PRESENT),
        Finding("productive_cough", Polarity.PRESENT),
        Finding("sudden_onset", Polarity.ABSENT),
        Finding("leg_swelling", Polarity.ABSENT),
        Finding("calf_tenderness", Polarity.ABSENT),
        Finding("recent_immobility", Polarity.ABSENT),
        Finding("imaging:ctpa_filling_defect", Polarity.PRESENT),
    ]

    def look(correlated: bool) -> tuple[int, float]:
        differential = BayesianProposer(
            build_knowledge_base(correlated=correlated)
        ).propose(evidence)
        labels = [h.label for h in differential.hypotheses]
        return (
            labels.index("pulmonary_embolism") + 1,
            differential.probability_of("pulmonary_embolism"),
        )

    plain_rank, plain_p = look(False)
    aware_rank, aware_p = look(True)

    assert (plain_rank, aware_rank) == (4, 2), "correlation still moves the rank"
    assert aware_p > 5 * plain_p, f"a lift, not a nudge: {plain_p:.3f} -> {aware_p:.3f}"

    # Pinned so that a later change putting it back on top is noticed rather
    # than quietly celebrated: on this evidence set alone it is still second.
    assert plain_p < 0.10 < aware_p < 0.50

    # And the case, run properly, is still answered. Correlated is the
    # shipped configuration -- every entry point in scripts/ builds it that
    # way -- so this is the arm whose result describes the system.
    case = next(c for c in build_cases() if c.case_id == "fx-009")
    kb = build_knowledge_base(correlated=True)
    outcome = DiagnosticAgent(
        kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb)
    ).run(case)
    assert outcome.prediction == case.diagnosis == "pulmonary_embolism"


# --------------------------------------------------------------------------
# provenance


def test_scope_document_still_matches_the_knowledge_base(kb):
    """SCOPE.md's discriminating-evidence table, as an executable assertion.

    The table states which findings raise and which lower each diagnosis.
    It was hand-written before sourcing, four of its claims turned out to
    contradict the sourced numbers, and it was regenerated from the knowledge
    base -- so this is a regression test on document/code drift, not a
    validation of the numbers. ``scripts/plausibility_check.py`` explains why
    that distinction matters and what would be needed to restore the stronger
    claim.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "plausibility_check",
        Path(__file__).resolve().parent.parent / "scripts" / "plausibility_check.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    contradictions = []
    for label, concept, direction in module.CLAIMS:
        entry = kb.get(label)
        if entry is None or concept not in entry.features:
            continue
        if sum(1 for d in kb.diseases() if concept in d.features) <= 1:
            continue
        finding = Finding(concept, Polarity.PRESENT)
        ratio = entry.likelihood(finding, kb.background(concept)) / (
            kb.background_likelihood(finding)
        )
        if (ratio > 1.0) != (direction == "raises"):
            contradictions.append(f"{label}/{concept} says {direction}, LR {ratio:.2f}")

    assert not contradictions, (
        "SCOPE.md and the knowledge base disagree:\n  " + "\n  ".join(contradictions)
    )


def test_unsourced_likelihoods_report_as_invented():
    """Absence is the honest default: a likelihood with no source is invented.

    Built on a bare knowledge base rather than the fixtures, which now carry 25
    sourced entries. Asserting the fixtures are 0% sourced was correct when
    written and became a test of how much work had been done rather than of the
    mechanism.
    """
    from dxagent.knowledge import DiseaseEntry, InMemoryKnowledgeBase
    from dxagent.provenance import report

    bare = InMemoryKnowledgeBase()
    bare.add(
        DiseaseEntry(label="x", prevalence=1.0, features={"a": 0.5, "b": 0.5})
    )
    coverage = report(bare)
    assert coverage.total == 2
    assert coverage.invented == 2
    assert coverage.coverage == 0.0


def test_ontology_edges_agree_with_the_reasoner(kb):
    """The graph must not be able to disagree with the arithmetic.

    An ontology built by hand alongside a knowledge base is two sources of
    truth that drift. This one is derived from the same likelihood ratio
    ``evidence_split`` uses, so the test is that they still say the same
    thing about the same finding.
    """
    from dxagent.ontology import Ontology

    graph = Ontology.build(kb)
    assert graph.edges, "the fixture knowledge base should produce edges"

    for entry in kb.diseases():
        findings = [Finding(c, Polarity.PRESENT) for c in entry.features]
        supporting, against = kb.evidence_split(entry.label, findings)
        graph_supports = {
            e.finding for e in graph.edges_into(entry.label) if e.relation == "supports"
        }
        graph_contradicts = {
            e.finding
            for e in graph.edges_into(entry.label)
            if e.relation == "contradicts"
        }
        # evidence_split's citations carry "<label>/<concept>" as the locator.
        split_supports = {c.locator.split("/", 1)[1] for c in supporting}
        split_against = {c.locator.split("/", 1)[1] for c in against}

        assert graph_supports == split_supports, entry.label
        assert graph_contradicts == split_against, entry.label


def test_ontology_edges_carry_provenance_and_are_mostly_invented(kb):
    """A graph that looks uniform would misrepresent the knowledge base.

    Most edges are drawn from invented numbers. That has to be visible on
    the edge and in the summary, or a dense diagram becomes a more effective
    way to mislead than the numbers were on their own.
    """
    from dxagent.ontology import Ontology

    graph = Ontology.build(kb)
    tiers = {e.provenance for e in graph.edges}
    assert tiers <= {"measured", "narrative", "invented"}
    assert "invented" in tiers

    invented = sum(1 for e in graph.edges if e.provenance == "invented")
    assert invented > len(graph.edges) / 2, "most edges should be invented today"
    assert "edge provenance" in graph.summary()

    # A sourced edge keeps its citation; an invented one has none to keep.
    for edge in graph.edges:
        if edge.provenance == "invented":
            assert edge.citation is None
        else:
            assert edge.citation is not None and edge.citation.source_id


def test_ontology_is_traversable_in_both_directions(kb):
    """The property that makes it a graph rather than a report."""
    from dxagent.ontology import Ontology

    graph = Ontology.build(kb)
    cause = "community_acquired_pneumonia"

    findings = graph.neighbours(cause)
    assert findings, "a cause should reach findings"
    for finding in findings:
        assert cause in graph.neighbours(finding), "edges must traverse both ways"

    # Strongest-first ordering, with contradicting edges ranked by distance
    # from parity rather than by raw ratio.
    strengths = [e.strength for e in graph.edges_into(cause)]
    assert strengths == sorted(strengths, reverse=True)


def test_ontology_dot_marks_invented_edges_dashed(kb):
    """The one visual encoding that carries meaning rather than decoration."""
    from dxagent.ontology import Ontology

    dot = Ontology.build(kb).to_dot(causes=["pericarditis"])
    assert dot.startswith("digraph ontology {") and dot.rstrip().endswith("}")
    assert "style=dashed" in dot, "invented edges must be visually distinct"
    assert "pericarditis" in dot


def test_the_hard_cases_still_break_the_model_in_exactly_one_place():
    """The replacement instrument, pinned to what it currently measures.

    The ten development fixtures are saturated: the shipped configuration
    ranks all ten correctly, commits nothing wrongly, and the gate's measured
    value on them is +0.0% accuracy gained. They cannot tell a good change
    from a bad one any more. ``build_hard_cases`` is five named diagnostic
    traps written to replace them, and its docstring records that the traps
    were fixed on clinical grounds before the model saw any of them.

    What it found on the first run, which is the argument for building it:
    **fx-h04 is a wrong commit at 70% confidence** -- a pneumonia in a COPD
    patient, read as a COPD exacerbation, with consolidation visible on the
    chest radiograph. That is the first wrong commit on any fixture set in
    this project, and the ten cases could never have surfaced it.

    Pinned rather than fixed, and pinned as a failure rather than deleted,
    for the same reason the masquerade test is: a number that moves should be
    noticed rather than discovered. If a later change fixes fx-h04, this test
    fails and the fix gets written down.
    """
    from dxagent.datasets.fixtures import build_hard_cases

    kb = build_knowledge_base(correlated=True)
    agent = DiagnosticAgent(kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb))

    top1, wrong = [], []
    for case in build_hard_cases():
        outcome = agent.run(case)
        if outcome.differential.hypotheses[0].label == case.diagnosis:
            top1.append(case.case_id)
        if outcome.verdict is Verdict.COMMITTED and outcome.prediction != case.diagnosis:
            wrong.append((case.case_id, outcome.prediction))

    assert len(top1) == 4, f"true diagnosis ranked first in {top1}"
    assert wrong == [("fx-h04", "copd_exacerbation")], f"wrong commits: {wrong}"

    # The instrument has to stay harder than the set it replaces, or it is not
    # an instrument. The ten commit nothing wrongly; these five commit one.
    plain = DiagnosticAgent(
        kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb)
    )
    assert not [
        case.case_id
        for case in build_cases()
        if plain.run(case).verdict is Verdict.COMMITTED
        and plain.run(case).prediction != case.diagnosis
    ]


def test_documented_coverage_figures_match_the_knowledge_base(kb):
    """The documents quote counts; this fails when they go stale.

    Three separate audits this session found the same class of error: a
    number sourced, and SCOPE.md / DESIGN.md / RESPONSIBLE_AI.md still
    quoting the previous count. Prose drifts silently because nothing
    executes it, and a stale figure reads exactly as confidently as a live
    one. Rather than re-checking by hand a fourth time, the figures are
    pinned here -- if sourcing moves a count, this fails and names the file.

    Deliberately only pins the *coverage* figures, not every number in the
    documents. Those are the ones that move whenever anyone does sourcing
    work, which is what makes them the ones that go stale.

    The denominator is taken from the knowledge base, not written into the
    pattern. The first version of this test hardcoded 135; when sourcing took
    the total to 153 the patterns stopped matching any document and the test
    went on passing, which is the exact failure it exists to prevent, one
    level up.
    """
    from pathlib import Path

    from dxagent.provenance import report

    coverage = report(kb)
    root = Path(__file__).resolve().parent.parent
    documents = {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("SCOPE.md", "DESIGN.md", "RESPONSIBLE_AI.md", "WRITEUP.md")
    }

    stale: list[str] = []
    invented = str(coverage.invented)
    percent = f"{coverage.coverage:.0%}"
    total = coverage.total

    # The denominator is read from the knowledge base rather than written in.
    # An earlier version of this test hardcoded 135, and when the total moved
    # to 153 the two patterns using it silently stopped matching anything --
    # a drift guard that had itself drifted, passing because it was looking
    # for a string no document contained any more.
    of_total = re.compile(rf"(\d+) of {total}\b")
    percent_of_total = re.compile(rf"(\d+)% (?:sourced|of the {total})")

    for name, text in documents.items():
        # Any "<n> of <total>" claim must use the live invented count.
        for quoted in of_total.findall(text):
            if quoted != invented:
                stale.append(
                    f"{name}: says '{quoted} of {total}', live count is {invented}"
                )
        # Any "<n>% sourced" or "<n>% of the <total>" claim must match.
        for quoted in percent_of_total.findall(text):
            if f"{quoted}%" != percent:
                stale.append(f"{name}: says '{quoted}% sourced', live figure is {percent}")
        # "Coverage is N%" must be *sourcing* coverage, not the abstention
        # gate's coverage, which is a different quantity that also lives in
        # DESIGN.md and reached 100% on the fixture set. Require the sourcing
        # context on the same clause -- the first draft of this test matched
        # both and reported the gate's number as a stale sourcing figure.
        for quoted in re.findall(r"[Cc]overage is (\d+)%[^.]*invented", text):
            if f"{quoted}%" != percent:
                stale.append(f"{name}: says 'coverage is {quoted}%', live figure is {percent}")

    assert not stale, "documented coverage figures are stale:\n  " + "\n  ".join(stale)


def test_priors_are_counted_separately_and_default_to_invented(kb):
    """The priors must be visible in the audit, and not folded into it.

    Two failure modes, both of which the report had at different times.
    Counting priors nowhere leaves eight invented numbers the audit cannot
    name -- silence a reader takes for absence of a problem. Counting them in
    the likelihood totals is the opposite error: it lets a well-sourced
    likelihood table carry eight unsourced priors inside one flattering
    percentage. They are a different quantity from a different literature and
    are reported on their own line.
    """
    from dxagent.knowledge import DiseaseEntry, InMemoryKnowledgeBase
    from dxagent.provenance import Provenance, LikelihoodSource, report

    coverage = report(kb)
    assert coverage.priors_total == len(kb.diseases())
    assert coverage.priors_sourced == coverage.priors_total, (
        "every prior now carries a source; see _PRIORS in fixtures.py"
    )
    assert coverage.priors_invented == 0
    assert "disease priors" in coverage.summary()

    # Each carries the band spanning the two source presentations, because
    # the point estimate assumes a 50/50 mix the sources do not supply.
    for entry in kb.diseases():
        assert entry.prior_source.band is not None
        low, high = entry.prior_source.band
        assert low <= entry.prevalence <= high

    # A prior is not counted in the likelihood totals.
    bare = InMemoryKnowledgeBase()
    bare.add(DiseaseEntry(label="x", prevalence=1.0, features={"a": 0.5}))
    assert report(bare).total == 1
    assert report(bare).priors_total == 1

    # And the mechanism works when one is supplied, so this tests the tier
    # rather than the fact that nobody has done the sourcing yet.
    sourced = InMemoryKnowledgeBase()
    sourced.add(
        DiseaseEntry(
            label="x",
            prevalence=1.0,
            features={"a": 0.5},
            prior_source=LikelihoodSource(provenance=Provenance.MEASURED),
        )
    )
    assert report(sourced).priors_sourced == 1
    assert "invented" in coverage.summary()


def test_fixture_coverage_is_partial_and_reported(kb):
    """The fixtures are part-sourced, and the figure has to be visible."""
    from dxagent.provenance import report

    coverage = report(kb)
    assert coverage.total == sum(len(e.features) for e in kb.diseases())
    assert coverage.measured > 0, "expected the DDXPlus-sourced entries"
    assert coverage.invented > 0, "and the rest still invented"
    assert coverage.measured + coverage.narrative + coverage.invented == coverage.total


def test_narrative_rubric_returns_a_band_not_a_point():
    """A converted phrase is a range; hardening it into a point loses the
    only honest thing about it."""
    from dxagent.provenance import Provenance, from_narrative

    value, source = from_narrative(
        "common", Citation("MSD", "cap", "fever is common")
    )
    assert source.provenance is Provenance.NARRATIVE
    assert source.band is not None
    low, high = source.band
    assert low < value < high
    assert source.is_sourced


def test_unknown_phrase_is_rejected_rather_than_defaulted():
    """A silent default would manufacture a number that looks derived and
    is not -- exactly the confusion the module exists to prevent."""
    from dxagent.provenance import from_narrative

    with pytest.raises(KeyError, match="not in the rubric"):
        from_narrative("fairly often ish", Citation("X", "y"))


def test_provenance_counts_by_tier(kb):
    import dataclasses

    from dxagent.provenance import Provenance, from_narrative, measured, report

    before = report(kb)
    label = kb.diseases()[0].label
    entry = kb.get(label)
    # Pick concepts that are not already sourced, so the delta is unambiguous.
    concepts = [c for c in entry.features if c not in entry.sources][:2]
    assert len(concepts) == 2, "fixture entry has too few unsourced features"

    # Both helpers return (value, source), so an entry destined for _SOURCED
    # reads the same either way and the value cannot disagree with itself.
    _, narrative_source = from_narrative("rare", Citation("MSD", "x", "rare"))
    measured_value, measured_source = measured(
        0.14, Citation("PIOPED", "table 2"), 0.10, 0.19
    )
    assert measured_value == pytest.approx(0.14)
    # Merged, not replaced: this entry already carries sourced likelihoods, and
    # overwriting the dict would remove them and make the delta below wrong in
    # a way that looks like a counting bug.
    sources = dict(entry.sources)
    sources[concepts[0]] = narrative_source
    sources[concepts[1]] = measured_source
    kb.entries[label] = dataclasses.replace(entry, sources=sources)

    # Counted as a delta against whatever the fixtures already carry, so this
    # tests the tallying rather than how many entries have been sourced so far.
    after = report(kb)
    assert after.measured == before.measured + 1
    assert after.narrative == before.narrative + 1
    assert after.sourced == before.sourced + 2
    assert after.invented == before.invented - 2


# --------------------------------------------------------------------------
# synthetic case generation


def test_synthetic_cases_are_marked_in_their_id(kb):
    """The quality gate depends on a synthetic case being visible if it leaks
    into an evaluation set, rather than on someone remembering."""
    from dxagent.synthesis import SyntheticCase

    case = SyntheticCase(
        case_id="pe-000",
        narrative="sudden breathlessness",
        present=("sudden_onset",),
        absent=("fever",),
        diagnosis="pulmonary_embolism",
        reasoning="",
    ).to_case()
    assert case.case_id.startswith("synthetic-")
    assert case.features == {"sudden_onset": True, "fever": False}


def test_confusability_is_not_an_artefact_of_sparse_entries(kb):
    """Comparing over the intersection makes thinly-described diseases look
    confusable with everything; the union with marginal backoff does not."""
    from dxagent.synthesis import confusable_pairs

    pairs = confusable_pairs(kb, limit=6)
    assert pairs
    assert all(0.0 <= distance <= 1.0 for _, _, distance in pairs)
    # Sorted closest-first.
    assert list(pairs) == sorted(pairs, key=lambda row: row[2])
    # The sparsest entry must not monopolise the ranking.
    sparsest = min(kb.diseases(), key=lambda e: len(e.features)).label
    involving_sparsest = sum(1 for a, b, _ in pairs if sparsest in (a, b))
    assert involving_sparsest < len(pairs)


def test_phi_scan_catches_structured_identifiers():
    from dxagent.synthesis import phi_scan

    found = dict(phi_scan("ring 555-123-4567 or email a@b.com re MRN 8827361"))
    assert "phone" in found
    assert "email" in found
    assert "record number" in found
    # And is honest about what it cannot do: a bare name is not an identifier
    # any regular expression can find. This is still true of phi_scan itself
    # -- phi_scan_ner exists specifically to close this exact gap.
    assert phi_scan("the patient, John Smith, reports chest pain") == ()


def test_phi_scan_ner_catches_the_name_the_regex_screen_misses():
    from dxagent.synthesis import ner_available, phi_scan_ner

    if not ner_available():
        pytest.skip("spaCy/en_core_web_sm not installed")
    hits = phi_scan_ner("the patient, John Smith, reports chest pain")
    assert any(text == "John Smith" for _, text in hits)
    # A clean clinical narrative with no name should not false-positive on
    # ordinary clinical vocabulary.
    assert phi_scan_ner("a 54-year-old presents with pleuritic chest pain") == ()


def test_screen_reports_whether_ner_actually_ran():
    from dxagent.synthesis import ner_available, screen

    kept, report = screen([])
    assert report["ner_screening_active"] == ner_available()


def test_screen_drops_a_case_whose_narrative_names_a_person():
    from dxagent.synthesis import SyntheticCase, ner_available, screen

    if not ner_available():
        pytest.skip("spaCy/en_core_web_sm not installed")
    case = SyntheticCase(
        case_id="synthetic-phi-001",
        diagnosis="community_acquired_pneumonia",
        present=("fever",),
        absent=(),
        narrative="the patient, John Smith, reports fever and cough",
        reasoning="fever and cough support pneumonia",
        generator="test",
        critique="",
        accepted=True,
    )
    kept, report = screen([case])
    assert kept == []
    assert report["rejected_for_phi"] == 1


def test_near_duplicates_are_flagged():
    from dxagent.synthesis import SyntheticCase, near_duplicates

    def make(case_id, narrative):
        return SyntheticCase(
            case_id=case_id,
            narrative=narrative,
            present=("fever",),
            absent=(),
            diagnosis="community_acquired_pneumonia",
            reasoning="",
        )

    text = "a patient presents with fever and a productive cough for three days"
    cases = [make("a", text), make("b", text), make("c", "sudden severe pleuritic pain on exertion today")]
    flagged = near_duplicates(cases, threshold=0.5)
    assert [(a, b) for a, b, _ in flagged] == [("a", "b")]


def test_failed_critique_rejects_rather_than_passes(kb):
    """An unusable checker must not let cases through looking validated."""
    from dxagent.synthesis import CaseGenerator, SyntheticCase

    case = SyntheticCase(
        case_id="x", narrative="n", present=("fever",), absent=(),
        diagnosis="community_acquired_pneumonia", reasoning="",
    )
    broken = CaseGenerator(ScriptedLLM(["not json at all"]), kb)
    checked = broken.critique(case)
    assert not checked.accepted
    assert "unavailable" in checked.critique


def test_json_parsing_tolerates_the_wrappers_models_actually_emit():
    """The system prompt asks for bare JSON; models comply unevenly.

    Handling only responses that begin at character zero passes whenever the
    first model tried happens to produce one, and drops every case the next
    model generates. Each shape here is one a real model returns.
    """
    from dxagent.synthesis import _parse_json

    payload = '{"cases": [{"present": ["fever"], "diagnosis": "d"}]}'
    for wrapped in (
        payload,
        f"```json\n{payload}\n```",
        f"```\n{payload}\n```",
        f"Here are the vignettes:\n```json\n{payload}\n```",
        f"{payload}\n\nLet me know if you need more.",
        f"Sure! Here you go:\n{payload}\nHope that helps!",
    ):
        assert _parse_json(wrapped)["cases"][0]["diagnosis"] == "d"


def test_unreadable_response_raises_rather_than_looking_empty():
    """Tolerant of prose around JSON, not of JSON that is not there.

    Returning ``{}`` for an unparseable response would make a refusal
    indistinguishable from a well-formed empty batch.
    """
    from dxagent.synthesis import _parse_json

    for unreadable in ("I can't help with that.", "", "```json\n{oops\n```"):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_json(unreadable)


def test_generation_failures_are_recorded_not_swallowed(kb):
    """An empty batch must say which of three problems produced it.

    A model that refused, a response that could not be parsed, and a network
    failure all return no cases. Without the record, the run reports
    "generated 0" and there is nothing to act on.
    """
    from dxagent.synthesis import CaseGenerator

    refused = CaseGenerator(ScriptedLLM(["I can't help with that."]), kb)
    assert refused.generate("community_acquired_pneumonia", count=1) == []
    assert len(refused.failures) == 1
    assert "community_acquired_pneumonia" in refused.failures[0]

    # A wrapped-but-valid response is not a failure.
    payload = (
        '{"cases": [{"narrative": "n", "present": ["fever"], "absent": [], '
        '"diagnosis": "community_acquired_pneumonia", "reasoning": "r"}]}'
    )
    ok = CaseGenerator(ScriptedLLM([f"Here you go:\n```json\n{payload}\n```"]), kb)
    assert len(ok.generate("community_acquired_pneumonia", count=1)) == 1
    assert ok.failures == []


def test_near_miss_cases_do_not_collide_with_their_own_diagnosis(kb):
    """Found by running the pipeline for real; invisible to every stub test.

    ``case_id`` omitted ``confusable_with``, so every near-miss batch for a
    diagnosis collided with that diagnosis's typical batch and with every
    other near-miss batch for it. Screening drops by id, so the collision
    silently deleted the near-miss cases -- and in the first live run, all
    four survived as zero. Those are the cases the abstention gate exists to
    be tested against, so this deleted precisely the most valuable part of
    the corpus while reporting success.
    """
    from dxagent.synthesis import CaseGenerator

    payload = json.dumps({"cases": [{
        "narrative": "n", "present": ["fever"], "absent": [],
        "diagnosis": "acute_coronary_syndrome", "reasoning": "r",
    }]})
    generator = CaseGenerator(ScriptedLLM([payload]), kb, seed=0)

    ids = [
        c.case_id
        for batch in (
            generator.generate("acute_coronary_syndrome", count=1),
            generator.generate("acute_coronary_syndrome", 1, confusable_with="pericarditis"),
            generator.generate("acute_coronary_syndrome", 1, confusable_with="panic_attack"),
        )
        for c in batch
    ]
    assert len(ids) == 3
    assert len(set(ids)) == 3, f"colliding case ids: {ids}"
    assert any("-as-pericarditis-" in i for i in ids)


def test_screen_accounts_for_every_case_it_was_given(kb):
    """The report must reconcile against the corpus, or it is worse than none.

    ``rejected_as_duplicate`` was derived from the number of distinct
    duplicate *identifiers* while the loop dropped *cases*. With colliding
    ids the report understated its own losses -- a live run showed 20
    generated, 3 rejections reported, 13 kept, and four cases unaccounted for.
    """
    from dxagent.synthesis import SyntheticCase, screen

    def make(case_id, narrative, accepted=True):
        return SyntheticCase(
            case_id=case_id, narrative=narrative, present=("fever",), absent=(),
            diagnosis="community_acquired_pneumonia", reasoning="",
            accepted=accepted,
        )

    cases = [
        make("a", "patient with fever and productive cough for three days"),
        make("b", "patient with fever and productive cough for three days"),
        make("c", "unrelated presentation of sudden severe chest wall pain"),
        make("d", "another distinct presentation with wheeze and long history", accepted=False),
        make("e", "call the clinic on 555-123-4567 about this patient"),
    ]
    kept, report = screen(cases)

    accounted = (
        len(kept)
        + report["rejected_by_critique"]
        + report["rejected_as_duplicate"]
        + report["rejected_for_phi"]
    )
    assert accounted == report["generated"] == len(cases)


def test_vocabulary_leakage_is_measured_not_assumed_away():
    """The prompt forbids it and models comply unevenly.

    The first live run wrote an identifier into all 13 kept narratives.
    Tightening the prompt cut it to 1 of 18, which is a reason to measure it
    rather than to declare it solved.
    """
    from dxagent.synthesis import leaks_vocabulary

    assert leaks_vocabulary("presents with productive_cough and lab:raised_wcc") == (
        "lab:raised_wcc",
        "productive_cough",
    )
    assert leaks_vocabulary("presents with a productive cough and a raised white cell count") == ()


def test_narrative_repair_verifies_its_own_output(kb):
    """A repair that reports success without checking is worse than none.

    Three cases: a clean narrative is left alone without spending a call, a
    genuine fix is accepted, and a rewrite that still leaks is rejected in
    favour of the original rather than trusted.
    """
    from dxagent.synthesis import CaseGenerator, SyntheticCase, leaks_vocabulary

    def make(narrative):
        return SyntheticCase(
            case_id="x", narrative=narrative, present=("fever",), absent=(),
            diagnosis="community_acquired_pneumonia", reasoning="",
        )

    clean = make("presents with a productive cough and a raised D-dimer")
    untouched = CaseGenerator(ScriptedLLM(["unused"]), kb)
    assert untouched.repair_narrative(clean) is clean

    good = json.dumps({"narrative": "presents with a raised D-dimer"})
    repaired = CaseGenerator(ScriptedLLM([good]), kb).repair_narrative(
        make("presents with a raised lab:raised_d_dimer")
    )
    assert leaks_vocabulary(repaired.narrative) == ()

    # A rewrite that still leaks must not replace the original.
    bad = json.dumps({"narrative": "still has productive_cough in it"})
    generator = CaseGenerator(ScriptedLLM([bad]), kb)
    original = make("presents with productive_cough")
    result = generator.repair_narrative(original)
    assert result.narrative == original.narrative
    assert generator.failures and "still leaks" in generator.failures[0]


def test_repair_preserves_the_findings_lists(kb):
    """The repair rewrites prose only; the data must be untouched."""
    from dxagent.synthesis import CaseGenerator, SyntheticCase

    case = SyntheticCase(
        case_id="x", narrative="shows productive_cough today",
        present=("fever", "productive_cough"), absent=("orthopnoea",),
        diagnosis="community_acquired_pneumonia", reasoning="r",
    )
    payload = json.dumps({"narrative": "shows a productive cough today"})
    repaired = CaseGenerator(ScriptedLLM([payload]), kb).repair_narrative(case)

    assert repaired.present == case.present
    assert repaired.absent == case.absent
    assert repaired.diagnosis == case.diagnosis
    assert repaired.reasoning == case.reasoning


def test_screen_reports_what_it_discarded_and_why(kb):
    from dxagent.synthesis import SyntheticCase, screen

    def make(case_id, narrative, accepted=True):
        return SyntheticCase(
            case_id=case_id, narrative=narrative, present=("fever",), absent=(),
            diagnosis="community_acquired_pneumonia", reasoning="",
            accepted=accepted,
        )

    cases = [
        make("clean", "a distinctive vignette about breathlessness at rest"),
        make("phi", "contact the patient on 555-123-4567 about the result"),
        make("rejected", "another quite different vignette entirely", accepted=False),
    ]
    kept, report = screen(cases)
    assert [c.case_id for c in kept] == ["clean"]
    assert report["generated"] == 3
    assert report["rejected_for_phi"] == 1
    assert report["rejected_by_critique"] == 1


# --------------------------------------------------------------------------
# verbalised confidence


def test_verbalised_confidence_is_parsed_and_bounded(kb):
    """A stated confidence is a separate claim from the probability assigned."""
    import json as _json
    import math as _math

    reply = _json.dumps(
        {
            "ranking": [{"label": kb.diseases()[0].label, "probability": 1.0, "why": ""}],
            "confidence": 0.9,
        }
    )
    proposer = LLMProposer(kb, ScriptedLLM([reply]))
    proposer.propose([Finding("fever", Polarity.PRESENT)])
    assert proposer.last_verbalised_confidence == pytest.approx(0.9)

    # Out of range and absent both read as "not stated", not as a number.
    for bad in ('{"ranking": [], "confidence": 4}', '{"ranking": []}', "junk"):
        p2 = LLMProposer(kb, ScriptedLLM([bad]))
        p2.propose([Finding("fever", Polarity.PRESENT)])
        assert _math.isnan(p2.last_verbalised_confidence)


def test_verbalised_calibrator_refuses_small_fits():
    from dxagent.gate import VerbalisedCalibrator

    calibrator = VerbalisedCalibrator().fit([(0.9, True)] * 5)
    assert not calibrator.fitted
    assert calibrator.apply(0.9) == 0.9  # unchanged passthrough


def test_verbalised_calibrator_corrects_overconfidence():
    """The documented failure: models claim more than they earn."""
    from dxagent.gate import VerbalisedCalibrator

    # States 0.9 every time, right half the time.
    samples = [(0.9, i % 2 == 0) for i in range(60)]
    calibrator = VerbalisedCalibrator().fit(samples)

    assert calibrator.fitted
    assert calibrator.apply(0.9) == pytest.approx(0.5, abs=0.05)
    assert calibrator.overconfidence == pytest.approx(0.4, abs=0.05)


def test_verbalised_calibrator_leaves_a_calibrated_model_alone():
    """A model that earns what it claims should come back roughly unchanged."""
    from dxagent.gate import VerbalisedCalibrator

    samples = [(0.9, i % 10 != 0) for i in range(40)]  # 0.9 stated, 90% correct
    samples += [(0.5, i % 2 == 0) for i in range(40)]  # 0.5 stated, 50% correct
    calibrator = VerbalisedCalibrator().fit(samples)

    assert calibrator.apply(0.9) == pytest.approx(0.9, abs=0.1)
    assert calibrator.apply(0.5) == pytest.approx(0.5, abs=0.1)
    assert abs(calibrator.overconfidence) < 0.05


def test_workup_is_a_floor_on_investigation_not_a_substitute(kb, cases):
    """The distinction that cost fx-009 when it was got wrong.

    A mandatory workup must guarantee its items are gathered before the loop
    commits. It must not seize the turn to do so: preempting the selector meant
    the D-dimer was ordered first and displaced the chest X-ray, whose negative
    result is what undermines the wrong diagnosis on that case. The checklist
    was cleared and the investigation was worse.

    Both halves are asserted -- the item is still gathered, and it is not
    gathered first -- because either alone is satisfiable by a broken version.
    """
    from dxagent.guidelines import PE_WORKUP

    case = {c.case_id: c for c in cases}["fx-009"]
    agent = DiagnosticAgent(
        kb=build_knowledge_base(correlated=True),
        limits=LoopLimits(require_workup=True),
    )
    outcome = agent.run(case)
    asked = [s.action.target for s in outcome.steps]

    required = set(PE_WORKUP.required)
    assert required & set(asked), "the workup item must still be gathered"
    assert outcome.differential.top.label == case.diagnosis

    # The property, asserted directly rather than by position. An earlier
    # version required the workup item not to be asked first, which held only
    # while some other action cleared the due-diligence bar on turn zero.
    # Sourcing lowered the first-turn information gains, nothing clears the
    # bar, and the floor correctly fills in -- so the positional assertion was
    # pinning an accident of the numbers, not the behaviour it named. What
    # must stay true is that the workup never displaces a *better* action.
    if asked[0] in required:
        # The floor fired on turn zero. Legitimate only if the action it
        # displaced was not worth the turn -- and the action it displaced is
        # whatever ``choose_action`` had settled on, which is not necessarily
        # the highest-information-gain one: rule-out and decision-flip
        # overrides run first. Comparing against the raw selector pick reads
        # a legitimate floor as a preemption.
        from dxagent.agent import choose_action

        correlated = build_knowledge_base(correlated=True)
        state = CaseState(
            case_id=case.case_id,
            presenting_complaint=case.presenting_complaint,
            findings=case.initial(),
        )
        state.asked.update(case.initial_findings)
        differential = BayesianProposer(correlated).propose(
            state.findings, case.presenting_complaint
        )
        displaced, _, _, _ = choose_action(
            InformationGainSelector(correlated),
            correlated,
            state,
            differential,
            LoopLimits(require_workup=False),
        )
        assert (
            displaced is None
            or displaced.expected_information_gain < LoopLimits().due_diligence_gain
        ), "the workup preempted an action that was worth taking"


def test_workup_still_fills_in_when_nothing_else_is_worth_a_turn(kb):
    """The floor has to bite, or it is not a floor.

    A differential with nothing informative left must still order the
    outstanding workup item rather than committing without it.
    """
    from dxagent.guidelines import PE_WORKUP

    case = Case(
        case_id="quiet",
        presenting_complaint="pleuritic chest pain",
        diagnosis="pulmonary_embolism",
        features={"pleuritic_pain": True, "lab:raised_d_dimer": True},
        initial_findings=("pleuritic_pain",),
    )
    outcome = DiagnosticAgent(kb=kb).run(case)
    asked = [s.action.target for s in outcome.steps]
    assert set(PE_WORKUP.required) & set(asked)


# --------------------------------------------------------------------------
# UMLS / SNOMED concept layer (brief section 6)


def test_concept_carries_umls_identifiers_alongside_hpo():
    """Section 6 mandates UMLS and SNOMED CT; HPO is not a substitute for them.

    Both are kept. HPO supplies the hierarchy this vocabulary already uses;
    SNOMED CT is what a clinical system would exchange. Collapsing them into
    one identifier would lose whichever question the other answers.
    """
    from dxagent.vocabulary import Concept

    coded = Concept(
        key="fever",
        hpo_id="HP:0001945",
        hpo_name="Fever",
        modality="history",
        cui="C0015967",
        snomed_ct="386661006",
        umls_name="Fever",
    )
    assert coded.is_grounded and coded.is_umls_grounded
    assert "HP:0001945" in str(coded) and "C0015967" in str(coded)

    # A concept with no CUI is still usable; the layer is reported, not required.
    plain = Concept(key="x", hpo_id="HP:1", hpo_name="x", modality="history")
    assert plain.is_grounded and not plain.is_umls_grounded


def test_umls_coverage_is_reported_rather_than_assumed():
    from dxagent.vocabulary import Concept, Vocabulary

    vocab = Vocabulary(
        concepts={
            "a": Concept("a", "HP:1", "a", "history", cui="C1"),
            "b": Concept("b", "HP:2", "b", "history"),
        }
    )
    assert vocab.umls_coverage == (1, 2)


def test_every_vocabulary_concept_has_a_umls_search_term():
    """A concept with no written term would resolve by accident or not at all."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_umls_map", Path("scripts/build_umls_map.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    concepts = {c for e in build_knowledge_base().diseases() for c in e.features}
    assert not concepts - set(module.SEARCH_TERMS)
    assert set(module.DISEASE_TERMS) == {e.label for e in build_knowledge_base().diseases()}


def test_real_cases_reference_diagnoses_the_kb_actually_has():
    from dxagent.datasets import REAL_CASES, build_knowledge_base

    labels = {e.label for e in build_knowledge_base().diseases()}
    for case in REAL_CASES:
        assert case.diagnosis in labels


def test_real_cases_leave_vocabulary_empty_so_silence_means_unknown():
    # The whole point of extracting real case reports this way: a concept
    # the source never mentions must read as unasked, not as a manufactured
    # negative. That guarantee lives in leaving `vocabulary` empty -- see
    # the Case docstring in environment.py for why.
    from dxagent.datasets import REAL_CASES

    for case in REAL_CASES:
        assert case.vocabulary == frozenset()


def test_real_cases_run_through_the_actual_agent_without_crashing():
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import REAL_CASES, build_knowledge_base

    kb = build_knowledge_base(correlated=True)
    agent = DiagnosticAgent(
        kb=kb,
        proposer=BayesianProposer(kb),
        gate=AbstentionGate(kb=kb),
        limits=LoopLimits(),
    )
    for case in REAL_CASES:
        outcome = agent.run(case)
        assert outcome.verdict is not None


def test_informative_turns_excludes_all_unknown_steps():
    from dxagent.schemas import Action, ActionKind, CaseState, Differential, Finding, Polarity

    state = CaseState(case_id="t", presenting_complaint="")
    diff = Differential.from_scores({"a": 0.6, "b": 0.4})
    action = Action(kind=ActionKind.ASK, target="x", cost=0.2, expected_information_gain=0.1)

    state.record(action, [Finding(concept="x", polarity=Polarity.UNKNOWN, provenance="ask")], diff)
    assert state.turn == 1
    assert state.informative_turns == 0

    state.record(action, [Finding(concept="y", polarity=Polarity.PRESENT, provenance="ask")], diff)
    assert state.turn == 2
    assert state.informative_turns == 1


def test_uninformative_turns_still_count_flag_changes_when_the_loop_stops():
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import build_knowledge_base
    from dxagent.environment import Case

    kb = build_knowledge_base(correlated=True)
    # A case that answers nothing -- every question comes back UNKNOWN. With
    # the default (True), max_turns is reached in max_turns raw turns
    # regardless of how uninformative they were. With False, an
    # all-UNKNOWN case can never accumulate an informative turn, so the loop
    # is bounded by cost instead, not by the turn count -- a real behaviour
    # difference, not just a different number.
    blank_case = Case(
        case_id="blank",
        presenting_complaint="nothing volunteered",
        features={},
        diagnosis="pulmonary_embolism",
        vocabulary=frozenset(),
    )

    counting = DiagnosticAgent(
        kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb),
        limits=LoopLimits(uninformative_turns_still_count=True, max_turns=3, max_cost=1000.0),
    )
    free = DiagnosticAgent(
        kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb),
        limits=LoopLimits(uninformative_turns_still_count=False, max_turns=3, max_cost=1000.0),
    )

    counting_outcome = counting.run(blank_case)
    free_outcome = free.run(blank_case)

    # Counting every turn stops at 3 raw turns. Not counting uninformative
    # ones keeps going well past 3, bounded only by the (very large) cost
    # budget or by exhausting the vocabulary -- a materially different turn
    # count, not a coincidence.
    assert counting_outcome.steps[-1].index + 1 <= 3
    assert free_outcome.steps[-1].index + 1 > 3


def test_unanswered_actions_still_cost_flag_changes_what_the_budget_buys():
    """A test the record never recorded was never performed, so never billed.

    Measured on the real PMC cases before the flag existed: 98% of one
    exhausted cost budget had gone on actions that came back UNKNOWN,
    including a CTPA charged at 20.0 against a report that never mentions
    one. The default keeps charging, so every earlier result stands.
    """
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import build_knowledge_base
    from dxagent.environment import Case

    kb = build_knowledge_base(correlated=True)
    # Answers nothing: every action necessarily comes back UNKNOWN.
    blank = Case(
        case_id="blank",
        presenting_complaint="nothing volunteered",
        features={},
        diagnosis="pulmonary_embolism",
        vocabulary=frozenset(),
    )

    def run(charge: bool):
        agent = DiagnosticAgent(
            kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb),
            limits=LoopLimits(
                unanswered_actions_still_cost=charge,
                uninformative_turns_still_count=False,
                max_turns=3,
            ),
        )
        return agent.run(blank)

    charged, free = run(True), run(False)

    assert charged.budget_spent > 0, "the default must still bill for actions"
    assert free.budget_spent == 0, "nothing was performed, so nothing is owed"
    # And the freed budget buys real reach: more of the vocabulary gets asked
    # rather than the loop stopping on a bill for tests that never happened.
    assert len(free.steps) > len(charged.steps)


def test_reading_unlisted_findings_as_atypical_buys_rank_and_costs_safety():
    """The measured negative result behind ``unlisted_as_atypical`` staying off.

    The marginal backoff has a real cost: a raised troponin scores 0.45 under
    asthma exacerbation because asthma lists no troponin entry, so findings
    that should demolish a hypothesis barely move it, and a confirmed NSTEMI
    ranked fifth. Reading an unlisted finding as "not typical" instead fixes
    exactly that -- mean true-diagnosis rank over the eight real cases
    improves from 2.62 to 2.00, and the NSTEMI climbs from fifth to second.

    It is still the wrong trade, and this test exists so that nobody adopts
    it on the ranking number alone. Every uncharacterised cell becomes
    strongly discriminating at once, the posterior sharpens everywhere, and
    the gate starts committing on cases it previously abstained from: real
    commits go from 3 to 7, but three of them are wrong -- pericarditis
    committed as acute coronary syndrome at 98%, and a confirmed NSTEMI
    committed as pericarditis at 87%. An unexcluded ACS that escalates is
    safe; a confident wrong commit on a real infarction is the failure this
    whole gate exists to prevent. A fixture case breaks too.

    Ranking improved while decisions got worse, which is precisely why this
    project reports coverage and error separately.
    """
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import REAL_CASES
    from dxagent.datasets.fixtures import build_knowledge_base as build

    def wrong_commits(atypical: bool) -> int:
        kb = build(correlated=True, unlisted_as_atypical=atypical)
        agent = DiagnosticAgent(
            kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb),
            limits=LoopLimits(
                uninformative_turns_still_count=False,
                unanswered_actions_still_cost=False,
            ),
        )
        bad = 0
        for case in REAL_CASES:
            outcome = agent.run(case)
            if outcome.verdict is Verdict.COMMITTED:
                bad += outcome.prediction != case.diagnosis
        return bad

    assert wrong_commits(False) == 0, "the shipped backoff commits nothing wrong"
    assert wrong_commits(True) > 0, (
        "if this stops being true the trade-off has changed and the default "
        "is worth revisiting -- re-measure rather than flipping the flag"
    )


def test_completing_the_grid_buys_rank_and_costs_one_wrong_commit():
    """The measured trade behind ``complete_grid`` being off by default.

    Filling the remaining uncharacterised cells removes indefensible values
    -- a pericardial friction rub asserted at 0.60 for panic attack -- and it
    works: mean true-diagnosis rank over the real cases improves 2.50 to
    1.875, and the number of cases the loop is willing to commit on rises
    from three to five.

    There were 77 such cells; PIOPED II has since measured 14 of them, so the
    flag now writes 63. That is the shape of the argument against it. It is
    63 invented numbers from one non-clinician in one sitting, it takes the
    sourced fraction from 33% back down to 24%, and it turns one abstention
    into a wrong commit -- acute coronary syndrome on a true pericarditis.

    That wrong commit errs *toward* the time-critical diagnosis, which is
    the direction this project's asymmetry says to prefer and the direction
    the source case's own clinicians took (they went to angiography). That
    is an argument for adopting it, not a proof, and the decision wants a
    clinician rather than a passing test.
    """
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import REAL_CASES
    from dxagent.datasets.fixtures import build_knowledge_base as build

    def measure(grid: bool) -> tuple[int, float]:
        kb = build(correlated=True, complete_grid=grid)
        agent = DiagnosticAgent(
            kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb),
            limits=LoopLimits(
                uninformative_turns_still_count=False,
                unanswered_actions_still_cost=False,
            ),
        )
        wrong, ranks = 0, []
        for case in REAL_CASES:
            outcome = agent.run(case)
            labels = [h.label for h in outcome.differential.hypotheses]
            ranks.append(labels.index(case.diagnosis) + 1)
            if outcome.verdict is Verdict.COMMITTED:
                wrong += outcome.prediction != case.diagnosis
        return wrong, sum(ranks) / len(ranks)

    off_wrong, off_rank = measure(False)
    on_wrong, on_rank = measure(True)

    assert off_wrong == 0, "the shipped grid still commits nothing wrong"
    assert on_wrong > off_wrong, "completing the grid trades an abstention away"
    assert on_rank < off_rank, "and buys ranking with it"

    # No cell is left to fall back on once the grid is complete.
    kb = build(correlated=True, complete_grid=True)
    every = {c for e in kb.diseases() for c in e.features}
    assert not [
        (e.label, c) for c in every for e in kb.diseases() if c not in e.features
    ]


def test_escalation_names_a_workup_item_that_was_sought_but_unavailable():
    """An escalation a clinician can act on names the missing test.

    ``outstanding_workup`` cannot tell "not asked yet" from "asked, and the
    record had no answer"; ``unavailable_workup`` intersects it with what was
    actually asked, which is the difference between "could not narrow the
    differential" and "a D-dimer was sought and never obtained".
    """
    from dxagent.agent import name_the_missing_workup
    from dxagent.guidelines import unavailable_workup
    from dxagent.schemas import CaseState, Finding, Polarity

    # Rest dyspnoea arms the PE workup, which requires a D-dimer. The D-dimer
    # was asked for and came back UNKNOWN -- sought, not available.
    state = CaseState(case_id="t", presenting_complaint="breathless")
    state.findings = [
        Finding(concept="dyspnoea_at_rest", polarity=Polarity.PRESENT),
        Finding(concept="lab:raised_d_dimer", polarity=Polarity.UNKNOWN),
    ]
    state.asked = {"dyspnoea_at_rest", "lab:raised_d_dimer"}

    gaps = unavailable_workup(state.findings, state.asked)
    assert [c for c, _ in gaps] == ["lab:raised_d_dimer"]

    reason = name_the_missing_workup("top-1 confidence below threshold", state)
    assert "lab:raised_d_dimer" in reason
    assert "sought but not available" in reason

    # Never asked at all is a different situation and must not be reported as
    # unavailable -- the loop simply has not got to it yet.
    unasked = CaseState(case_id="t2", presenting_complaint="breathless")
    unasked.findings = [Finding(concept="dyspnoea_at_rest", polarity=Polarity.PRESENT)]
    assert unavailable_workup(unasked.findings, unasked.asked) == ()
    assert name_the_missing_workup("reason", unasked) == "reason"


def test_default_cost_accounting_is_unchanged_by_the_new_flag(kb, cases):
    """The flag is opt-in: on the fixtures the shipped default is identical."""
    from dxagent import DiagnosticAgent, LoopLimits

    for case in cases:
        a = DiagnosticAgent(kb=kb, limits=LoopLimits()).run(case)
        b = DiagnosticAgent(
            kb=kb, limits=LoopLimits(unanswered_actions_still_cost=True)
        ).run(case)
        assert a.budget_spent == b.budget_spent
        assert a.differential.top.label == b.differential.top.label
