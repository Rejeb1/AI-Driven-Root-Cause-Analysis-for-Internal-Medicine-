"""The diagnostic reasoning loop.

One turn:

    1. Update the differential from all findings so far (grounded in the KB).
    2. Calibrate it.
    3. Ask the gate whether to commit, escalate, or continue.
    4. If continuing, select the highest information-value action within budget
       and execute it against the environment.

Termination is guaranteed by three independent conditions -- turn limit, cost
budget, and exhaustion of informative actions -- because a loop that can only
stop when it becomes confident is a loop that can fail to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import InformationGainSelector, classify
from .belief import BayesianProposer, ConsensusProposer, Proposer
from .environment import Case, CaseOracle, Environment
from .gate import AbstentionGate
from .guidelines import outstanding_workup
from .knowledge import InMemoryKnowledgeBase
from .schemas import (
    Action,
    ActionKind,
    CaseOutcome,
    CaseState,
    Escalation,
    Verdict,
)


@dataclass
class LoopLimits:
    max_turns: int = 12
    max_cost: float = 40.0

    # Even when the gate is satisfied, do not stop while an affordable action
    # remains whose expected information gain exceeds this. Without it the loop
    # committed on the two symptoms the patient volunteered -- 85% confident,
    # wrong, and having asked nothing. A posterior can be peaked because the
    # evidence is decisive or because there is barely any evidence, and top-1
    # probability alone cannot tell those apart. Set to 0.0 to disable.
    due_diligence_gain: float = 0.15

    # Do not commit while an affordable unasked test could change the leading
    # diagnosis. As a *detector* this separates the fixture failures cleanly --
    # both cases the loop gets wrong have such a test outstanding, six of the
    # eight it gets right do not. As a *remedy* it does nothing: enabling it
    # leaves accuracy at 8/10 while raising mean cost from 5.5 to 8.6, because
    # the tests it orders are the ones the current posterior already considers
    # relevant, and the current posterior is what is wrong. Default off on that
    # measurement; the signal is still worth reading, so ``flip_action``
    # remains available to the gate and to analysis.
    require_decisive_tests: bool = False

    # Gather the evidence a published guideline requires for the presentation,
    # before committing, regardless of the current differential. The only
    # stopping criterion here that does not consult the posterior -- which is
    # what lets it recover a diagnosis the posterior has dismissed. Costs
    # unnecessary tests on patients who do not need them; that trade is a
    # clinical policy decision, not an engineering one.
    require_workup: bool = True


@dataclass
class DiagnosticAgent:
    """Orchestrates proposer, selector, gate, and environment."""

    kb: InMemoryKnowledgeBase
    proposer: Proposer | None = None
    selector: InformationGainSelector | None = None
    gate: AbstentionGate | None = None
    limits: LoopLimits = field(default_factory=LoopLimits)
    trace: bool = False

    def __post_init__(self) -> None:
        self.proposer = self.proposer or BayesianProposer(self.kb)
        self.selector = self.selector or InformationGainSelector(self.kb)
        self.gate = self.gate or AbstentionGate(self.kb)

    def run(self, case: Case, environment: Environment | None = None) -> CaseOutcome:
        environment = environment or CaseOracle(case)
        state = CaseState(
            case_id=case.case_id,
            presenting_complaint=case.presenting_complaint,
            findings=case.initial(),
        )
        state.asked.update(case.initial_findings)

        while True:
            differential = self.proposer.propose(
                state.findings, case.presenting_complaint
            )
            calibrated = self.gate.calibrate(differential)
            state.differential = calibrated
            disagreement = (
                self.proposer.last_disagreement
                if isinstance(self.proposer, ConsensusProposer)
                else 0.0
            )
            # Only the Bayesian proposer can report how well the KB accounts
            # for the findings; an LLM ranking exposes no such quantity. Absent
            # it the gate falls back to inf and the check is inert, rather than
            # escalating everything.
            evidence_fit = getattr(self.proposer, "last_evidence_fit", float("inf"))
            decision = self.gate.evaluate(
                calibrated, disagreement=disagreement, evidence_fit=evidence_fit
            )

            # Presentation-triggered workup outranks everything below it,
            # because it is the only criterion here that does not read the
            # posterior. The others all ask "given what I believe, what should
            # I check?", which cannot recover a diagnosis the belief has
            # already dismissed.
            mandated = (
                self._mandatory_action(state) if self.limits.require_workup else None
            )

            action = self.selector.select(state, calibrated)
            # Priority order, most overriding first: excluding a dangerous
            # diagnosis, then settling a question that would change the answer,
            # then general information gain. Safety outranks decisiveness, and
            # decisiveness outranks sharpening a posterior that is not in
            # doubt.
            flip = (
                self.selector.flip_action(state, calibrated)
                if self.limits.require_decisive_tests
                else None
            )
            if flip is not None:
                action = flip
            ruleout = self.selector.ruleout_action(state, calibrated)
            if ruleout is not None:
                action = ruleout
            if mandated is not None:
                action = mandated
            affordable = (
                action is not None
                and state.budget_spent + action.cost <= self.limits.max_cost
            )
            out_of_turns = state.turn >= self.limits.max_turns

            # Due diligence: confident is not the same as finished.
            outstanding = (
                affordable
                and not out_of_turns
                and (
                    mandated is not None
                    or ruleout is not None
                    or flip is not None
                    or (
                        self.limits.due_diligence_gain > 0.0
                        and action.expected_information_gain
                        >= self.limits.due_diligence_gain
                    )
                )
            )

            self._log(state, calibrated, decision, action)

            if decision.should_commit and not outstanding:
                return CaseOutcome(
                    case_id=case.case_id,
                    verdict=Verdict.COMMITTED,
                    differential=calibrated,
                    confidence=decision.confidence,
                    escalation=None,
                    steps=tuple(state.history),
                    budget_spent=state.budget_spent,
                )

            if not decision.should_commit:
                if out_of_turns:
                    return self._escalate(
                        state, calibrated, decision.confidence,
                        "turn limit reached before the confidence threshold was "
                        f"met; {decision.reason}",
                        action.target if action else None,
                    )
                if action is None:
                    return self._escalate(
                        state, calibrated, decision.confidence,
                        "no remaining action would meaningfully narrow the "
                        f"differential; {decision.reason}",
                        None,
                    )
                if not affordable:
                    return self._escalate(
                        state, calibrated, decision.confidence,
                        f"cost budget exhausted; the next informative step "
                        f"({action.target}) exceeds the remaining allowance",
                        action.target,
                    )

            findings = list(environment.respond(action))
            state.record(action, findings, calibrated)

    def _mandatory_action(self, state: CaseState) -> Action | None:
        """The next unsatisfied item of a triggered guideline workup, if any.

        Deliberately reads only ``state.findings`` -- never the differential.
        Consulting the posterior here would reintroduce exactly the dependence
        this rule exists to escape.
        """
        for concept, workup in outstanding_workup(state.findings):
            if concept in state.asked or state.observed(concept) is not None:
                continue
            return Action(
                kind=classify(concept),
                target=concept,
                cost=self.kb.cost_of(concept),
                expected_information_gain=0.0,
                rationale=f"required by guideline workup: {workup}",
            )
        return None

    def _escalate(
        self,
        state: CaseState,
        differential,
        confidence: float,
        reason: str,
        unresolved: str | None,
    ) -> CaseOutcome:
        return CaseOutcome(
            case_id=state.case_id,
            verdict=Verdict.ESCALATED,
            differential=differential,
            confidence=confidence,
            escalation=Escalation(
                reason=reason,
                differential=differential,
                unresolved_question=unresolved,
                findings=tuple(state.findings),
                turns_used=state.turn,
            ),
            steps=tuple(state.history),
            budget_spent=state.budget_spent,
        )

    def _log(self, state: CaseState, differential, decision, action) -> None:
        if not self.trace:
            return
        top = ", ".join(
            f"{h.label} {h.probability:.2f}" for h in differential.hypotheses[:3]
        )
        nxt = (
            f"next={action.target} (gain {action.expected_information_gain:.2f})"
            if action
            else "next=none"
        )
        print(
            f"  turn {state.turn:>2} | H={differential.entropy:.3f} | "
            f"cost={state.budget_spent:>5.1f} | {top} | {nxt} | {decision.reason}"
        )


__all__ = ["DiagnosticAgent", "LoopLimits"]
