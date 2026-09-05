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
    # diagnosis. A good detector and a poor remedy, and the evidence for that
    # has now been taken three times. On the invented knowledge base it left
    # accuracy at 8/10 and raised cost. On DDXPlus-sourced likelihoods it
    # appeared to earn its keep, reaching 10/10 beside the correlation
    # structure. With four of those replaced by frequencies from a real cohort
    # it drops back: correlation alone reaches 10/10, and adding this returns
    # 9/10 at half again the cost. The middle result was an artefact of the
    # simulator overstating pleuritic pain twofold. Default off; ``flip_action``
    # stays available to the gate and to analysis.
    require_decisive_tests: bool = False

    # Gather the evidence a published guideline requires for the presentation,
    # before committing, regardless of the current differential. The only
    # stopping criterion here that does not consult the posterior -- which is
    # what lets it recover a diagnosis the posterior has dismissed. Costs
    # unnecessary tests on patients who do not need them; that trade is a
    # clinical policy decision, not an engineering one.
    #
    # It is a floor, not a substitute: see ``choose_action``. Letting it pick
    # the turn as well as the requirement cost fx-009, and the two are now
    # separated.
    require_workup: bool = True

    # Whether a turn whose answer was UNKNOWN still counts against max_turns.
    # Default True matches every existing result exactly -- this flag did not
    # exist before, and the ten fixtures answer everything, so it changes
    # nothing for them. It matters for real, incompletely-documented case
    # reports, where several turns can come back UNKNOWN because the source
    # text simply never addressed that finding -- not a question a smarter
    # selector could have skipped, since it could not have known the answer
    # was unrecorded until it asked. Set False to stop the turn budget being
    # spent on questions the record cannot answer, so a scarce turn limit is
    # reserved for questions that can actually move the differential.
    uninformative_turns_still_count: bool = True

    # Whether an action whose findings all came back UNKNOWN still draws on
    # the cost budget. Default True preserves every existing result exactly.
    #
    # The same argument as the flag above, applied to the other budget, and
    # measured before it was believed: on the real PMC cases, 98% and 78% of
    # the exhausted cost budget had gone on actions that returned UNKNOWN --
    # in one case a CTPA charged at 20.0 against a report that never mentions
    # a CTPA. Cost models performing a test. If the record never recorded one,
    # no test was performed and no cost was incurred, so charging for it
    # prices a scan that did not happen and then reports the resulting
    # early stop as "cost budget exhausted", which is a statement about the
    # source document rather than about the agent.
    #
    # Affordability is still checked at full price before the action runs:
    # the agent must be able to afford the test it orders. Only afterwards,
    # on learning nothing was performed, is the charge dropped. With this and
    # the turn flag both off the binding constraint becomes exhaustion of
    # informative actions, which is the honest one for an incomplete record.
    unanswered_actions_still_cost: bool = True


def mandatory_action(kb, state: CaseState) -> Action | None:
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
            cost=kb.cost_of(concept),
            expected_information_gain=0.0,
            rationale=f"required by guideline workup: {workup}",
        )
    return None


def choose_action(
    selector, kb, state: CaseState, differential, limits: "LoopLimits"
) -> tuple[Action | None, Action | None, Action | None, Action | None]:
    """Pick the next action, and report which rules were live.

    Shared by the loop and the LangGraph engine. It lives here rather than in
    each because the two had their own copies and drifted the moment one was
    changed -- the equivalence test caught it, which is what that test is for,
    but the fix is to remove the duplication rather than to keep re-syncing it.

    Priority, most overriding first: excluding a dangerous diagnosis, then
    settling a question that would change the answer, then general information
    gain. Safety outranks decisiveness, and decisiveness outranks sharpening a
    posterior that is not in doubt.

    The guideline workup sits outside that order. It is a floor on
    investigation rather than a substitute for it, so it fills in only when
    nothing else is worth a turn. Letting it preempt cost fx-009: forcing the
    D-dimer to turn zero displaced the chest X-ray the selector would have
    chosen, and that X-ray's negative result is what undermines the wrong
    diagnosis. The checklist is still cleared before committing, enforced by
    ``still_outstanding`` rather than by seizing the turn.
    """
    mandated = mandatory_action(kb, state) if limits.require_workup else None

    action = selector.select(state, differential)
    flip = (
        selector.flip_action(state, differential)
        if limits.require_decisive_tests
        else None
    )
    if flip is not None:
        action = flip
    ruleout = selector.ruleout_action(state, differential)
    if ruleout is not None:
        action = ruleout
    if mandated is not None and (
        action is None
        or action.expected_information_gain < limits.due_diligence_gain
    ):
        action = mandated
    return action, mandated, ruleout, flip


def still_outstanding(
    action: Action | None,
    mandated: Action | None,
    ruleout: Action | None,
    flip: Action | None,
    affordable: bool,
    out_of_turns: bool,
    limits: "LoopLimits",
) -> bool:
    """Whether anything remains that should stop the loop committing.

    Confident is not the same as finished.
    """
    return bool(
        affordable
        and not out_of_turns
        and (
            mandated is not None
            or ruleout is not None
            or flip is not None
            or (
                limits.due_diligence_gain > 0.0
                and action.expected_information_gain >= limits.due_diligence_gain
            )
        )
    )


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

            action, mandated, ruleout, flip = choose_action(
                self.selector, self.kb, state, calibrated, self.limits
            )
            affordable = (
                action is not None
                and state.budget_spent + action.cost <= self.limits.max_cost
            )
            turns_used = (
                state.turn
                if self.limits.uninformative_turns_still_count
                else state.informative_turns
            )
            out_of_turns = turns_used >= self.limits.max_turns
            outstanding = still_outstanding(
                action, mandated, ruleout, flip, affordable, out_of_turns, self.limits
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
            state.record(
                action,
                findings,
                calibrated,
                charge_uninformative=self.limits.unanswered_actions_still_cost,
            )

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
