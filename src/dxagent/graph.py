"""The reasoning loop as a LangGraph state machine.

The brief mandates LangGraph with explicit nodes and state persisted between
them, so the chain of reasoning is inspectable rather than buried in a
``while`` loop's local variables. This module provides that without
reimplementing any reasoning: every node delegates to the same proposer,
selector and gate that ``DiagnosticAgent`` uses, and to the same
``choose_action`` -- which is shared rather than copied because the copies
drifted the first time one side was changed.

That equivalence is the property worth protecting, and it is asserted by test.
A refactor that quietly changes behaviour while claiming to be a refactor is
worse than no refactor, because every earlier measurement silently stops
applying.

The nodes
---------
    propose   -> differential from all findings so far, grounded in the KB
    calibrate -> temperature scaling, then the conformal set
    decide    -> the gate; also resolves what evidence is still outstanding
    select    -> choose the next action; the workup is a floor, not first
    observe   -> execute it against the environment and record the result

``decide`` is the conditional branch: it routes to ``select`` to keep going or
to ``finish`` to stop. Everything the graph knows sits in ``ReasoningState``,
which is a plain dict subclass so LangGraph can checkpoint it and a reviewer
can read it.

Why keep the hand-rolled loop as well
-------------------------------------
It has no dependencies, so the test suite and the evaluation harness still run
on a clean checkout with nothing installed. The graph is the mandated
orchestration; the loop is the reference implementation the graph is checked
against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from .actions import InformationGainSelector, classify
from .agent import LoopLimits, choose_action, still_outstanding
from .belief import BayesianProposer, ConsensusProposer, Proposer
from .environment import Case, CaseOracle, Environment
from .gate import AbstentionGate
from .guidelines import outstanding_workup
from .knowledge import InMemoryKnowledgeBase
from .schemas import (
    Action,
    CaseOutcome,
    CaseState,
    Escalation,
    Verdict,
)


class ReasoningState(TypedDict, total=False):
    """What passes between nodes, and what a checkpoint contains.

    ``CaseState`` is carried whole rather than unpacked into scalars: it
    already accumulates findings, asked concepts, spend and per-turn history,
    and splitting that across graph keys would create two places where the
    same fact lives.
    """

    case: Case
    state: CaseState
    environment: Environment
    differential: Any
    decision: Any
    action: Action | None
    mandated: bool
    outstanding: bool
    finished: bool
    outcome: CaseOutcome | None


@dataclass
class GraphAgent:
    """LangGraph orchestration of the same components as ``DiagnosticAgent``."""

    kb: InMemoryKnowledgeBase
    proposer: Proposer | None = None
    selector: InformationGainSelector | None = None
    gate: AbstentionGate | None = None
    limits: LoopLimits = field(default_factory=LoopLimits)
    _compiled: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.proposer = self.proposer or BayesianProposer(self.kb)
        self.selector = self.selector or InformationGainSelector(self.kb)
        self.gate = self.gate or AbstentionGate(self.kb)

    # -- nodes -------------------------------------------------------------

    def _propose(self, state: ReasoningState) -> ReasoningState:
        case = state["case"]
        differential = self.proposer.propose(
            state["state"].findings, case.presenting_complaint
        )
        return {"differential": differential}

    def _calibrate(self, state: ReasoningState) -> ReasoningState:
        calibrated = self.gate.calibrate(state["differential"])
        state["state"].differential = calibrated
        return {"differential": calibrated}

    def _decide(self, state: ReasoningState) -> ReasoningState:
        differential = state["differential"]
        disagreement = (
            self.proposer.last_disagreement
            if isinstance(self.proposer, ConsensusProposer)
            else 0.0
        )
        evidence_fit = getattr(self.proposer, "last_evidence_fit", float("inf"))
        decision = self.gate.evaluate(
            differential, disagreement=disagreement, evidence_fit=evidence_fit
        )

        case_state = state["state"]
        action, mandated, ruleout, flip = choose_action(
            self.selector, self.kb, case_state, differential, self.limits
        )
        affordable = (
            action is not None
            and case_state.budget_spent + action.cost <= self.limits.max_cost
        )
        turns_used = (
            case_state.turn
            if self.limits.uninformative_turns_still_count
            else case_state.informative_turns
        )
        out_of_turns = turns_used >= self.limits.max_turns
        outstanding = still_outstanding(
            action, mandated, ruleout, flip, affordable, out_of_turns, self.limits
        )

        outcome: CaseOutcome | None = None
        if decision.should_commit and not outstanding:
            outcome = CaseOutcome(
                case_id=case_state.case_id,
                verdict=Verdict.COMMITTED,
                differential=differential,
                confidence=decision.confidence,
                escalation=None,
                steps=tuple(case_state.history),
                budget_spent=case_state.budget_spent,
            )
        elif not decision.should_commit:
            if out_of_turns:
                outcome = self._escalate(
                    case_state, differential, decision.confidence,
                    "turn limit reached before the confidence threshold was met; "
                    f"{decision.reason}",
                    action.target if action else None,
                )
            elif action is None:
                outcome = self._escalate(
                    case_state, differential, decision.confidence,
                    "no remaining action would meaningfully narrow the "
                    f"differential; {decision.reason}",
                    None,
                )
            elif not affordable:
                outcome = self._escalate(
                    case_state, differential, decision.confidence,
                    f"cost budget exhausted; the next informative step "
                    f"({action.target}) exceeds the remaining allowance",
                    action.target,
                )

        return {
            "decision": decision,
            "action": action,
            "outstanding": outstanding,
            "finished": outcome is not None,
            "outcome": outcome,
        }

    def _observe(self, state: ReasoningState) -> ReasoningState:
        action = state["action"]
        findings = list(state["environment"].respond(action))
        state["state"].record(
            action,
            findings,
            state["differential"],
            charge_uninformative=self.limits.unanswered_actions_still_cost,
        )
        return {}

    @staticmethod
    def _route(state: ReasoningState) -> str:
        return "finish" if state.get("finished") else "observe"

    # -- helpers shared with DiagnosticAgent -------------------------------

    @staticmethod
    def _escalate(
        state: CaseState, differential, confidence: float, reason: str,
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

    # -- graph -------------------------------------------------------------

    def compile(self):
        """Build the state graph. Imported lazily so the package works without
        LangGraph installed -- the hand-rolled loop stays the dependency-free
        path, and only this module needs the extra."""
        if self._compiled is not None:
            return self._compiled

        from langgraph.graph import END, StateGraph

        builder = StateGraph(ReasoningState)
        builder.add_node("propose", self._propose)
        builder.add_node("calibrate", self._calibrate)
        builder.add_node("decide", self._decide)
        builder.add_node("observe", self._observe)

        builder.set_entry_point("propose")
        builder.add_edge("propose", "calibrate")
        builder.add_edge("calibrate", "decide")
        builder.add_conditional_edges(
            "decide", self._route, {"observe": "observe", "finish": END}
        )
        # The cycle: an observation sends the loop back to re-propose over the
        # enlarged evidence set rather than incrementally patching the previous
        # differential, which keeps each turn's differential a function of all
        # findings so far and nothing else.
        builder.add_edge("observe", "propose")

        # recursion_limit is set per-invocation in run(); termination is
        # actually guaranteed by the same three conditions as the loop.
        self._compiled = builder.compile()
        return self._compiled

    def run(self, case: Case, environment: Environment | None = None) -> CaseOutcome:
        graph = self.compile()
        environment = environment or CaseOracle(case)
        state = CaseState(
            case_id=case.case_id,
            presenting_complaint=case.presenting_complaint,
            findings=case.initial(),
        )
        state.asked.update(case.initial_findings)

        initial: ReasoningState = {
            "case": case,
            "state": state,
            "environment": environment,
            "finished": False,
            "outcome": None,
        }
        # Four nodes per turn plus a margin, so LangGraph's own recursion guard
        # never fires before the loop's turn limit does. If it ever did, the
        # failure would look like a graph bug rather than an exhausted budget.
        final = graph.invoke(
            initial, {"recursion_limit": 4 * self.limits.max_turns + 10}
        )
        outcome = final.get("outcome")
        if outcome is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "graph terminated without an outcome; this should be "
                "unreachable given the loop's termination conditions"
            )
        return outcome


__all__ = ["GraphAgent", "ReasoningState"]
