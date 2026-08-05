"""Evaluation harness.

Runs an agent over a case set, optionally fitting the calibrator on a held-out
split first. The split is not optional in spirit: thresholds tuned on the same
cases they are evaluated on will look excellent and mean nothing, and this is
the single easiest mistake to make in selective-prediction work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..agent import DiagnosticAgent
from ..belief import BayesianProposer
from ..environment import Case, CaseOracle, NoisyOracle
from ..schemas import CaseOutcome
from .metrics import (
    AbstentionMetrics,
    CalibrationMetrics,
    DifferentialMetrics,
    RankingMetrics,
    SelectiveMetrics,
    abstention_metrics,
    calibration_metrics,
    differential_metrics,
    format_report,
    ranking_metrics,
    risk_coverage_curve,
    selective_metrics,
)


@dataclass
class EvaluationResult:
    ranking: RankingMetrics
    calibration: CalibrationMetrics
    selective: SelectiveMetrics
    outcomes: list[CaseOutcome] = field(default_factory=list)
    truths: dict[str, str] = field(default_factory=dict)
    temperature: float = 1.0
    # Populated only when the case source carries gold differentials; the
    # fixtures label a single diagnosis, so these stay empty there and the
    # corresponding report sections are omitted rather than printed as zeros.
    differential: DifferentialMetrics | None = None
    abstention: AbstentionMetrics | None = None

    def report(self) -> str:
        header = f"temperature (fitted)          {self.temperature:.2f}"
        return header + "\n" + format_report(
            self.ranking,
            self.calibration,
            self.selective,
            self.differential,
            self.abstention,
        )

    def to_json(self, path: Path) -> None:
        """Persist metrics and per-case outcomes for later comparison."""
        payload = {
            "temperature": self.temperature,
            "ranking": asdict(self.ranking),
            "calibration": asdict(self.calibration),
            "selective": asdict(self.selective),
            "risk_coverage": risk_coverage_curve(self.outcomes, self.truths),
            "cases": [
                {
                    "case_id": o.case_id,
                    "truth": self.truths[o.case_id],
                    "verdict": o.verdict.value,
                    "prediction": o.prediction,
                    "confidence": round(o.differential.top.probability, 4),
                    "top3": list(o.differential.top_k(3)),
                    "turns": len(o.steps),
                    "cost": round(o.budget_spent, 2),
                    "actions": [s.action.target for s in o.steps],
                    "escalation_reason": o.escalation.reason if o.escalation else None,
                }
                for o in self.outcomes
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_cases(
    cases: list[Case], calibration_fraction: float = 0.3
) -> tuple[list[Case], list[Case]]:
    """Deterministic split by case id hash, so runs are reproducible.

    Stratification by diagnosis is not implemented and should be before any
    reported result: with small case sets an unstratified split can put every
    instance of a diagnosis on one side.
    """
    ordered = sorted(cases, key=lambda c: c.case_id)
    cut = max(1, int(len(ordered) * calibration_fraction))
    return ordered[:cut], ordered[cut:]


def run_agent(
    agent: DiagnosticAgent,
    cases: list[Case],
    noisy: bool = False,
    recall_failure: float = 0.15,
    seed: int = 0,
) -> list[CaseOutcome]:
    outcomes = []
    for index, case in enumerate(cases):
        environment = (
            NoisyOracle(case, recall_failure=recall_failure, seed=seed + index)
            if noisy
            else CaseOracle(case)
        )
        outcomes.append(agent.run(case, environment))
    return outcomes


def evaluate(
    agent: DiagnosticAgent,
    cases: list[Case],
    calibration_cases: list[Case] | None = None,
    noisy: bool = False,
    fit_calibration: bool = True,
) -> EvaluationResult:
    """Fit the calibrator on held-out cases, then evaluate.

    Calibration is fitted on the *initial* differential from the presenting
    findings alone rather than on the loop's terminal differential. That is a
    simplification: the temperature appropriate to a two-finding posterior is
    not necessarily right for a ten-finding one. Per-turn or per-evidence-count
    calibration is the correct treatment and is an open item.
    """
    if fit_calibration and calibration_cases:
        proposer = BayesianProposer(agent.kb)
        samples = [
            (proposer.propose(case.initial(), case.presenting_complaint), case.diagnosis)
            for case in calibration_cases
        ]
        agent.gate.scaler.fit(samples)

    outcomes = run_agent(agent, cases, noisy=noisy)
    truths = {case.case_id: case.diagnosis for case in cases}

    gold = {c.case_id: c.differential for c in cases if c.differential}
    ambiguity = {c.case_id: c.ambiguity for c in cases if c.differential}

    return EvaluationResult(
        ranking=ranking_metrics(outcomes, truths),
        calibration=calibration_metrics(outcomes, truths),
        selective=selective_metrics(outcomes, truths),
        outcomes=outcomes,
        truths=truths,
        temperature=agent.gate.scaler.temperature,
        differential=differential_metrics(outcomes, gold) if gold else None,
        abstention=abstention_metrics(outcomes, ambiguity) if ambiguity else None,
    )


__all__ = ["EvaluationResult", "evaluate", "run_agent", "split_cases"]
