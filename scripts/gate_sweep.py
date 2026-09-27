#!/usr/bin/env python3
"""Show what the gate's invented thresholds trade, without choosing them.

    python scripts/gate_sweep.py

The abstention gate commits only above ``min_confidence``, with at least
``min_margin`` over the runner-up and no time-critical rival above
``red_flag_tolerance``. All three are invented. They cannot be sourced --
how cautious a triage tool should be is a clinical policy, not a frequency
-- and picking them from this sweep would be fitting the gate to twenty
cases. So this script only reports: for each value, how many cases commit,
how many of those are wrong, and how many escalate, on every case set. The
table is for a clinician to read, not for the code to act on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_cases, build_knowledge_base  # noqa: E402
from dxagent.datasets.fixtures import build_hard_cases  # noqa: E402

REAL_LIMITS = LoopLimits(
    uninformative_turns_still_count=False, unanswered_actions_still_cost=False
)


def score(kb, gate_kwargs, cases, limits=None) -> str:
    agent = DiagnosticAgent(
        kb=kb,
        proposer=BayesianProposer(kb),
        gate=AbstentionGate(kb=kb, **gate_kwargs),
        **({"limits": limits} if limits else {}),
    )
    correct = wrong = escalated = 0
    for case in cases:
        outcome = agent.run(case)
        if outcome.verdict is Verdict.COMMITTED:
            if outcome.prediction == case.diagnosis:
                correct += 1
            else:
                wrong += 1
        else:
            escalated += 1
    return f"{correct:>2}/{wrong}/{escalated:<2}"


def sweep(name: str, values: list[float], shipped: float) -> None:
    kb = build_knowledge_base(correlated=True)
    fixtures, hard = build_cases(), build_hard_cases()
    print(f"\n{name}  (shipped {shipped})     correct/wrong/escalated")
    print(f"  {'value':>6}   {f'real ({len(REAL_CASES)})':>10}   {'fixtures':>9}   {'hard':>6}")
    for value in values:
        kwargs = {name: value}
        mark = "*" if value == shipped else " "
        print(
            f" {mark}{value:>6.2f}   {score(kb, kwargs, REAL_CASES, REAL_LIMITS):>10}"
            f"   {score(kb, kwargs, fixtures):>9}   {score(kb, kwargs, hard):>6}"
        )


def main() -> int:
    sweep("min_confidence", [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80], 0.65)
    sweep("min_margin", [0.05, 0.10, 0.15, 0.20, 0.25], 0.15)
    sweep("red_flag_tolerance", [0.05, 0.10, 0.15, 0.20], 0.10)
    print(
        "\nReported, not chosen from: each threshold is a clinical policy, "
        "and\nthe shipped values stay until a clinician picks different ones."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
