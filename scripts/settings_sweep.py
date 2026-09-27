#!/usr/bin/env python3
"""Show what the loop's invented budget and cost settings trade.

    python scripts/settings_sweep.py

Companion to ``gate_sweep.py``. The loop stops at ``max_turns`` questions or
``max_cost`` cost units, and every question and test has an invented cost
(``fixtures.COSTS``: history 0.2, bedside 0.5-2, bloods 4-5, imaging 8-20).
None of these can be sourced -- they encode how much investigation is
acceptable, which is a service-design choice. Correlation weights, the other
invented group, were swept in an earlier pass (see WRITEUP.md). This script
reports outcomes per setting on every case set and chooses nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dxagent.knowledge as knowledge  # noqa: E402
from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_cases, build_knowledge_base  # noqa: E402
from dxagent.datasets.fixtures import COSTS, build_hard_cases  # noqa: E402

REAL_FLAGS = dict(uninformative_turns_still_count=False, unanswered_actions_still_cost=False)


def score(kb, cases, **limits) -> str:
    agent = DiagnosticAgent(
        kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb),
        limits=LoopLimits(**limits),
    )
    correct = wrong = escalated = 0
    spent = 0.0
    for case in cases:
        outcome = agent.run(case)
        spent += outcome.budget_spent
        if outcome.verdict is Verdict.COMMITTED:
            correct += outcome.prediction == case.diagnosis
            wrong += outcome.prediction != case.diagnosis
        else:
            escalated += 1
    return f"{correct:>2}/{wrong}/{escalated:<2} cost {spent / len(cases):>4.1f}"


def row(label: str, kb, **limits) -> None:
    print(
        f"  {label:<26} {score(kb, REAL_CASES, **REAL_FLAGS, **limits):>19}"
        f"   {score(kb, build_cases(), **limits):>19}"
        f"   {score(kb, build_hard_cases(), **limits):>19}"
    )


def header(title: str) -> None:
    print(f"\n{title}          correct/wrong/escalated, mean cost")
    print(f"  {'':<26} {f'real ({len(REAL_CASES)})':>19}   {'fixtures (10)':>19}   {'hard (5)':>19}")


def scaled_costs(kb_factory, factor: float, prefixes: tuple[str, ...]):
    kb = kb_factory()
    knowledge._COSTS.update(
        {k: v * factor for k, v in COSTS.items() if k.startswith(prefixes)}
    )
    return kb


def main() -> int:
    factory = lambda: build_knowledge_base(correlated=True)  # noqa: E731

    header("max_cost (shipped 40)")
    for value in (20.0, 30.0, 40.0, 60.0, 80.0):
        row(f"{'*' if value == 40.0 else ' '}max_cost {value:.0f}", factory(), max_cost=value)

    header("max_turns (shipped 12)")
    for value in (6, 8, 12, 16, 24):
        row(f"{'*' if value == 12 else ' '}max_turns {value}", factory(), max_turns=value)

    header("relative cost of labs and imaging (history and bedside fixed)")
    for factor in (0.5, 1.0, 2.0):
        kb = scaled_costs(factory, factor, ("lab:", "imaging:"))
        row(f"{'*' if factor == 1.0 else ' '}labs+imaging x{factor}", kb)
    knowledge._COSTS.update(COSTS)

    print(
        "\nReported, not chosen from: how much investigation is acceptable is "
        "a\nservice-design decision, and the shipped values stay until someone "
        "makes it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
