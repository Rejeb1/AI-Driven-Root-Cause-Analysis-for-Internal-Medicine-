#!/usr/bin/env python3
"""What the escalations would have become had the missing tests been done.

    python scripts/counterfactual_workup.py            # original 25
    python scripts/counterfactual_workup.py --holdout  # holdout 28

Zero wrong commits on the real cases is a measured fact about the records
as they are. Most escalations on those records stop because a test was
never reported -- a troponin, an ECG, a D-dimer -- so the zero is partly a
property of missing data: a loop that is never told the troponin cannot be
wrong about what it means. This script asks the question the zero leaves
open: if those tests had been done, how often would the same loop and gate
have committed, and to what?

For each escalated case, the tests are the escalation's own candidates
(``dxagent.resolution``: missing workup items, each live diagnosis's
confirming finding, then the most informative of the rest), the first
three. Every combination of their results is enumerated -- a disjoint
outcome tree -- and each leaf is weighted by how likely those results are
**in a patient who has the case's true diagnosis**, read from that
diagnosis's knowledge-base likelihoods. The same commit test the loop uses
decides each leaf: gate passes and no triggered workup is outstanding.

Two caveats, stated where the numbers are. The weights come from the
knowledge base, 85 of whose 177 likelihoods are invented, so this measures
the model against itself as much as against medicine. And it fills at most
three tests and never asks a fourth, so "still escalated" includes leaves a
longer workup might have resolved either way.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_knowledge_base  # noqa: E402
from dxagent.datasets.holdout_cases import HOLDOUT_CASES  # noqa: E402
from dxagent.gate import real_case_scaler  # noqa: E402
from dxagent.resolution import _candidates, _commits  # noqa: E402
from dxagent.schemas import Finding, Polarity  # noqa: E402

TESTS = 3


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = HOLDOUT_CASES if "--holdout" in sys.argv else REAL_CASES
    kb = build_knowledge_base(correlated=True)
    gate = AbstentionGate(kb=kb, scaler=real_case_scaler())
    agent = DiagnosticAgent(
        kb=kb,
        proposer=BayesianProposer(kb),
        gate=gate,
        limits=LoopLimits(
            uninformative_turns_still_count=False, unanswered_actions_still_cost=False
        ),
    )
    proposer = BayesianProposer(kb)
    total_right = total_wrong = total_still = 0.0
    escalated = 0
    print(f"{'case':14} {'true diagnosis':28} right  wrong  still  tests")
    for case in cases:
        outcome = agent.run(case)
        if not outcome.abstained:
            continue
        escalated += 1
        findings = list(outcome.escalation.findings)
        truth = kb.get(case.diagnosis)
        tests = _candidates(kb, findings, outcome.differential)[:TESTS]
        right = wrong = still = 0.0
        for results in itertools.product((False, True), repeat=len(tests)):
            weight = 1.0
            for concept, present in zip(tests, results):
                p = truth.features.get(concept, kb.backoff(concept))
                weight *= p if present else 1.0 - p
            hypothetical = [f for f in findings if f.concept not in tests] + [
                Finding(
                    concept=c,
                    polarity=Polarity.PRESENT if present else Polarity.ABSENT,
                    provenance="counterfactual",
                )
                for c, present in zip(tests, results)
            ]
            committed = _commits(proposer, gate, hypothetical, case.presenting_complaint)
            if committed is None:
                still += weight
            elif committed[0] == case.diagnosis:
                right += weight
            else:
                wrong += weight
        total_right += right
        total_wrong += wrong
        total_still += still
        print(
            f"{case.case_id:14} {case.diagnosis:28} {right:5.0%}  {wrong:5.0%}  "
            f"{still:5.0%}  {', '.join(tests)}"
        )
    print(
        f"\n{escalated} escalations. Had the three tests been done, expected "
        f"{total_right:.1f} more correct commits, {total_wrong:.1f} wrong "
        f"commits, {total_still:.1f} still escalated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
