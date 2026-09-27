#!/usr/bin/env python3
"""A language model as a single-pass proposer on the real cases.

    python scripts/llm_real_cases.py                    # Gemini, 3 runs per case
    python scripts/llm_real_cases.py --repeats 2 --no-complaint

Each case's complete extracted record goes to the model in one prompt, and
its top-ranked diagnosis is scored against the confirmed one; the Bayesian
proposer is scored on the same record for comparison. Two things this
script is careful about, because each would inflate the model's number
silently:

- **Fallbacks are counted, not scored.** ``LLMProposer`` returns the
  Bayesian posterior when the model's answer cannot be parsed. Scoring that
  as the model's answer would credit it with Bayes' results.
- **``--no-complaint`` isolates the findings.** The model also reads the
  free-text presenting complaint, which the Bayesian proposer ignores, so
  the default comparison is not like for like. Run both.

Needs ``GEMINI_API_KEY`` (free tier) or ``--provider`` with its own
settings. Any model here is a documented substitute for the brief's
mandated one, and the published case reports may be in its training data:
it sees only the extracted findings, never the report or its identifier,
which reduces that risk without removing it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.baselines import _observe_everything  # noqa: E402
from dxagent.belief import BayesianProposer, LLMProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_knowledge_base  # noqa: E402
from dxagent.llm import from_provider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-complaint", action="store_true")
    args = parser.parse_args()

    kb = build_knowledge_base(correlated=True)
    bayes = BayesianProposer(kb)
    llm = from_provider(args.provider, args.model)
    model = LLMProposer(kb=kb, llm=llm)
    print(f"{args.provider}:{getattr(llm, 'model', '?')}, {args.repeats} runs per case, "
          f"complaint {'withheld' if args.no_complaint else 'included'}\n")

    totals: Counter = Counter()
    for case in REAL_CASES:
        findings = _observe_everything(case)
        bayes_top = bayes.propose(findings, case.presenting_complaint).top.label
        tops = []
        for _ in range(args.repeats):
            complaint = "" if args.no_complaint else case.presenting_complaint
            differential = model.propose(findings, complaint)
            tops.append("FALLBACK" if model.last_was_fallback else differential.top.label)
        correct = sum(t == case.diagnosis for t in tops)
        totals["correct"] += correct
        totals["fallback"] += tops.count("FALLBACK")
        totals["stable"] += len(set(tops)) == 1
        totals["bayes"] += bayes_top == case.diagnosis
        print(
            f"{case.case_id:<14} {case.diagnosis:<29} "
            f"bayes {'ok  ' if bayes_top == case.diagnosis else 'MISS'}  "
            f"model {correct}/{args.repeats}  {dict(Counter(tops))}"
        )

    runs = len(REAL_CASES) * args.repeats
    print(
        f"\nmodel top-1 {totals['correct']}/{runs} runs ({totals['correct'] / runs:.1%}), "
        f"{totals['fallback']} fallbacks, same answer every run on "
        f"{totals['stable']}/{len(REAL_CASES)} cases"
    )
    print(f"Bayesian top-1 on the same record {totals['bayes']}/{len(REAL_CASES)}")
    print(
        "\nTop-1 on a complete record only: no loop, no abstention, no calibration.\n"
        "A system that must commit on every case is not comparable to one that\n"
        "may escalate, and n=25 settles nothing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
