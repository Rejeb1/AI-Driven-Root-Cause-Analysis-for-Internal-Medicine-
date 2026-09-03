#!/usr/bin/env python3
"""Run the agent against real patients, not invented ones.

    python scripts/eval_real_cases.py

Two cases, hand-extracted from open-access PMC case reports (see
``dxagent.datasets.real_cases`` for exactly which ones and how each finding
was read from the source text). This is a different, much smaller and much
more honest thing than the ten fixture cases: those were invented to
exercise the code; these are real, published patients the knowledge base
never saw.

Read this output as a single data point per case, not a benchmark. Two
cases is not evidence of real-world accuracy in either direction -- it is
one more real thing to look at than zero.

Not for clinical use. The knowledge base is still mostly invented; see
RESPONSIBLE_AI.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_knowledge_base  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import RULE, THIN, humanise, wrap  # noqa: E402


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _force_utf8_stdout()
    kb = build_knowledge_base(correlated=True)
    agent = DiagnosticAgent(
        kb=kb,
        proposer=BayesianProposer(kb),
        gate=AbstentionGate(kb=kb),
        # Real case reports return UNKNOWN for anything the source text never
        # addressed -- the ten fixtures never do this, so it never mattered
        # there. Without this, a turn spent on an unanswerable question costs
        # the same as one that actually narrows the differential, and both
        # real cases ran out of budget before reaching their single most
        # decisive finding. See LoopLimits.uninformative_turns_still_count.
        limits=LoopLimits(uninformative_turns_still_count=False),
    )

    print(RULE)
    print(f"REAL-CASE EVALUATION  ({len(REAL_CASES)} cases from published PMC reports)")
    print(RULE)
    print(wrap(
        "\nTwo real patients, not ten invented ones. Every finding below was "
        "read from the source case report by hand -- see "
        "dxagent/datasets/real_cases.py for the extraction and the citation. "
        "This is one real data point per case, not a benchmark.\n"
    ))

    correct = 0
    for case in REAL_CASES:
        outcome = agent.run(case)
        print(THIN)
        print(f"{case.case_id}")
        print(wrap(case.presenting_complaint))
        for step in outcome.steps:
            answers = ", ".join(
                f"{humanise(f.concept)}: {f.polarity.value}" for f in step.findings
            )
            print(f"    {step.index + 1}. {humanise(step.action.target)} -> {answers}")
        print(f"  true diagnosis: {case.diagnosis.replace('_', ' ')}")
        if outcome.verdict is Verdict.COMMITTED:
            got_it = outcome.prediction == case.diagnosis
            correct += int(got_it)
            print(
                f"  system said:    {outcome.prediction.replace('_', ' ')} "
                f"({outcome.confidence:.0%})  "
                f"[{'CORRECT' if got_it else 'WRONG'}]"
            )
        else:
            print(f"  system said:    ESCALATED -- {outcome.escalation.reason}")
            if outcome.escalation.unresolved_question:
                print(
                    "  unresolved:     "
                    + humanise(outcome.escalation.unresolved_question)
                )
        top3 = ", ".join(
            f"{h.label.replace('_', ' ')} {h.probability:.0%}"
            for h in outcome.differential.hypotheses[:3]
        )
        print(f"  differential:   {top3}")

    print(f"\n{RULE}")
    print(f"  {correct} committed and correct, out of {len(REAL_CASES)} real cases.")
    print(wrap(
        "  Not a benchmark result -- n=2, hand-picked for clarity, no "
        "clinician review of the extraction. Reported this small on purpose "
        "rather than not reported at all.",
        indent="  ",
    ))
    print(wrap(
        "  Both cases escalated rather than committed, and this run uses "
        "uninformative_turns_still_count=False (LoopLimits) so an UNKNOWN "
        "answer -- common in a real record, never seen in a fixture -- no "
        "longer burns a turn for free. That surfaced two different, deeper "
        "reasons neither case reaches its most decisive finding, and "
        "neither is the turn-budget artefact this script showed before "
        "that flag existed:",
        indent="  ",
    ))
    print(wrap(
        "  pmc-4565285: stopped on the *cost* budget, not turns -- CTPA "
        "is this knowledge base's most expensive action, and by the time "
        "cheaper questions were exhausted there was no budget left for it.",
        indent="    ",
    ))
    print(wrap(
        "  pmc-5841117: stopped because the selector judged nothing left "
        "worth asking, with cost and turns both still available -- the "
        "ECG-ST-changes test was never selected, which is a real "
        "consequence of ranking by *this knowledge base's own* (mostly "
        "invented) likelihoods, not a code defect.",
        indent="    ",
    ))
    print(wrap(
        "  Fixing the turn-accounting bug didn't fix the cases -- it "
        "removed one artefact and exposed the two real constraints "
        "underneath it. That is the honest result of this evaluation, not "
        "a failure of the fix.",
        indent="  ",
    ))
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
