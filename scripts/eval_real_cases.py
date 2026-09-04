#!/usr/bin/env python3
"""Run the agent against real patients, not invented ones.

    python scripts/eval_real_cases.py

A handful of cases, hand-extracted from open-access PMC case reports (see
``dxagent.datasets.real_cases`` for exactly which ones and how each finding
was read from the source text). This is a different, much smaller and much
more honest thing than the ten fixture cases: those were invented to
exercise the code; these are real, published patients the knowledge base
never saw.

Read this output as one data point per case, not a benchmark. A handful of
cases is not evidence of real-world accuracy in either direction -- it is
more real evidence than zero.

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
        "\nReal patients, not invented ones. Every finding below was "
        "read from the source case report by hand -- see "
        "dxagent/datasets/real_cases.py for the extraction and the citation. "
        "Each is one real data point, not a benchmark.\n"
    ))

    correct = 0
    escalation_reasons: list[tuple[str, str]] = []
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
            escalation_reasons.append((case.case_id, outcome.escalation.reason))
        top3 = ", ".join(
            f"{h.label.replace('_', ' ')} {h.probability:.0%}"
            for h in outcome.differential.hypotheses[:3]
        )
        print(f"  differential:   {top3}")

    print(f"\n{RULE}")
    print(f"  {correct} committed and correct, out of {len(REAL_CASES)} real cases.")
    print(wrap(
        f"  Not a benchmark result -- n={len(REAL_CASES)}, hand-picked for "
        "clarity, no clinician review of the extraction. Reported this "
        "small on purpose rather than not reported at all.",
        indent="  ",
    ))
    if escalation_reasons:
        print(wrap(
            "  This run uses uninformative_turns_still_count=False "
            "(LoopLimits), so an UNKNOWN answer -- common in a real record, "
            "never seen in a fixture -- no longer burns a turn for free. "
            "Escalations still happen, but for reasons that are now real "
            "rather than a turn-budget artefact:",
            indent="  ",
        ))
        for case_id, reason in escalation_reasons:
            print(wrap(f"  {case_id}: {reason}", indent="    "))
        print(wrap(
            "  Fixing the turn-accounting bug didn't make these cases "
            "commit -- it removed one artefact and let whatever real "
            "constraint sits underneath (cost, or the selector's own "
            "ranking under this knowledge base's mostly-invented "
            "likelihoods) show through instead. That is the honest result "
            "of this evaluation, not a failure of the fix.",
            indent="  ",
        ))
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
