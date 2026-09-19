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
        # Both budget flags are set for the same reason, and the second was
        # only believed after being measured: on these cases 98% and 78% of
        # the exhausted cost budget had gone on actions that returned
        # UNKNOWN, a CTPA charged at 20.0 against a report that never
        # mentions one among them. A finding the source never recorded means
        # no test was performed, so it costs neither a turn nor money. Both
        # default the other way, so nothing else in the project moves.
        limits=LoopLimits(
            uninformative_turns_still_count=False,
            unanswered_actions_still_cost=False,
        ),
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

    correct = wrong = 0
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
            wrong += int(not got_it)
            print(
                f"  system said:    {outcome.prediction.replace('_', ' ')} "
                f"({outcome.confidence:.0%})  "
                f"[{'CORRECT' if got_it else 'WRONG'}]"
            )
            for rival in outcome.unexamined:
                print(
                    f"  not examined:   {rival.label.replace('_', ' ')} -- "
                    f"{', '.join(humanise(c) for c in rival.present)} present; "
                    f"{', '.join(humanise(c) for c in rival.unexamined)} never asked"
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
    print(
        f"  {correct} committed and correct, {wrong} committed and WRONG, "
        f"{len(REAL_CASES) - correct - wrong} escalated, out of "
        f"{len(REAL_CASES)} real cases."
    )
    print(wrap(
        f"  Not a benchmark result -- n={len(REAL_CASES)}: ten hand-picked, "
        "eight chosen by a rule fixed before any was read, no clinician "
        "review of the extraction. Reported this small on purpose rather "
        "than not reported at all.",
        indent="  ",
    ))
    if escalation_reasons:
        print(wrap(
            "  Both LoopLimits budget flags are off for this run "
            "(uninformative_turns_still_count, unanswered_actions_still_cost), "
            "because a finding the source never recorded means no test was "
            "performed: it costs neither a turn nor money. Charging for it "
            "billed one case 20.0 for a CTPA its report never mentions, and "
            "98% of that case's exhausted budget went the same way. The "
            "escalations that remain are these:",
            indent="  ",
        ))
        for case_id, reason in escalation_reasons:
            print(wrap(f"  {case_id}: {reason}", indent="    "))
        print(wrap(
            "  Neither budget fix raised the number that commits. What they "
            "changed is what the escalations mean: no case now stops because "
            "it ran out of turns or money, so every remaining escalation is "
            "the agent having asked what it could and still being unable to "
            "separate the diagnoses on what these reports actually record. "
            "That is a statement about the evidence rather than about the "
            "harness, which is the whole reason for removing the artefacts.",
            indent="  ",
        ))
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
