#!/usr/bin/env python3
"""A readable consultation transcript, for showing the system to a person.

    python scripts/demo.py                 # a case the system gets right
    python scripts/demo.py --case fx-009   # a case it gets wrong
    python scripts/demo.py --list          # what is available
    python scripts/demo.py --grounded      # cite retrieved guideline text

``run_eval.py`` prints metrics, which answer "how good is it" for someone who
already knows what it does. This prints one consultation as prose: what the
patient said, what the system asked and why, what it concluded, and what
argues against that conclusion. It is the artefact for a room, not a terminal.

Showing a failure is a feature, not an omission -- ``--case fx-009`` walks
through a pulmonary embolism the system confidently calls pneumonia, which is
a more honest demonstration than a cherry-picked success and is the case the
project's findings are built on.

Not for clinical use. The knowledge base is invented; see fixtures.py.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import build_cases, build_knowledge_base  # noqa: E402

def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and mangle anything outside it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


RULE = "=" * 74
THIN = "-" * 74


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        text, width=74, initial_indent=indent, subsequent_indent=indent
    )


def humanise(concept: str) -> str:
    """`exam:raised_jvp` -> `raised jvp (examination)`."""
    kind, _, rest = concept.partition(":")
    label = (rest or kind).replace("_", " ")
    prefix = {"exam": "examination", "lab": "blood test", "imaging": "imaging"}
    return f"{label} ({prefix[kind]})" if rest and kind in prefix else label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="fx-002", help="case id to run")
    parser.add_argument("--list", action="store_true", help="list available cases")
    parser.add_argument("--grounded", action="store_true", help="retrieve citations")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument(
        "--no-workup",
        action="store_true",
        help="disable the guideline workup (see fx-009: it costs that case)",
    )
    args = parser.parse_args()
    _force_utf8_stdout()

    kb = build_knowledge_base(correlated=True)
    cases = {c.case_id: c for c in build_cases()}

    if args.list:
        print("available cases\n")
        for case_id, case in cases.items():
            print(f"  {case_id}   {case.diagnosis:<30} {case.presenting_complaint}")
        return 0

    if args.case not in cases:
        print(f"no case {args.case!r}. Try --list.")
        return 1
    case = cases[args.case]

    proposer = BayesianProposer(kb)
    if args.grounded:
        try:
            from dxagent.retrieval import GroundedProposer, GuidelineIndex

            print("building the guideline index (first run downloads BGE-M3)...\n")
            proposer = GroundedProposer(proposer, GuidelineIndex().build())
        except ImportError:
            print("retrieval extras not installed; continuing without them\n")

    agent = DiagnosticAgent(
        kb=kb,
        proposer=proposer,
        gate=AbstentionGate(kb=kb, min_confidence=args.min_confidence),
        limits=LoopLimits(require_workup=not args.no_workup),
    )
    outcome = agent.run(case)

    print(RULE)
    print(f"CONSULTATION  {case.case_id}")
    print(RULE)
    print(f"\nThe patient reports:\n")
    print(wrap(case.presenting_complaint))

    print(f"\n{THIN}\nWHAT THE SYSTEM ASKED, AND WHY\n{THIN}\n")
    if not outcome.steps:
        print("    Nothing. It committed on the presenting complaint alone.")
    for step in outcome.steps:
        answers = ", ".join(
            f"{humanise(f.concept)}: {f.polarity.value}" for f in step.findings
        )
        print(f"  {step.index + 1}. {humanise(step.action.target)}")
        print(wrap(f"why: {step.action.rationale}", indent="        "))
        print(wrap(f"answer: {answers}", indent="        "))
        print()

    print(f"{THIN}\nWHAT IT CONCLUDED\n{THIN}\n")
    for rank, hypothesis in enumerate(outcome.differential.top_k(4), start=1):
        entry = outcome.differential.hypotheses[rank - 1]
        print(f"  {rank}. {entry.label.replace('_', ' ')}  —  {entry.probability:.0%}")

        supporting = [c for c in entry.support if c.snippet]
        against = list(entry.against)
        if supporting:
            print("       supporting:")
            for citation in supporting[:3]:
                print(wrap(f"[{citation.source_id}] {citation.snippet}", "         "))
        else:
            print("       supporting: no finding here argues for this")
        if against:
            print("       against:")
            for citation in against[:3]:
                print(wrap(f"[{citation.source_id}] {citation.snippet}", "         "))
        if not against:
            print("       against: nothing here argues against it, which is weaker")
            print("                than it sounds -- a diagnosis claiming few")
            print("                findings strongly is hard to argue against")
        # Probabilities normalise across the eight causes, so a diagnosis can
        # rise because its rivals fell rather than because anything argued
        # for it. Printing the thin supporting list without saying so makes a
        # confident number look unjustified when the justification is simply
        # elsewhere.
        if len(supporting) < 2 and entry.probability >= 0.4:
            ruled = [
                (other.label, c)
                for other in outcome.differential.hypotheses
                if other.label != entry.label
                for c in other.against
                if c.snippet
            ]
            if ruled:
                print("       reached mainly by elimination; what ruled out the rivals:")
                for label, citation in ruled[:3]:
                    print(wrap(
                        f"[{citation.source_id}] against {label.replace('_', ' ')}: "
                        f"{citation.snippet}",
                        "         ",
                    ))
        # The entry's own citation, kept apart from the evidence. For this
        # knowledge base it reads "synthetic entry, not sourced", and it used
        # to be printed at the head of the supporting list.
        for citation in entry.grounding:
            if citation.snippet:
                print(wrap(
                    f"provenance: [{citation.source_id}] {citation.snippet}",
                    "       ",
                ))
        print()

    print(f"{THIN}\nTHE DECISION\n{THIN}\n")
    if outcome.verdict is Verdict.COMMITTED:
        print(wrap(f"COMMITTED to {outcome.prediction.replace('_', ' ')}."))
        print(wrap(f"Confidence {outcome.confidence:.0%}. Cost {outcome.budget_spent:.1f}."))
    else:
        print(wrap("ESCALATED to a clinician rather than committing."))
        print(wrap(f"Reason: {outcome.escalation.reason}"))
        if outcome.escalation.unresolved_question:
            print(
                wrap(
                    "Unresolved: "
                    + humanise(outcome.escalation.unresolved_question)
                )
            )

    correct = outcome.differential.top.label == case.diagnosis
    print(f"\n{THIN}")
    print(f"  true diagnosis: {case.diagnosis.replace('_', ' ')}")
    print(f"  system said:    {outcome.differential.top.label.replace('_', ' ')}")
    print(f"  verdict:        {'CORRECT' if correct else 'WRONG'}")
    if not correct:
        print(
            wrap(
                "This failure is documented in DESIGN.md. The guideline workup "
                "forces the D-dimer to the first turn, which displaces the chest "
                "X-ray the loop would otherwise have ordered early; that X-ray "
                "comes back negative and is what undermines the wrong diagnosis. "
                "Without the workup this case is correct. Run with "
                "--no-workup to see it.",
                indent="  ",
            )
        )
    print(THIN)
    print("\n  Not for clinical use: the knowledge base is invented.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
