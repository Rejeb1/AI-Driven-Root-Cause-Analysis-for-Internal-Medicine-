#!/usr/bin/env python3
"""A live consultation: you play the patient, typed answers, not a fixture.

    python scripts/consult.py

Every other script in this project reasons over a pre-written case -- a
dictionary of true/false answers written in advance. This one asks *you* live,
the same way ``CaseOracle`` answers from a dictionary, except the dictionary
is your keyboard. Same agent, same gate, same knowledge base as
``demo.py``; the only new code is ``InteractiveOracle``, which answers
``Environment.respond()`` by prompting instead of by lookup.

There is no gold diagnosis for a patient you invent, so this prints no
CORRECT/WRONG verdict -- there is nothing to grade against. What it shows is
the same thing demo.py shows: what it asked, why, and what it concluded.

Not for clinical use. The knowledge base is invented; see fixtures.py.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.environment import Case, Environment  # noqa: E402
from dxagent.schemas import Action, Finding, Polarity  # noqa: E402

# Reuse demo.py's formatting rather than re-implement it -- two copies of the
# same "how to print a finding" logic is exactly how the run_eval.py baseline
# mislabelling happened, and that is not a mistake worth repeating on purpose.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import RULE, THIN, humanise, wrap  # noqa: E402


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


@dataclass
class InteractiveOracle:
    """Answers ``Environment.respond()`` by asking a live human, not a lookup.

    The only genuinely new piece here. Everything downstream of this class
    (the agent, the gate, the selector, the printing) is exactly what
    demo.py already runs and tests already cover -- this class is the sole
    new surface, kept as small as the ``Environment`` protocol allows.
    """

    def respond(self, action: Action) -> list[Finding]:
        if not action.target:
            return []
        label = humanise(action.target)
        while True:
            try:
                raw = input(f"  {label}? [y/n/u=unknown] ").strip().lower()
            except EOFError:
                raw = "u"
            if raw in ("y", "yes", "present", "p"):
                polarity = Polarity.PRESENT
                break
            if raw in ("n", "no", "absent", "a"):
                polarity = Polarity.ABSENT
                break
            if raw in ("u", "unknown", ""):
                polarity = Polarity.UNKNOWN
                break
            print("    (answer y, n, or u)")
        # Echoed explicitly rather than relying on terminal echo of the typed
        # answer: a piped or captured transcript (this file's own test run,
        # or someone recording the session) is otherwise unreadable -- every
        # prompt runs together on one line with no visible answer between them.
        print(f"    -> {polarity.value}")
        return [
            Finding(
                concept=action.target,
                polarity=polarity,
                provenance=f"{action.kind.value} (user-reported)",
            )
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument(
        "--no-workup",
        action="store_true",
        help="disable the mandatory guideline workup",
    )
    args = parser.parse_args()
    _force_utf8_stdout()

    print(RULE)
    print("LIVE CONSULTATION -- you are the patient")
    print(RULE)
    print(wrap(
        "\nAnswer each question the system asks. 'y' = present, 'n' = "
        "absent, 'u' = don't know / not tested. This patient has no gold "
        "diagnosis -- you're making them up as you go -- so there's no "
        "CORRECT/WRONG at the end, only what the system concluded and why.\n"
    ))
    try:
        complaint = input("What brings the patient in today? ").strip()
    except EOFError:
        complaint = ""
    print()
    if not complaint:
        complaint = "unspecified presentation"
    print(wrap(f"The patient reports:\n\n{complaint}"))

    case = Case(
        case_id="live-consultation",
        presenting_complaint=complaint,
        features={},
        diagnosis="unknown",
        initial_findings=(),
    )

    kb = build_knowledge_base(correlated=True)
    agent = DiagnosticAgent(
        kb=kb,
        proposer=BayesianProposer(kb),
        gate=AbstentionGate(kb=kb, min_confidence=args.min_confidence),
        limits=LoopLimits(require_workup=not args.no_workup),
    )

    print(f"\n{THIN}\nCONSULTATION\n{THIN}\n")
    outcome = agent.run(case, environment=InteractiveOracle())

    print(f"\n{THIN}\nWHAT IT CONCLUDED\n{THIN}\n")
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
        print(wrap("ESCALATED rather than committing."))
        print(wrap(f"Reason: {outcome.escalation.reason}"))
        if outcome.escalation.unresolved_question:
            print(wrap("Unresolved: " + humanise(outcome.escalation.unresolved_question)))

    print(f"\n{THIN}")
    print("  No gold diagnosis: you invented this patient, so there's nothing")
    print("  to grade the answer against -- only whether the reasoning above")
    print("  looks right to you.")
    print(THIN)
    print("\n  Not for clinical use: the knowledge base is invented.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
