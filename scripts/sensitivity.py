#!/usr/bin/env python3
"""Find which invented likelihoods actually change the answer.

    python scripts/sensitivity.py
    python scripts/sensitivity.py --delta 0.2 --top 25

The knowledge base holds 135 numbers and most of them are invented. Sourcing
all of them is weeks of work, and much of it would be wasted: a likelihood can
be badly wrong and change no diagnosis at all, because it is never decisive
for any case in the set.

This script perturbs each likelihood on its own and counts how many cases
change their top-1 diagnosis as a result. The output is a ranked shopping
list -- the numbers worth finding a citation for, in order -- which turns
"source the whole knowledge base" into "source these four".

Three things it is not, the third learned the hard way.

It is not a claim that the unlisted numbers are correct; they are unverified
either way, and a number that is not decisive on ten fixture cases may be
decisive on the eleventh. It measures sensitivity of *this* case set, so a
narrow set produces a short list for the wrong reason -- report the case count
alongside the list.

And it is emphatically not a claim that the unlisted numbers could be removed.
Perturbation moves one number a little while every other number stays where it
is; it says nothing about whether the number is needed. Run ``--ablate`` for
that question, and the answer is that deleting the invented likelihoods and
falling back to the knowledge base's own marginals takes the fixture set from
9/10 to 4/10. Individually insensitive, collectively load-bearing. An earlier
version of this script ended by calling the unlisted numbers "not currently
load-bearing", which was the wrong conclusion drawn from the right
measurement.

Uses single-pass proposal over the complete record rather than the full agent
loop: the question is which likelihood the *inference* depends on, and letting
evidence-gathering vary as well would confound that with which questions the
agent happens to ask.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import build_cases, build_knowledge_base  # noqa: E402
from dxagent.provenance import report as provenance_report  # noqa: E402
from dxagent.schemas import Finding, Polarity  # noqa: E402


def observed(case) -> list[Finding]:
    return [
        Finding(
            concept=concept,
            polarity=Polarity.PRESENT if present else Polarity.ABSENT,
        )
        for concept, present in case.features.items()
    ]


def top_labels(kb, cases) -> list[str]:
    proposer = BayesianProposer(kb)
    return [proposer.propose(observed(c), c.presenting_complaint).top.label for c in cases]


def perturbed(kb, label: str, concept: str, value: float):
    """A copy of the KB with one likelihood changed."""
    entry = kb.get(label)
    features = dict(entry.features)
    features[concept] = value
    clone = dataclasses.replace(kb, entries=dict(kb.entries))
    clone.entries[label] = dataclasses.replace(entry, features=features)
    clone._marginals.clear()
    return clone


def band_sweep(kb, cases, baseline) -> int:
    """Re-run at the edges of every recorded uncertainty band.

    A likelihood taken from a reference text is a range, not a point, and the
    question that matters is whether a conclusion survives the range. If it
    does, the uncertainty is quantified rather than merely admitted; if it does
    not, the sweep names the number the conclusion actually rests on, which is
    the one worth a better source.

    Only sourced likelihoods have bands, so with none recorded there is nothing
    to sweep -- and saying that is more useful than sweeping invented numbers
    across invented bands, which would produce a robustness figure with no
    content.
    """
    banded = [
        (entry.label, concept, source)
        for entry in kb.diseases()
        for concept, source in entry.sources.items()
        if source.band is not None
    ]
    if not banded:
        print(
            "\nNo likelihood carries an uncertainty band yet, so there is "
            "nothing to sweep.\nBands arrive with provenance: see "
            "dxagent.provenance.from_narrative, which\nrecords the range a "
            "reference-text phrase implies. Sweeping invented numbers\nacross "
            "invented bands would produce a robustness figure with no content."
        )
        return 0

    print(f"\nsweeping {len(banded)} banded likelihoods across their ranges\n")
    fragile = []
    for label, concept, source in banded:
        low, high = source.band
        changed = max(
            sum(
                1
                for before, after in zip(
                    baseline, top_labels(perturbed(kb, label, concept, edge), cases)
                )
                if before != after
            )
            for edge in (low, high)
        )
        if changed:
            fragile.append((changed, label, concept, low, high))

    fragile.sort(reverse=True)
    if not fragile:
        print(
            "Every conclusion survives every band. The uncertainty in the "
            "sourced\nlikelihoods does not change any diagnosis on this case "
            "set."
        )
    else:
        print(f"{'cases':>6}  {'disease':<30} {'finding':<26} band")
        print("-" * 80)
        for changed, label, concept, low, high in fragile:
            print(f"{changed:>6}  {label:<30} {concept:<26} {low:.2f}-{high:.2f}")
        print(
            f"\n{len(fragile)} conclusions depend on where inside its band a "
            f"number sits.\nThose need a measured frequency, not a converted "
            f"phrase."
        )
    return 0


def ablate(kb, cases) -> int:
    """Remove likelihoods entirely, rather than moving them a little.

    Perturbation asks "is this number's *value* load-bearing"; ablation asks
    "is the number needed at all". The knowledge base already backs off to its
    own marginal for any feature a disease does not characterise, so deleting
    an invented likelihood is a real option -- and a more honest one than
    inventing a value, since it states that nothing is claimed rather than
    claiming something unfounded.

    The measurement says not to. The invented numbers turn out to be
    individually insensitive and collectively essential, which is why the
    perturbation list must not be read as a list of the only ones that matter.
    """
    from dxagent.agent import DiagnosticAgent

    def accuracy(knowledge) -> int:
        agent = DiagnosticAgent(kb=knowledge)
        return sum(
            agent.run(case).differential.top.label == case.diagnosis
            for case in cases
        )

    def without(predicate) -> tuple:
        trimmed = dataclasses.replace(kb, entries=dict(kb.entries))
        removed = 0
        for label, entry in list(trimmed.entries.items()):
            kept = {c: v for c, v in entry.features.items() if not predicate(entry, c)}
            removed += len(entry.features) - len(kept)
            trimmed.entries[label] = dataclasses.replace(entry, features=kept)
        trimmed._marginals.clear()
        return trimmed, removed

    print(f"\n{len(cases)} cases | ablation: delete, do not perturb\n")
    print(f"{'knowledge base':<44} {'correct':>9}")
    print("-" * 55)
    print(f"{'complete':<44} {accuracy(kb):>6}/{len(cases)}")

    trimmed, removed = without(lambda e, c: c not in e.sources)
    print(f"{f'{removed} invented deleted -> marginal backoff':<44} "
          f"{accuracy(trimmed):>6}/{len(cases)}")

    trimmed, removed = without(lambda e, c: c in e.sources)
    print(f"{f'{removed} sourced deleted, invented kept':<44} "
          f"{accuracy(trimmed):>6}/{len(cases)}")

    print(
        "\nThe invented numbers cannot simply be dropped: they carry most of\n"
        "the discriminating structure, even though perturbing any one of them\n"
        "individually changes nothing. That is the honest description of this\n"
        "knowledge base -- not 'four numbers matter and the rest are padding'."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.15,
        help="how far to move each likelihood, up and down (default 0.15)",
    )
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    parser.add_argument(
        "--correlated",
        action="store_true",
        help="use the correlation-aware knowledge base",
    )
    parser.add_argument(
        "--bands",
        action="store_true",
        help="sweep sourced likelihoods across their recorded uncertainty bands",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="remove likelihoods entirely and measure the cost, rather than "
        "moving them a little",
    )
    args = parser.parse_args()

    kb = build_knowledge_base(correlated=args.correlated)
    cases = build_cases()
    baseline = top_labels(kb, cases)

    coverage = provenance_report(kb)
    print(coverage.summary())

    if args.ablate:
        return ablate(kb, cases)
    if args.bands:
        return band_sweep(kb, cases, baseline)

    parameters = [
        (entry.label, concept)
        for entry in kb.diseases()
        for concept in entry.features
    ]
    print(
        f"{len(parameters)} likelihoods | {len(cases)} cases | "
        f"perturbation +/-{args.delta}"
    )

    results: list[tuple[int, str, str, float, str]] = []
    for label, concept in parameters:
        current = kb.get(label).features[concept]
        worst = 0
        worst_note = ""
        for direction in (-1, 1):
            moved = min(max(current + direction * args.delta, 0.01), 0.99)
            if moved == current:
                continue
            changed = sum(
                1
                for before, after in zip(
                    baseline, top_labels(perturbed(kb, label, concept, moved), cases)
                )
                if before != after
            )
            if changed > worst:
                worst = changed
                worst_note = f"{current:.2f} -> {moved:.2f}"
        if worst:
            results.append((worst, label, concept, current, worst_note))

    results.sort(key=lambda row: (-row[0], row[1], row[2]))

    print()
    if not results:
        print(
            "No single likelihood changes any diagnosis at this perturbation.\n"
            "Either the case set is too small to be discriminating, or --delta\n"
            "is smaller than the margins involved. Try a larger delta before\n"
            "concluding the knowledge base is robust."
        )
        return 0

    print(f"{'cases':>6}  {'disease':<30} {'finding':<30} change")
    print("-" * 84)
    for changed, label, concept, _, note in results[: args.top]:
        print(f"{changed:>6}  {label:<30} {concept:<30} {note}")

    print(
        f"\n{len(results)} of {len(parameters)} likelihoods change at least one "
        f"diagnosis when perturbed by {args.delta:+.2f}.\nSource those first."
    )
    print(
        "\nThis measures LOCAL sensitivity -- moving one number a little, with\n"
        "every other number held where it is. It does not say the rest are\n"
        "dispensable, and reading it that way is a mistake this script used to\n"
        "invite: deleting every likelihood outside this list and letting the\n"
        "knowledge base fall back to its marginals drops the fixture set from\n"
        "9/10 to 4/10. They are individually insensitive and collectively\n"
        "load-bearing, which are different properties. Run --ablate to measure\n"
        "the second one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
