#!/usr/bin/env python3
"""Generate, critique and screen synthetic cases (brief section 5.2).

    python scripts/synthesize.py --dry-run          # show the plan, call nothing
    python scripts/synthesize.py --per-disease 3    # needs ANTHROPIC_API_KEY
    python scripts/synthesize.py --out cases.json

Runs the recipe: seed from the knowledge base, generate vignettes, critique
them in a separate pass, target the measured near-miss pairs, then screen for
identifiers and duplicates. Prints the batch a clinician should spot-check.

The output is never an evaluation set. Synthetic cases are for coverage,
balance and stress-testing; scoring the system on cases generated from the
knowledge base it reasons over measures self-consistency. Every case written
here carries a ``synthetic-`` prefix so one reaching a metrics table is visible
in the per-case output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.synthesis import (  # noqa: E402
    CaseGenerator,
    confusable_pairs,
    review_sample,
    screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-disease", type=int, default=3)
    parser.add_argument("--near-miss-pairs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the surviving cases here")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be generated without calling a model",
    )
    parser.add_argument(
        "--provider",
        choices=("anthropic", "gemini"),
        default="anthropic",
        help=(
            "anthropic is the brief's mandated model (needs ANTHROPIC_API_KEY, "
            "paid). gemini is a free-tier substitute (needs GEMINI_API_KEY) -- "
            "using it is a deviation from the spec and belongs in the write-up."
        ),
    )
    args = parser.parse_args()

    kb = build_knowledge_base()
    pairs = confusable_pairs(kb, limit=args.near_miss_pairs)

    print(f"knowledge base: {len(kb.diseases())} conditions")
    print(f"\ntypical cases: {args.per_disease} per condition")
    print("near-miss targets, by measured likelihood-profile distance:")
    for first, second, distance in pairs:
        print(f"   {distance:.3f}  {first} presenting as {second}")

    planned = len(kb.diseases()) * args.per_disease + len(pairs) * args.per_disease
    print(f"\nwould generate {planned} cases before screening")

    if args.dry_run:
        print("\n--dry-run: nothing generated.")
        return 0

    env_var = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "GEMINI_API_KEY"
    if not os.environ.get(env_var):
        print(
            f"\n{env_var} is not set, so there is no model to generate with.\n"
            "Run with --dry-run to see the plan, or set the key.\n"
            "Nothing was written."
        )
        return 1

    if args.provider == "gemini":
        from dxagent.llm import GeminiLLM

        print(
            "\nusing Gemini, not the brief's mandated model -- record this "
            "substitution in the write-up."
        )
        llm = GeminiLLM()
    else:
        from dxagent.llm import AnthropicLLM

        llm = AnthropicLLM()

    generator = CaseGenerator(llm, kb, seed=args.seed)

    generated = []
    for entry in kb.diseases():
        generated.extend(generator.generate(entry.label, args.per_disease))
    for first, second, _ in pairs:
        generated.extend(
            generator.generate(first, args.per_disease, confusable_with=second)
        )

    if generator.failures:
        print(f"\n{len(generator.failures)} call(s) produced nothing usable:")
        for note in generator.failures[:10]:
            print(f"   {note}")
        if len(generator.failures) > 10:
            print(f"   ... and {len(generator.failures) - 10} more")

    if not generated:
        print(
            "\nNothing was generated. The failures above say why; an empty "
            "batch\nwith no failures listed means the model returned "
            "well-formed JSON\ncontaining no cases."
        )
        return 1

    print(f"\ngenerated {len(generated)}; critiquing...")
    critiqued = [generator.critique(case) for case in generated]

    kept, report = screen(critiqued)
    print("\nscreening")
    for key in (
        "generated",
        "rejected_by_critique",
        "rejected_as_duplicate",
        "rejected_for_phi",
        "kept",
    ):
        print(f"   {key:<24} {report[key]}")
    for case_id, hits in report["phi_detail"]:
        print(f"   PHI in {case_id}: {[kind for kind, _ in hits]}")

    leaking = report["narratives_leaking_vocabulary"]
    if leaking:
        print(
            f"\n{leaking} of {report['kept']} kept narratives write vocabulary "
            "identifiers\ninto the prose verbatim. Kept, because the findings "
            "lists are still\nvalid, but they do not read like clinical text."
        )

    print(
        "\nThe PHI screen matches structured identifiers only. It does not "
        "find\na name in running prose, and a clean report is not a clean "
        "corpus."
    )

    sample = review_sample(kept, size=min(10, len(kept)), seed=args.seed)
    print(f"\nfor clinician spot-check ({len(sample)} of {len(kept)}):")
    for case in sample:
        print(f"   {case.case_id}: {case.narrative[:70]}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "report": {k: v for k, v in report.items() if k != "phi_detail"},
                    "cases": [
                        {
                            "case_id": c.case_id,
                            "narrative": c.narrative,
                            "present": list(c.present),
                            "absent": list(c.absent),
                            "diagnosis": c.diagnosis,
                            "reasoning": c.reasoning,
                        }
                        for c in kept
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
