#!/usr/bin/env python3
"""Turn fixture concepts into measured likelihoods, using DDXPlus as the source.

    python scripts/map_concepts.py --search cough      # find the right code
    python scripts/map_concepts.py --status            # what is mapped so far
    python scripts/map_concepts.py --emit --root PATH  # produce _SOURCED entries

The workflow. ``--search`` shows DDXPlus questions matching a word, so you can
read the actual wording before deciding two things mean the same. Write the
code into ``CONCEPT_MAP`` in ``datasets/ddxplus_mapping.py``. ``--emit`` then
counts co-occurrence across the release and prints entries ready to paste into
``_SOURCED`` in ``fixtures.py``.

Why the reading step cannot be skipped. A mapping asserts that a fixture
concept and a DDXPlus question mean the same thing, and a wrong assertion
produces a confidently sourced wrong number -- worse than the invented one it
replaced, because it arrives with 1.3M patients behind it. Keyword matching
gets roughly a third of them and is wrong about several of those.

How much of the knowledge base this can actually source: about a third
--------------------------------------------------------------------
Measured over 60,000 patients with the mapping as it stands, 28 of 88
disease/finding pairs come back non-zero. The other 60 are hard zeros, and a
zero here is not a frequency. DDXPlus samples evidence from a rule base listing
which findings belong to which pathology, so a finding the rules omit is
generated for nobody: 0 of 508 pulmonary embolism patients report fever, and 0
report smoking. Both are false of real patients and true of the simulator.

``--emit`` therefore refuses to write a zero as ``measured``. Attaching a
citation and a patient count to a clinically wrong number is worse than leaving
the invented value in place, because the invented one is labelled invented and
nobody trusts it.

DDXPlus is also a telemedicine questionnaire, so it holds no examination signs,
no laboratory results and no imaging -- half the vocabulary, and the half
containing the numbers that decide fx-010. Those need the literature.

What does come out is measured from a simulator rather than from patients, and
the DDXPLUS citation carries that distinction into the knowledge base.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.datasets.ddxplus_mapping import (  # noqa: E402
    CONCEPT_MAP,
    DISEASE_MAP,
    EXPECTED_ABSENT,
)


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def load_evidences(root: Path) -> dict:
    path = root / "release_evidences.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Pass --root pointing at the release.")
    return json.loads(path.read_text(encoding="utf-8"))


def search(meta: dict, term: str) -> None:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    shown = 0
    for code, entry in meta.items():
        question = entry.get("question_en", "") or entry.get("name", "")
        if not pattern.search(question):
            continue
        kind = "antecedent" if entry.get("is_antecedent") else "symptom"
        values = entry.get("possible-values") or []
        print(f"  {code:<14} [{kind:<10}] {question}")
        if values and entry.get("data_type") != "B":
            meanings = entry.get("value_meaning") or {}
            for value in values[:8]:
                label = meanings.get(value, {}).get("en", value)
                print(f"        {code}_@_{value:<10} {label}")
            if len(values) > 8:
                print(f"        ... {len(values) - 8} more values")
        shown += 1
    if not shown:
        print(f"  nothing matches {term!r}")


def status() -> None:
    kb = build_knowledge_base()
    concepts = sorted({c for e in kb.diseases() for c in e.features})
    mapped = [c for c in concepts if CONCEPT_MAP.get(c)]
    absent = [c for c in concepts if c in EXPECTED_ABSENT]
    todo = [c for c in concepts if c not in mapped and c not in absent]

    print(f"{len(mapped)}/{len(concepts)} concepts mapped, "
          f"{len(absent)} marked as not covered by DDXPlus\n")
    for concept in mapped:
        print(f"  mapped   {concept:<32} -> {CONCEPT_MAP[concept]}")
    for concept in absent:
        print(f"  n/a      {concept:<32} (source from the literature)")
    for concept in todo:
        print(f"  TODO     {concept}")


def emit(root: Path, limit: int | None) -> None:
    from dxagent.datasets.ddxplus import DDXPlusLoader

    if not CONCEPT_MAP:
        raise SystemExit(
            "CONCEPT_MAP is empty. Use --search to find codes and fill it in "
            "first;\nemitting from an empty mapping would produce nothing and "
            "look like a failure."
        )

    loader = DDXPlusLoader(root=root)
    cases = loader.load_cases("release_train_patients", limit=limit)
    print(f"# measured over {len(cases):,} DDXPlus patients", file=sys.stderr)

    positives: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for case in cases:
        totals[case.diagnosis] += 1
        for concept, present in case.features.items():
            if present:
                positives[case.diagnosis][concept] += 1

    skipped: list[str] = []
    print("    # generated by scripts/map_concepts.py -- measured from DDXPlus,")
    print("    # which is a simulator, so these describe its generating model.")
    for label, pathologies in DISEASE_MAP.items():
        n = sum(totals.get(p, 0) for p in pathologies)
        if not n:
            print(f"    # {label}: no DDXPlus cases found for {pathologies}",
                  file=sys.stderr)
            continue
        for concept, code in CONCEPT_MAP.items():
            if not code:
                continue
            codes = (code,) if isinstance(code, str) else tuple(code)
            # A disjunction is counted per *patient*, not per code. Summing the
            # per-code counts would double-count anyone with swelling in both
            # the calf and the ankle and can push the frequency above 1.
            hits = sum(
                1
                for case in cases
                if case.diagnosis in pathologies
                and any(case.features.get(c) for c in codes)
            )
            value = hits / n
            shown = codes[0] if len(codes) == 1 else f"{codes[0]} +{len(codes) - 1}"

            if hits == 0:
                # A zero here is not a measured frequency. DDXPlus generates
                # evidence from a rule base that lists which findings belong to
                # which pathology, so a finding the rules omit is sampled for
                # nobody -- 0 of 508 pulmonary embolism patients report fever,
                # which is false of real patients and true of the simulator.
                # Emitting it as `measured` would attach a citation and a
                # patient count to a clinically wrong number, which is worse
                # than leaving the invented value in place and saying so.
                skipped.append(f"{label}/{concept}")
                continue

            print(
                f'    ("{label}", "{concept}"): measured(\n'
                f"        {value:.3f},\n"
                f'        Citation("DDXPLUS", "{shown}", '
                f'"{hits} of {n} {label} patients"),\n'
                f"    ),"
            )

    if skipped:
        print(
            f"\n# {len(skipped)} pairs left invented. DDXPlus records no patient "
            f"at all with\n# that finding for that pathology, which reflects its "
            f"rule base rather than\n# a measured frequency of zero. Source these "
            f"from the literature:",
            file=sys.stderr,
        )
        for item in skipped[:10]:
            print(f"#   {item}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"#   ... and {len(skipped) - 10} more", file=sys.stderr)


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"C:\Users\azizb\Downloads\figshare")
    parser.add_argument("--search", help="find DDXPlus questions matching a word")
    parser.add_argument("--status", action="store_true", help="mapping progress")
    parser.add_argument("--emit", action="store_true", help="print _SOURCED entries")
    parser.add_argument("--limit", type=int, default=200_000)
    args = parser.parse_args()

    if args.status:
        status()
        return 0

    root = Path(args.root)
    if args.search:
        search(load_evidences(root), args.search)
        return 0
    if args.emit:
        emit(root, args.limit)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
