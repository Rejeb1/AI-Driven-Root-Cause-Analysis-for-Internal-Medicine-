#!/usr/bin/env python3
"""Download HPO, build the findings vocabulary, verify it, and freeze it.

    python scripts/build_vocabulary.py            # build + report
    python scripts/build_vocabulary.py --refresh  # re-download hp.obo

Verification is not optional and prints every resolved label, because the only
mapping error that matters is the one that looks fine in a coverage count. This
table shipped with 'productive_cough' pointing at HP:0031246 'Nonproductive
cough' -- 100% coverage, correct-looking id, exactly backwards.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dxagent.vocabulary import CURATED, HPO_SOURCE, Vocabulary  # noqa: E402

DATA = ROOT / "data"
OBO = DATA / "hp.obo"
FROZEN = DATA / "vocabulary.json"

# Prefixes that indicate a negated concept. Used to catch the antonym class.
NEGATIONS = ("non", "absent ", "decreased ", "reduced ", "lack of ")


def download(force: bool = False) -> None:
    DATA.mkdir(exist_ok=True)
    if OBO.exists() and not force:
        print(f"using cached {OBO} ({OBO.stat().st_size // 1_000_000} MB)")
        return
    print(f"downloading {HPO_SOURCE}")
    urllib.request.urlretrieve(HPO_SOURCE, OBO)
    print(f"  -> {OBO} ({OBO.stat().st_size // 1_000_000} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    download(force=args.refresh)
    # Picks up data/umls_concepts.json when scripts/build_umls_map.py has been
    # run, so the frozen vocabulary carries section 6's identifiers too.
    vocab = Vocabulary.build(OBO, umls_path=Path("data/umls_concepts.json"))
    coverage = vocab.coverage()

    print(f"\nHPO release   {coverage['release']}")
    print(f"terms parsed  {coverage['hpo_terms']:,}")
    print(f"concepts      {coverage['mapped']}/{coverage['concepts']} mapped")
    for modality, ratio in sorted(coverage["by_modality"].items()):
        print(f"  {modality:<9} {ratio}")

    print("\nresolved mappings (check each label against the intended concept)")
    problems: list[str] = []
    for key, (hpo_id, _, _) in sorted(CURATED.items()):
        if hpo_id is None:
            continue
        term = vocab.term(hpo_id)
        label = term.name if term else "*** MISSING ***"
        flag = ""
        if term and term.name.lower().startswith(NEGATIONS):
            if not any(n in key for n in ("decreased", "reduced", "absent")):
                flag = "  <-- NEGATED TERM, verify"
                problems.append(key)
        if not term:
            problems.append(key)
        print(f"  {key:<32} {hpo_id:<13} {label}{flag}")

    print("\nunmapped, with stated reason")
    for key in coverage["unmapped"]:
        print(f"  {key:<32} {vocab.concept(key).note[:80]}")

    if problems:
        print(f"\nFAILED: {len(problems)} mapping(s) need review: {problems}")
        return 1

    vocab.to_json(FROZEN)
    print(f"\nfrozen -> {FROZEN}")
    print("Pin this file in version control; HPO releases monthly and an "
          "unpinned vocabulary makes two evaluation runs incomparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
