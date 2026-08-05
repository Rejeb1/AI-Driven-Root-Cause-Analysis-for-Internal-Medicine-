#!/usr/bin/env python3
"""Measure how many disease -> finding edges UMLS actually gives us.

This answers one question before anyone downloads a 30 GB release:

    For the conditions in this project, does UMLS assert enough
    disease-to-finding relationships to reason over?

If a disease comes back with 30 manifestation edges, the knowledge base can be
grounded in UMLS. If it comes back with two, the edges must be hand-authored
from guideline text and the full download would be wasted effort.

Usage
-----
    set UMLS_API_KEY=your-key-here        (Windows PowerShell: $env:UMLS_API_KEY="...")
    python scripts/umls_probe.py

Get the key from https://uts.nlm.nih.gov -> sign in -> My Profile.
Never paste the key into source control or a chat window.

Note on interpreting the output
-------------------------------
This script does NOT assume it knows which relationship labels matter. It
prints every label it finds, ranked by frequency, precisely because the useful
ones may not be the ones expected. Filtering to a guessed list first would hide
whatever is actually there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://uts-ws.nlm.nih.gov/rest"

# The conditions this project reasons over.
DISEASES = [
    "pulmonary embolism",
    "community acquired pneumonia",
    "acute coronary syndrome",
    "congestive heart failure",
    "chronic obstructive pulmonary disease",
    "asthma",
    "pericarditis",
    "panic attack",
]

# Relationship labels that plausibly mean "this finding is seen in this disease".
# Used ONLY for the summary line -- the full label census is printed regardless,
# so a wrong guess here costs nothing.
MANIFESTATION_RELA = {
    "has_manifestation",
    "manifestation_of",
    "is_finding_of_disease",
    "disease_has_finding",
    "associated_finding_of",
    "has_associated_finding",
    "cause_of",
    "due_to",
    "associated_with",
    "has_associated_morphology",
    "occurs_in",
}


def request(path: str, api_key: str, **params) -> dict:
    """GET a UMLS endpoint and return parsed JSON."""
    params["apiKey"] = api_key
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise SystemExit(
                "401 Unauthorized. The API key was rejected.\n"
                "Check it at https://uts.nlm.nih.gov -> My Profile."
            ) from exc
        if exc.code == 404:
            return {}  # no results for this concept; not an error
        raise


def find_cui(name: str, api_key: str) -> tuple[str, str] | None:
    """Resolve a disease name to its best-matching CUI.

    Restricted to disease-ish semantic types so that searching 'asthma' returns
    the disorder rather than, say, a drug indication or a questionnaire item.
    T047 Disease or Syndrome, T046 Pathologic Function, T033 Finding,
    T184 Sign or Symptom, T048 Mental or Behavioral Dysfunction (panic attack
    lives here, not under T047/T046 -- v1 missed it for exactly this reason).
    """
    payload = request(
        "/search/current",
        api_key,
        string=name,
        sabs="SNOMEDCT_US",
        semanticTypes="T047,T046,T033,T184,T048",
        pageSize=5,
    )
    results = payload.get("result", {}).get("results", [])
    for item in results:
        if item.get("ui") and item["ui"] != "NONE":
            return item["ui"], item.get("name", name)
    return None


def fetch_relations(cui: str, api_key: str, max_pages: int = 200) -> list[dict]:
    """Page through all relations for a CUI.

    max_pages=200 (20,000 relations) is a safety bound, not an expected
    ceiling -- v1's max_pages=10 silently truncated at 1000 and five of eight
    diseases hit that ceiling, so the truncation was invisible in the output.
    """
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        payload = request(
            f"/content/current/CUI/{cui}/relations",
            api_key,
            pageNumber=page,
            pageSize=100,
        )
        results = payload.get("result", [])
        if not results:
            break
        out.extend(results)
        if len(results) < 100:
            break
        time.sleep(0.2)  # be polite to NLM
    else:
        print(f"  [warning] hit max_pages={max_pages} cap for CUI {cui}; results are truncated")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("UMLS_API_KEY"))
    parser.add_argument("--json", help="write full results here")
    args = parser.parse_args()

    if not args.api_key:
        print(
            "No API key found.\n\n"
            "PowerShell:  $env:UMLS_API_KEY=\"your-key\"\n"
            "then:        python scripts/umls_probe.py\n\n"
            "Or:          python scripts/umls_probe.py --api-key your-key\n\n"
            "Key is at https://uts.nlm.nih.gov -> sign in -> My Profile."
        )
        return 1

    all_labels: Counter[str] = Counter()
    summary: list[tuple[str, str, int, int]] = []
    detail: dict[str, dict] = {}
    label_examples: dict[str, list[tuple[str, str, str]]] = {}

    for name in DISEASES:
        found = find_cui(name, args.api_key)
        if not found:
            print(f"{name:<40} no CUI found")
            summary.append((name, "-", 0, 0))
            continue
        cui, label = found
        relations = fetch_relations(cui, args.api_key)

        labels = Counter(
            (r.get("additionalRelationLabel") or r.get("relationLabel") or "?")
            for r in relations
        )
        all_labels.update(labels)
        manifestations = sum(v for k, v in labels.items() if k in MANIFESTATION_RELA)

        for r in relations:
            label = r.get("additionalRelationLabel") or r.get("relationLabel") or "?"
            bucket = label_examples.setdefault(label, [])
            if len(bucket) < 6:
                bucket.append((name, r.get("relatedIdName") or "?", r.get("rootSource") or "?"))

        print(
            f"{name:<40} {cui:<10} {len(relations):>4} relations, "
            f"{manifestations:>3} manifestation-type"
        )
        summary.append((name, cui, len(relations), manifestations))
        detail[name] = {
            "cui": cui,
            "preferred_name": label,
            "total_relations": len(relations),
            "labels": dict(labels.most_common()),
            "manifestation_examples": [
                {
                    "label": r.get("additionalRelationLabel") or r.get("relationLabel"),
                    "related": r.get("relatedIdName"),
                    "source": r.get("rootSource"),
                }
                for r in relations
                if (r.get("additionalRelationLabel") or "") in MANIFESTATION_RELA
            ][:15],
        }
        time.sleep(0.3)

    print("\n" + "=" * 68)
    print("every relationship label seen, most common first")
    print("=" * 68)
    for label, count in all_labels.most_common(30):
        mark = "  <-- counted as manifestation" if label in MANIFESTATION_RELA else ""
        print(f"  {label:<40} {count:>5}{mark}")

    print("\n" + "=" * 68)
    print("sample targets -- what the top labels actually point to")
    print("=" * 68)
    print("This is the part the counts can't tell you: symptom-level edges")
    print("(useful for a differential) vs disease-to-disease edges (not).\n")
    for label, count in all_labels.most_common(10):
        print(f"  {label} ({count} total)")
        for disease, related, source in label_examples.get(label, []):
            print(f"      {disease:<38} -> {related:<45} [{source}]")
        print()

    total_manifestations = sum(m for _, _, _, m in summary)
    diseases_found = sum(1 for _, cui, _, _ in summary if cui != "-")
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    if diseases_found == 0:
        print("No concepts resolved. Check the API key and network access.")
    else:
        mean = total_manifestations / diseases_found
        print(f"{diseases_found}/{len(DISEASES)} diseases resolved to a CUI")
        print(f"mean manifestation-type edges per disease: {mean:.1f}")
        print()
        if mean >= 15:
            print("DENSE. UMLS can ground the knowledge base.")
            print("Next: download the full release and extract MRREL properly.")
        elif mean >= 5:
            print("THIN but usable as a starting skeleton.")
            print("Next: extract what exists, then hand-author the gaps from")
            print("guideline text. Budget time for the manual pass.")
        else:
            print("TOO SPARSE to ground a differential.")
            print("Next: do NOT download the 30 GB release for this purpose.")
            print("Hand-author edges from open guideline text instead, and use")
            print("UMLS only as the concept vocabulary (which it does well).")
        print()
        print("Either way, note that NONE of these edges carry frequencies, and")
        print("none of them mean 'contradicts'. Those remain unsolved.")

    if args.json:
        Path(args.json).write_text(json.dumps(detail, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
