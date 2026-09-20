#!/usr/bin/env python3
"""Propose a real case from a PMC report, for a human to check.

    python scripts/extract_case.py PMC12708975
    python scripts/extract_case.py PMC12708975 --model qwen2.5:7b
    python scripts/extract_case.py --text report.txt --diagnosis pericarditis

Fetches the full text through NCBI efetch (or reads a file), hands the case
section to a local model with the extraction rules from real_cases.py, and
prints a ``Case(...)`` block in which every finding carries the verbatim
quote that justified it -- plus everything the model proposed that was
rejected, and why. Nothing here goes into the case set on its own: the
output is a draft whose quotes a human checks against the report, and the
rejection list is where to look for what the model wanted to invent.

Output is JSON too (--json), so a batch can be compared against hand
extractions -- which is how the model's usefulness at this job is measured.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.extraction import extract_findings  # noqa: E402
from dxagent.llm import from_provider  # noqa: E402

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pmc(pmcid: str) -> str:
    """Full text of an open-access PMC article as plain text, references dropped."""
    pid = pmcid.upper().replace("PMC", "")
    url = EFETCH + "?" + urllib.parse.urlencode({"db": "pmc", "id": pid, "rettype": "xml"})
    with urllib.request.urlopen(url, timeout=60) as response:
        xml = response.read().decode("utf-8", "replace")
    xml = re.split(r"<ref-list", xml)[0]
    xml = re.sub(r"</(p|title|sec)>", "\n", xml)
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text)


def humanise(concept: str) -> str:
    kind, _, rest = concept.partition(":")
    label = (rest or kind).replace("_", " ")
    prefix = {"exam": "on examination", "lab": "blood test", "imaging": "imaging"}
    return f"{label} ({prefix[kind]})" if rest and kind in prefix else label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pmcid", nargs="?", help="e.g. PMC12708975")
    parser.add_argument("--text", help="read the report from a file instead of fetching")
    parser.add_argument("--diagnosis", default="unknown", help="label to write into the Case block")
    parser.add_argument("--provider", default="ollama", choices=("ollama", "anthropic", "gemini", "openai-compatible"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", help="also write the proposal and rejections here")
    args = parser.parse_args()

    if args.text:
        text = Path(args.text).read_text(encoding="utf-8")
        case_id = Path(args.text).stem
    elif args.pmcid:
        text = fetch_pmc(args.pmcid)
        case_id = "pmc-" + args.pmcid.upper().replace("PMC", "")
    else:
        parser.error("give a PMC id or --text")

    kb = build_knowledge_base()
    concepts = {c: humanise(c) for e in kb.diseases() for c in e.features}
    concepts = dict(sorted(concepts.items()))

    llm = from_provider(args.provider, args.model)
    result = extract_findings(text, llm, concepts)

    print(f"# {case_id}: {len(result.accepted)} findings accepted, "
          f"{len(result.rejected)} rejected, {result.chunks} chunk(s), "
          f"model {args.provider}:{getattr(llm, 'model', '?')}")
    print("# Every quote below must be checked against the report before this")
    print("# goes anywhere. The model proposed; the string search filtered;")
    print("# you decide.")
    print("Case(")
    print(f'    case_id="{case_id}",')
    print('    presenting_complaint="<write this from the report>",')
    print(f'    diagnosis="{args.diagnosis}",')
    print("    features={")
    for e in sorted(result.accepted, key=lambda x: x.concept):
        quote = e.quote.replace("\n", " ")
        print(f"        # {quote!r}")
        print(f'        "{e.concept}": {e.polarity.value == "present"},')
    print("    },")
    print("    vocabulary=frozenset(),")
    print(")")
    if result.rejected:
        print("\n# rejected -- what the model wanted and did not get:")
        for r in result.rejected:
            print(f"#   {r.concept:30} {r.polarity:8} {r.reason}  | {r.quote[:70]!r}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "case_id": case_id,
            "features": result.features,
            "quotes": {e.concept: e.quote for e in result.accepted},
            "rejected": [r.__dict__ for r in result.rejected],
            "chunks": result.chunks,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
