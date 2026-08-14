#!/usr/bin/env python3
"""Export the finding/cause graph, so the ontology can be read rather than run.

    python scripts/build_ontology.py                     # summary + provenance split
    python scripts/build_ontology.py --cause pericarditis
    python scripts/build_ontology.py --finding fever
    python scripts/build_ontology.py --confusable        # which causes look alike
    python scripts/build_ontology.py --dot ontology.dot  # Graphviz
    python scripts/build_ontology.py --json ontology.json

Section 2 of the brief asks for an ontology whose nodes are findings and causes
and whose edges are labelled supports/contradicts. ``dxagent.ontology`` builds
it from the knowledge base -- no new clinical claims, the same likelihood
ratios the reasoner already uses -- and this exposes it.

Render the DOT with Graphviz if it is installed:

    dot -Tsvg ontology.dot -o ontology.svg

Invented edges are dashed and sourced edges solid, which is the one piece of
visual encoding that carries meaning here: a reader whose diagram is mostly
dashes has understood this knowledge base correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.ontology import Ontology  # noqa: E402


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cause", help="show every finding bearing on one cause")
    parser.add_argument("--finding", help="show every cause one finding bears on")
    parser.add_argument(
        "--confusable",
        action="store_true",
        help="cause pairs sharing supporting findings, at a threshold low "
        "enough for the query to return anything",
    )
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--dot", help="write Graphviz DOT here")
    parser.add_argument("--json", dest="as_json", help="write JSON here")
    args = parser.parse_args()

    kb = build_knowledge_base()
    graph = Ontology.build(kb, min_ratio=args.min_ratio)
    print(graph.summary())

    if args.cause:
        edges = graph.edges_into(args.cause)
        if not edges:
            print(f"\nno edges for {args.cause!r}. Known causes:")
            for node in graph.causes():
                print(f"   {node.id}")
            return 1
        print(f"\nfindings bearing on {args.cause}, strongest first:\n")
        for edge in edges:
            cite = f"  [{edge.citation.source_id}]" if edge.citation else ""
            print(
                f"  {edge.relation:<11} {edge.finding:<30} "
                f"LR {edge.likelihood_ratio:>5.2f}   {edge.provenance}{cite}"
            )

    if args.finding:
        edges = graph.edges_from(args.finding)
        if not edges:
            print(f"\nno edges for {args.finding!r}.")
            return 1
        print(f"\ncauses {args.finding} bears on, strongest first:\n")
        for edge in edges:
            cite = f"  [{edge.citation.source_id}]" if edge.citation else ""
            print(
                f"  {edge.relation:<11} {edge.cause:<30} "
                f"LR {edge.likelihood_ratio:>5.2f}   {edge.provenance}{cite}"
            )

    if args.confusable:
        # Deliberately a second, looser graph. At the default threshold this
        # query returns nothing for every pair, because a finding two causes
        # share is one neither stands far above the marginal on -- see
        # Ontology.shared_findings.
        loose = Ontology.build(kb, min_ratio=1.05)
        causes = [n.id for n in loose.causes()]
        print("\ncause pairs sharing supporting findings (threshold 1.05):\n")
        found = False
        for i, first in enumerate(causes):
            for second in causes[i + 1 :]:
                shared = loose.shared_findings(first, second)
                if shared:
                    found = True
                    print(f"  {first} / {second}")
                    print(f"      {', '.join(shared)}")
        if not found:
            print("  none, even at 1.05")

    if args.dot:
        Path(args.dot).write_text(graph.to_dot(), encoding="utf-8")
        print(f"\nwrote {args.dot}  (render: dot -Tsvg {args.dot} -o ontology.svg)")

    if args.as_json:
        Path(args.as_json).write_text(
            json.dumps(graph.to_json(), indent=2), encoding="utf-8"
        )
        print(f"wrote {args.as_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
