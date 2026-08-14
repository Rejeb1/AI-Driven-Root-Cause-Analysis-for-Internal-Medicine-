"""The finding/cause graph, as a structure rather than a computation.

Section 2 of the brief asks for "a lightweight symptom/causal ontology (nodes =
findings and causes, edges = 'supports'/'contradicts')". The *relation* has
always existed here -- ``InMemoryKnowledgeBase.evidence_split`` derives it per
call from likelihood ratios, with citations -- but only as arithmetic performed
on demand. Nothing could be walked, exported, or drawn, and "here is our
ontology" answered with a function call is not the same deliverable as one
answered with a graph.

This builds the graph. It adds no clinical knowledge: every edge is computed
from the same likelihood ratio ``evidence_split`` already uses, so the graph
and the reasoner cannot disagree. What it adds is a *form* -- an object with
nodes and labelled edges that a reader can traverse, filter, and render.

Two things it is not, both worth stating before anyone quotes it
----------------------------------------------------------------
**It is not causal, and the brief's word for it is wrong here.** An edge says
"this finding is more expected under this cause than in the population this
knowledge base describes". That is association, measured or invented. Fever
supports pneumonia in this graph because pneumonia patients have fevers, not
because the graph knows pneumonia causes fever -- and the distinction is not
pedantic: commit ``0624bdf`` exists specifically to stop risk factors
resolving to the disorders they cause. Calling this a causal graph would
license inferences it cannot support.

**An edge is exactly as trustworthy as the number under it.** 101 of 135
likelihoods are invented, so most edges here are drawn from guesses. Every
edge carries its provenance tier for that reason, and ``Ontology.summary()``
reports the split rather than presenting a uniform-looking graph. A dense,
confident-looking diagram built from invented numbers is a more effective way
to mislead than the numbers alone, which is the risk this module has to be
careful about.

One thing the graph makes visible that the reasoner does not
-------------------------------------------------------------
At the default threshold **asthma exacerbation has no supporting edges at
all**. Every finding it carries is claimed at least as strongly by something
else -- almost always COPD, whose wheeze rate (91%) sits just above asthma's
(87%) and whose productive cough and dyspnoea are higher too. The knowledge
base can still reach the diagnosis, because absent findings and the relative
shape of the whole posterior carry information a per-edge threshold does not
see. But there is no single finding that argues *for* asthma above the
population this knowledge base describes.

That is a real property of the differential, not an artefact: SCOPE.md's own
justification for including asthma is that separating it from COPD "is a real
clinical task". Seeing it as an empty row is the kind of thing a graph shows
and a likelihood table hides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .knowledge import InMemoryKnowledgeBase
from .schemas import Citation, Finding, Polarity

# Below this, a finding is too close to the population rate to be worth an
# edge. Same default as ``evidence_split``'s ``min_ratio``, and for the same
# reason: near-parity findings would otherwise connect everything to
# everything and the graph would carry no information.
_MIN_RATIO = 1.5


@dataclass(frozen=True)
class Node:
    """A finding or a cause."""

    id: str
    kind: str  # "finding" | "cause"
    label: str
    # Identifiers, when the frozen vocabulary carries them. Findings usually
    # have all three; causes carry a CUI and SNOMED code but no HPO id, since
    # HPO is a phenotype ontology and a disease is not a phenotype.
    hpo_id: str | None = None
    cui: str | None = None
    snomed_ct: str | None = None
    modality: str = ""
    red_flag: bool = False


@dataclass(frozen=True)
class Edge:
    """A finding's relation to a cause, with the evidence for the relation."""

    finding: str
    cause: str
    relation: str  # "supports" | "contradicts"
    likelihood_ratio: float
    probability: float  # P(finding | cause), the number the ratio came from
    provenance: str  # "measured" | "narrative" | "invented"
    citation: Citation | None = None

    @property
    def strength(self) -> float:
        """Distance from parity, so supports and contradicts compare directly.

        A contradicting edge at LR 0.25 and a supporting one at LR 4.0 are
        equally strong in opposite directions; the raw ratio makes the first
        look weaker because it is bounded below by zero and the second is not.
        """
        return self.likelihood_ratio if self.relation == "supports" else 1.0 / self.likelihood_ratio


@dataclass
class Ontology:
    """Findings and causes, joined by supports/contradicts edges."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: tuple[Edge, ...] = ()

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        kb: InMemoryKnowledgeBase,
        vocabulary: Any = None,
        min_ratio: float = _MIN_RATIO,
    ) -> "Ontology":
        """Derive the graph from a knowledge base.

        ``vocabulary`` is optional and only supplies identifiers; the graph's
        shape comes entirely from the knowledge base, so a missing vocabulary
        costs labels rather than edges.
        """
        concepts = getattr(vocabulary, "concepts", {}) if vocabulary else {}

        nodes: dict[str, Node] = {}
        edges: list[Edge] = []

        for entry in kb.diseases():
            nodes[entry.label] = Node(
                id=entry.label,
                kind="cause",
                label=entry.label.replace("_", " "),
                red_flag=entry.red_flag,
            )

        for entry in kb.diseases():
            for concept, probability in entry.features.items():
                if concept not in nodes:
                    term = concepts.get(concept)
                    nodes[concept] = Node(
                        id=concept,
                        kind="finding",
                        label=(
                            getattr(term, "hpo_name", None)
                            or concept.split(":")[-1].replace("_", " ")
                        ),
                        hpo_id=getattr(term, "hpo_id", None),
                        cui=getattr(term, "cui", None),
                        snomed_ct=getattr(term, "snomed_ct", None),
                        modality=getattr(term, "modality", ""),
                    )

                finding = Finding(concept, Polarity.PRESENT)
                background = kb.background_likelihood(finding)
                if background <= 0:
                    continue
                ratio = entry.likelihood(finding, kb.background(concept)) / background

                if ratio >= min_ratio:
                    relation = "supports"
                elif ratio <= 1.0 / min_ratio:
                    relation = "contradicts"
                else:
                    continue

                source = entry.sources.get(concept)
                edges.append(
                    Edge(
                        finding=concept,
                        cause=entry.label,
                        relation=relation,
                        likelihood_ratio=ratio,
                        probability=probability,
                        provenance=source.provenance.value if source else "invented",
                        citation=source.citation if source else None,
                    )
                )

        return cls(nodes=nodes, edges=tuple(edges))

    # -- traversal ---------------------------------------------------------

    def edges_from(self, finding: str) -> tuple[Edge, ...]:
        """Every cause this finding bears on, strongest first."""
        found = [e for e in self.edges if e.finding == finding]
        return tuple(sorted(found, key=lambda e: -e.strength))

    def edges_into(self, cause: str) -> tuple[Edge, ...]:
        """Every finding bearing on this cause, strongest first."""
        found = [e for e in self.edges if e.cause == cause]
        return tuple(sorted(found, key=lambda e: -e.strength))

    def neighbours(self, node_id: str) -> tuple[str, ...]:
        """Node ids one edge away, in either direction."""
        node = self.nodes.get(node_id)
        if node is None:
            return ()
        if node.kind == "cause":
            return tuple(dict.fromkeys(e.finding for e in self.edges_into(node_id)))
        return tuple(dict.fromkeys(e.cause for e in self.edges_from(node_id)))

    def shared_findings(self, first: str, second: str) -> tuple[str, ...]:
        """Findings that support both causes -- where two diagnoses look alike.

        The graph query the differential actually turns on: two causes joined
        by many shared supporting findings are the pair a clinician has to
        work to separate, and the pair this system is most likely to confuse.

        **Sensitive to the ``min_ratio`` the graph was built with**, more than
        any other query here, because it needs a finding to clear the bar for
        two causes at once. At the 1.5 default it returns nothing for any pair
        in the fixture knowledge base; at 1.05 pneumonia and pulmonary
        embolism share hypoxia, tachycardia and a raised D-dimer, which is the
        clinically recognisable overlap. Build a second graph at a lower
        threshold to ask this question rather than reading an empty result as
        "these two are easy to tell apart".

        The reason is the marginal. A ratio is taken against the
        knowledge-base-wide rate for that finding, and that rate is pulled up
        by whichever cause claims the finding hardest -- so a finding two
        causes share is, by construction, one neither of them stands far above
        the average on. It is the same effect that made four hand-written
        claims in SCOPE.md false.
        """
        def supporters(cause: str) -> set[str]:
            return {e.finding for e in self.edges_into(cause) if e.relation == "supports"}

        return tuple(sorted(supporters(first) & supporters(second)))

    def causes(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes.values() if n.kind == "cause")

    def findings(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes.values() if n.kind == "finding")

    # -- reporting and export ---------------------------------------------

    def summary(self) -> str:
        supports = sum(1 for e in self.edges if e.relation == "supports")
        tiers: dict[str, int] = {}
        for edge in self.edges:
            tiers[edge.provenance] = tiers.get(edge.provenance, 0) + 1
        split = ", ".join(f"{n} {tier}" for tier, n in sorted(tiers.items()))
        return (
            f"{len(self.causes())} causes, {len(self.findings())} findings, "
            f"{len(self.edges)} edges ({supports} supports, "
            f"{len(self.edges) - supports} contradicts)\n"
            f"edge provenance: {split}"
        )

    def to_json(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "label": n.label,
                    "hpo_id": n.hpo_id,
                    "cui": n.cui,
                    "snomed_ct": n.snomed_ct,
                    "modality": n.modality,
                    "red_flag": n.red_flag,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "finding": e.finding,
                    "cause": e.cause,
                    "relation": e.relation,
                    "likelihood_ratio": round(e.likelihood_ratio, 3),
                    "probability": e.probability,
                    "provenance": e.provenance,
                    "citation": (
                        {
                            "source_id": e.citation.source_id,
                            "locator": e.citation.locator,
                        }
                        if e.citation
                        else None
                    ),
                }
                for e in self.edges
            ],
        }

    def to_dot(self, causes: Iterable[str] | None = None) -> str:
        """Graphviz DOT, so the graph can actually be looked at.

        Invented edges are drawn dashed and sourced ones solid. That is the
        one piece of visual encoding here that carries meaning rather than
        decoration: a reader seeing mostly dashes has understood the knowledge
        base correctly, and a uniform diagram would have misled them.
        """
        wanted = set(causes) if causes else {n.id for n in self.causes()}
        lines = [
            "digraph ontology {",
            "  rankdir=LR;",
            '  node [shape=box, fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=8];',
        ]

        drawn: set[str] = set()
        for edge in self.edges:
            if edge.cause not in wanted:
                continue
            drawn.add(edge.finding)
            drawn.add(edge.cause)

        for node_id in sorted(drawn):
            node = self.nodes[node_id]
            if node.kind == "cause":
                shape = "ellipse"
                extra = ', penwidth=2' if node.red_flag else ""
            else:
                shape = "box"
                extra = ""
            lines.append(
                f'  "{node_id}" [label="{node.label}", shape={shape}{extra}];'
            )

        for edge in self.edges:
            if edge.cause not in wanted:
                continue
            style = "solid" if edge.provenance != "invented" else "dashed"
            colour = "black" if edge.relation == "supports" else "grey50"
            arrow = "normal" if edge.relation == "supports" else "tee"
            lines.append(
                f'  "{edge.finding}" -> "{edge.cause}" '
                f'[label="{edge.likelihood_ratio:.1f}", style={style}, '
                f"color={colour}, arrowhead={arrow}];"
            )

        lines.append("}")
        return "\n".join(lines)


__all__ = ["Edge", "Node", "Ontology"]
