"""The vetted knowledge base.

Every hypothesis the system ranks must be traceable to an entry here. The
backend is behind a Protocol because the real KB is blocked on the UMLS licence
and MIMIC credentialing; the in-memory backend below is a working stand-in with
the same interface, so no downstream code changes when the real one lands.

The KB stores, per disease, a conditional feature likelihood table:

    P(feature present | disease)

This is a naive-Bayes factorisation and it is *wrong* -- clinical features are
correlated (fever and tachycardia co-occur far more than independence implies).
It is used deliberately anyway, for two reasons:

  1. It gives an auditable, non-LLM posterior to compare the LLM's proposed
     differential against. Divergence between the two is a useful signal.
  2. Miscalibration from the independence assumption is exactly what the
     calibration layer is there to absorb, and it is better to have a
     transparently wrong prior that a physician can inspect than an opaque one.

The correlation structure is the obvious first thing to revisit once real
prevalence data is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

from .schemas import Citation, Finding, Polarity

# Likelihood floor/ceiling. Prevents a single unexpected finding from driving a
# posterior to exactly zero, which would make the diagnosis unrecoverable no
# matter what later evidence says.
_EPS = 0.01


@dataclass(frozen=True)
class DiseaseEntry:
    """One disease as represented in the KB."""

    label: str
    prevalence: float
    features: dict[str, float]  # concept -> P(present | disease)
    citations: tuple[Citation, ...] = ()
    red_flag: bool = False  # time-critical: raises the bar for committing
    notes: str = ""

    def likelihood(self, finding: Finding, background: float = 0.5) -> float:
        """P(this finding | this disease).

        When the feature is not characterised for this disease we back off to
        ``background``, the KB-wide marginal for that feature. Returning 1.0
        instead -- "uninformative" -- looks neutral but is not: it makes a
        sparsely-described disease strictly cheaper to explain any finding with
        than a thoroughly-described one, so the posterior drifts toward whichever
        entries happen to be least complete. That is an artefact of KB coverage
        masquerading as a clinical inference, and it is the single easiest way
        for this design to be quietly wrong.
        """
        p_present = self.features.get(finding.concept)
        if p_present is None:
            p_present = background
        p_present = min(max(p_present, _EPS), 1.0 - _EPS)
        if finding.polarity is Polarity.PRESENT:
            return p_present
        if finding.polarity is Polarity.ABSENT:
            return 1.0 - p_present
        return 1.0


@runtime_checkable
class KnowledgeBase(Protocol):
    """Interface the reasoner depends on."""

    def diseases(self) -> tuple[DiseaseEntry, ...]: ...

    def get(self, label: str) -> DiseaseEntry | None: ...

    def discriminating_features(self, labels: Iterable[str]) -> tuple[str, ...]: ...


@dataclass
class InMemoryKnowledgeBase:
    """Dict-backed KB. Deterministic, no network, no credentials."""

    entries: dict[str, DiseaseEntry] = field(default_factory=dict)

    _marginals: dict[str, float] = field(default_factory=dict, repr=False)

    def add(self, entry: DiseaseEntry) -> None:
        self.entries[entry.label] = entry
        self._marginals.clear()

    def background(self, concept: str) -> float:
        """Prevalence-weighted marginal P(concept present) across the KB.

        Used as the backoff when a disease does not characterise a feature.
        """
        if not self._marginals:
            self._recompute_marginals()
        return self._marginals.get(concept, 0.5)

    def _recompute_marginals(self) -> None:
        weights = {e.label: max(e.prevalence, 1e-9) for e in self.entries.values()}
        total = sum(weights.values()) or 1.0
        concepts = {c for e in self.entries.values() for c in e.features}
        for concept in concepts:
            described = [
                (weights[e.label], e.features[concept])
                for e in self.entries.values()
                if concept in e.features
            ]
            mass = sum(w for w, _ in described)
            if mass <= 0:
                self._marginals[concept] = 0.5
            else:
                self._marginals[concept] = sum(w * p for w, p in described) / mass
        self._marginals["__total__"] = total

    def likelihood(self, label: str, finding: Finding) -> float:
        """P(finding | label), with marginal backoff for uncharacterised features."""
        entry = self.entries.get(label)
        if entry is None:
            return 1.0
        return entry.likelihood(finding, background=self.background(finding.concept))

    def background_likelihood(self, finding: Finding) -> float:
        """P(finding) under the KB-wide marginal, ignoring any disease.

        The null model an atypicality check measures against: how likely this
        finding is in the population the KB describes, rather than under any
        particular diagnosis.
        """
        p_present = min(max(self.background(finding.concept), _EPS), 1.0 - _EPS)
        if finding.polarity is Polarity.PRESENT:
            return p_present
        if finding.polarity is Polarity.ABSENT:
            return 1.0 - p_present
        return 1.0

    def evidence_split(
        self, label: str, findings: Iterable[Finding], min_ratio: float = 1.5
    ) -> tuple[tuple[Citation, ...], tuple[Citation, ...]]:
        """Split observed findings into those that support this disease and those
        that argue against it, as citations.

        Scored by likelihood ratio against the KB-wide marginal:

            LR = P(finding | disease) / P(finding)

        A ratio above one means the finding is more expected under this disease
        than in the population the KB describes, and below one that it is less.
        The ratio is used rather than the raw likelihood because the raw number
        cannot express contradiction: a finding present in 30% of cases of a
        disease looks weak in isolation and is strong evidence *for* it if the
        population rate is 3%.

        Handling both polarities matters as much as handling both directions.
        An expected finding that is *absent* argues against a diagnosis --
        no pleuritic pain in a suspected pulmonary embolism -- and a
        support-only account of the evidence silently drops that entire class
        of reasoning.

        ``min_ratio`` sets how far from parity a finding must sit to be worth
        citing, in either direction, so that near-neutral findings are omitted
        rather than padding both lists.
        """
        entry = self.entries.get(label)
        if entry is None:
            return (), ()

        supporting: list[Citation] = []
        against: list[Citation] = []
        for finding in findings:
            if finding.polarity is Polarity.UNKNOWN:
                continue
            background = self.background_likelihood(finding)
            if background <= 0:
                continue
            ratio = entry.likelihood(finding, self.background(finding.concept)) / background
            if ratio >= min_ratio:
                supporting.append(self._evidence_citation(entry, finding, ratio))
            elif ratio <= 1.0 / min_ratio:
                against.append(self._evidence_citation(entry, finding, ratio))
        return tuple(supporting), tuple(against)

    @staticmethod
    def _evidence_citation(
        entry: DiseaseEntry, finding: Finding, ratio: float
    ) -> Citation:
        source = entry.citations[0].source_id if entry.citations else "KB"
        observed = finding.polarity.value
        probability = entry.features.get(finding.concept)
        basis = (
            f"P({finding.concept} | {entry.label}) = {probability:.2f}"
            if probability is not None
            else f"{finding.concept} uncharacterised for {entry.label}; "
            "backed off to the knowledge-base marginal"
        )
        return Citation(
            source_id=source,
            locator=f"{entry.label}/{finding.concept}",
            snippet=f"{finding.concept} {observed}; {basis}; likelihood ratio {ratio:.2f}",
        )

    def diseases(self) -> tuple[DiseaseEntry, ...]:
        return tuple(self.entries.values())

    def get(self, label: str) -> DiseaseEntry | None:
        return self.entries.get(label)

    def citations_for(self, label: str) -> tuple[Citation, ...]:
        entry = self.entries.get(label)
        return entry.citations if entry else ()

    def discriminating_features(self, labels: Iterable[str]) -> tuple[str, ...]:
        """Features characterised for at least one of ``labels``.

        Ordered by how much the labels disagree about them, most contested
        first, so the action selector examines promising candidates early.
        """
        labels = list(labels)
        relevant = [self.entries[l] for l in labels if l in self.entries]
        if not relevant:
            return ()
        spread: dict[str, float] = {}
        for concept in {c for e in relevant for c in e.features}:
            values = [e.features.get(concept, 0.5) for e in relevant]
            spread[concept] = max(values) - min(values)
        return tuple(
            c for c, _ in sorted(spread.items(), key=lambda kv: (-kv[1], kv[0]))
        )

    def cost_of(self, concept: str) -> float:
        """Acquisition cost for a feature, in arbitrary consistent units.

        Placeholder tariff. Real costs (and, more importantly, patient burden
        and turnaround time, which are not the same as price) are a clinician
        input, not something to guess at here.
        """
        return _COSTS.get(concept, 1.0)


# Cost tiers: history ~free, exam cheap, bedside labs moderate, imaging dear.
_COSTS: dict[str, float] = {}


def register_costs(costs: dict[str, float]) -> None:
    _COSTS.update(costs)


__all__ = [
    "DiseaseEntry",
    "InMemoryKnowledgeBase",
    "KnowledgeBase",
    "register_costs",
]
