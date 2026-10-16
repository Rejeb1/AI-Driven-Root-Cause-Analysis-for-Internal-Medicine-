"""What would resolve an escalation: the missing results that would let it commit.

An escalation that says "ACS not excluded; troponin and ECG sought but not
available" is already more useful than a shrug. It still leaves the
clinician to work out the consequence: *if* those two come back normal, what
then? On the real cases that is the commonest situation by far -- 12 of 17
ACS red-flag escalations are reports that record neither a troponin nor an
ECG -- and the answer is computable, because the gate is a function of the
findings.

So on an escalation the same model and the same gate are re-run on
hypothetical results for the tests that are missing: one, two or three of
them, every combination of present and absent. A combination is reported
only if the *whole* commit path would pass on it -- the gate's eight
conditions, the guideline vetoes, and no triggered workup still outstanding
-- so a resolution never says more than the loop itself would have said
given those results.

What this is not:

- It is not a commit. The verdict stays ESCALATED and nothing that counts
  correct or wrong commits reads these. The point is the conditional: "if
  these results are normal, this is X at p%; if not, it is not".
- It is not a prediction of the results. Each path carries the model's own
  predictive probability of that combination, computed from the current
  posterior under the same naive-Bayes independence the model already
  assumes; a path that resolves the case but is unlikely says so.
- It does not use a language model. The resolution is simulated with the
  Bayesian proposer only, so it is deterministic and costs nothing, even
  when the loop itself ran with a consensus proposer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .actions import InformationGainSelector
from .belief import BayesianProposer
from .gate import AbstentionGate
from .guidelines import diagnostic_criteria_met, outstanding_workup
from .knowledge import InMemoryKnowledgeBase
from .schemas import Differential, Finding, Polarity

# How many tests a resolution may name, and how many candidates are tried.
# Three covers the deepest real chain (troponin and ECG for the ACS workup,
# plus a D-dimer when PE is also armed); six candidates keep the search at a
# few hundred cheap posterior updates. Engineering choices, not fitted.
MAX_TESTS = 3
MAX_CANDIDATES = 8
MAX_REPORTED = 3
# A diagnosis below this share of the posterior gets no confirming test of
# its own in the search; the red-flag tolerance, so "live" means the same
# thing here as it does to the gate.
LIVE = 0.10


@dataclass(frozen=True)
class Resolution:
    """One set of hypothetical results under which the loop would commit."""

    results: tuple[tuple[str, bool], ...]  # (concept, present?) in a stable order
    probability: float  # the model's predictive probability of these results
    diagnosis: str
    confidence: float

    def describe(self) -> str:
        parts = ", ".join(
            f"{concept} {'present' if present else 'absent'}"
            for concept, present in self.results
        )
        return (
            f"if {parts} -> {self.diagnosis} at {self.confidence:.0%} "
            f"(these results {self.probability:.0%} likely)"
        )


def _definite(findings: list[Finding]) -> set[str]:
    return {f.concept for f in findings if f.polarity is not Polarity.UNKNOWN}


def _signature(
    kb: InMemoryKnowledgeBase, label: str, differential: Differential, unknown: set[str]
) -> str | None:
    """The unanswered finding that most favours ``label`` over its live rivals.

    Ranked by the likelihood ratio against the posterior-weighted rivals, read
    straight off the knowledge base -- the test a clinician would order to
    confirm this diagnosis, by the model's own numbers.
    """
    entry = kb.get(label)
    if entry is None:
        return None
    rivals = [h for h in differential.hypotheses if h.label != label]
    mass = sum(h.probability for h in rivals) or 1.0

    def ratio(concept: str) -> float:
        own = entry.features.get(concept, kb.backoff(concept))
        other = sum(
            h.probability * kb.get(h.label).features.get(concept, kb.backoff(concept))
            for h in rivals
            if kb.get(h.label) is not None
        ) / mass
        return own / max(other, 1e-6)

    scored = sorted((c for c in unknown if c in entry.features), key=ratio, reverse=True)
    return scored[0] if scored else None


def _candidates(
    kb: InMemoryKnowledgeBase, findings: list[Finding], differential: Differential
) -> list[str]:
    """Missing workup items, then each live diagnosis's confirming finding,
    then the most informative of the rest.

    The first version took workup items and then expected information gain
    only. Measured on the real escalations, its fork contained the true
    diagnosis in 3 of 16 and 7 of 24: a test that discriminates well *on
    average* is rarely the one that would confirm a diagnosis sitting at 15%,
    so the fork kept naming the leader and never the runner-up.
    """
    known = _definite(findings)
    concepts = {c for e in kb.diseases() for c in e.features} - known
    ordered: list[str] = []

    def add(concept: str | None) -> None:
        if concept is not None and concept in concepts and concept not in ordered:
            ordered.append(concept)

    for concept, _ in outstanding_workup(findings):
        add(concept)
    for h in differential.hypotheses:
        if h.probability < LIVE:
            continue
        add(_signature(kb, h.label, differential, concepts - set(ordered)))
    selector = InformationGainSelector(kb)
    for concept in sorted(concepts, key=lambda c: -selector.expected_gain(differential, c)):
        if len(ordered) >= MAX_CANDIDATES:
            break
        add(concept)
    return ordered[:MAX_CANDIDATES]


def _probability(
    kb: InMemoryKnowledgeBase,
    differential: Differential,
    results: tuple[tuple[str, bool], ...],
) -> float:
    total = 0.0
    for h in differential.hypotheses:
        entry = kb.get(h.label)
        if entry is None:
            continue
        p = h.probability
        for concept, present in results:
            q = entry.features.get(concept, kb.backoff(concept))
            p *= q if present else 1.0 - q
        total += p
    return total


def _commits(
    proposer: BayesianProposer,
    gate: AbstentionGate,
    findings: list[Finding],
    complaint: str,
) -> tuple[str, float] | None:
    differential = gate.calibrate(proposer.propose(findings, complaint))
    decision = gate.evaluate(
        differential,
        evidence_fit=getattr(proposer, "last_evidence_fit", float("inf")),
        criteria_met=diagnostic_criteria_met(findings),
    )
    if not decision.should_commit or outstanding_workup(findings):
        return None
    return differential.top.label, decision.confidence


def resolutions(
    kb: InMemoryKnowledgeBase,
    gate: AbstentionGate,
    findings: list[Finding] | tuple[Finding, ...],
    differential: Differential,
    complaint: str = "",
) -> tuple[Resolution, ...]:
    """The most likely minimal result sets under which the loop would commit."""
    findings = list(findings)
    proposer = BayesianProposer(kb)
    candidates = _candidates(kb, findings, differential)
    found: list[Resolution] = []
    for size in range(1, MAX_TESTS + 1):
        for subset in itertools.combinations(candidates, size):
            for outcome in itertools.product((False, True), repeat=size):
                results = tuple(zip(subset, outcome))
                # Minimal only: a superset of a path that already resolves the
                # case adds a test the clinician does not need.
                if any(set(r.results) <= set(results) for r in found):
                    continue
                hypothetical = [f for f in findings if f.concept not in subset] + [
                    Finding(
                        concept=c,
                        polarity=Polarity.PRESENT if present else Polarity.ABSENT,
                        provenance="hypothetical",
                    )
                    for c, present in results
                ]
                committed = _commits(proposer, gate, hypothetical, complaint)
                if committed is None:
                    continue
                label, confidence = committed
                found.append(
                    Resolution(
                        results=results,
                        probability=_probability(kb, differential, results),
                        diagnosis=label,
                        confidence=confidence,
                    )
                )
    # One path per diagnosis, the likeliest, so the packet reads as a fork --
    # "raised troponin -> ACS; normal D-dimer with a high BNP -> oedema" --
    # rather than three near-identical routes to the same answer. The single
    # likeliest path is not a prediction: on the real cases it named the true
    # diagnosis in under half of escalations (DESIGN.md), so the fork is the
    # useful shape, not the top line.
    found.sort(key=lambda r: (-r.probability, len(r.results)))
    best: dict[str, Resolution] = {}
    for r in found:
        best.setdefault(r.diagnosis, r)
    return tuple(list(best.values())[:MAX_REPORTED])


__all__ = ["Resolution", "resolutions"]
