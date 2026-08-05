"""The environment the agent queries for evidence.

``CaseOracle`` is a faithful, noiseless responder: it knows the full feature
vector of a labelled case and answers truthfully. That makes it useful for
measuring the reasoning loop in isolation, and useless for measuring anything
about real information gathering, because real patients under-report, misreport,
and answer questions they were not asked.

``NoisyOracle`` adds recall failure and reporting noise, which is the minimum
needed before any claim about robustness is meaningful. A dialogue-based patient
simulator is the next step up and the natural place to reuse the AgentClinic
setup.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .schemas import Action, Finding, Polarity


@dataclass(frozen=True)
class Case:
    """One labelled case.

    ``features`` may be sparse. ``vocabulary`` says how to read that sparsity:
    a concept listed there but missing from ``features`` is genuinely absent,
    while a concept in neither was never recorded. Sources that enumerate every
    feature they observed (the fixtures) leave ``vocabulary`` empty and store
    explicit ``False`` entries; sources that record positives only (DDXPlus)
    store the positives and declare the vocabulary those positives were drawn
    from. Without the distinction a closed-world dataset answers every question
    about an absent symptom with UNKNOWN, which carries no information, and the
    agent gathers nothing.
    """

    case_id: str
    presenting_complaint: str
    features: dict[str, bool]  # concept -> truly present
    diagnosis: str
    initial_findings: tuple[str, ...] = ()  # concepts volunteered up front
    demographics: dict[str, object] = field(default_factory=dict)
    vocabulary: frozenset[str] = frozenset()  # concepts the record is complete for
    # Gold differential as (label, probability), most likely first. Empty when
    # the source labels a single diagnosis only. ``diagnosis`` stays the single
    # ground truth either way, so nothing that reads it needs to change.
    differential: tuple[tuple[str, float], ...] = ()

    def known_absent(self, concept: str) -> bool:
        """True when the record is complete enough to call this a real negative."""
        return concept not in self.features and concept in self.vocabulary

    @property
    def differential_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.differential)

    @property
    def gold_confidence(self) -> float:
        """Probability the gold differential puts on the true diagnosis.

        0.0 when there is no gold differential, or when it does not name the
        true diagnosis at all.
        """
        for label, probability in self.differential:
            if label == self.diagnosis:
                return probability
        return 0.0

    @property
    def ambiguity(self) -> float:
        """How uncertain the gold labelling was about this case, in [0, 1].

        The stand-in for "would a clinician call this genuinely uncertain".
        Defined as ``1 - gold_confidence``: when the gold differential gives
        the true diagnosis 0.25, four fifths of the mass sits on other causes
        and the case is hard; when it gives 0.9, it is not.

        Measured against ``differential_entropy`` on DDXPlus, this is the one
        that discriminates. Entropy is near-saturated there -- median 0.96,
        first quartile 0.93 -- because the differentials are long whether or
        not the case is difficult, so thresholding it flags almost every case
        and the proxy carries no signal. Breadth of a differential and
        uncertainty about it are not the same quantity.

        Returns 0.0 with no gold differential, so a source that provides none
        reads as unambiguous rather than as missing data. Check ``differential``
        before trusting this on a mixed case set.
        """
        if not self.differential:
            return 0.0
        return 1.0 - self.gold_confidence

    @property
    def differential_entropy(self) -> float:
        """Normalised entropy of the gold differential, in [0, 1].

        Kept as a secondary signal: it measures how broadly the gold spread its
        mass, which is worth reporting even though it is the wrong basis for an
        abstention threshold (see ``ambiguity``). Normalising by ``log(len)``
        keeps differentials of different lengths comparable.
        """
        probabilities = [p for _, p in self.differential if p > 0.0]
        if len(probabilities) < 2:
            return 0.0
        total = sum(probabilities)
        if total <= 0.0:
            return 0.0
        entropy = -sum(
            (p / total) * math.log(p / total) for p in probabilities
        )
        return entropy / math.log(len(probabilities))

    def initial(self) -> list[Finding]:
        out: list[Finding] = []
        for concept in self.initial_findings:
            present = self.features.get(concept, False)
            out.append(
                Finding(
                    concept=concept,
                    polarity=Polarity.PRESENT if present else Polarity.ABSENT,
                    provenance="presenting history",
                )
            )
        return out


@runtime_checkable
class Environment(Protocol):
    def respond(self, action: Action) -> list[Finding]: ...


@dataclass
class CaseOracle:
    """Truthful responder. Unqueried features stay unobserved."""

    case: Case

    def respond(self, action: Action) -> list[Finding]:
        if not action.target:
            return []
        if action.target not in self.case.features:
            if self.case.known_absent(action.target):
                # The record is complete over this vocabulary, so silence is a
                # genuine negative rather than a gap.
                return [
                    Finding(
                        concept=action.target,
                        polarity=Polarity.ABSENT,
                        provenance=action.kind.value,
                    )
                ]
            # Asked about something the case does not characterise. Reporting
            # UNKNOWN rather than ABSENT is important: it stops the agent from
            # treating a gap in the dataset as a genuine negative.
            return [
                Finding(
                    concept=action.target,
                    polarity=Polarity.UNKNOWN,
                    provenance=f"{action.kind.value} (not recorded)",
                )
            ]
        present = self.case.features[action.target]
        return [
            Finding(
                concept=action.target,
                polarity=Polarity.PRESENT if present else Polarity.ABSENT,
                provenance=action.kind.value,
            )
        ]


@dataclass
class NoisyOracle:
    """Truthful for tests and imaging; lossy for history.

    ``recall_failure`` is the probability that a genuinely present symptom is
    reported absent when *asked about* (not volunteered). Modelled as one-sided
    because under-reporting of symptoms is far more common than fabrication.
    """

    case: Case
    recall_failure: float = 0.15
    seed: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def respond(self, action: Action) -> list[Finding]:
        findings = CaseOracle(self.case).respond(action)
        out: list[Finding] = []
        for finding in findings:
            degrade = (
                action.kind.value == "ask"
                and finding.polarity is Polarity.PRESENT
                and self._rng.random() < self.recall_failure
            )
            out.append(
                Finding(
                    concept=finding.concept,
                    polarity=Polarity.ABSENT if degrade else finding.polarity,
                    provenance=finding.provenance + (" (recall failure)" if degrade else ""),
                )
            )
        return out


__all__ = ["Case", "CaseOracle", "Environment", "NoisyOracle"]
