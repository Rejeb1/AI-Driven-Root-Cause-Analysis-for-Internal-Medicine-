"""Core data model for the diagnostic reasoning loop.

Design note
-----------
This schema is deliberately thin. The Week 1 clinician meeting is expected to
settle the final reasoning schema (what a "finding" is allowed to be, how
severity/acuity are represented, what an escalation packet must contain), so
everything here is structured to be extended rather than rewritten:

  * ``Finding`` carries a free-text ``raw`` alongside a normalised ``concept``,
    so a UMLS/SNOMED identifier can be carried alongside the HPO one
    without touching call sites.
  * ``Hypothesis`` carries ``support`` as a list of ``Citation`` objects rather
    than prose, so the grounding requirement is enforced structurally.
  * ``Action`` is an open vocabulary of ``ActionKind`` plus a target, so new
    evidence-gathering modalities do not require schema changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    """The kinds of evidence-gathering moves the agent may make.

    Split by cost class rather than by clinical taxonomy, because the action
    selector reasons about information value per unit cost.
    """

    ASK = "ask"  # history question, ~free
    EXAMINE = "examine"  # physical exam manoeuvre, cheap
    LAB = "lab"  # laboratory test
    IMAGING = "imaging"  # radiology
    COMMIT = "commit"  # stop and state the diagnosis
    ESCALATE = "escalate"  # hand off to the collaborating physician


class Polarity(str, Enum):
    """Whether a finding is present, absent, or was sought but indeterminate.

    Explicit ABSENT is load-bearing: a negative result is informative, and
    conflating "not asked" with "asked and absent" is a known failure mode in
    naive differential-ranking implementations.
    """

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Citation:
    """A pointer into the vetted knowledge base.

    ``locator`` is intentionally opaque (a document id, a section anchor, a
    guideline clause) so the KB backend can change without schema churn.
    """

    source_id: str
    locator: str
    snippet: str = ""
    # True when the passage was retrieved for this case rather than attached to
    # the knowledge-base entry. The distinction is the whole of what
    # citation-constrained generation claims: an entry citation says where a
    # number came from, a retrieved one says which text a clinician can read to
    # check the reasoning. Collapsing them would let the weaker claim pass for
    # the stronger one.
    retrieved: bool = False

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.source_id}#{self.locator}"


@dataclass(frozen=True)
class Finding:
    """One piece of observed clinical evidence."""

    concept: str
    polarity: Polarity = Polarity.PRESENT
    value: Any = None
    raw: str = ""
    provenance: str = "unspecified"

    @property
    def is_positive(self) -> bool:
        return self.polarity is Polarity.PRESENT


@dataclass(frozen=True)
class Hypothesis:
    """A candidate diagnosis with calibrated probability and grounding."""

    label: str
    probability: float
    support: tuple[Citation, ...] = ()
    against: tuple[Citation, ...] = ()
    rationale: str = ""

    @property
    def is_grounded(self) -> bool:
        """True if the hypothesis cites at least one KB source.

        Ungrounded hypotheses are surfaced to the physician rather than
        silently dropped -- an LLM-proposed diagnosis with no KB support is a
        signal worth reading, not noise.
        """
        return len(self.support) > 0

    @property
    def is_retrieval_grounded(self) -> bool:
        """True if at least one supporting citation was retrieved for this case.

        Stricter than ``is_grounded``: a hypothesis can cite the knowledge-base
        entry it came from while no passage in the corpus actually speaks to
        this presentation. That is the case citation-constrained generation is
        meant to flag, and only this property can see it.
        """
        return any(c.retrieved for c in self.support)


@dataclass(frozen=True)
class Differential:
    """A ranked, normalised set of hypotheses."""

    hypotheses: tuple[Hypothesis, ...]

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("differential must contain at least one hypothesis")
        total = sum(h.probability for h in self.hypotheses)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"probabilities must sum to 1.0, got {total:.6f}")

    @classmethod
    def from_scores(
        cls,
        scores: dict[str, float],
        support: dict[str, tuple[Citation, ...]] | None = None,
        rationales: dict[str, str] | None = None,
        against: dict[str, tuple[Citation, ...]] | None = None,
    ) -> Differential:
        """Build a normalised differential from unnormalised positive scores.

        ``against`` is optional so that callers with nothing to say about
        contradicting evidence stay valid, but a proposer that never supplies
        it produces hypotheses whose case against them is silently empty --
        which reads as "nothing argues against this" rather than "this was
        never assessed".
        """
        if not scores:
            raise ValueError("cannot build a differential from no scores")
        support = support or {}
        rationales = rationales or {}
        against = against or {}
        total = sum(scores.values())
        if total <= 0:
            # Degenerate posterior: fall back to uniform rather than crashing,
            # and let the abstention gate catch the resulting low confidence.
            uniform = 1.0 / len(scores)
            normalised = {k: uniform for k in scores}
        else:
            normalised = {k: v / total for k, v in scores.items()}
        ranked = sorted(normalised.items(), key=lambda kv: (-kv[1], kv[0]))
        return cls(
            tuple(
                Hypothesis(
                    label=label,
                    probability=prob,
                    support=support.get(label, ()),
                    against=against.get(label, ()),
                    rationale=rationales.get(label, ""),
                )
                for label, prob in ranked
            )
        )

    @property
    def top(self) -> Hypothesis:
        return self.hypotheses[0]

    @property
    def margin(self) -> float:
        """Probability gap between the top two hypotheses.

        Used alongside top-1 probability in the gate: a peaked-but-contested
        posterior (0.45 vs 0.44) warrants escalation even when a bare
        probability threshold would pass.
        """
        if len(self.hypotheses) == 1:
            return self.hypotheses[0].probability
        return self.hypotheses[0].probability - self.hypotheses[1].probability

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats. Drives information-gain action selection."""
        return -sum(
            h.probability * math.log(h.probability)
            for h in self.hypotheses
            if h.probability > 0
        )

    def probability_of(self, label: str) -> float:
        for h in self.hypotheses:
            if h.label == label:
                return h.probability
        return 0.0

    def top_k(self, k: int) -> tuple[str, ...]:
        return tuple(h.label for h in self.hypotheses[:k])

    def rescaled(self, scores: dict[str, float]) -> Differential:
        """Return a new differential with the same labels and new scores."""
        return Differential.from_scores(
            scores,
            support={h.label: h.support for h in self.hypotheses},
            rationales={h.label: h.rationale for h in self.hypotheses},
        )


@dataclass(frozen=True)
class Action:
    """A proposed evidence-gathering move."""

    kind: ActionKind
    target: str = ""
    cost: float = 0.0
    expected_information_gain: float = 0.0
    rationale: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.kind in (ActionKind.COMMIT, ActionKind.ESCALATE)


@dataclass(frozen=True)
class Step:
    """One turn of the loop, retained for auditability."""

    index: int
    action: Action
    findings: tuple[Finding, ...]
    differential_after: Differential
    entropy_before: float
    entropy_after: float

    @property
    def realised_information_gain(self) -> float:
        return self.entropy_before - self.entropy_after


@dataclass
class CaseState:
    """Mutable state carried through a single case."""

    case_id: str
    presenting_complaint: str
    findings: list[Finding] = field(default_factory=list)
    differential: Differential | None = None
    history: list[Step] = field(default_factory=list)
    budget_spent: float = 0.0
    asked: set[str] = field(default_factory=set)

    @property
    def turn(self) -> int:
        return len(self.history)

    @property
    def informative_turns(self) -> int:
        """Turns that actually learned something, as opposed to spending the
        budget on a question the record simply never answers.

        A step whose findings are all UNKNOWN could not have been avoided by
        a smarter selector -- the selector picks the best *expected*
        information gain before asking, and UNKNOWN is precisely the answer
        it could not have predicted. Counting such a turn the same as an
        informative one is fair for a fixture, which never returns UNKNOWN,
        and punitive for a real case report, which frequently does simply
        because the source text never addressed that finding one way or the
        other. See ``LoopLimits.uninformative_turns_are_free``.
        """
        return sum(
            1
            for step in self.history
            if any(f.polarity is not Polarity.UNKNOWN for f in step.findings)
        )

    def observed(self, concept: str) -> Finding | None:
        for f in self.findings:
            if f.concept == concept:
                return f
        return None

    def record(self, action: Action, findings: list[Finding], differential: Differential) -> None:
        entropy_before = self.differential.entropy if self.differential else math.log(
            max(len(differential.hypotheses), 2)
        )
        self.findings.extend(findings)
        if action.target:
            self.asked.add(action.target)
        self.budget_spent += action.cost
        self.differential = differential
        self.history.append(
            Step(
                index=len(self.history),
                action=action,
                findings=tuple(findings),
                differential_after=differential,
                entropy_before=entropy_before,
                entropy_after=differential.entropy,
            )
        )


class Verdict(str, Enum):
    COMMITTED = "committed"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class Escalation:
    """The packet handed to the collaborating physician on abstention.

    Contents are the point of the abstention design: an abstention that says
    only "I don't know" transfers no work off the clinician. This carries the
    posterior, the discriminating question the agent could not resolve, and the
    evidence it did gather.
    """

    reason: str
    differential: Differential
    unresolved_question: str | None
    findings: tuple[Finding, ...]
    turns_used: int


@dataclass(frozen=True)
class CaseOutcome:
    """Terminal result for one case."""

    case_id: str
    verdict: Verdict
    differential: Differential
    confidence: float
    escalation: Escalation | None
    steps: tuple[Step, ...]
    budget_spent: float

    @property
    def prediction(self) -> str | None:
        """The committed diagnosis, or None if the agent abstained."""
        return self.differential.top.label if self.verdict is Verdict.COMMITTED else None

    @property
    def abstained(self) -> bool:
        return self.verdict is Verdict.ESCALATED


__all__ = [
    "Action",
    "ActionKind",
    "CaseOutcome",
    "CaseState",
    "Citation",
    "Differential",
    "Escalation",
    "Finding",
    "Hypothesis",
    "Polarity",
    "Step",
    "Verdict",
    "replace",
]
