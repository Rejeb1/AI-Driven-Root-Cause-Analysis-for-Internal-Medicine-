"""The baselines the agentic loop has to beat.

The project brief names two: a non-agentic single-prompt model and a
retrieval-only ranker. Both are implemented here behind the same interface as
``DiagnosticAgent`` -- ``run(case, environment) -> CaseOutcome`` -- so the
existing harness and every metric apply to them unchanged. That constraint is
the point: a baseline evaluated by a different code path is not a comparison,
it is two numbers next to each other.

What each one isolates
----------------------
``SinglePassBaseline`` removes the *loop*. It is handed the complete patient
record at once, proposes a differential, and commits. Comparing it against the
agent measures what sequential evidence gathering buys, and nothing else --
same knowledge base, same inference, same metrics. This is also the setting
MEDDxAgent criticises as flattering: complete profiles and a single attempt.
If the agent cannot beat it, the loop is decoration.

``RetrievalOnlyBaseline`` removes the *reasoning*. It ranks diseases by how
well their recorded features overlap the observed findings, with no likelihood
model, no priors and no evidence weighting. It is deliberately close to what a
vector search over disease descriptions would return, and it exists to answer
the question a reviewer will ask first: how much of the result comes from
retrieving the right document, and how much from reasoning over it?

Neither abstains. That is not an oversight -- the point of the comparison is
that they *cannot*, and the coverage-risk curve of a system that always answers
is the floor the gate is supposed to improve on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .belief import BayesianProposer, Proposer
from .environment import Case, CaseOracle, Environment
from .knowledge import InMemoryKnowledgeBase
from .schemas import CaseOutcome, Differential, Finding, Polarity, Verdict


def _observe_everything(case: Case) -> list[Finding]:
    """The complete record as findings, which is what 'non-agentic' means here.

    Includes the negatives: a baseline handed only the positive findings would
    be solving an easier problem than the agent, which can ask about anything
    and receive an answer either way.
    """
    findings = [
        Finding(
            concept=concept,
            polarity=Polarity.PRESENT if present else Polarity.ABSENT,
            provenance="complete record",
        )
        for concept, present in case.features.items()
    ]
    # Under a closed-world record, everything in the vocabulary and absent from
    # the feature map is a genuine negative and the baseline should see it too.
    findings.extend(
        Finding(concept=concept, polarity=Polarity.ABSENT, provenance="complete record")
        for concept in sorted(case.vocabulary)
        if concept not in case.features
    )
    return findings


def _committed(case: Case, differential: Differential) -> CaseOutcome:
    return CaseOutcome(
        case_id=case.case_id,
        verdict=Verdict.COMMITTED,
        differential=differential,
        confidence=differential.top.probability,
        escalation=None,
        steps=(),
        budget_spent=0.0,
    )


@dataclass
class SinglePassBaseline:
    """One shot over the complete record. No evidence gathering, no gate.

    ``proposer`` defaults to the Bayesian one so the baseline runs offline with
    no credentials. Passing an ``LLMProposer`` gives the literal comparison the
    brief asks for -- a non-agentic single-prompt model -- over the same KB, and
    the difference between the two proposers is then itself a reportable result.
    """

    kb: InMemoryKnowledgeBase
    proposer: Proposer | None = None

    def __post_init__(self) -> None:
        self.proposer = self.proposer or BayesianProposer(self.kb)

    def run(self, case: Case, environment: Environment | None = None) -> CaseOutcome:
        # environment is accepted and ignored: this baseline does not interact,
        # which is exactly the property under test.
        differential = self.proposer.propose(
            _observe_everything(case), case.presenting_complaint
        )
        return _committed(case, differential)


@dataclass
class RetrievalOnlyBaseline:
    """Rank by feature overlap. No likelihood model, no priors, no loop.

    Scoring is deliberately crude, because a sophisticated scorer would stop
    being a retrieval baseline and start being a second reasoner. Each disease
    scores the sum, over observed positive findings, of how strongly it records
    that feature, normalised by how many features it records at all.

    The normalisation is the one piece that is not naive, and it is there for a
    documented reason: without it a disease described in exhaustive detail
    outranks a well-matched one purely for having more entries, which is a
    knowledge-base coverage artefact rather than a retrieval result -- the same
    failure ``DiseaseEntry.likelihood`` backs off to the marginal to avoid.
    """

    kb: InMemoryKnowledgeBase
    match_threshold: float = 0.5
    _floor: float = field(default=1e-9, repr=False)

    def run(self, case: Case, environment: Environment | None = None) -> CaseOutcome:
        observed = {
            f.concept for f in _observe_everything(case) if f.polarity is Polarity.PRESENT
        }
        scores: dict[str, float] = {}
        for entry in self.kb.diseases():
            described = [
                weight
                for concept, weight in entry.features.items()
                if weight >= self.match_threshold
            ]
            matched = sum(
                weight
                for concept, weight in entry.features.items()
                if weight >= self.match_threshold and concept in observed
            )
            scores[entry.label] = matched / max(len(described), 1) + self._floor

        support = {e.label: e.citations for e in self.kb.diseases()}
        return _committed(case, Differential.from_scores(scores, support=support))


__all__ = ["RetrievalOnlyBaseline", "SinglePassBaseline"]
