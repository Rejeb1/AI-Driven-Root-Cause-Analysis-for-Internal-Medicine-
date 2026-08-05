"""Choosing the next evidence-gathering action.

The selector scores each unasked, KB-characterised feature by expected
reduction in posterior entropy, divided by acquisition cost, and takes the best.
This is the standard myopic (one-step-lookahead) value-of-information rule.

Two known limitations, recorded rather than hidden:

  * It is myopic. A pair of cheap questions that are jointly decisive but
    individually uninformative will be passed over. Non-myopic lookahead is
    tractable at this branching factor and is a candidate improvement.
  * Entropy reduction is diagnosis-agnostic: it does not know that ruling out a
    time-critical diagnosis is worth more than sharpening the posterior between
    two benign ones. ``red_flag_bonus`` is a blunt first pass at this and the
    weighting is a clinician decision, not an engineering one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .knowledge import InMemoryKnowledgeBase
from .schemas import Action, ActionKind, CaseState, Differential, Finding, Polarity

# Concepts whose names begin with these prefixes map to costlier action kinds.
# A stopgap until the KB carries a modality field per feature.
_MODALITY_PREFIXES: tuple[tuple[str, ActionKind], ...] = (
    ("imaging:", ActionKind.IMAGING),
    ("lab:", ActionKind.LAB),
    ("exam:", ActionKind.EXAMINE),
)


def classify(concept: str) -> ActionKind:
    for prefix, kind in _MODALITY_PREFIXES:
        if concept.startswith(prefix):
            return kind
    return ActionKind.ASK


@dataclass
class InformationGainSelector:
    """Myopic value-of-information action selection."""

    kb: InMemoryKnowledgeBase
    red_flag_bonus: float = 1.5

    # Below this expected gain (nats) an action is not worth a turn. Set from
    # observed behaviour: at 1e-4 the agent burned its entire turn budget on
    # near-worthless history questions and never reached a decisive test.
    min_gain: float = 0.02

    # Added to cost in the value-per-cost denominator. Without it, pure
    # gain/cost is dominated by whatever is cheapest: a 0.2-cost question with
    # 0.01 nats of gain outranks a 5.0-cost troponin with 0.6 nats, so the loop
    # asks twelve trivial questions and times out. The offset says, in effect,
    # that every action also costs a turn -- which is the real scarce resource
    # in a consultation, and the thing bare monetary cost fails to capture.
    cost_offset: float = 1.5

    # Probability above which a red-flag diagnosis must be actively excluded
    # rather than merely out-ranked. Well below the gate's reporting bar.
    ruleout_floor: float = 0.03
    min_discrimination: float = 0.25

    # An outcome must be at least this likely to count as a reason to withhold
    # a commit. Guards against chasing flips that hinge on a result the agent
    # believes will not happen.
    min_flip_outcome: float = 0.05

    def candidates(self, state: CaseState, differential: Differential, top_k: int = 5) -> list[Action]:
        """Rank available actions, best first."""
        considered = differential.top_k(top_k)
        features = self.kb.discriminating_features(considered)
        actions: list[Action] = []
        for concept in features:
            if concept in state.asked or state.observed(concept) is not None:
                continue
            gain = self.expected_gain(differential, concept)
            if gain < self.min_gain:
                continue
            cost = max(self.kb.cost_of(concept), 1e-6)
            weight = self._red_flag_weight(concept, considered)
            actions.append(
                Action(
                    kind=classify(concept),
                    target=concept,
                    cost=cost,
                    expected_information_gain=gain,
                    rationale=(
                        f"expected entropy reduction {gain:.3f} nats "
                        f"at cost {cost:.2f}"
                    ),
                )
            )
        actions.sort(key=lambda a: (-self.value(a, considered), a.target))
        return actions

    def value(self, action: Action, considered: tuple[str, ...]) -> float:
        """Information gain per unit of (cost + turn), red-flag weighted."""
        return (
            action.expected_information_gain
            * self._red_flag_weight(action.target, considered)
            / (action.cost + self.cost_offset)
        )

    def select(self, state: CaseState, differential: Differential) -> Action | None:
        ranked = self.candidates(state, differential)
        return ranked[0] if ranked else None

    def expected_gain(self, differential: Differential, concept: str) -> float:
        """E[H(posterior_before) - H(posterior_after)] over the two outcomes.

        P(feature present) is the posterior-weighted average of the per-disease
        likelihoods, i.e. the agent's current belief about what it will observe.
        """
        p_present = self._probability_of_presence(differential, concept)
        p_present = min(max(p_present, 1e-6), 1.0 - 1e-6)

        h_present = self._posterior_entropy(differential, concept, Polarity.PRESENT)
        h_absent = self._posterior_entropy(differential, concept, Polarity.ABSENT)
        expected_after = p_present * h_present + (1 - p_present) * h_absent
        # Clamp at zero: observing evidence cannot increase expected entropy,
        # and small negatives here are pure float noise.
        return max(differential.entropy - expected_after, 0.0)

    def _posterior_after(
        self, differential: Differential, concept: str, polarity: Polarity
    ) -> dict[str, float]:
        """The differential re-weighted by one hypothetical observation."""
        finding = Finding(concept=concept, polarity=polarity)
        scores: dict[str, float] = {}
        for h in differential.hypotheses:
            scores[h.label] = h.probability * self.kb.likelihood(h.label, finding)
        total = sum(scores.values())
        if total <= 0:
            uniform = 1.0 / max(len(scores), 1)
            return {label: uniform for label in scores}
        return {label: value / total for label, value in scores.items()}

    def _posterior_entropy(
        self, differential: Differential, concept: str, polarity: Polarity
    ) -> float:
        posterior = self._posterior_after(differential, concept, polarity)
        if not posterior:
            return math.log(2)
        return -sum(p * math.log(p) for p in posterior.values() if p > 0)

    def _probability_of_presence(
        self, differential: Differential, concept: str
    ) -> float:
        """Posterior-weighted belief that the feature will be observed present."""
        p_present = 0.0
        for h in differential.hypotheses:
            entry = self.kb.get(h.label)
            if entry is None:
                continue
            p_present += h.probability * entry.features.get(
                concept, self.kb.background(concept)
            )
        return min(max(p_present, 0.0), 1.0)

    def flip_action(
        self, state: CaseState, differential: Differential, top_k: int = 5
    ) -> Action | None:
        """The cheapest unasked test whose result would change the leading diagnosis.

        A different question from ``select``, and the distinction is the point.
        Expected information gain asks whether an observation would *sharpen*
        the posterior; this asks whether it would *overturn* it. A test can
        carry substantial entropy reduction while never changing the answer,
        and a test can be decisive while barely moving the entropy, so a
        due-diligence rule keyed to gain will happily commit with the decisive
        test unasked.

        As a detector this works: both fixture cases the loop gets wrong have
        such a test outstanding at the moment they commit, and six of the eight
        it gets right do not.

        As a remedy it does not. Ordering those tests leaves accuracy
        unchanged at 8/10 and raises mean cost from 5.5 to 8.6, and the reason
        is worth stating because it also disposes of three other candidate
        fixes. In fx-009 the true diagnosis sits at 0.5% -- not because the
        evidence excludes it but because four weak negatives multiplied under
        the independence assumption -- and every criterion here reads the
        posterior to decide what to pursue. Ranking candidates by decisiveness
        rather than price does not help, and neither does dropping
        ``min_flip_outcome`` to zero: the agent does not order the CTPA because
        it does not believe the CTPA will be positive, and that belief is the
        thing at fault. A criterion conditioned on the posterior cannot
        investigate what the posterior has dismissed.

        Only outcomes with at least ``min_flip_outcome`` probability of
        occurring count. Without that floor, a result the agent believes is
        almost impossible still counts as a reason to withhold a commit, and
        with enough candidate features some such result nearly always exists.
        """
        leader = differential.top.label
        considered = differential.top_k(top_k)
        best: tuple[float, Action] | None = None

        for concept in self.kb.discriminating_features(considered):
            if concept in state.asked or state.observed(concept) is not None:
                continue
            p_present = self._probability_of_presence(differential, concept)
            outcomes = (
                (Polarity.PRESENT, p_present),
                (Polarity.ABSENT, 1.0 - p_present),
            )
            for polarity, chance in outcomes:
                if chance < self.min_flip_outcome:
                    continue
                posterior = self._posterior_after(differential, concept, polarity)
                alternative = max(posterior, key=posterior.get)
                if alternative == leader:
                    continue
                cost = max(self.kb.cost_of(concept), 1e-6)
                action = Action(
                    kind=classify(concept),
                    target=concept,
                    cost=cost,
                    expected_information_gain=self.expected_gain(differential, concept),
                    rationale=(
                        f"decisive: {polarity.value} would move the leading "
                        f"diagnosis from {leader} to {alternative} "
                        f"(p={chance:.0%})"
                    ),
                )
                if best is None or cost < best[0]:
                    best = (cost, action)
                break

        return best[1] if best else None

    def ruleout_action(
        self, state: CaseState, differential: Differential
    ) -> Action | None:
        """Find a test that would exclude a live time-critical diagnosis.

        Deliberately *not* driven by information gain. Expected entropy
        reduction is computed under the current posterior, so a diagnosis the
        agent has already pushed down to 8% contributes almost nothing to it --
        and the agent therefore never investigates it. That is fine for a
        differential between benign conditions and unacceptable for a pulmonary
        embolism, where 8% is not a rounding error but a reason to order the
        test.

        The rule: while any red-flag diagnosis holds more than ``ruleout_floor``,
        seek the single most discriminating affordable feature for it, chosen by
        likelihood ratio against the rest of the differential. Note that the
        floor is much lower than the gate's ``red_flag_tolerance`` -- the bar for
        *investigating* a dangerous diagnosis should sit well below the bar for
        *reporting* one.
        """
        live = [
            h
            for h in differential.hypotheses
            if h.probability > self.ruleout_floor
            and (entry := self.kb.get(h.label)) is not None
            and entry.red_flag
        ]
        if not live:
            return None

        best: tuple[float, Action] | None = None
        for hypothesis in live:
            entry = self.kb.get(hypothesis.label)
            for concept, p_given in entry.features.items():
                if concept in state.asked or state.observed(concept) is not None:
                    continue
                # Posterior-weighted likelihood among the other hypotheses.
                others = [h for h in differential.hypotheses if h.label != hypothesis.label]
                mass = sum(h.probability for h in others) or 1e-9
                p_rest = sum(
                    h.probability
                    * (self.kb.get(h.label).features.get(concept, self.kb.background(concept))
                       if self.kb.get(h.label) else 0.5)
                    for h in others
                ) / mass
                discrimination = abs(p_given - p_rest)
                if discrimination < self.min_discrimination:
                    continue
                cost = max(self.kb.cost_of(concept), 1e-6)
                score = discrimination / (cost + self.cost_offset)
                action = Action(
                    kind=classify(concept),
                    target=concept,
                    cost=cost,
                    expected_information_gain=self.expected_gain(differential, concept),
                    rationale=(
                        f"rule out {hypothesis.label} "
                        f"({hypothesis.probability:.0%}); discrimination "
                        f"{discrimination:.2f}"
                    ),
                )
                if best is None or score > best[0]:
                    best = (score, action)
        return best[1] if best else None

    def _red_flag_weight(self, concept: str, labels: tuple[str, ...]) -> float:
        """Up-weight actions that discriminate a time-critical diagnosis."""
        for label in labels:
            entry = self.kb.get(label)
            if entry and entry.red_flag and concept in entry.features:
                return self.red_flag_bonus
        return 1.0


__all__ = ["InformationGainSelector", "classify"]
