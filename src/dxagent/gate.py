"""The abstain / escalate gate, and the calibration it depends on.

The gate is the part of this system that most needs to be defensible, so it is
deliberately simple and fully inspectable: a small set of named conditions, each
of which can be reported to the physician as the reason for a handoff. No
learned gating policy, because "the model decided to escalate" is not an
auditable answer.

A gate is only as good as the confidence it reads. Raw posteriors from a
naive-Bayes model over correlated features are systematically overconfident, so
``TemperatureScaler`` is fitted on held-out cases before the gate's thresholds
mean anything. Order matters: calibrate, then threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .knowledge import InMemoryKnowledgeBase
from .schemas import Differential


@dataclass
class TemperatureScaler:
    """Single-parameter calibration: p_i ** (1/T), renormalised.

    T > 1 softens an overconfident posterior; T < 1 sharpens an underconfident
    one. One parameter is chosen over isotonic regression or a full Dirichlet
    recalibration because the number of labelled held-out cases available at
    this stage is small, and a one-parameter fit is the only thing that will not
    overfit it.
    """

    temperature: float = 1.0
    min_fit_samples: int = 30
    fitted: bool = False
    fit_n: int = 0

    def apply(self, differential: Differential) -> Differential:
        if math.isclose(self.temperature, 1.0):
            return differential
        power = 1.0 / self.temperature
        scores = {
            h.label: max(h.probability, 1e-12) ** power
            for h in differential.hypotheses
        }
        return differential.rescaled(scores)

    def fit(
        self,
        samples: list[tuple[Differential, str]],
        grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
    ) -> "TemperatureScaler":
        """Grid-search T minimising negative log-likelihood of the true label.

        Grid search rather than gradient descent: the objective is
        one-dimensional and smooth, the grid is cheap, and it cannot diverge.
        """
        if len(samples) < self.min_fit_samples:
            # Fitting one parameter on a handful of cases produces a confident
            # number with no content. On the first run this fitted T=0.5 from
            # two cases, i.e. it *sharpened* an already-overconfident posterior
            # and made calibration worse. Leaving T=1.0 and saying so is more
            # useful than a fitted value nobody should trust.
            self.fitted = False
            self.fit_n = len(samples)
            return self
        best_t, best_nll = self.temperature, float("inf")
        for t in grid:
            scaler = TemperatureScaler(t)
            nll = 0.0
            for differential, truth in samples:
                p = scaler.apply(differential).probability_of(truth)
                nll -= math.log(max(p, 1e-12))
            if nll < best_nll:
                best_t, best_nll = t, nll
        self.temperature = best_t
        self.fitted = True
        self.fit_n = len(samples)
        return self


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the gate, with the reason exposed."""

    should_commit: bool
    reason: str
    confidence: float

    @property
    def should_escalate(self) -> bool:
        return not self.should_commit


@dataclass
class AbstentionGate:
    """Decides whether to commit to the top diagnosis or hand off.

    Conditions, all of which must pass to commit:

      1. Calibrated top-1 probability >= ``min_confidence``.
      2. Top-two margin >= ``min_margin``. Catches contested posteriors that a
         probability threshold alone would wave through.
      3. Top hypothesis is grounded in the KB (has at least one citation).
      4. If any red-flag diagnosis retains more than ``red_flag_tolerance``
         probability mass, do not commit -- even to a different, more likely
         diagnosis. Asymmetric by design: missing a time-critical diagnosis and
         missing a benign one are not equivalent errors, and a single confidence
         threshold cannot express that.
      5. Proposer disagreement (if a consensus proposer is in use) below
         ``max_disagreement``.
      6. The evidence is explained by *something* in the KB
         (``min_evidence_fit``). Conditions 1 and 2 read the shape of the
         posterior; this one reads whether the findings are accounted for at
         all, and so covers presentations whose cause is absent from the KB.
         It does *not* catch confident errors where the wrong diagnosis
         explains the evidence perfectly well -- see ``BayesianProposer.
         evidence_fit`` for the measurement. Nothing in this gate catches those
         yet.
    """

    kb: InMemoryKnowledgeBase
    min_confidence: float = 0.65
    min_margin: float = 0.15
    red_flag_tolerance: float = 0.10
    max_disagreement: float = 0.40
    require_grounding: bool = True

    # Per-finding log-likelihood ratio of the KB mixture over the population
    # marginal, below which the presentation is treated as unexplained. Zero is
    # the principled setting rather than a tuned one: it is the point at which
    # knowing the KB's diseases tells you nothing about the findings that the
    # base rates did not already. Raise it to demand a positive explanation,
    # and sweep it rather than quoting one value.
    min_evidence_fit: float = 0.0
    scaler: TemperatureScaler = field(default_factory=TemperatureScaler)

    def calibrate(self, differential: Differential) -> Differential:
        return self.scaler.apply(differential)

    def evaluate(
        self,
        differential: Differential,
        disagreement: float = 0.0,
        evidence_fit: float = float("inf"),
    ) -> GateDecision:
        """Assess an already-calibrated differential.

        ``evidence_fit`` defaults to ``inf`` so a caller that does not supply it
        gets the previous behaviour rather than a silent escalation on every
        case.
        """
        confidence = differential.top.probability

        if self.require_grounding and not differential.top.is_grounded:
            return GateDecision(
                False,
                f"top hypothesis '{differential.top.label}' has no knowledge-base "
                "citation, so it cannot be reported as grounded",
                confidence,
            )

        # Only *competing* red flags block a commit. The first version of this
        # rule blocked whenever any red-flag diagnosis held mass, which meant
        # that correctly identifying a pulmonary embolism at 98% confidence
        # triggered an escalation -- the gate refusing to report exactly the
        # finding it exists to protect. Committing to the red flag is the
        # desired outcome; committing to something benign while a red flag is
        # still live is not.
        red_flags = [
            h
            for h in differential.hypotheses
            if h.label != differential.top.label
            and (entry := self.kb.get(h.label)) is not None
            and entry.red_flag
            and h.probability > self.red_flag_tolerance
        ]
        if red_flags:
            names = ", ".join(f"{h.label} ({h.probability:.0%})" for h in red_flags)
            return GateDecision(
                False,
                f"time-critical diagnosis not excluded: {names}",
                confidence,
            )

        # Checked before the confidence conditions, because a case nothing
        # explains should be reported as unexplained rather than as one the
        # model happened to feel uncertain about. The two failures call for
        # different things from the clinician.
        if evidence_fit < self.min_evidence_fit:
            return GateDecision(
                False,
                f"findings are not explained by any diagnosis in the knowledge "
                f"base (evidence fit {evidence_fit:+.2f} below "
                f"{self.min_evidence_fit:+.2f}); the presentation may be "
                f"atypical or its cause may be absent from the knowledge base",
                confidence,
            )

        if confidence < self.min_confidence:
            return GateDecision(
                False,
                f"top-1 confidence {confidence:.0%} below threshold "
                f"{self.min_confidence:.0%}",
                confidence,
            )

        if differential.margin < self.min_margin:
            return GateDecision(
                False,
                f"top-two margin {differential.margin:.0%} below threshold "
                f"{self.min_margin:.0%}; differential remains contested",
                confidence,
            )

        if disagreement > self.max_disagreement:
            return GateDecision(
                False,
                f"proposers disagree (total variation {disagreement:.2f} > "
                f"{self.max_disagreement:.2f})",
                confidence,
            )

        return GateDecision(
            True,
            f"calibrated confidence {confidence:.0%} with margin "
            f"{differential.margin:.0%}",
            confidence,
        )


__all__ = ["AbstentionGate", "GateDecision", "TemperatureScaler"]
