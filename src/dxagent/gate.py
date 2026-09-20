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


@dataclass
class VerbalisedCalibrator:
    """Recalibrates a model's stated confidence against what it earns.

    The brief pairs conformal prediction with verbalised-confidence
    elicitation: asking the model how sure it is, in words. The number that
    comes back is usable and must not be used raw. Verbalised confidence is
    systematically overconfident -- a well-documented finding, and one this
    project has independent reason to expect, since a model asked to grade
    itself is not observing anything it did not already use to produce the
    answer.

    So it is treated as a *score*, not a probability: something monotonically
    related to correctness whose mapping onto correctness must be measured.
    ``fit`` bins held-out cases by stated confidence and records the accuracy
    actually achieved in each bin; ``apply`` maps a new statement through that
    empirical curve.

    Binned rather than a fitted curve, because the shape of the miscalibration
    is not known in advance and assuming one -- a temperature, a Platt sigmoid
    -- would impose it. Bins assume only monotonicity, and ``overconfidence``
    reports the gap so the raw claim and the earned one stay comparable.
    """

    bins: int = 5
    min_fit_samples: int = 30
    fitted: bool = False
    fit_n: int = 0
    # Empirical accuracy per bin, index 0 the lowest stated confidence.
    curve: tuple[float, ...] = ()
    mean_stated: float = 0.0
    mean_correct: float = 0.0

    @property
    def overconfidence(self) -> float:
        """Positive when the model claims more than it delivers."""
        return self.mean_stated - self.mean_correct

    def _bin(self, stated: float) -> int:
        index = int(min(max(stated, 0.0), 0.999999) * self.bins)
        return min(index, self.bins - 1)

    def fit(self, samples: list[tuple[float, bool]]) -> "VerbalisedCalibrator":
        """``samples`` are (stated confidence, was the answer correct)."""
        if len(samples) < self.min_fit_samples:
            # Same refusal as TemperatureScaler: an accuracy computed from three
            # cases per bin is noise, and a calibrator that reports it as a
            # correction is worse than one that declines.
            self.fitted = False
            self.fit_n = len(samples)
            return self

        buckets: list[list[bool]] = [[] for _ in range(self.bins)]
        for stated, correct in samples:
            buckets[self._bin(stated)].append(correct)

        overall = sum(1 for _, c in samples if c) / len(samples)
        curve: list[float] = []
        for bucket in buckets:
            # An empty bin means the model never made a claim in that range;
            # fall back to the overall accuracy rather than to zero, which
            # would read as "claims in this range are always wrong".
            curve.append(sum(bucket) / len(bucket) if bucket else overall)

        self.curve = tuple(curve)
        self.fitted = True
        self.fit_n = len(samples)
        self.mean_stated = sum(s for s, _ in samples) / len(samples)
        self.mean_correct = overall
        return self

    def apply(self, stated: float) -> float:
        """The accuracy this level of stated confidence actually earned."""
        if not self.fitted:
            return stated
        return self.curve[self._bin(stated)]


@dataclass
class ConformalPredictor:
    """Split conformal prediction over the differential.

    Temperature scaling makes a probability *mean* what it says on average.
    Conformal prediction answers a different question: give me a set of
    diagnoses that contains the truth at least (1 - alpha) of the time. The
    guarantee is distribution-free and finite-sample -- it needs no assumption
    that the model is well specified, which matters here because the naive-Bayes
    independence assumption is known to be wrong.

    The two belong together rather than in competition. A calibrated top-1
    probability says how much to trust the leading answer; a conformal set says
    how many answers still have to stay on the table. A set of size one is a
    clean commit; a set of size six is a differential that has not been
    narrowed, whatever the top-1 confidence claims.

    Method. On held-out cases, score each by nonconformity ``1 - p(truth)``.
    Take the ceil((n+1)(1-alpha))/n empirical quantile of those scores as the
    threshold ``q``, and at prediction time return every label with
    ``p(label) >= 1 - q``. The (n+1) correction is what makes the coverage
    guarantee exact rather than approximate, and it is the part most
    implementations quietly drop.

    Validity depends on exchangeability between the calibration and test sets.
    Fit it on cases drawn the same way as those it will be used on, and refit
    when the case mix changes -- a threshold carried over from a different
    population is a guarantee in name only.
    """

    alpha: float = 0.1  # target miscoverage; 0.1 means 90% coverage
    threshold: float = 0.0  # 1 - q; labels at or above this enter the set
    fitted: bool = False
    fit_n: int = 0
    min_fit_samples: int = 30

    def fit(
        self, samples: list[tuple[Differential, str]]
    ) -> "ConformalPredictor":
        """Calibrate on (differential, true label) pairs."""
        if len(samples) < self.min_fit_samples:
            # Same reasoning as TemperatureScaler.fit: a quantile of a handful
            # of scores is noise wearing the costume of a guarantee. Report the
            # refusal rather than emitting an interval nobody should rely on.
            self.fitted = False
            self.fit_n = len(samples)
            return self

        scores = sorted(
            1.0 - differential.probability_of(truth) for differential, truth in samples
        )
        n = len(scores)
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        if rank > n:
            # Too few samples to certify this alpha at all: the quantile falls
            # outside the sample. Admit everything rather than pretend.
            q = 1.0
        else:
            q = scores[rank - 1]
        self.threshold = 1.0 - q
        self.fitted = True
        self.fit_n = n
        return self

    def predict_set(self, differential: Differential) -> tuple[str, ...]:
        """Labels retained at the calibrated coverage level, most likely first.

        Never returns empty: if no label clears the threshold the top one is
        returned anyway, because an empty prediction set is not a usable answer
        for a clinician and the honest signal in that case is set size one with
        low confidence, which the gate already reads.
        """
        if not self.fitted:
            return (differential.top.label,)
        retained = tuple(
            h.label for h in differential.hypotheses if h.probability >= self.threshold
        )
        return retained or (differential.top.label,)

    def coverage(
        self, samples: list[tuple[Differential, str]]
    ) -> tuple[float, float]:
        """Empirical (coverage, mean set size) on held-out cases.

        Report both. Coverage alone is trivially satisfiable by returning every
        label, and set size alone says nothing about correctness; the pair is
        the result.
        """
        if not samples:
            return float("nan"), float("nan")
        hits = 0
        total_size = 0
        for differential, truth in samples:
            predicted = self.predict_set(differential)
            hits += truth in predicted
            total_size += len(predicted)
        n = len(samples)
        return hits / n, total_size / n


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

    # Largest conformal prediction set the gate will still commit on. One means
    # "commit only when the calibrated set has narrowed to a single diagnosis".
    # Inert until the predictor is fitted, so an unfitted gate behaves exactly
    # as it did before conformal prediction existed.
    max_conformal_set: int = 1
    scaler: TemperatureScaler = field(default_factory=TemperatureScaler)
    conformal: ConformalPredictor = field(default_factory=ConformalPredictor)

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

        # Read before the point estimate: the set is a statement about how many
        # diagnoses remain admissible at the calibrated coverage level, which is
        # a stronger thing to act on than one number's distance from a
        # hand-picked threshold.
        if self.conformal.fitted:
            predicted = self.conformal.predict_set(differential)
            if len(predicted) > self.max_conformal_set:
                shown = ", ".join(predicted[:4])
                more = f" and {len(predicted) - 4} more" if len(predicted) > 4 else ""
                return GateDecision(
                    False,
                    f"at {1 - self.conformal.alpha:.0%} coverage the differential "
                    f"still admits {len(predicted)} diagnoses ({shown}{more}); "
                    f"more than the {self.max_conformal_set} this gate commits on",
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

        # NaN means no second opinion this turn, which is the situation with
        # no consensus proposer at all: it neither blocks nor counts as
        # agreement. A comparison with NaN is False, so this is explicit
        # rather than relied upon.
        if disagreement == disagreement and disagreement > self.max_disagreement:
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


__all__ = [
    "AbstentionGate",
    "ConformalPredictor",
    "GateDecision",
    "TemperatureScaler",
    "VerbalisedCalibrator",
]
