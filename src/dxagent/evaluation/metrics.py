"""Metrics.

Three families, and the third is the one that matters for this project:

  Ranking       -- top-k accuracy, mean reciprocal rank. Standard, and
                   insufficient on its own: they say nothing about whether the
                   confidence attached to a prediction can be trusted.
  Calibration   -- expected calibration error, Brier score. Whether a stated
                   70% means 70%.
  Selective     -- coverage, selective accuracy, risk-coverage curve, AURC. How
                   good the abstention decision is.

A note on reporting, because it is easy to accidentally cheat here: selective
accuracy always looks better than full-coverage accuracy for any non-trivial
gate, so quoting it alone overstates the system. Coverage must be reported
alongside it, and the honest headline comparison is a risk-coverage curve
against the no-abstention baseline, not a single accuracy number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schemas import CaseOutcome


@dataclass(frozen=True)
class RankingMetrics:
    n: int
    top1: float
    top3: float
    top5: float
    mrr: float


@dataclass(frozen=True)
class CalibrationMetrics:
    n: int
    ece: float
    brier: float
    mean_confidence: float
    accuracy: float

    @property
    def overconfidence(self) -> float:
        """Positive when the system claims more certainty than it earns."""
        return self.mean_confidence - self.accuracy


@dataclass(frozen=True)
class SelectiveMetrics:
    n: int
    coverage: float
    selective_accuracy: float
    full_coverage_accuracy: float
    abstention_precision: float
    aurc: float
    mean_cost: float
    mean_turns: float

    @property
    def accuracy_gain(self) -> float:
        return self.selective_accuracy - self.full_coverage_accuracy


@dataclass(frozen=True)
class DifferentialMetrics:
    """Agreement with a gold *differential* rather than a single gold label."""

    n: int
    recall_at_5: float
    recall_at_gold_length: float
    precision_at_gold_length: float
    f1_at_gold_length: float
    mean_gold_length: float


@dataclass(frozen=True)
class AbstentionMetrics:
    """Whether the gate defers on the cases the gold differential calls uncertain.

    A stand-in for the physician judgement the project's evaluation plan asks
    for. It is weaker than that judgement in one specific way worth stating
    when reporting it: a broad gold differential means the *labelling process*
    was uncertain, which is correlated with but not identical to a case being
    genuinely hard.
    """

    n: int
    n_ambiguous: float
    abstention_rate: float
    proxy_precision: float  # of the cases it deferred on, how many were ambiguous
    proxy_recall: float  # of the ambiguous cases, how many did it defer on
    mean_ambiguity: float


def differential_metrics(
    outcomes: list[CaseOutcome],
    gold: dict[str, tuple[tuple[str, float], ...]],
    k: int = 5,
) -> DifferentialMetrics:
    """DDx recall and precision against the gold differential.

    Two lengths are reported because they answer different questions. Recall at
    a fixed k asks "does a k-item list cover the gold", which is the number
    comparable across systems. Recall at the gold's own length asks "given the
    gold named m causes, did the top m contain them", which does not reward a
    system for padding the list. Precision is only meaningful at the second.

    Cases with no gold differential are skipped rather than scored zero, so a
    mixed case set does not silently deflate the result.
    """
    scored = [o for o in outcomes if gold.get(o.case_id)]
    if not scored:
        return DifferentialMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    recall_k = 0.0
    recall_m = 0.0
    precision_m = 0.0
    lengths = 0

    for outcome in scored:
        truth = {label for label, _ in gold[outcome.case_id]}
        predicted = [h.label for h in outcome.differential.hypotheses]
        m = len(truth)
        lengths += m

        recall_k += len(truth & set(predicted[:k])) / m
        hits_m = len(truth & set(predicted[:m]))
        recall_m += hits_m / m
        precision_m += hits_m / max(len(predicted[:m]), 1)

    n = len(scored)
    recall_at_m = recall_m / n
    precision_at_m = precision_m / n
    denominator = recall_at_m + precision_at_m
    return DifferentialMetrics(
        n=n,
        recall_at_5=recall_k / n,
        recall_at_gold_length=recall_at_m,
        precision_at_gold_length=precision_at_m,
        f1_at_gold_length=(
            2 * recall_at_m * precision_at_m / denominator if denominator else 0.0
        ),
        mean_gold_length=lengths / n,
    )


def abstention_metrics(
    outcomes: list[CaseOutcome],
    ambiguity: dict[str, float],
    threshold: float = 0.5,
) -> AbstentionMetrics:
    """Does the gate defer on the cases the gold differential calls uncertain?

    ``threshold`` is on normalised gold entropy, so 0.5 means "mass spread over
    the equivalent of a genuine multi-way tie". It is a reporting choice, not a
    tuned parameter -- sweep it rather than quoting one value, for the same
    reason a single point on a risk-coverage curve is not a result.
    """
    scored = [o for o in outcomes if o.case_id in ambiguity]
    if not scored:
        return AbstentionMetrics(0, 0.0, 0.0, float("nan"), float("nan"), 0.0)

    ambiguous = {o.case_id for o in scored if ambiguity[o.case_id] >= threshold}
    deferred = {o.case_id for o in scored if o.abstained}
    overlap = ambiguous & deferred

    n = len(scored)
    return AbstentionMetrics(
        n=n,
        n_ambiguous=len(ambiguous),
        abstention_rate=len(deferred) / n,
        proxy_precision=(len(overlap) / len(deferred)) if deferred else float("nan"),
        proxy_recall=(len(overlap) / len(ambiguous)) if ambiguous else float("nan"),
        mean_ambiguity=sum(ambiguity[o.case_id] for o in scored) / n,
    )


def ranking_metrics(outcomes: list[CaseOutcome], truths: dict[str, str]) -> RankingMetrics:
    """Computed over all cases, ignoring the commit/escalate decision.

    Deliberately: this measures the differential itself, so that a change in the
    gate cannot flatter or damage the ranking numbers.
    """
    if not outcomes:
        return RankingMetrics(0, 0.0, 0.0, 0.0, 0.0)
    hits = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    for outcome in outcomes:
        truth = truths[outcome.case_id]
        labels = [h.label for h in outcome.differential.hypotheses]
        for k in hits:
            if truth in labels[:k]:
                hits[k] += 1
        if truth in labels:
            reciprocal += 1.0 / (labels.index(truth) + 1)
    n = len(outcomes)
    return RankingMetrics(
        n=n,
        top1=hits[1] / n,
        top3=hits[3] / n,
        top5=hits[5] / n,
        mrr=reciprocal / n,
    )


def calibration_metrics(
    outcomes: list[CaseOutcome], truths: dict[str, str], bins: int = 10
) -> CalibrationMetrics:
    """ECE with equal-width bins, plus the top-1 Brier score.

    Equal-width rather than equal-mass bins: with the case counts available at
    this stage, equal-mass bins would each hold a handful of cases and the
    resulting ECE would be dominated by sampling noise. Revisit once n is in the
    hundreds -- and treat ECE from fewer than ~100 cases as indicative only.
    """
    if not outcomes:
        return CalibrationMetrics(0, 0.0, 0.0, 0.0, 0.0)

    records = []
    for outcome in outcomes:
        truth = truths[outcome.case_id]
        confidence = outcome.differential.top.probability
        correct = outcome.differential.top.label == truth
        records.append((confidence, correct))

    n = len(records)
    accuracy = sum(1 for _, c in records if c) / n
    mean_confidence = sum(p for p, _ in records) / n
    brier = sum((p - (1.0 if c else 0.0)) ** 2 for p, c in records) / n

    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            (p, c) for p, c in records if (p > low or index == 0) and p <= high
        ]
        if not bucket:
            continue
        bucket_conf = sum(p for p, _ in bucket) / len(bucket)
        bucket_acc = sum(1 for _, c in bucket if c) / len(bucket)
        ece += (len(bucket) / n) * abs(bucket_conf - bucket_acc)

    return CalibrationMetrics(
        n=n,
        ece=ece,
        brier=brier,
        mean_confidence=mean_confidence,
        accuracy=accuracy,
    )


def selective_metrics(
    outcomes: list[CaseOutcome], truths: dict[str, str]
) -> SelectiveMetrics:
    """Coverage/accuracy trade-off actually realised by the gate."""
    if not outcomes:
        return SelectiveMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(outcomes)
    committed = [o for o in outcomes if not o.abstained]
    escalated = [o for o in outcomes if o.abstained]

    correct_committed = sum(
        1 for o in committed if o.prediction == truths[o.case_id]
    )
    # Counterfactual: had the agent been forced to answer, would it have been
    # wrong? This is what makes an abstention worth its cost -- an abstention on
    # a case the system would have got right is a false alarm on the clinician's
    # time.
    would_have_erred = sum(
        1 for o in escalated if o.differential.top.label != truths[o.case_id]
    )
    all_correct = sum(
        1 for o in outcomes if o.differential.top.label == truths[o.case_id]
    )

    return SelectiveMetrics(
        n=n,
        coverage=len(committed) / n,
        selective_accuracy=(correct_committed / len(committed)) if committed else float("nan"),
        full_coverage_accuracy=all_correct / n,
        abstention_precision=(would_have_erred / len(escalated)) if escalated else float("nan"),
        aurc=area_under_risk_coverage(outcomes, truths),
        mean_cost=sum(o.budget_spent for o in outcomes) / n,
        mean_turns=sum(len(o.steps) for o in outcomes) / n,
    )


def risk_coverage_curve(
    outcomes: list[CaseOutcome], truths: dict[str, str]
) -> list[tuple[float, float]]:
    """Sweep a confidence threshold; return (coverage, error rate) points.

    Cases are ordered by confidence descending, then answered greedily. This
    measures the *ranking quality of the confidence signal* independently of
    where the gate's thresholds happen to sit, which is the right way to compare
    two confidence estimators.
    """
    if not outcomes:
        return []
    ordered = sorted(
        outcomes, key=lambda o: o.differential.top.probability, reverse=True
    )
    points: list[tuple[float, float]] = []
    errors = 0
    for index, outcome in enumerate(ordered, start=1):
        if outcome.differential.top.label != truths[outcome.case_id]:
            errors += 1
        points.append((index / len(ordered), errors / index))
    return points


def area_under_risk_coverage(
    outcomes: list[CaseOutcome], truths: dict[str, str]
) -> float:
    """AURC by trapezoidal integration. Lower is better."""
    curve = risk_coverage_curve(outcomes, truths)
    if len(curve) < 2:
        return curve[0][1] if curve else 0.0
    total = 0.0
    for (c0, r0), (c1, r1) in zip(curve, curve[1:]):
        total += (c1 - c0) * (r0 + r1) / 2
    return total / (curve[-1][0] - curve[0][0]) if curve[-1][0] > curve[0][0] else total


def format_report(
    ranking: RankingMetrics,
    calibration: CalibrationMetrics,
    selective: SelectiveMetrics,
    differential: DifferentialMetrics | None = None,
    abstention: AbstentionMetrics | None = None,
) -> str:
    def pct(value: float) -> str:
        return "  n/a" if math.isnan(value) else f"{value:6.1%}"

    lines = [
        f"cases evaluated              {ranking.n}",
        "",
        "ranking (all cases, gate ignored)",
        f"  top-1                    {pct(ranking.top1)}",
        f"  top-3                    {pct(ranking.top3)}",
        f"  top-5                    {pct(ranking.top5)}",
        f"  MRR                       {ranking.mrr:.3f}",
        "",
        "calibration",
        f"  ECE                       {calibration.ece:.3f}",
        f"  Brier (top-1)             {calibration.brier:.3f}",
        f"  mean confidence          {pct(calibration.mean_confidence)}",
        f"  accuracy                 {pct(calibration.accuracy)}",
        f"  overconfidence           {calibration.overconfidence:+7.1%}",
        "",
        "selective prediction (gate active)",
        f"  coverage                 {pct(selective.coverage)}",
        f"  selective accuracy       {pct(selective.selective_accuracy)}",
        f"  full-coverage accuracy   {pct(selective.full_coverage_accuracy)}",
        f"  accuracy gained          {selective.accuracy_gain:+7.1%}",
        f"  abstention precision     {pct(selective.abstention_precision)}",
        f"  AURC (lower better)       {selective.aurc:.3f}",
        "",
        "cost",
        f"  mean cost per case        {selective.mean_cost:.2f}",
        f"  mean turns per case       {selective.mean_turns:.2f}",
    ]

    if differential is not None and differential.n:
        lines += [
            "",
            f"differential agreement ({differential.n} cases with a gold differential)",
            f"  DDx recall @5            {pct(differential.recall_at_5)}",
            f"  DDx recall @gold         {pct(differential.recall_at_gold_length)}",
            f"  DDx precision @gold      {pct(differential.precision_at_gold_length)}",
            f"  DDx F1 @gold             {pct(differential.f1_at_gold_length)}",
            f"  mean gold length          {differential.mean_gold_length:.1f}",
        ]

    if abstention is not None and abstention.n:
        lines += [
            "",
            "abstention against the gold-differential proxy",
            f"  ambiguous cases           {abstention.n_ambiguous:.0f} / {abstention.n}",
            f"  abstention rate          {pct(abstention.abstention_rate)}",
            f"  proxy precision          {pct(abstention.proxy_precision)}",
            f"  proxy recall             {pct(abstention.proxy_recall)}",
            f"  mean gold ambiguity       {abstention.mean_ambiguity:.3f}",
        ]

    return "\n".join(lines)


__all__ = [
    "AbstentionMetrics",
    "CalibrationMetrics",
    "DifferentialMetrics",
    "RankingMetrics",
    "SelectiveMetrics",
    "abstention_metrics",
    "area_under_risk_coverage",
    "calibration_metrics",
    "differential_metrics",
    "format_report",
    "ranking_metrics",
    "risk_coverage_curve",
    "selective_metrics",
]
