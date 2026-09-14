"""Where each likelihood came from, and how sure we are of it.

Every number in the knowledge base is currently invented, and the honest
response to that is not to replace them with better-looking invented numbers
but to record which ones have a source and which do not. A knowledge base that
reports "30 of 135 likelihoods are decisive, 22 of those are sourced, 8 remain
estimates" is worth more than one that looks uniformly precise and is not.

Three tiers, in descending order of what they license you to claim:

``MEASURED``
    A frequency from a study: "fever occurred in 14% of confirmed PE in this
    cohort". Carries a citation to the paper. The only tier that supports a
    claim about the world rather than about a text.

``NARRATIVE``
    A qualitative statement from a reference text -- "fever is common in X" --
    converted by the fixed rubric below. Legitimate, and weaker than it looks:
    the rubric is a convention, so two people applying it agree with each other
    but neither is measuring anything.

``INVENTED``
    A plausible-looking guess. Useful for exercising code paths and for nothing
    else.

The rubric is deliberately coarse and fixed in advance. A finer one would imply
a precision the source text does not contain, and choosing the mapping after
seeing which values produce good results is how a knowledge base becomes fitted
to its own benchmark without anyone deciding to do that.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .schemas import Citation


class Provenance(str, Enum):
    MEASURED = "measured"
    NARRATIVE = "narrative"
    INVENTED = "invented"


# Qualitative phrasing -> (point estimate, low, high).
# The band is the part that matters: it is what a sensitivity sweep varies, and
# carrying it is what stops a converted phrase from hardening into a fact.
NARRATIVE_RUBRIC: dict[str, tuple[float, float, float]] = {
    "always": (0.95, 0.90, 0.99),
    "characteristic": (0.85, 0.70, 0.95),
    "hallmark": (0.85, 0.70, 0.95),
    "usually": (0.75, 0.60, 0.90),
    "common": (0.60, 0.40, 0.80),
    "frequent": (0.60, 0.40, 0.80),
    "often": (0.55, 0.35, 0.75),
    # Synonym of "may occur", added because the Merck Manual uses both and a
    # rubric that recognises one and not the other converts by accident of
    # phrasing rather than by meaning.
    "can occur": (0.30, 0.15, 0.50),
    "may occur": (0.30, 0.15, 0.50),
    "sometimes": (0.30, 0.15, 0.50),
    # Comparative rather than absolute: "less common symptoms include cough"
    # says cough is rarer than the cardinal ones without saying how rare. The
    # band is therefore wider than "occasional" and spans it, which is the
    # honest way to represent a phrase that carries less information than the
    # absolute terms around it.
    "less common": (0.20, 0.08, 0.40),
    "occasional": (0.20, 0.08, 0.35),
    "uncommon": (0.08, 0.02, 0.20),
    "rare": (0.05, 0.01, 0.15),
    "not typical": (0.05, 0.01, 0.15),
    "absent": (0.02, 0.01, 0.08),
}


@dataclass(frozen=True)
class LikelihoodSource:
    """Provenance for one P(finding | disease) entry."""

    provenance: Provenance
    citation: Citation | None = None
    low: float | None = None
    high: float | None = None
    note: str = ""

    @property
    def is_sourced(self) -> bool:
        return self.provenance is not Provenance.INVENTED

    @property
    def band(self) -> tuple[float, float] | None:
        if self.low is None or self.high is None:
            return None
        return (self.low, self.high)


def from_narrative(
    phrase: str, citation: Citation, note: str = ""
) -> tuple[float, LikelihoodSource]:
    """Convert a reference-text phrase into a value and its provenance.

    Raises on an unrecognised phrase rather than falling back to a default.
    A silent default here would quietly manufacture a number that looks
    narrative-derived and is not, which is precisely the confusion this module
    exists to prevent -- extend the rubric instead, so the extension is a
    visible decision.
    """
    key = phrase.strip().lower()
    if key not in NARRATIVE_RUBRIC:
        raise KeyError(
            f"{phrase!r} is not in the rubric. Known phrases: "
            f"{', '.join(sorted(NARRATIVE_RUBRIC))}. Add it deliberately rather "
            "than mapping it to a default."
        )
    value, low, high = NARRATIVE_RUBRIC[key]
    return value, LikelihoodSource(
        provenance=Provenance.NARRATIVE,
        citation=citation,
        low=low,
        high=high,
        note=note or f'reference text: "{phrase}"',
    )


def measured(
    value: float, citation: Citation, low: float | None = None,
    high: float | None = None, note: str = "",
) -> tuple[float, LikelihoodSource]:
    """A frequency taken from a study, with its provenance.

    Returns the pair rather than the source alone, matching
    ``from_narrative``, so both go into ``_SOURCED`` the same way. An earlier
    version took ``value`` and discarded it, which meant the value had to be
    written twice per entry and could disagree with itself.
    """
    return value, LikelihoodSource(
        provenance=Provenance.MEASURED,
        citation=citation,
        low=low,
        high=high,
        note=note,
    )


@dataclass(frozen=True)
class ProvenanceReport:
    total: int
    measured: int
    narrative: int
    invented: int
    # Disease priors, counted separately rather than folded into the totals
    # above. They are a different quantity -- P(disease) against
    # P(finding | disease) -- sourced from different literature, and merging
    # them would let a well-sourced likelihood table hide eight unsourced
    # priors inside a single flattering percentage.
    priors_total: int = 0
    priors_sourced: int = 0

    @property
    def sourced(self) -> int:
        return self.measured + self.narrative

    @property
    def coverage(self) -> float:
        return self.sourced / self.total if self.total else 0.0

    @property
    def priors_invented(self) -> int:
        return self.priors_total - self.priors_sourced

    def summary(self) -> str:
        lines = [
            f"{self.total} likelihoods: {self.measured} measured, "
            f"{self.narrative} narrative-derived, {self.invented} invented "
            f"({self.coverage:.0%} sourced)"
        ]
        if self.priors_total:
            lines.append(
                f"{self.priors_total} disease priors: {self.priors_sourced} "
                f"sourced, {self.priors_invented} invented"
            )
        return "\n".join(lines)


def report(kb) -> ProvenanceReport:
    """Count likelihoods and priors by provenance across a knowledge base."""
    total = measured_count = narrative_count = 0
    priors_total = priors_sourced = 0
    for entry in kb.diseases():
        priors_total += 1
        prior = getattr(entry, "prior_source", None)
        if prior is not None and prior.is_sourced:
            priors_sourced += 1
        for concept in entry.features:
            total += 1
            source = entry.sources.get(concept)
            if source is None:
                continue
            if source.provenance is Provenance.MEASURED:
                measured_count += 1
            elif source.provenance is Provenance.NARRATIVE:
                narrative_count += 1
    return ProvenanceReport(
        total=total,
        measured=measured_count,
        narrative=narrative_count,
        invented=total - measured_count - narrative_count,
        priors_total=priors_total,
        priors_sourced=priors_sourced,
    )


# The numbers this module did not count
# ---------------------------------------------------------------------------
# ``report`` above covers the likelihood table and 8 disease priors, and for a
# long time this project quoted its coverage over those and called that its
# honesty metric. It was not counting the rest of the model.
#
# There were 47 further invented numbers when first counted -- five
# correlation weights, twenty-seven acquisition costs, five gate thresholds,
# three loop budget limits and seven selector constants -- and 50 after the
# pericardial columns added two costs and one correlation pairing. None of them is a likelihood, so none
# was tracked, and their absence from the audit was not a decision anyone
# made.
#
# SCOPE.md already states why that matters, about the disease priors, which
# had exactly this problem until they were pulled into the report:
#
#     They were invented and untracked for most of this project's life, which
#     was the more dangerous state -- an invented number the audit cannot name
#     reads as an absence of a problem.
#
# The hole was closed for priors and nobody asked whether it existed
# elsewhere. It did, five times over, and the worst case is the correlation
# weights: five judgement calls that are the difference between zero and three
# wrong commits on the real patients, with nothing anywhere recording that
# they are invented.
#
# They are reported separately rather than folded into the likelihood
# percentage, for the same reason the priors are. A correlation weight and a
# P(finding | disease) are different quantities answering different questions,
# and averaging them would produce a number that is easier to quote and means
# less.


@dataclass(frozen=True)
class ParameterGroup:
    """One family of non-likelihood numbers, and where it lives."""

    name: str
    count: int
    provenance: Provenance
    where: str
    note: str

    @property
    def is_sourced(self) -> bool:
        return self.provenance is not Provenance.INVENTED


@dataclass(frozen=True)
class ParameterReport:
    groups: tuple[ParameterGroup, ...]

    @property
    def total(self) -> int:
        return sum(g.count for g in self.groups)

    @property
    def invented(self) -> int:
        return sum(g.count for g in self.groups if not g.is_sourced)

    def summary(self) -> str:
        lines = [
            f"{self.total} model parameters outside the likelihood table: "
            f"{self.invented} invented"
        ]
        for group in self.groups:
            tier = group.provenance.value
            lines.append(f"  {group.count:>3} {group.name:<26} {tier:<9} {group.where}")
        return "\n".join(lines)


def parameter_report() -> ParameterReport:
    """Count the invented numbers that are not likelihoods or priors.

    Imports are deferred because ``datasets.fixtures`` imports this module,
    and the point of the function is to reach into the places these numbers
    actually live rather than maintain a second copy of them here that could
    drift.
    """
    from .actions import InformationGainSelector
    from .agent import LoopLimits
    from .datasets import fixtures
    from .gate import AbstentionGate

    def float_fields(cls) -> int:
        return sum(
            1
            for f in cls.__dataclass_fields__.values()
            if isinstance(f.default, float) and not isinstance(f.default, bool)
        )

    def numeric_fields(cls) -> int:
        return sum(
            1
            for f in cls.__dataclass_fields__.values()
            if isinstance(f.default, (int, float)) and not isinstance(f.default, bool)
        )

    return ParameterReport(
        groups=(
            ParameterGroup(
                "correlation weights",
                len(fixtures._CORRELATION_GROUPS),
                Provenance.INVENTED,
                "datasets/fixtures.py",
                "How much each cluster of findings is one clinical picture. "
                "The most load-bearing invented numbers in the system: without "
                "correlation weighting the ten real patients produce three "
                "wrong commits instead of none.",
            ),
            ParameterGroup(
                "acquisition costs",
                len(fixtures.COSTS),
                Provenance.INVENTED,
                "datasets/fixtures.py",
                "What each question or test costs the budget, and so which "
                "ones the selector can afford to ask.",
            ),
            ParameterGroup(
                "gate thresholds",
                float_fields(AbstentionGate),
                Provenance.INVENTED,
                "gate.py",
                "Confidence, margin, red-flag tolerance, proposer "
                "disagreement and evidence fit. Every commit or escalation "
                "turns on these five.",
            ),
            ParameterGroup(
                "loop budget",
                numeric_fields(LoopLimits),
                Provenance.INVENTED,
                "agent.py",
                "Turn and cost ceilings, and the due-diligence gain floor.",
            ),
            ParameterGroup(
                "selector constants",
                float_fields(InformationGainSelector),
                Provenance.INVENTED,
                "actions.py",
                "Information-gain tuning: the cost offset, the red-flag "
                "bonus, and the thresholds that make a test count as "
                "decisive.",
            ),
        )
    )


__all__ = [
    "LikelihoodSource",
    "ParameterGroup",
    "ParameterReport",
    "parameter_report",
    "NARRATIVE_RUBRIC",
    "Provenance",
    "ProvenanceReport",
    "from_narrative",
    "measured",
    "report",
]
