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


__all__ = [
    "LikelihoodSource",
    "NARRATIVE_RUBRIC",
    "Provenance",
    "ProvenanceReport",
    "from_narrative",
    "measured",
    "report",
]
