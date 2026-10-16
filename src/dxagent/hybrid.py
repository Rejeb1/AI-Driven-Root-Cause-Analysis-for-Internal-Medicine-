"""A language model proposes; the gate's safety conditions still decide.

The measured gap this exists for: given each real case's complete record, a
free Gemini model ranked the true diagnosis first on 85% of runs when it
could read the patient's own words, 64% on the extracted findings alone, and
the Bayesian model 56% (DESIGN.md). The words carry what the vocabulary
cannot -- "worse leaning forward", "fear of dying". Inside the loop as a
second opinion it changed no outcome in 461 turns, because the consensus
design can only *block* a commit, and the Bayesian loop rarely wanted to
make one the model would block.

So the model is given the opposite job, and only one narrow occasion to do
it. When the Bayesian loop has finished and escalated **only because it is
not confident enough** -- top-1 below the threshold, or a contested margin
-- the model reads the complaint and the findings the loop gathered, once,
and names a diagnosis. That proposal commits only if every safety condition
the gate enforces still holds for it, evaluated on the Bayesian posterior
and the guideline rules, not on the model's say-so:

1. no triggered workup is outstanding (a missing troponin is never
   overridden);
2. no guideline criterion met by the findings names a different diagnosis;
3. no time-critical diagnosis other than the proposal keeps more than the
   red-flag tolerance in the Bayesian posterior;
4. the Bayesian posterior does not rule the proposal out (it keeps at least
   the same tolerance);
5. the model states a confidence at or above the gate's own threshold.

Every threshold is one the gate already has; none is new or fitted.
Escalations for any other reason -- a red flag, a veto, unexplained
evidence, an unavailable workup -- are left exactly as they are.
"""

from __future__ import annotations

import dataclasses
import math

from .belief import BayesianProposer, LLMProposer
from .gate import AbstentionGate
from .guidelines import diagnostic_criteria_met, outstanding_workup
from .knowledge import InMemoryKnowledgeBase
from .schemas import CaseOutcome, Verdict

# The gate reasons this layer may act on. Everything else is a safety
# escalation and stays one.
_CONFIDENCE_ONLY = ("top-1 confidence", "top-two margin")


def eligible(outcome: CaseOutcome) -> bool:
    """Escalated, and the gate's only objection was confidence or margin."""
    if not outcome.abstained or outcome.escalation is None:
        return False
    reason = outcome.escalation.reason
    return reason.startswith(_CONFIDENCE_ONLY) and "could not be completed" not in reason


def second_reading(
    outcome: CaseOutcome,
    complaint: str,
    kb: InMemoryKnowledgeBase,
    gate: AbstentionGate,
    reader: LLMProposer,
) -> tuple[CaseOutcome, str]:
    """The outcome after the second reading, and why it did or did not commit."""
    if not eligible(outcome):
        return outcome, "not eligible: a safety escalation, not a confidence one"
    findings = list(outcome.escalation.findings)
    if outstanding_workup(findings):
        return outcome, "not eligible: a triggered workup is outstanding"

    proposal = reader.propose(findings, complaint)
    if reader.last_was_fallback:
        return outcome, "no reading: the model did not answer"
    label = proposal.top.label
    stated = reader.last_verbalised_confidence
    if math.isnan(stated) or stated < gate.min_confidence:
        return outcome, f"declined: {label} at stated confidence {stated:.0%}"

    bayes = gate.calibrate(BayesianProposer(kb).propose(findings, complaint))
    vetoing = [c for c in diagnostic_criteria_met(findings) if c != label]
    if vetoing:
        return outcome, f"declined: {label}, but criteria for {', '.join(vetoing)} are met"
    rivals = [
        h for h in bayes.hypotheses
        if h.label != label
        and (entry := kb.get(h.label)) is not None
        and entry.red_flag
        and h.probability > gate.red_flag_tolerance
    ]
    if rivals:
        names = ", ".join(f"{h.label} ({h.probability:.0%})" for h in rivals)
        return outcome, f"declined: {label}, but time-critical {names} not excluded"
    if bayes.probability_of(label) < gate.red_flag_tolerance:
        return outcome, (
            f"declined: {label}, which the Bayesian posterior rules out "
            f"({bayes.probability_of(label):.0%})"
        )

    committed = dataclasses.replace(
        outcome,
        verdict=Verdict.COMMITTED,
        differential=proposal,
        confidence=stated,
        escalation=None,
        committed_by="language model, safety conditions checked",
    )
    return committed, f"committed: {label} at stated {stated:.0%}"


__all__ = ["eligible", "second_reading"]
