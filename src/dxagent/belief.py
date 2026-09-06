"""Differential proposal and update.

Two independent proposers are provided, behind one interface:

  ``BayesianProposer``  -- naive-Bayes posterior over the KB likelihood tables.
                           Deterministic, auditable, runs offline.
  ``LLMProposer``       -- asks a model to rank the differential given the
                           findings, constrained to KB labels.

``ConsensusProposer`` runs both and reports disagreement. The point is not to
average them into a smoother number; it is that disagreement between a
transparent statistical posterior and a language model's clinical intuition is
itself a reason to escalate. Averaging would destroy that signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .knowledge import InMemoryKnowledgeBase, KnowledgeBase
from .llm import LLMClient
from .schemas import Differential, Finding, Polarity


@runtime_checkable
class Proposer(Protocol):
    def propose(self, findings: list[Finding], complaint: str) -> Differential: ...


@dataclass
class BayesianProposer:
    """Naive-Bayes posterior over KB diseases.

    Works in log space; a case with twenty findings and likelihoods near the
    floor underflows float64 otherwise.

    Also reports ``last_evidence_fit`` -- how well the KB as a whole explains
    the findings, as distinct from how confident it is in the winner. See
    ``evidence_fit`` below for why the two need separating.
    """

    kb: InMemoryKnowledgeBase
    # Positive means the KB's disease mixture explains the evidence better than
    # the population marginal does. inf until a proposal has been made, so a
    # gate reading it before then does not treat "no evidence yet" as "evidence
    # nothing can explain".
    last_evidence_fit: float = float("inf")

    def propose(self, findings: list[Finding], complaint: str = "") -> Differential:
        import math

        entries = self.kb.diseases()
        if not entries:
            raise ValueError("knowledge base is empty")

        # Correlated findings are down-weighted so the product stops counting
        # the same information twice. With no correlations supplied every
        # weight is 1.0 and this is the naive-Bayes product unchanged.
        weights = (
            self.kb.redundancy_weights(f.concept for f in findings)
            if self.kb.correlations
            else {}
        )

        log_scores: dict[str, float] = {}
        for entry in entries:
            total = math.log(max(entry.prevalence, 1e-9))
            for finding in findings:
                weight = weights.get(finding.concept, 1.0)
                total += weight * math.log(self.kb.likelihood(entry.label, finding))
            log_scores[entry.label] = total

        # Shift before exponentiating to keep the largest term at 1.0.
        peak = max(log_scores.values())
        scores = {label: math.exp(v - peak) for label, v in log_scores.items()}

        self.last_evidence_fit = self.evidence_fit(findings, log_scores)

        # The entry's own citation grounds the *existence* of the hypothesis;
        # the per-finding ones ground the *evidence* for and against it. They
        # are kept in separate fields, which they were not originally: the
        # entry citation used to be concatenated onto the front of support, so
        # every hypothesis displayed a line under "supporting evidence" that
        # read "synthetic entry, not sourced" -- a disclosure rendered as its
        # own opposite. Grounding still satisfies the gate's requirement that
        # a committed hypothesis be in the knowledge base at all, so a
        # hypothesis nothing currently supports is still grounded rather than
        # escalating for missing provenance.
        support: dict[str, tuple] = {}
        against: dict[str, tuple] = {}
        grounding: dict[str, tuple] = {}
        for entry in entries:
            supporting, contradicting = self.kb.evidence_split(entry.label, findings)
            support[entry.label] = supporting
            against[entry.label] = contradicting
            grounding[entry.label] = entry.citations

        rationales = {
            e.label: self._rationale(e.label, findings) for e in entries
        }
        return Differential.from_scores(
            scores,
            support=support,
            rationales=rationales,
            against=against,
            grounding=grounding,
        )

    def evidence_fit(
        self, findings: list[Finding], log_scores: dict[str, float]
    ) -> float:
        """Per-finding log-likelihood ratio of the KB mixture over the marginal.

        The quantity a confidence threshold cannot see. Top-1 probability is a
        statement about the *shape* of the posterior -- it is high whenever one
        disease outranks the others, whether or not any of them accounts for
        what was actually observed. A pulmonary embolism presenting with fever
        and cough scores 0.73 for pneumonia: peaked, confident, and wrong, and
        no threshold on that 0.73 can catch it because the number is not about
        the evidence at all.

        This asks the other question. P(E) = sum_d P(d) P(E|d) is the mass the
        whole KB assigns to the findings; dividing by the same findings'
        probability under the KB-wide marginal gives a ratio that is positive
        when *some* diagnosis explains the case better than chance and near
        zero when none does.

        What it catches, and what it does not. Measured on the fixture set, the
        two cases the loop gets wrong score +0.06 and +0.18, inside the +0.01
        to +0.32 range of the eight it gets right -- no threshold separates
        them. That is not a tuning failure but the wrong instrument: those
        errors are *masquerade*, not anomaly. A pulmonary embolism presenting
        with fever and cough produces evidence that pneumonia genuinely
        explains well, so the marginal likelihood is high and this check stays
        silent. It detects presentations that *nothing* in the KB explains --
        a cause outside the KB's coverage, or a genuinely bizarre combination
        -- which is a real safety concern and a different one. Do not read a
        healthy fit as evidence the answer is right.

        Dividing by the number of informative findings keeps cases with three
        and with fifteen of them comparable; both terms grow with that count,
        so the raw difference would just measure how many questions were asked.
        UNKNOWN findings contribute a likelihood of 1.0 to both sides and are
        excluded from the count rather than being allowed to dilute it.

        Returns ``inf`` when there is no informative evidence: nothing has been
        observed, so nothing is unexplained. That is the loop's due-diligence
        rule to enforce, not this one's.
        """
        import math

        informative = [f for f in findings if f.polarity is not Polarity.UNKNOWN]
        if not informative:
            return float("inf")

        # logsumexp over log P(d) + log P(E|d), which is log P(E).
        peak = max(log_scores.values())
        log_evidence = peak + math.log(
            sum(math.exp(v - peak) for v in log_scores.values())
        )
        log_background = sum(
            math.log(self.kb.background_likelihood(f)) for f in informative
        )
        return (log_evidence - log_background) / len(informative)

    def _rationale(self, label: str, findings: list[Finding]) -> str:
        entry = self.kb.get(label)
        if entry is None:
            return ""
        supporting = [
            f.concept
            for f in findings
            if f.is_positive and entry.features.get(f.concept, 0.0) >= 0.6
        ]
        conflicting = [
            f.concept
            for f in findings
            if f.is_positive and entry.features.get(f.concept, 0.5) <= 0.15
        ]
        parts = []
        if supporting:
            parts.append("consistent with " + ", ".join(sorted(supporting)))
        if conflicting:
            parts.append("poorly explained by " + ", ".join(sorted(conflicting)))
        return "; ".join(parts)


_PROPOSE_SYSTEM = """You are a diagnostic reasoning component in a clinical \
decision-support system. You do not talk to patients and you do not give \
advice. Your only job is to distribute probability mass over a fixed list of \
candidate diagnoses given the observed findings.

Rules:
- Use ONLY labels from the provided candidate list. Do not invent labels.
- Return probabilities that sum to 1.0.
- If the findings are weakly discriminating, say so with a flat distribution. \
Do not manufacture confidence.
- Respond with JSON only, no prose and no markdown fences, in the form:
  {"ranking": [{"label": "...", "probability": 0.0, "why": "..."}]}"""


@dataclass
class LLMProposer:
    """Constrained LLM ranking over the KB's label set."""

    kb: InMemoryKnowledgeBase
    llm: LLMClient
    fallback: BayesianProposer | None = None
    # Optional GuidelineIndex. Supplied, the retrieved passages go into the
    # prompt and the model is told to reason from them -- which is what makes
    # this retrieval-augmented *generation* rather than retrieval that decorates
    # a generated answer afterwards. Absent, the model answers from parametric
    # memory and nothing marks the difference, so the distinction is worth
    # keeping visible at the call site.
    index: object | None = None
    passages_per_disease: int = 2
    # The model's own stated confidence in its top choice, from the last call.
    # Kept separate from the probability it assigned that label: the two are
    # different claims, and treating a stated confidence as a probability is
    # the error VerbalisedCalibrator exists to correct. NaN until a call has
    # returned one, so "never stated" is distinguishable from "stated zero".
    last_verbalised_confidence: float = float("nan")

    def propose(self, findings: list[Finding], complaint: str = "") -> Differential:
        labels = [e.label for e in self.kb.diseases()]
        prompt = self._render(labels, findings, complaint)
        try:
            raw = self.llm.complete(prompt, system=_PROPOSE_SYSTEM)
            scores = self._parse(raw, labels)
            self.last_verbalised_confidence = self._parse_confidence(raw)
        except Exception:
            # A malformed or failed generation must not take the case down.
            # Degrade to the statistical posterior and let the gate see the
            # resulting confidence for what it is.
            scores = {}
        if not scores:
            if self.fallback is None:
                self.fallback = BayesianProposer(self.kb)
            return self.fallback.propose(findings, complaint)
        grounding = {e.label: e.citations for e in self.kb.diseases()}
        return Differential.from_scores(scores, grounding=grounding)

    def _render(self, labels: list[str], findings: list[Finding], complaint: str) -> str:
        observed = "\n".join(
            f"- {f.concept}: {f.polarity.value}"
            + (f" (value: {f.value})" if f.value is not None else "")
            for f in findings
        ) or "- none yet"
        prompt = (
            f"Presenting complaint: {complaint or 'unspecified'}\n\n"
            f"Observed findings:\n{observed}\n\n"
            f"Candidate diagnoses:\n"
            + "\n".join(f"- {l}" for l in labels)
        )
        evidence = self._retrieved_context(labels, findings, complaint)
        if evidence:
            prompt += (
                "\n\nRetrieved guideline passages. Reason from these rather than "
                "from memory, and say so when they do not cover a candidate:\n"
                + evidence
            )
        return prompt

    def _retrieved_context(
        self, labels: list[str], findings: list[Finding], complaint: str
    ) -> str:
        """Guideline text for each candidate, as prompt context.

        Silent when no index is configured, so the LLM path keeps working
        without the retrieval extras installed. A retrieval failure degrades to
        an ungrounded answer rather than taking the case down, which is the
        same choice ``propose`` already makes for a malformed generation --
        but the resulting hypotheses then carry no retrieved citation, so the
        gate can still see that they are unsupported.
        """
        if self.index is None:
            return ""
        from .retrieval import case_query

        query = case_query(findings, complaint)
        blocks: list[str] = []
        for label in labels:
            try:
                hits = self.index.search(
                    query, k=self.passages_per_disease, target=label
                )
            except Exception:
                continue
            for passage, score in hits:
                blocks.append(
                    f"- [{passage.citation.source_id}] ({label}, similarity "
                    f"{score:.2f}) {passage.text}"
                )
        return "\n".join(blocks)

    @staticmethod
    def _parse_confidence(raw: str) -> float:
        """The stated confidence, or NaN when the model did not give one.

        NaN rather than a default, because a missing statement and a stated
        0.5 are different facts and a calibrator fitted on invented 0.5s would
        be measuring this parser.
        """
        import math

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1].removeprefix("json").strip()
            value = float(json.loads(text).get("confidence"))
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError, IndexError):
            return float("nan")
        return value if 0.0 <= value <= 1.0 else float("nan")

    @staticmethod
    def _parse(raw: str, labels: list[str]) -> dict[str, float]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            text = text.removeprefix("json").strip()
        payload = json.loads(text)
        allowed = set(labels)
        scores: dict[str, float] = {}
        for item in payload.get("ranking", []):
            label = item.get("label")
            prob = float(item.get("probability", 0.0))
            # Silently drop hallucinated labels rather than trusting them.
            if label in allowed and prob > 0:
                scores[label] = prob
        return scores


@dataclass
class ConsensusProposer:
    """Runs two proposers and exposes their disagreement."""

    primary: Proposer
    secondary: Proposer
    last_disagreement: float = 0.0
    last_labels_agree: bool = True

    def propose(self, findings: list[Finding], complaint: str = "") -> Differential:
        a = self.primary.propose(findings, complaint)
        b = self.secondary.propose(findings, complaint)
        self.last_disagreement = _total_variation(a, b)
        self.last_labels_agree = a.top.label == b.top.label
        return a


def _total_variation(a: Differential, b: Differential) -> float:
    """Total variation distance between two differentials, in [0, 1]."""
    labels = {h.label for h in a.hypotheses} | {h.label for h in b.hypotheses}
    return 0.5 * sum(
        abs(a.probability_of(label) - b.probability_of(label)) for label in labels
    )


__all__ = [
    "BayesianProposer",
    "ConsensusProposer",
    "LLMProposer",
    "Proposer",
]
