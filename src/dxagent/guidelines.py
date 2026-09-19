"""Published clinical decision rules, encoded as citable structures.

Not for clinical use. This module transcribes published rules so that a
research system can reason with them and cite them; it has not been verified
against the primary sources by a clinician, and the vocabulary mapping in
particular is an engineering judgement about which recorded concept stands for
which criterion. Treat every mapping as a claim to be checked.

Why this exists
---------------
Everything else in this system reads the posterior to decide what to do next.
Measured on the fixture set, that is precisely why it fails the cases it fails:
in fx-009 the true diagnosis sits at 0.5% because four weak negatives multiplied
under the independence assumption, and no criterion conditioned on that belief
will investigate it. Confidence thresholds, expected information gain,
marginal likelihood and decision-flip lookahead were each tried and each
inherited the same blind spot.

A published decision rule does not have that problem, because it is not
conditioned on the model's belief at all. PERC does not ask how likely you
think a pulmonary embolism is; it asks eight questions triggered by the
presentation. That independence from the posterior is the entire point, and it
is why ``MandatoryWorkup`` below is checked before the gate rather than by it.

What a rule is and is not
-------------------------
These rules are decision aids with defined populations and defined purposes,
and conflating those purposes is the easiest way to misuse them:

  * PERC is a *rule-out*. All eight criteria negative, in a population already
    judged low-risk, means further testing is not indicated. It does not
    establish a diagnosis and it is not valid outside that population.
  * Wells is a *risk stratifier* for pulmonary embolism, feeding a testing
    decision rather than producing a diagnosis.
  * CURB-65 is a *severity* score for already-diagnosed pneumonia, used to
    decide site of care. It says nothing about whether pneumonia is present,
    and using it as diagnostic evidence would be a category error.
  * HEART stratifies major-adverse-cardiac-event risk in undifferentiated
    chest pain.

``RuleKind`` records which is which so that downstream code cannot quietly
treat a severity score as diagnostic support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .schemas import Citation, Finding, Polarity


class RuleKind(str, Enum):
    RULE_OUT = "rule_out"  # negative result argues against the target
    RISK_STRATIFY = "risk_stratify"  # graded likelihood, feeds a testing decision
    SEVERITY = "severity"  # assumes the diagnosis; grades how bad it is


@dataclass(frozen=True)
class Criterion:
    """One line of a decision rule.

    ``concept`` is the vocabulary term standing for this criterion, or None
    when the current vocabulary cannot express it. Unmappable criteria are kept
    rather than dropped: a rule scored on five of its eight lines is not that
    rule, and silently omitting the other three would hide that.
    """

    description: str
    weight: float
    concept: str | None = None
    polarity: Polarity = Polarity.PRESENT  # which observation satisfies it


@dataclass(frozen=True)
class ClinicalRule:
    name: str
    target: str  # disease label this rule speaks about
    kind: RuleKind
    criteria: tuple[Criterion, ...]
    citation: Citation
    bands: tuple[tuple[float, str], ...] = ()  # (lower bound, interpretation)
    note: str = ""

    @property
    def mappable(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.concept)

    @property
    def unmappable(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if not c.concept)

    @property
    def coverage(self) -> float:
        """Fraction of the rule's total weight the vocabulary can express."""
        total = sum(abs(c.weight) for c in self.criteria)
        if total <= 0:
            return 0.0
        return sum(abs(c.weight) for c in self.mappable) / total

    def score(self, findings: list[Finding]) -> float:
        """Score the mappable criteria against observed findings.

        Only findings actually observed count. An unasked criterion is not the
        same as a negative one, and scoring it as absent would let an unasked
        rule read as reassuring -- the failure mode PERC is most often misused
        for in practice.
        """
        observed = {f.concept: f.polarity for f in findings}
        total = 0.0
        for criterion in self.mappable:
            polarity = observed.get(criterion.concept)
            if polarity is not None and polarity is criterion.polarity:
                total += criterion.weight
        return total

    def unobserved(self, findings: list[Finding]) -> tuple[Criterion, ...]:
        """Mappable criteria still unanswered. Empty means the rule is complete."""
        observed = {
            f.concept for f in findings if f.polarity is not Polarity.UNKNOWN
        }
        return tuple(c for c in self.mappable if c.concept not in observed)

    def interpret(self, findings: list[Finding]) -> str:
        value = self.score(findings)
        band = ""
        for lower, label in sorted(self.bands, reverse=True):
            if value >= lower:
                band = label
                break
        missing = len(self.unobserved(findings))
        suffix = f"; {missing} criteria unanswered" if missing else ""
        gap = (
            f"; {len(self.unmappable)} criteria not expressible in the current "
            "vocabulary"
            if self.unmappable
            else ""
        )
        return f"{self.name} = {value:g}{f' ({band})' if band else ''}{suffix}{gap}"


@dataclass(frozen=True)
class MandatoryWorkup:
    """Evidence that must be gathered for a presentation, whatever the model believes.

    Triggered by observed findings rather than by the posterior. That is the
    property that makes it work where four posterior-conditioned mechanisms
    did not: a diagnosis the model has already dismissed to 0.5% is exactly the
    one it will never choose to investigate, and a rule that never consults the
    model's belief is unaffected by the belief being wrong.

    The cost of this is real and should be stated rather than discovered: it
    orders tests on patients who do not need them. That trade -- unnecessary
    testing against missed time-critical diagnoses -- is a clinical policy
    decision, not an engineering one, and the thresholds here are transcribed
    defaults awaiting a clinician's judgement.
    """

    name: str
    trigger_any: tuple[str, ...]  # any of these findings present arms the rule
    required: tuple[str, ...]  # concepts that must be observed before committing
    rationale: str
    citation: Citation

    def is_triggered(self, findings: list[Finding]) -> bool:
        present = {f.concept for f in findings if f.polarity is Polarity.PRESENT}
        return any(concept in present for concept in self.trigger_any)

    def outstanding(self, findings: list[Finding]) -> tuple[str, ...]:
        if not self.is_triggered(findings):
            return ()
        answered = {
            f.concept for f in findings if f.polarity is not Polarity.UNKNOWN
        }
        return tuple(c for c in self.required if c not in answered)


@dataclass(frozen=True)
class ConfirmatoryWorkup:
    """A workup owed once an earlier mandatory result itself comes back positive.

    ``MandatoryWorkup`` fixes the blind spot where a dismissed posterior never
    orders its own screening test. It does not fix the next blind spot: a
    screening test that comes back positive and then, because it was answered
    late enough that cheaper questions had already spent the turn or cost
    budget, never gets followed by the confirmatory test the positive result
    actually indicates. On the PE real-case set that gap was the entire
    difference between escalating and reaching a CTPA that was, by then,
    affordable in principle but never reached in practice.

    The trigger is a *result*, not the presentation -- ``antecedent`` must
    itself have been observed present, not merely asked about. That is what
    keeps this from mandating imaging on every patient who happens to have a
    nonspecific D-dimer elevation with no PE-consistent presentation: the
    antecedent concept only reads PRESENT here because ``MandatoryWorkup``
    already required the presentation to be armed before ordering it.
    """

    name: str
    antecedent: str
    required: tuple[str, ...]
    rationale: str
    citation: Citation

    def outstanding(self, findings: list[Finding]) -> tuple[str, ...]:
        present = {f.concept for f in findings if f.polarity is Polarity.PRESENT}
        if self.antecedent not in present:
            return ()
        answered = {
            f.concept for f in findings if f.polarity is not Polarity.UNKNOWN
        }
        return tuple(c for c in self.required if c not in answered)


def _cite(source: str, locator: str, snippet: str) -> Citation:
    return Citation(source_id=source, locator=locator, snippet=snippet)


# ---------------------------------------------------------------------------
# The rules. Concept names follow the fixture vocabulary; criteria that it
# cannot express carry concept=None and are reported by `vocabulary_gaps()`.

WELLS_PE = ClinicalRule(
    name="Wells score (PE)",
    target="pulmonary_embolism",
    kind=RuleKind.RISK_STRATIFY,
    criteria=(
        Criterion("Clinical signs of DVT", 3.0, "calf_tenderness"),
        Criterion("Unilateral leg swelling", 3.0, "leg_swelling"),
        Criterion("PE is the most likely diagnosis", 3.0, None),
        Criterion("Heart rate > 100", 1.5, "exam:tachycardia"),
        Criterion("Immobilisation >= 3 days or surgery in past 4 weeks", 1.5,
                  "recent_immobility"),
        Criterion("Previous objectively diagnosed PE or DVT", 1.5, None),
        Criterion("Haemoptysis", 1.0, None),
        Criterion("Malignancy treated within 6 months or palliative", 1.0, None),
    ),
    bands=((7.0, "high probability"), (2.0, "moderate"), (0.0, "low")),
    citation=_cite(
        "WELLS-2000",
        "Wells PS et al., Thromb Haemost 2000;83:416-20",
        "Derivation of a simple clinical model to categorise patients "
        "probability of pulmonary embolism",
    ),
    note="Stratifies risk to guide testing. Does not establish a diagnosis.",
)

PERC = ClinicalRule(
    name="PERC rule (PE rule-out)",
    target="pulmonary_embolism",
    kind=RuleKind.RULE_OUT,
    criteria=(
        Criterion("Age >= 50", 1.0, None),
        Criterion("Heart rate >= 100", 1.0, "exam:tachycardia"),
        Criterion("SaO2 < 95%", 1.0, "exam:hypoxia"),
        Criterion("Unilateral leg swelling", 1.0, "leg_swelling"),
        Criterion("Haemoptysis", 1.0, None),
        Criterion("Recent surgery or trauma", 1.0, "recent_immobility"),
        Criterion("Prior PE or DVT", 1.0, None),
        Criterion("Exogenous oestrogen use", 1.0, None),
    ),
    bands=((1.0, "PERC positive - cannot rule out"), (0.0, "PERC negative")),
    citation=_cite(
        "PERC-2008",
        "Kline JA et al., J Thromb Haemost 2008;6:772-80",
        "Prospective multicentre evaluation of the pulmonary embolism "
        "rule-out criteria",
    ),
    note=(
        "Valid only in a population already judged low-risk. All eight "
        "criteria negative means further testing is not indicated; it is not "
        "a positive finding of health."
    ),
)

CURB65 = ClinicalRule(
    name="CURB-65 (pneumonia severity)",
    target="community_acquired_pneumonia",
    kind=RuleKind.SEVERITY,
    criteria=(
        Criterion("Confusion", 1.0, None),
        Criterion("Urea > 7 mmol/L", 1.0, None),
        Criterion("Respiratory rate >= 30", 1.0, None),
        Criterion("Systolic BP < 90 or diastolic <= 60", 1.0, None),
        Criterion("Age >= 65", 1.0, None),
    ),
    bands=((3.0, "severe - consider ICU"), (2.0, "moderate"), (0.0, "low")),
    citation=_cite(
        "CURB65-2003",
        "Lim WS et al., Thorax 2003;58:377-82",
        "Defining community acquired pneumonia severity on presentation to "
        "hospital",
    ),
    note=(
        "Assumes pneumonia is already diagnosed and grades severity to decide "
        "site of care. Using it as diagnostic evidence is a category error."
    ),
)

HEART = ClinicalRule(
    name="HEART score (chest pain)",
    target="acute_coronary_syndrome",
    kind=RuleKind.RISK_STRATIFY,
    criteria=(
        Criterion("History moderately or highly suspicious", 2.0,
                  "exertional_chest_pain"),
        Criterion("Significant ST deviation on ECG", 2.0, "exam:ecg_st_changes"),
        Criterion("Troponin above reference range", 2.0, "lab:raised_troponin"),
        Criterion("Age >= 65", 2.0, None),
        Criterion("Three or more risk factors, or known atherosclerosis", 2.0,
                  "smoking_history"),
    ),
    bands=((7.0, "high risk"), (4.0, "moderate risk"), (0.0, "low risk")),
    citation=_cite(
        "HEART-2008",
        "Six AJ, Backus BE, Kelder JC, Neth Heart J 2008;16:191-6",
        "Chest pain in the emergency room: value of the HEART score",
    ),
    note=(
        "Predicts major adverse cardiac events, not the presence of ACS. The "
        "risk-factor criterion is approximated here by smoking history alone, "
        "which understates it."
    ),
)

RULES: tuple[ClinicalRule, ...] = (WELLS_PE, PERC, CURB65, HEART)


# ---------------------------------------------------------------------------
# Mandatory workups: presentation-triggered, posterior-independent.

PE_WORKUP = MandatoryWorkup(
    name="Pulmonary embolism exclusion",
    trigger_any=("pleuritic_pain", "dyspnoea_at_rest", "exam:hypoxia", "sudden_onset"),
    required=("lab:raised_d_dimer",),
    rationale=(
        "Pleuritic pain, rest dyspnoea, hypoxia or sudden onset arms a PE "
        "workup regardless of the current differential. On the fixture set the "
        "loop dismissed a true PE to 0.5% on four weak negatives and then "
        "declined to order the D-dimer because it no longer believed the "
        "result would be positive; a presentation-triggered rule does not "
        "consult that belief."
    ),
    citation=_cite(
        "PERC-2008",
        "Kline JA et al., J Thromb Haemost 2008;6:772-80",
        "PERC applies to patients in whom PE is being considered, identified "
        "by presentation rather than by a computed probability",
    ),
)

ACS_WORKUP = MandatoryWorkup(
    name="Acute coronary syndrome exclusion",
    trigger_any=("exertional_chest_pain", "palpitations", "exam:ecg_st_changes"),
    required=("lab:raised_troponin", "exam:ecg_st_changes"),
    rationale=(
        "Chest pain of possible cardiac character requires a troponin and an "
        "ECG before a benign alternative is committed to, whatever the model "
        "currently ranks first."
    ),
    citation=_cite(
        "HEART-2008",
        "Six AJ, Backus BE, Kelder JC, Neth Heart J 2008;16:191-6",
        "Troponin and ECG are two of the five HEART components and are "
        "obtained on presentation, not on suspicion",
    ),
)

PERICARDITIS_WORKUP = MandatoryWorkup(
    name="Pericarditis assessment",
    trigger_any=("pleuritic_pain", "exam:friction_rub"),
    required=(
        "exam:ecg_st_changes",
        "exam:ecg_pr_depression",
        "imaging:pericardial_effusion",
    ),
    rationale=(
        "Pleuritic chest pain or a rub is the presentation of pericarditis, "
        "and a diagnosis the loop has dismissed to 4% is one it will never "
        "spend a turn on: a real pericarditis with a large effusion was "
        "committed as pneumonia without the effusion ever being asked for. "
        "The ESC criteria for the diagnosis are two of four -- pericarditic "
        "pain, rub, ECG change, effusion -- so the ECG and the echocardiogram "
        "are what settle it either way. Transcription judgement, stated: the "
        "guideline says 'suspected pericarditis'; pleuritic pain is how that "
        "suspicion is expressed in this vocabulary, and it arms the workup on "
        "pleuritic pneumonias and embolisms too. That is the same cost the "
        "PE workup already pays, and it is a clinical policy decision "
        "awaiting a clinician."
    ),
    citation=_cite(
        "ESC-PERICARDIAL-2015",
        "Adler Y et al., Eur Heart J 2015;36:2921-64, section 3.1, "
        "recommendations for the diagnosis of acute pericarditis",
        "In all cases of suspected acute pericarditis, ECG and transthoracic "
        "echocardiography are recommended (Class I, Level C)",
    ),
)

WORKUPS: tuple[MandatoryWorkup, ...] = (PE_WORKUP, ACS_WORKUP, PERICARDITIS_WORKUP)

# A positive D-dimer, in a patient where PE_WORKUP was already armed, is not
# one more piece of evidence to weigh against its acquisition cost -- it is
# the indication for the confirmatory scan. Wells' own rule pairs risk
# stratification with exactly this D-dimer-then-imaging sequence; engineering
# judgement here, transcribing standard sequential-testing practice rather
# than a specific verified sentence from the paper, and stated as such.
PE_CONFIRM = ConfirmatoryWorkup(
    name="Pulmonary embolism confirmation",
    antecedent="lab:raised_d_dimer",
    required=("imaging:ctpa_filling_defect",),
    rationale=(
        "A positive D-dimer, once PE has been armed as a live consideration "
        "by the presentation, is the indication for CTPA rather than one "
        "more finding to weigh by expected information gain -- the same "
        "escalation from screening to confirmatory test that PE_WORKUP "
        "already enforces for ordering the D-dimer, carried one step "
        "further."
    ),
    citation=_cite(
        "WELLS-2000",
        "Wells PS et al., Thromb Haemost 2000;83:416-20",
        "risk stratification is paired with D-dimer testing to guide the "
        "decision to pursue definitive imaging, not read in isolation",
    ),
)

CONFIRMATIONS: tuple[ConfirmatoryWorkup, ...] = (PE_CONFIRM,)


def outstanding_workup(findings: list[Finding]) -> tuple[tuple[str, str], ...]:
    """Every triggered-but-incomplete workup, as (concept, workup name) pairs."""
    out: list[tuple[str, str]] = []
    for workup in WORKUPS:
        for concept in workup.outstanding(findings):
            out.append((concept, workup.name))
    for confirm in CONFIRMATIONS:
        for concept in confirm.outstanding(findings):
            out.append((concept, confirm.name))
    return tuple(out)


def unavailable_workup(
    findings: list[Finding], asked: set[str]
) -> tuple[tuple[str, str], ...]:
    """Workup items that were sought and came back UNKNOWN.

    ``outstanding_workup`` cannot tell "not asked yet" from "asked, and the
    record had no answer" -- both read as unanswered, which is right for
    deciding what to do next and wrong for explaining why the loop stopped.
    Intersecting it with what was actually asked separates them.

    This is the difference between an escalation that says the differential
    could not be narrowed and one that says a troponin was sought and never
    obtained. The second is the one a clinician can act on, and it is the
    stated purpose of the escalation packet.
    """
    return tuple(
        (concept, name)
        for concept, name in outstanding_workup(findings)
        if concept in asked
    )


def unexamined_signatures(
    findings: list[Finding], asked: set[str], committed: str
) -> tuple["UnexaminedRival", ...]:
    """Rivals whose defining picture is partly present and partly unexamined.

    The ordering claims say which findings define each disease. A rival is
    reported when at least one of those findings was observed present and at
    least one was never asked for -- a partial signature the loop walked past.
    On the real pericarditis committed as pneumonia this would have read
    "pericarditis: pleuritic pain present; effusion, PR depression, rub
    unexamined", at the moment of the commit, in the packet.

    Deliberately reads findings and the asked set, never the differential:
    every posterior-conditioned criterion this project tried failed for the
    same reason, that a diagnosis the model has dismissed is exactly the one
    it will not investigate. This is disclosure, not a gate condition -- it
    orders nothing and blocks nothing. Whether a listed rival should have been
    examined is the reader's call, and now the reader can make it.
    """
    from collections import defaultdict

    from .claims import ORDERING_CLAIMS
    from .schemas import UnexaminedRival

    defining: dict[str, set[str]] = defaultdict(set)
    for concept, leader, _rivals, _why in ORDERING_CLAIMS:
        defining[leader].add(concept)

    present = {f.concept for f in findings if f.polarity is Polarity.PRESENT}
    out = []
    for label in sorted(defining):
        if label == committed:
            continue
        hit = defining[label] & present
        missing = defining[label] - asked
        if hit and missing:
            out.append(UnexaminedRival(label, tuple(sorted(hit)), tuple(sorted(missing))))
    return tuple(out)


def cheap_unexamined(
    findings: list[Finding], asked: set[str], cost_of, ceiling: float
) -> str | None:
    """The cheapest unasked defining finding of any partially-present picture.

    The action that follows from ``unexamined_signatures`` without reading the
    posterior: if some disease has one defining finding present and another
    never asked, and asking costs no more than ``ceiling`` -- a history
    question or a bedside sign, never a test -- ask it before committing.
    Every disease is eligible, the current leader included; excluding the
    leader would mean reading the differential, which is the one thing this
    must not do. Returns None when nothing cheap is outstanding.
    """
    from collections import defaultdict

    from .claims import ORDERING_CLAIMS

    defining: dict[str, set[str]] = defaultdict(set)
    for concept, leader, _rivals, _why in ORDERING_CLAIMS:
        defining[leader].add(concept)
    present = {f.concept for f in findings if f.polarity is Polarity.PRESENT}

    best: tuple[float, str] | None = None
    for label, concepts in defining.items():
        if not concepts & present:
            continue
        for concept in concepts - asked:
            cost = cost_of(concept)
            if cost <= ceiling and (best is None or (cost, concept) < best):
                best = (cost, concept)
    return best[1] if best else None


def vocabulary_gaps() -> dict[str, tuple[str, ...]]:
    """Criteria the current vocabulary cannot express, per rule.

    Reported rather than silently dropped, because a rule scored on part of
    itself is a different and weaker rule. This is also the concrete shopping
    list for extending the finding vocabulary.
    """
    return {
        rule.name: tuple(c.description for c in rule.unmappable) for rule in RULES
    }


__all__ = [
    "ACS_WORKUP",
    "CONFIRMATIONS",
    "CURB65",
    "ConfirmatoryWorkup",
    "Criterion",
    "ClinicalRule",
    "HEART",
    "MandatoryWorkup",
    "PERC",
    "PE_CONFIRM",
    "PE_WORKUP",
    "RULES",
    "RuleKind",
    "WELLS_PE",
    "WORKUPS",
    "outstanding_workup",
    "unavailable_workup",
    "vocabulary_gaps",
]
