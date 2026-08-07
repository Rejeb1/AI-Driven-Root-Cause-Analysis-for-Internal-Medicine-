# Design notes and open questions

For the clinician meeting. Written after building the scaffold, so the questions
below are the ones the implementation actually raised rather than the ones
anticipated in advance.

## What the loop does

Each turn: update the differential from all findings so far → calibrate it → ask
the gate whether to commit, escalate, or continue → if continuing, take the
highest-value action within budget.

Termination is guaranteed by three independent conditions (turn limit, cost
budget, exhaustion of informative actions), because a loop that can only stop
once it becomes confident is a loop that can fail to stop.

## Six things that broke, and what they imply

These are recorded because each one is a general failure mode rather than a
typo, and several have direct design consequences. All are covered by regression
tests.

**1. Sparse KB entries won the posterior.** Treating an uncharacterised feature
as "uninformative" (likelihood 1.0) makes a thinly-described disease strictly
cheaper to explain any finding with than a thoroughly-described one. The
posterior drifts toward whichever entries are least complete — a KB *coverage*
artefact wearing the costume of a clinical inference. Fixed by backing off to the
KB-wide marginal. **Implication:** KB completeness must be audited per-feature,
not per-disease.

**2. Information gain per unit cost asked twelve trivial questions.** With
history at cost 0.2, a question worth 0.01 nats outranked a troponin worth 0.6
nats, so the agent exhausted its turn budget on trivia and never ordered the
decisive test. Fixed with a cost offset — every action also costs a *turn*, which
is the genuinely scarce resource in a consultation and the thing monetary cost
fails to capture. **Question for the clinician:** what is a turn actually worth
relative to a test, and does that differ between clinic and acute settings?

**3. The gate refused to report the diagnosis it exists to protect.** Blocking a
commit whenever any red-flag diagnosis held mass meant that correctly
identifying an ACS at 98% triggered an escalation. Only *competing* red flags
should block. **Implication:** asymmetric-risk logic needs to be stated as
"don't commit to something benign while a dangerous thing is live", not "be
cautious around dangerous things".

**4. The loop committed on the volunteered history alone.** 85% confident,
wrong, having asked nothing. A posterior can be peaked because the evidence is
decisive or because there is barely any evidence, and top-1 probability cannot
distinguish those. Fixed with a due-diligence rule: don't stop while affordable
informative evidence remains. **Question:** is there a mandatory minimum workup
per presenting complaint? That is a clinical standard, not an engineering
threshold, and it is the cleanest place to encode one.

**5. Fitting calibration on two cases produced T = 0.5.** It *sharpened* an
already-overconfident posterior and made ECE worse, while looking like a fitted
parameter. Now refuses below 30 samples and reports that it declined.

**6. Entropy-based selection will not investigate what it has dismissed.**
Expected gain is computed under the current posterior, so a PE at 8%
contributes almost nothing to it and the D-dimer is never ordered — 8% is not a
rounding error for a PE. Fixed by making red-flag exclusion a *separate*
criterion driven by likelihood ratio rather than a weight on entropy gain.
**Implication:** value of information and value of *safety* are different
objectives and should not be collapsed into one score.

## The headline negative result

> **Revised 5 August 2026 — read the revision below before citing this section.**
> The claim as written held on the fixture set and on a small DDXPlus sample.
> It does not hold on a 400-case DDXPlus run, where the gate gains +10.9%
> accuracy at 78.2% coverage. The mechanism described here is still real; the
> scope of the claim was wrong.

**On the fixture set, the abstention gate provides no value.** Coverage is 100%
at every threshold up to 0.8; at 0.9 it fires with abstention precision 0%, i.e.
every case it escalated was one it would have got right.

This is not a tuning problem. The two cases the system fails are atypical
presentations, and it fails them *confidently* — a PE presenting with fever and
cough scores 0.73 for pneumonia. Confidence-threshold abstention cannot catch
confident errors by construction, so the entire selective-prediction apparatus
is aimed at the wrong failure mode.

The mechanism is the naive-Bayes independence assumption. An atypical
presentation contributes several individually-unlikely features, and under
independence those multiply, so the true diagnosis is driven far below any
sane rule-out floor. Lowering the floor to catch it would order every test on
every patient.

Three candidate responses, none yet implemented:

1. **Correlation structure in the KB.** Attacks the cause. Needs real
   co-occurrence data, so it is gated on MIMIC.
2. **Atypicality detection.** Flag cases whose *evidence* is poorly explained by
   the entire differential (low marginal likelihood), rather than cases where
   the top hypothesis is uncertain. This catches confident errors, which
   confidence thresholds cannot, and it is implementable now — currently the
   most promising direction.
3. **Mandatory rule-out sets per complaint.** Blunt, clinician-specified, and
   robust to the model being wrong for reasons nobody anticipated. Cheapest to
   implement and hardest to argue with.

Worth noting for the survey's positioning: this is a concrete instance of
something the Wen et al. framing implies but does not emphasise — abstention
mechanisms keyed to confidence are structurally blind to the errors that matter
most clinically.

## Revision, 5 August 2026: when the gate does and does not earn its keep

Measured on 400 DDXPlus cases with the knowledge base fitted on 400 training
cases:

    coverage 78.2%   selective accuracy 97.4%   full-coverage 86.5%
    accuracy gained +10.9%   abstention precision 52.9%

So the gate is not useless. It abstains on one case in five and better than
half of those it would genuinely have got wrong. That contradicts the section
above, which generalised from the fixture set and from a 60-case DDXPlus run.

What appears to differ, and this is a hypothesis rather than a result: in the
earlier runs the KB was fitted on 5,000 cases, and DDXPlus generates evidence
independently given the pathology, so a naive-Bayes table fitted on that much
data recovers the generating process almost exactly and the posterior becomes
extremely peaked. A confidence threshold then has nothing to bite on. Fitted on
400 cases the same table is appropriately uncertain, and the threshold finds
real signal. The fitted temperature moved the same way, 0.75 to 0.50.

### The controlled run, and what it settled

Three factors varied one at a time, on 120 held-out DDXPlus cases:

    KB train  conformal  workup   coverage  sel.acc    gain  abst.prec
         400      False   False      88.3%    97.2%   +8.0%     71.4%
         400      False    True      88.3%    97.2%   +8.0%     71.4%
         400       True   False      78.3%    98.9%   +8.9%     42.3%
         400       True    True      78.3%    98.9%   +8.9%     42.3%
        5000      False   False      91.7%    99.1%   +0.8%     10.0%
        5000      False    True      91.7%    99.1%   +0.8%     10.0%
        5000       True   False      88.3%   100.0%   +0.8%      7.1%
        5000       True    True      88.3%   100.0%   +0.8%      7.1%

**Knowledge-base size decides it.** Ten times the training data takes the
gate's accuracy gain from +8.0% to +0.8% and its abstention precision from
71.4% to 10.0%. The hypothesis above is confirmed: **a posterior fitted well
enough to recover its own generating process becomes too confident for
confidence-based abstention to have anything left to find.** The blindness to
confident errors documented earlier is a property of over-specified models
rather than of abstention gates as such.

**Conformal prediction trades precision for coverage.** It abstains more --
coverage 88.3% to 78.3% -- and less accurately, precision 71.4% to 42.3%.
Worth having for the coverage guarantee, and not an improvement to the gate's
judgement; report the two separately rather than as one calibration story.

**The guideline workup has no effect on abstention at all.** Every paired row
is identical to three significant figures. That is a clean null and it is
consistent with what the workup does: it changes which evidence is gathered,
not how confident the posterior becomes, and this gate reads confidence.

The practical consequence is uncomfortable and worth stating: on this
benchmark, the better the knowledge base fits, the less the safety mechanism
contributes. Any abstention result here has to be reported with the size of
the knowledge base that produced it.

## What five mechanisms could not fix

Recorded because the pattern is more useful than any one attempt. To catch the
cases the loop gets confidently wrong, these were each built and measured:

1. Confidence threshold — blind to confident errors by construction.
2. Marginal likelihood (`evidence_fit`) — the failures score inside the range
   of the successes. They are masquerade, not anomaly: the wrong diagnosis
   explains the evidence well.
3. Decision-flip lookahead — a good detector (2/2 failures flagged) and no
   remedy; acting on it cost 60% more and fixed nothing.
4. Red-flag rule-out — its 3% floor is itself a posterior threshold, so a
   diagnosis pushed below it is never revisited.
5. Guideline workup — does order the D-dimer the posterior would not, because
   it is triggered by presentation rather than belief.

The fifth works as designed and still does not fix the diagnosis, and the
measurement that explains why is the important one: in fx-009, fever and
productive cough multiply under the independence assumption until a **positive
CTPA** leaves pulmonary embolism at 13.8%, behind COPD. Perfect evidence
gathering produces the wrong answer.

The defect is therefore in the likelihood model, not in evidence gathering, and
no test-selection policy can reach it. That leaves option 1 above — correlation
structure — as the only remaining path, and it is no longer gated on MIMIC:
1.3M DDXPlus patients with full evidence vectors make feature co-occurrence
directly estimable.

## Correlation structure: the fix, and what it does not fix

Option 1 above is now implemented. `InMemoryKnowledgeBase` accepts pairwise
correlations and discounts redundant evidence:

    w_f = 1 / (1 + sum over other observed g of |rho(f, g)|)

Two findings correlated at 1.0 contribute what one would. With no correlations
supplied every weight is 1 and inference is the naive-Bayes product unchanged,
so nothing measured before this existed has been silently restated.

**On fx-009 it is worth roughly an order of magnitude.** The four VTE
negatives -- no sudden onset, no leg swelling, no calf tenderness, no recent
immobility -- are one absent clinical picture, and each now weighs 0.28 rather
than 1.0, so together they count as a little over one finding instead of four:

    evidence                  PE (naive)   PE (correlated)
    fever + cough                 0.0080            0.0304
    + four VTE negatives          0.0019            0.0213
    + positive CTPA               0.0361            0.2388

**And it is not enough.** The true diagnosis still finishes behind COPD. The
residual is not independence but the likelihood values themselves: the fixture
gives P(productive cough | PE) = 0.10, and at clinically plausible values the
same case resolves correctly.

    P(fever|PE)  P(cough|PE)     PE      top-1
    0.15         0.10          0.239    copd
    0.14         0.20          0.309    copd
    0.25         0.30          0.440    pulmonary embolism

So fx-009 had **two** independent causes, and either alone was sufficient to
sink it. That is why five successive mechanisms failed to move it: each
addressed at most one. It also sharpens what "the KB is invented" costs -- not
merely unverifiable numbers, but numbers extreme enough to change the answer.

### DDXPlus cannot validate this

Correlation correction makes DDXPlus results slightly *worse* (top-1 98.0% to
96.7%; DDx recall 32.9% to 33.4%). That is the expected result rather than a
disappointment: the dataset samples evidence independently given the pathology,
so there is no dependence to recover and correcting for absent correlations
only adds noise.

Measured on the validate split, only 14 of 3,796 within-disease pairs exceed
|phi| = 0.3, and the strongest are all *negative* correlations between values
of the same categorical evidence -- `E_54_@_V_181` with `E_54_@_V_161` -- which
are alternative answers to one question rather than two findings. That is a
flattening artefact of the adapter, not clinical structure, and
`categorical_siblings()` ties them at correlation 1.0 from the release metadata
without needing to estimate anything.

The wider point for the write-up: this is the third distinct improvement whose
value DDXPlus is structurally incapable of showing. A benchmark that a
naive-Bayes model recovers by construction cannot reward relaxing the naive
assumption.

## What sourcing changed, and one conclusion it took back

Likelihoods were sourced in two passes: 17 from DDXPlus co-occurrence, then 4
from a published cohort of 360 real pulmonary embolism patients (Miniati et
al., PLoS ONE 2012;7(2):e30891). Coverage is 14%, and the remaining 116 are
still invented.

    knowledge base            plain   +corr   +decisive   +both
    invented                   8/10    8/10        8/10    8/10
    17 from DDXPlus            9/10      --          --   10/10
    + 4 from the literature    9/10   10/10        9/10    9/10

Two things follow.

**fx-010 is fixed, and by symptom frequencies rather than by the tests that
decide it.** The sensitivity analysis named BNP and troponin as the numbers
that case turns on; DDXPlus holds neither, and replacing the surrounding
symptom likelihoods was enough. Worth remembering when reading a sensitivity
list: it names what is decisive given everything else, not the only route in.

**The decisive-test rule appeared to start working and then stopped.** On
DDXPlus-sourced numbers it reached 10/10; on literature-sourced numbers
correlation alone reaches 10/10 and adding decisive tests drops back to 9/10 at
half again the cost. The difference is one figure: DDXPlus records pleuritic
pain in 71% of its pulmonary embolism patients, Miniati's real cohort shows
33%. The simulator overstates it twofold, and a mechanism that looked
vindicated on the inflated number was not.

That is the caution for every number sourced from DDXPlus in this project.
They are measured, and measured from a generating model; a conclusion about
mechanism that rests on them has not yet met a real frequency.

## The guideline workup costs a case, and the reason is instructive

With the correlation structure on and likelihoods part-sourced:

    correlated  workup  decisive   top-1   cost
          yes      no        no    10/10   16.9
          yes      no       yes     9/10   26.0
          yes     yes        no     9/10   17.0
          yes     yes       yes     9/10   22.1

The workup is on by default and it is what loses fx-009.

The mechanism is not that the D-dimer is unhelpful. Left to itself the loop
orders a chest X-ray early, it comes back negative, and that negative is what
undermines the pneumonia hypothesis and keeps the embolism alive long enough
for the CTPA to be ordered eleven turns in -- final probability 0.68, correct.
With the workup the D-dimer is forced to turn zero, the X-ray is never ordered
at all, pneumonia stays high, the embolism falls below the 3% rule-out floor,
and the loop commits after five turns having never ordered the confirmatory
test. Final probability 0.013.

Two distinct faults, and the second is the design error:

**The mandated test displaced a more informative one.** Ordering by checklist
rather than by value is the point of a mandatory workup, and the cost of that
is a test sequence chosen without regard to what the case needs.

**Satisfying the checklist removed a reason to continue.** The loop treats an
outstanding workup item as one of several conditions that make it keep going.
Once the D-dimer is done that condition is discharged, and if nothing else
happens to be outstanding the loop stops. A mandatory workup should be a floor
on investigation, not a substitute for it, and this implementation makes it
both.

The floor reading is fixable -- the workup should not be able to license
stopping -- and whether the default should change before that is a clinical
policy question rather than an engineering one. Ordering a D-dimer for a
hypoxic patient with pleuritic pain is correct whatever it costs this case set,
and ten fixture cases are not grounds for switching off a safety rule.

## Open questions for the meeting

1. Mandatory minimum workup per presenting complaint (see #4)?
2. Relative worth of a turn vs a test, by setting (see #2)?
3. What must an escalation packet contain to actually save the clinician time?
   Current contents — posterior, unresolved discriminating question, evidence
   gathered, reason — are a guess.
4. Red-flag investigate-floor vs report-floor: currently 3% and 10%, both
   invented.
5. Is "diagnosis" one label, or a label plus acuity plus disposition? The schema
   assumes one label and this is the change most expensive to make later.
6. Should proposer disagreement escalate, or be shown to the clinician as two
   competing readings?

## Known gaps

- Calibration is fitted on the initial differential, not per-evidence-count; the
  temperature right for a 2-finding posterior is not right for a 10-finding one.
- Train/test split is unstratified — must be fixed before any reported number.
- Action selection is myopic; jointly-decisive question pairs are missed.
- The KB estimated from DDXPlus makes "grounded in the KB" collapse into
  "consistent with the training data", which is not the claim the project
  intends. The deployed KB must be independently sourced.
