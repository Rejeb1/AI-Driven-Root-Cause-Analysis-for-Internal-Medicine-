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

Likelihoods were sourced in five passes: 17 from DDXPlus co-occurrence, 4 from
a published cohort of 360 real pulmonary embolism patients (Miniati et al.,
PLoS ONE 2012;7(2):e30891), then 9 and 5 from the Merck Manual's narrative
text through the fixed rubric, 3 by targeted search, and 14 from PIOPED II's
arm of patients investigated for embolism who turned out not to have one
(Stein PD et al., Am J Med 2007;120(10):871-9). Coverage was 42% at that
point, with 89 still invented.

That last pass is the one worth copying. The other four asked what a disease
looks like. It asked what the *rivals* look like, which is the half of a
likelihood ratio this project had been guessing at throughout: a study that
enrols everyone suspected of a condition and reports the arm that turned out
not to have it is measuring exactly the population a differential reasons
over. Immobility and calf tenderness in the seven non-PE diseases had been
carrying invented values of 0.61 and 0.40; PIOPED II measures 0.19 and 0.23.
The guess was wrong by two to three times in the direction that flattered the
model, and taking it back cost a claim — see below.

    knowledge base            plain   +corr   +decisive   +both
    invented                   8/10    8/10        8/10    8/10
    17 from DDXPlus            9/10      --          --   10/10
    + 4 from the literature    9/10   10/10        9/10    9/10
    + 14 from Merck           10/10    9/10       10/10    9/10

**Read the last row against the one above it before drawing anything from
this table.** The column that wins changes on every pass. Correlation was the
best arm on literature-sourced numbers and is the worst on Merck-sourced ones;
the plain loop was never best until it suddenly was. The full account of why
is in *Five sentences of textbook overturned the headline negative result*
below, and the short version is that correlation weighting was compensating
for over-extreme invented likelihoods and the compensation outlived the
problem.

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

After the Merck passes the rule is neutral rather than harmful — 10/10 with or
without it on the plain knowledge base, and it still costs turns for nothing.
Three passes, three verdicts on the same rule. None of them was a measurement
error; each was correct about a different knowledge base, which is the whole
difficulty with concluding anything about mechanism while the numbers underneath
are moving.

That is the caution for every number sourced from DDXPlus in this project.
They are measured, and measured from a generating model; a conclusion about
mechanism that rests on them has not yet met a real frequency.

## The guideline workup costs a case, and the reason is instructive

With the correlation structure on and likelihoods part-sourced:

    correlated  workup  decisive   top-1   cost
          yes      no        no     9/10   13.4
          yes      no       yes     9/10   16.8
          yes     yes        no     8/10   15.5
          yes     yes       yes     9/10   17.3

The workup is on by default and it is what loses fx-009. (These figures are
from the Merck-sourced knowledge base; the numbers moved but the conclusion
did not. Note that the whole grid is now below the plain, uncorrelated loop's
10/10 — on current numbers the correlation structure is the larger cost, and
this table holds it fixed to isolate the workup.)

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

## Running the synthesis pipeline found three bugs the tests could not

Section 5.2's generator was written, unit-tested, and never run against a
model. Every test passed. The first live run found three defects, and the
pattern in all three is the same: a stub returns what the test author expected,
so the code was only ever exercised on the shapes it already handled.

**Silent deletion of the most valuable cases.** `case_id` was
`f"{diagnosis}-{seed}-{index:03d}"` and ignored `confusable_with`, so every
near-miss batch collided with its diagnosis's typical batch and with every
other near-miss batch for it. Screening drops by id, so the collision deleted
the colliding cases without comment. The first run reported 20 generated, 13
kept, and looked healthy; in fact every near-miss case was gone — the cases
that exist specifically to stress the abstention gate, destroyed while the run
reported success. A stub test could not see this because it never generated two
batches for the same diagnosis.

**A report that could not be reconciled.** `screen()` derived
`rejected_as_duplicate` from the count of distinct duplicate *identifiers*
while its loop dropped *cases*. With colliding ids the report understated its
own losses and four cases vanished unaccounted for. The module docstring calls
the report "the deliverable as much as the corpus is", which is exactly why
this mattered: an unreconcilable report is worse than none, because it is
trusted. Counts are now taken where the drop happens and the function asserts
that every case is either kept or counted under exactly one reason.

**Parsing that would have emptied the corpus.** `_parse_json` unwrapped a
fenced block only when the response began with the fence, and `generate()`
caught every exception and returned `[]`. A model that answered correctly but
wrapped its answer in one sentence of prose was therefore indistinguishable
from one that refused, and the run would have printed "generated 0" with no
reason. Gemini adds a preamble routinely.

Two smaller findings are worth recording because they are the kind that get
assumed away. The model read "use ONLY findings from the provided vocabulary"
as an instruction about prose and wrote raw identifiers into all thirteen
narratives of the first run; tightening the prompt cut that to one in eighteen,
and a repair pass that verifies its own output takes it to zero — but it is
*measured* in the report rather than declared fixed, because prompt compliance
is not a guarantee. And `gemini-2.0-flash` now returns 429 with `limit: 0` on
the free tier, which reads like exhausted usage and is actually no quota at
all.

### The 20/20 is not a result

The agent scores 20 of 20 on the generated corpus. That number is a
self-consistency artefact and must not appear as a headline metric: the cases
were generated from the knowledge base the agent reasons over, so the score
measures agreement between a model's reading of the KB and the KB's own
arithmetic. It says nothing about diagnostic accuracy. The `synthetic-` prefix
on every generated id exists so that one reaching a metrics table shows up in
the per-case output instead of relying on someone remembering.

What the corpus is legitimately for: coverage, class balance, and stress cases
— particularly the near-miss vignettes, which are the reason the collision bug
above was worth finding rather than working around.

## Five sentences of textbook overturned the headline negative result

The second Merck pass added five citations. Three of them are on pulmonary
embolism and they moved the project's central finding.

| | invented | cited | source |
|---|---|---|---|
| P(fever \| PE) | 0.15 | 0.30 | "fever can occur" |
| P(productive cough \| PE) | 0.10 | 0.20 | "less common symptoms include cough" |
| P(crackles \| PE) | 0.15 | 0.20 | "less commonly ... crackles or wheezing" |

The invented values said pulmonary embolism essentially never presents with a
fever or a cough. The textbook says both occur, uncommonly. Under a product of
independent likelihoods that gap compounds, and it was holding four separate
conclusions in place.

**The buried diagnosis is no longer buried.** The headline negative result was
that two pneumonia-typical findings sank pulmonary embolism far enough that a
*positive CTPA* could not retrieve it — recorded as evidence that the remaining
work was the likelihood model rather than the loop. That was the right
diagnosis, and repairing three likelihoods repaired the failure. PE is now the
top hypothesis on that evidence, though still below 50%: the independence
assumption has not been fixed, only made less punishing on one disease.

**Correlation weighting now costs a case.** It was introduced because four
correlated negatives were being counted as four independent penalties, and it
helped — while the underlying numbers were too extreme. With the extremity
removed at its origin, the discount corrects an error that is no longer there:
the plain loop reaches 10/10 on the weakened configuration and the correlated
one gives a case back. The mechanism was never wrong; it was compensating, and
a compensation outlives its usefulness the moment the thing it compensated for
improves. This is the strongest argument in the project for fixing knowledge
bases rather than adding machinery to survive them.

**The masquerade failures are gone from the weakened configuration**, which is
where they were most visible. The shipped defaults still lose fx-009, now to
pneumonia rather than COPD.

**And one repair was itself reversed.** An earlier round of sourcing appeared
to make evidence fit separate the loop's failures from its successes; this
round put it back inside the range. That separation was a property of one
particular failure on ten cases, not of evidence fit, and it lasted exactly as
long as the numbers that produced it. The original conclusion — that evidence
fit reads whether the findings are explained by *something*, and in a
masquerade they are — was the durable one.

### What this says about the invented numbers

Not that they are harmless, and not that they are uniformly harmful. Three
numbers on one disease were holding a documented failure in place, while
`--ablate` shows the invented set as a whole is carrying most of the
discriminating structure. The cost of an invented likelihood is concentrated,
not spread — which is an argument for sourcing selectively and for expecting
each round of sourcing to overturn something, rather than for treating the
knowledge base as uniformly unreliable.

Nine tests changed in this round. Four were pinning conclusions the new numbers
falsified, and are superseded with the measurement that replaced them. Three
were pinning accidents of the old values rather than the behaviour they named —
a workup assertion that held only while some action cleared the due-diligence
bar on turn zero, a due-diligence comparison confounded by the workup's own
effect on `still_outstanding`, and a retrieval assertion naming whichever
source a meaningless hash embedding happened to rank first. Those three were
brittle when written and the sourcing merely exposed them.

## Sourcing the priors, and the fourth answer on correlation

The priors were the last wholly invented quantity, and the reason they stayed
that way was a search failure rather than an absence of data. DDXPlus cannot
supply them and the Merck chapters give population epidemiology, which is the
wrong quantity — what a prior needs is *presentation-conditional*: of patients
arriving with this complaint, how many have each cause. Both disease-side
searches missed it because the answer is in symptom-side articles. StatPearls
has two:

    Dyspnoea (NBK499965)      pneumonia 20-26%, heart failure 15-28%,
                              COPD 13-18%, asthma 13-15%,
                              PE / ACS / psychogenic each "fewer than 5%"
    Chest pain (NBK470557)    ACS 31%, GERD 30%, musculoskeletal 28%,
                              pericarditis 4%, PE 2%, pneumonia 2%

**The two disagree by twentyfold on acute coronary syndrome**, and that is the
finding rather than a nuisance. Renormalised over this differential, ACS is 3%
of a dyspnoea presentation and 63% of a chest-pain one. This project's
presentation is both, and nothing in either source says how they mix, so every
prior carries a band spanning the two and the point estimate assumes 50/50.
Three assumptions stack here — renormalisation, reading "fewer than 5%" and
"not named" both as 2.5%, and the mixing ratio — and they are listed in
`fixtures.py` beside the numbers rather than left implicit. These are derived
values, not measured ones.

What it changed:

- **The shipped configuration reaches 10/10**, from 9/10. fx-009 was the case
  the guideline workup had been costing since that rule was introduced.
- **Pulmonary embolism is rare in this presentation.** Its prior fell from an
  invented 0.10 to a derived 0.034, and acute coronary syndrome doubled to
  0.25. The invented priors were closest on pneumonia and pulmonary oedema and
  furthest on exactly the two conditions the chest-pain and dyspnoea
  literatures disagree about.
- **Correlation weighting matters again**, for the fourth different reason.
  With PE starting threefold further back, the plain knowledge base can no
  longer retrieve it even from a positive CTPA; the correlated one still can.

That fourth reversal is worth stating plainly, because the earlier three could
each be read as a correction and this one cannot. Correlation helped on
invented numbers, stopped helping once Merck sourcing removed the extremity it
was compensating for, and helps again now that the priors are real. None of the
four measurements was wrong. Each was correct about a different knowledge base.
**"Does correlation weighting help" has no answer independent of the numbers
underneath it**, and neither does any other question of that shape — which is
the strongest argument this project has for sourcing before tuning.

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
- The eight disease priors are **sourced** as of the StatPearls pass, and the
  entry below describes what that did. Superseded text follows for the record:
  the priors were invented. They are no longer *untracked*:
  `DiseaseEntry.prior_source` carries the same provenance tier as a
  likelihood, and `provenance.report` prints them on their own line rather
  than folding them into the likelihood percentage. Tracking is not sourcing —
  all eight remain guesses — but the audit can now name them, which it
  previously could not. DDXPlus cannot supply the real figures: its paper
  states the generation rates were capped into a 10–100% band to avoid a
  dataset dominated by a few pathologies, so its per-pathology counts are a
  rebalancing artefact and using them as prevalence would be worse than the
  invented values.
- Supports/contradicts is computed, not stored. Section 2 asks for an ontology
  whose nodes are findings and causes and whose edges are labelled; what exists
  derives that relation per call from likelihood ratios. The reasoning is
  auditable and every edge is cited, but there is no structure to traverse.

## The uncharacterised-cell backoff: better ranking, worse decisions

A real PMC patient — pmc-13070269, NSTEMI confirmed on angiography, troponin
raised, ST depressions present — ranked **fifth at 6%**, behind asthma at 31%.
Decomposing it found the mechanism, and it was not the cost model or the
selector:

    finding                observed    P(·|ACS)   P(·|asthma)
    smoking_history        absent          0.28          0.80
    pleuritic_pain         present         0.10          0.31
    exam:ecg_st_changes    present         0.75          0.71
    lab:raised_troponin    present         0.90          0.45

Asthma lists no entry for troponin, ECG changes or pleuritic pain, so each
backs off to the KB-wide marginal — which the cardiac diagnoses drag upward.
A raised troponin therefore scores 0.45 under asthma rather than something
near 0.02, and two findings that should demolish the hypothesis barely move
it, while *absence of smoking history* counts nearly three times harder
against ACS than a raised troponin counts for it.

One correction came out of this and stands on its own: the Merck troponin
sentence for pericarditis had been transcribed as "almost always elevated"
and converted through the `always` bucket to 0.95, which put a raised
troponin at higher probability under pericarditis than under ACS. The source
says "often"; the published frequency is 30–50%, inside the band `often`
carries. Corrected to 0.55, from the rubric rather than by choice. It moved
the case from 6% to 7% and fixed nothing, which is the honest report.

**The backoff policy itself was then changed, measured, and rejected.**
`unlisted_as_atypical` reads a finding missing from a disease's profile as
weak evidence of absence, at the rubric's "not typical" value, on the
argument that a text enumerating a disease's features and omitting one has
said something. As a ranking change it works:

    policy            fixtures   real correct   commits   mean true rank
    marginal (ship)      10/10          3/8         3          2.62
    not-typical           9/10          4/8         7          2.00

And as a safety change it is a disaster. Commits rise from 3 to 7 because
every uncharacterised cell becomes strongly discriminating at once and the
posterior sharpens everywhere — but **three of those seven commits are
wrong**: pericarditis committed as acute coronary syndrome at 98%, and the
confirmed NSTEMI this whole investigation started from committed as
pericarditis at 87%. The case did climb from fifth to second, and then the
gate acted on it. An unexcluded ACS that escalates is safe; a confident wrong
commit on a real infarction is the failure the gate exists to prevent.

Kept as a flag, off, with a test asserting the shipped policy makes no wrong
commits and the alternative does. The lesson is the one this project keeps
relearning in different clothes: mean rank improved while decisions got
worse, which is why coverage and error are reported separately and why a
ranking metric must never be the thing a change is adopted on.

## Completing the grid: 77 cells nobody had chosen

The backoff investigation left a question it did not answer. Filling four
cardiac-marker cells helped and held safety, but the audit that prompted it
found the same defect in **77 cells** — every place a disease does not list a
finding and the KB-wide marginal answers for it. The marginal is computed
only over the diseases that *do* describe a finding, so the more specific a
sign is, the higher the number it hands to every disease that lacks it:

    finding                listed for            asserted for everyone else
    wheeze_subjective      COPD, asthma                              0.89
    exam:friction_rub      pericarditis only                         0.60
    recent_immobility      pulmonary embolism only                   0.61
    calf_tenderness        pulmonary embolism only                   0.40

Read the last two: they are Wells criteria, which exist to identify pulmonary
embolism, and the model was asserting recent immobility at 61% in panic
attack and calf tenderness at 40% in pneumonia. A pericardial friction
rub — among the most specific signs in this differential — was a coin flip
for every disease that is not pericarditis.

**The backoff was never "no claim".** The reasoner used a number either way.
The only question was whether anyone had chosen it, and nobody had.

`_COMPLETIONS` chooses the rest. Not by one rule: blanket-low was already
measured and rejected above, and the reason it failed is the split the
completions turn on. A finding can be missing from a profile because the
pathology does not produce it — a friction rub needs an inflamed pericardium,
a filling defect needs clot — or because it is a *risk factor the disease
does not cause at all*. A pneumonia patient can perfectly well have been
immobile for three days or smoked for forty years, so those cells get base
rates, not floors. Two of those three risk factors have since been measured
rather than reasoned about, which took the block from 77 cells to 63 and
showed the reasoning had been right in kind and wrong in size: the base rates
chosen here for immobility and calf tenderness were 0.03 to 0.10, against
PIOPED II's measured 0.19 and 0.23. That is the same trap that nearly produced an invented
0.05 for troponin in COPD before the literature said 32%.

    config                fixtures  no-workup arm  real correct  wrong  rank
    marginal (shipped)      10/10           8/10           3/8      0   2.50
    complete_grid           10/10           8/10           4/8      1   1.88

It buys real ground: the confirmed NSTEMI climbs from fourth to second and
escalates there rather than committing wrongly, mean rank matches the
rejected blanket policy without its three wrong commits, and the shipped
configuration still gets every fixture case.

It is off anyway, and the reasons are worth being explicit about. It is 63
invented numbers from one non-clinician in a single sitting. It takes the
sourced fraction from 42% to 24% — not a regression, but the count finally
including claims the model was already making. And it turns one abstention
into a wrong commit: acute coronary syndrome at 82% on a true pericarditis.

That last one is the closest call in this document. The error runs *toward*
the time-critical diagnosis, which is the direction section 2 of
RESPONSIBLE_AI.md argues the system should prefer, and the direction the
source case's own clinicians took — they went to angiography. An argument,
not a proof. Adopting it means re-measuring four findings recorded above
that it revises, and that is a decision for a clinician or the supervisor
rather than for whichever configuration scores better today.

### Adopted, measured, and reverted

The entry above ends "It is off anyway". It was then turned on, the four
findings it revises were re-measured rather than re-asserted, and it was
turned off again once the real-case calibration was measured. The
measurements below all describe the flag-on configuration and remain
true of it; the decision at the end of this section is what changed.

**The fifth reversal on correlation and decisive tests, and the first that
inverts the heading three sections above.** With every cell chosen:

    corr   decisive   workup   top-1
    no        no        no      9/10   (fx-009)
    no        yes       no      9/10   (fx-009)
    yes       no        no      9/10   (fx-009)
    yes       yes       no     10/10
    yes       no        yes    10/10   <- shipped

Correlation alone no longer reaches ten and the plain loop no longer trails
it, while the decisive-test rule — written off three times above, at real
length — is now one of the two things that gets every case. The reason is
mechanical rather than vindicating: a decisiveness rule asks what a
hypothetical result would do to the posterior, and a posterior built on
filled cells actually moves. It never had that before. Five measurements,
five answers, and the durable claim is still the one the original entry
made: this reports what the numbers underneath are, not what a mechanism is
worth.

**The weakened arm's single failure moved from fx-001 to fx-009.** The
pneumonia misread as COPD is no longer the last one standing; the buried
pulmonary embolism is.

**A positive CTPA now carries pulmonary embolism past 0.5** — 0.537, against
the earlier 0.4-something — because the seven rivals state low values for a
filling defect instead of inheriting a marginal that flattered them. The
test's bound moved from "< 0.5" to a range, since the claim worth protecting
was never a particular number.

**And the finding that should give a reader most pause.** The escalation
packet test sets a deliberately unclearable gate — 99% confidence, 98%
margin — and on the *uncorrelated* knowledge base fx-001 now clears it:
99.26% confident with a 98.68% margin after eight turns. Twenty-seven filled
cells multiplying as independent evidence is all that takes. The same case
on the correlated base sits at 42% and abstains.

So correlation weighting is no longer worth "about one fixture case", which
is how every earlier entry here describes it. It is now the only thing
between this model and unusable overconfidence — and the correlation values
are themselves invented, in a file that says so. Completing the grid did not
remove the project's dependence on unsourced numbers. It moved that
dependence somewhere less visible, and this paragraph exists so the move is
on the record rather than discovered later.

Real cases under the shipped default: 4 of 8 correct, one wrong commit
(acute coronary syndrome at 82% on a true pericarditis), the confirmed
NSTEMI escalating from second place rather than committing wrongly.

### Correcting the overconfidence claim, and damping what caused it

The entry above says correlation weighting is "the only thing between this
model and unusable overconfidence", on the evidence that fx-001 cleared a
99%/98% gate on the uncorrelated base. That was the wrong evidence and the
claim was overstated. fx-001 is a case the model gets *right*, so 99.26%
there is confidence, not overconfidence, and on the fixture set completing
the grid actually **improved** calibration in every arm — ECE 0.286 to 0.143
correlated, 0.199 to 0.074 plain, with every arm underconfident rather than
over.

The concern survives, but only on the evidence that matters. Fixtures are
generated from the knowledge base the model reasons over, so their
calibration measures self-consistency. On the eight real PMC cases, which
are not:

    complete_grid   ECE     Brier   mean conf   accuracy   overconfidence
    off             0.303   0.103       0.520       0.50           +0.020
    on              0.347   0.207       0.695       0.50           +0.195

Top-1 accuracy did not move. The Brier score doubled and twenty points of
unearned confidence appeared, which is the real cost of adopting the grid
and is what the wrong commit is made of.

The mechanism was findable. Nine findings — pleuritic pain, rest dyspnoea,
palpitations, tachycardia, friction rub, hypoxia, white count, D-dimer and
CTPA — sat in no correlation group at all. That cost nothing while most of
them backed off to the marginal for most diseases and contributed a
likelihood ratio near one. Completing the grid turned all nine into real
evidence multiplying independently.

Four groups were added on the same basis as the original five, findings a
clinician reads as one story: respiratory distress (rest dyspnoea, hypoxia,
tachycardia), infection (fever, white count), inflamed pleura or pericardium
(pleuritic pain, friction rub), and palpitations with tachycardia. The last
changed no metric at all and is kept because the claim is true, not because
it earned its place.

    real cases          ECE     Brier   overconfidence
    grid off          0.303     0.103          +0.020
    grid on           0.347     0.207          +0.195
    grid on + groups  0.288     0.171          +0.156

ECE now sits below where it was before the grid was completed; Brier and
overconfidence are recovered part of the way and no further. Ranking, the
fixture set and the real-case commits are all unchanged by the damping —
4 of 8 correct, one wrong, mean rank 2.00, fixtures 10/10.

**D-dimer and CTPA were deliberately left ungrouped.** They are genuinely
correlated, and damping them together is precisely what would undo the
fx-009 repair: the point of that case is that a positive CTPA must survive a
cluster of absent VTE findings. Correlating the decisive test with the
picture it exists to overrule would be a fix that breaks the thing this
knowledge base was most recently repaired to do.

What remains true, in weaker form than the entry above claimed: the model
now leans harder on correlation values that are themselves invented, and the
residual +0.156 overconfidence on real patients is unexplained by anything
measured here.

### The decision, and why calibration outranked ranking

Two configurations, both measured, both defensible on their own terms:

    config                      wrong   Brier   overconf   real ok   rank   invented
    marginal backoff (shipped)      0   0.103     +0.020       3/8   2.38    102/139
    completed grid + damping        1   0.171     +0.156       4/8   2.00    179/216

Those figures are as measured on the day of the decision and are left as
recorded. The knowledge base has since taken the PIOPED II pass and lost the
damping groups, so the current numbers are 102/153 invented, mean rank 2.50
shipped against 1.88 with the grid on, and the wrong-commit count on each arm
is unchanged. The decision below is unaffected by the move; the argument
never turned on the size of the ranking gain.

The completed grid wins on ranking and on one more correct commit. It is not
adopted, and the reasoning is worth stating because the losing column is the
one that looks better at a glance.

**It degrades the claim this project is actually making.** Calibrated
abstention is the thesis; ranking is not, and this document already records
the trap — mean rank improving while decisions get worse — twice, in the
blanket-backoff entry and again above. Adopting a milder instance of the
same trade with that written down would be choosing the metric that flatters
rather than the one that matters.

**It is inconsistent with a decision already taken.** The blanket-low
backoff was rejected for this exact trade and a test enforces it. Accepting
a gentler version of the same harm, because it is gentler, is motivated
reasoning rather than a new finding.

**It triples the invented count on one non-clinician's judgement**, in a
single sitting, with no clinician available to review it, in a project whose
central limitation is precisely how many of its numbers are invented and how
well the reader can tell which.

**And its main cost is unexplained.** The residual +0.156 overconfidence on
real patients survives the correlation damping and is not accounted for by
anything measured here. A change whose principal harm cannot be explained
does not belong on the default path.

What that leaves standing is a real defect, stated rather than fixed: 63
cells still answer from the marginal, so the model asserts a pericardial
friction rub at 0.60 in panic attack. The Wells half of that complaint has
since been answered by measurement rather than by this flag — see the
entry below. Those are indefensible values,
they are documented here, and the completions that would replace them sit
behind `complete_grid` fully measured, for a clinician to adopt rather than
an engineer to assume.

The four correlation groups added to damp the completions went with them.
They help only when the grid is complete: with it off they cost two fixture
cases (10/10 to 8/10), which is the clearest possible evidence that they
were compensating for the completions rather than correcting the model.


## Sourcing the rivals: PIOPED II, and a repair that was resting on a bad number

Every entry above this one asks the same question of a source: what does this
disease look like? The likelihood ratio needs the other half too — what do
the *rivals* look like — and this project had been answering that half by
judgement throughout, including in the two values it complained loudest
about. Recent immobility sat at 0.61 and calf tenderness at 0.40 for the
seven non-PE diseases, both inherited from pulmonary embolism's own marginal,
which amounts to asserting that a pneumonia patient usually has a swollen
calf.

PIOPED II (Stein PD et al., Am J Med 2007;120(10):871-9) answers it directly,
because of how it was built: it enrolled patients *investigated* for
pulmonary embolism and reports the arm that turned out not to have one. That
arm is the population a differential actually reasons over. Table 3 gives
immobilisation in 121 of 632 without PE (19%); Table 6 gives calf or thigh
signs of DVT in 146 of 632 (23%). Fourteen cells, two findings across seven
diseases, invented to measured:

    finding             invented   PIOPED II (no-PE arm)
    recent_immobility       0.61                    0.19
    calf_tenderness         0.40                    0.23

Coverage goes 27% to 33%; the completions block shrinks from 77 cells to 63.

**And it cost a claim, which is the part worth recording.** This document
previously stated that correlation weighting plus Merck sourcing together
repaired fx-009 — a pulmonary embolism reading as pneumonia — and a test
asserted that the correlated proposer put PE top of that evidence set. Both
were true. Part of the reason was not defensible: the rivals were being
penalised for the *absence* of venous-thromboembolism findings at 0.61 and
0.40, so a large share of PE's winning margin came from a number nobody had
chosen. With the measured values the penalty is small, and on that frozen
evidence set the correlated proposer now returns COPD.

What survives is the mechanism, which was always the point:

    fx-009 evidence set        PE rank   PE probability
    without correlation              4            0.039
    with correlation                 2            0.276

Four correlated negatives are still not four independent penalties, and
weighting them is still worth a seven-fold lift on identical evidence. The
case itself is still answered: fx-009 is a case rather than a snapshot, and
the shipped loop gathers the evidence this set freezes out and commits to
pulmonary embolism correctly. The test was narrowed to assert exactly that
and no more.

The weakened arm — no workup floor, no decisive-test rule, no correlation
weighting — went from one failure to two, fx-001 and fx-009. That is the
same effect seen from the other side, and it is not a regression in the
model: the prop was removed, and the arm that had been leaning on it now
reports what it was always worth. Correlation weighting alone carries both
cases back. The shipped configuration gets all ten throughout.

The general lesson is the one this project keeps relearning in different
forms. An invented number does not announce itself by making results worse.
This one made a result better, held up a documented claim, and was wrong by
two to three times.


## Which cells are sourceable at all: a taxonomy, and one negative result

The PIOPED pass raised an obvious question: what else is reachable? The answer
turned out to depend on the *kind* of finding, not on how much literature
exists about it, and the two attempts that established this are worth keeping
because one of them failed.

**Natriuretic peptide was attempted first and abandoned.** It looked like the
best target: `lab:raised_bnp` was 0 of 6 sourced, the presentation studied in
that literature is acute dyspnoea, which is exactly this project's, and
separating heart failure from COPD and pneumonia in a breathless patient is
the question those studies were built to answer. Three cohorts were checked.
Ray 2006 (Crit Care, n=514) stratifies natriuretic peptide by *death*, not by
diagnosis. Morrison 2002 (JACC) gives means and standard deviations by
diagnosis, not proportions above a threshold. The Breathing Not Properly
by-diagnosis breakdown is not in free full text.

One usable figure did surface — 104 of 167 (62%) of acute COPD
exacerbations without left ventricular dysfunction had elevated NT-proBNP —
and it was still not taken, for the reason that decides this whole question.
That figure sits on an age-specific NT-proBNP threshold. A heart-failure
figure would sit on BNP > 100 pg/mL. Different analyte, different cutoff. A
column assembled from them would manufacture a comparison the sources do not
support, which is the same artefact class this document already refuses to
reconcile when DDXPlus says 71% and a real cohort says 33% for pleuritic pain
in pulmonary embolism.

**Chest radiography was attempted second and worked**, for the complementary
reason. Consolidation is a binary radiological observation, so studies saying
"consolidation", "new infiltrate" and "pneumonic infiltrate" are answering one
question and can be pooled. `("copd_exacerbation", "imaging:cxr_consolidation")`
is now measured at 0.18.

That gives a taxonomy of the remaining 101, and it is more useful than the
sourcing itself:

    kind of finding                                     cells   sourceable
    binary observation, universal definition              ~20   yes, one study at a time
    continuous biomarker, assay-dependent cutoff          ~27   only from a single cohort
                                                                reporting every disease
    differentially caused by the rivals                   ~26   only from a by-diagnosis
                                                                cohort; pooling smears it
    genuinely unpublished (P(raised BNP | panic attack))  ~28   no

The third row is the one that keeps catching people, this author included.
PIOPED II's Table 6 reports crackles in 112 of 632 and decreased breath sounds
in 109 of 632 in its no-PE arm, and those numbers are *right there*, already
extracted. They are unusable pooled: pneumonia and pulmonary oedema cause
crackles and asthma causes wheeze, so a figure averaged over the seven rivals
would smear asthma's rate across pericarditis. The PIOPED pass worked only
because immobility and calf tenderness are risk factors that none of the eight
diseases causes.

The practical consequence is that the earlier estimate of 45-50% achievable
coverage was too optimistic. Without a cohort reporting findings by final
diagnosis — the one source shape that would fill whole columns, and the one
none of these searches found — the realistic ceiling is nearer 36-40%.


## The headline evaluation was measuring the wrong knowledge base

Found while writing WRITEUP.md, which is the argument for writing the
document: assembling every number in one place is what made two of them
visibly disagree.

`scripts/run_eval.py` built its knowledge base with `build_knowledge_base()`
— correlation weighting off. Every other entry point in `scripts/`
(`consult.py`, `demo.py`, `eval_real_cases.py`, and the web UI) builds it with
`correlated=True`. So for most of this project's life the headline metrics,
including both required baselines, described a configuration that nothing
shipped.

It understated the system, and that is probably why it lasted. A number that
comes out too low does not prompt anyone to go looking for a bug. Every
mechanism this document credits with fixing a case was being credited on a
knowledge base the evaluation was not using.

    metric                        before    after
    loop top-1 (held-out, n=7)     85.7%     100%
    single-pass baseline           85.7%     100%
    MRR                            0.929     1.000
    Brier                          0.103     0.097
    mean evidence seen               9.6      10.1
    abstention precision           50.0%      0.0%
    accuracy gained by the gate    +14.3%    +0.0%

The first four rows are not an improvement. Nothing about the system changed;
only which configuration was measured. The last two rows are the real news and
they run the other way: with the model now correct on all seven held-out
cases, the gate abstains twice and gains nothing, because there is no error
left to decline.

**That is a fact about the fixture set, not about the gate.** Ten hand-written
cases were always described here as a smoke test; they are now demonstrably
saturated, and a saturated set cannot measure a mechanism whose entire purpose
is knowing when to stop. Calibration says the same thing from the other side:
mean confidence 69.7% against 100% accuracy is 30 points of *under*confidence,
the opposite of the failure this project spends most of its effort guarding
against. The evidence that the gate earns its keep is the DDXPlus runs
recorded above (+10.9% accuracy gained, 52.9% abstention precision) and the
zero wrong commits across the eight real cases — not this split.

Two scripts were suspected alongside it and cleared, which is worth recording
so nobody re-opens it. `sensitivity.py` takes an explicit `--correlated` flag
and SCOPE.md reports its results both ways, so that is a deliberate design.
`plausibility_check.py` produces byte-identical output either way, because the
evidence table is built from likelihood ratios against the marginal and
correlation weighting does not enter that computation. Only `run_eval.py` was
wrong.


## A replacement test set, and the wrong commit it found immediately

Fixing `run_eval.py` left this project without a working instrument. The ten
development fixtures are saturated: the shipped configuration ranks all ten
correctly, commits none of them wrongly, and the gate's measured value on them
is +0.0% accuracy gained at 0.0% abstention precision, because there is no
error left to decline. Every mechanism argued for in this document was
justified as being worth some number of fixture cases. That measurement no
longer discriminates anything.

`build_hard_cases` is five named diagnostic traps written to replace it, kept
deliberately separate from `build_cases` so that every measurement already
recorded here keeps its denominator.

**How they were chosen matters more than what they contain.** Each is a
recognised trap in this differential, written from the clinical picture before
the model was run on any of them, and none was kept or dropped according to
whether the model got it right. The easy way to manufacture a discriminating
set is to generate cases, keep the failures, and report a set the system
fails; that measures a willingness to search and nothing else.

    fx-h01  cardiac asthma          LV failure wheezing, reads as COPD
    fx-h02  myopericarditis         troponin + ST changes, reads as ACS
    fx-h03  silent ischaemia        elderly diabetic, no chest pain at all
    fx-h04  pneumonia in a COPD pt  every COPD feature present, both true
    fx-h05  asthma, never smoked    the row with no supporting edges

First run: four of five ranked first, three committed correctly, one escalated
correctly (fx-h02 at 61%, just under the threshold), and **fx-h04 committed
wrongly at 70% confidence** — a pneumonia read as a COPD exacerbation with
consolidation visible on the radiograph. That is the first wrong commit on any
fixture set in this project, and the ten cases were structurally incapable of
producing it.

### Why fx-h04 fails, and the fix that does not fix it

The proposer without correlation weighting gets it right and confidently:
pneumonia 0.929. With correlation weighting it flips to COPD 0.485 against
pneumonia 0.470.

The first hypothesis was uncharacterised cells. Pneumonia states no value at
all for wheeze, smoking history or reduced breath sounds, so COPD claims all
three unopposed at 0.91, 0.82 and 0.60 while pneumonia falls back to the
marginal. That hypothesis is wrong: `complete_grid=True` fills exactly those
cells and fx-h04 still commits to COPD.

The real cause is a grouping this document already argued against, in another
case, and then made anyway:

    (0.75, ("fever", "productive_cough", "exam:crackles", "imaging:cxr_consolidation"))

Consolidation on the radiograph is the decisive test that separates these two
diagnoses — 0.95 against 0.18 — and it is damped together with the three soft
symptoms it exists to overrule. The entry above on the damping groups states
the principle explicitly for the other decisive test: "Correlating the
decisive test with the picture it exists to overrule would be a fix that
breaks the thing this knowledge base was most recently repaired to do." That
is why D-dimer and CTPA were left ungrouped. The consolidation group does the
thing that paragraph forbids.

Removing `imaging:cxr_consolidation` from that group was measured across every
set:

    arm                 fixtures      hard        real cases          calibration
    grouped (current)   10/10, 0 wrong  4/5, 1 wrong  3/8, rank 2.50  ECE .298 Brier .102 oc +.017
    consolidation out   10/10, 0 wrong  4/5, 1 wrong  3/8, rank 2.38  ECE .274 Brier .091 oc +.059

**It does not fix fx-h04 either**, which is the strongest evidence available
that it is not tuning: it was reached from a stated principle, it fails to
rescue the case that prompted looking at it, and it still improves the
fixtures from seven commits to eight, real-case rank, ECE and Brier. Its one
cost is overconfidence, +0.017 to +0.059.

Not adopted. The trade is mixed and this document's own precedent, set when
the completed grid was rejected, is that calibration outranks ranking; a
change that improves Brier while tripling overconfidence needs a reason
better than one case pointing at it. Recorded here so the next person has the
measurement rather than the intuition.


## Closing DDXPlus for good, and recovering frequencies from likelihood ratios

Two attempts at the invented fraction: one ended a route, one opened a method.

### DDXPlus is exhausted, provably rather than presumptively

SCOPE.md has said for some time that the DDXPlus route was exhausted. What
that rested on was the extractor emitting nothing new — and the extractor only
ever looks at cells the fixture entry already carries, a deliberate choice
recorded in `map_concepts.py`: "adding one changes the shape of the model
rather than sourcing it." That argument weakened once the PIOPED pass seeded
fourteen new cells on purpose, so the question was reopened and answered
properly, by computing all eighty pairs the mapping can reach whether the cell
exists or not.

    mapped concepts x diseases                     80
    cells DDXPlus can source that already exist    15
    new cells with a non-zero frequency             0
    new cells reading exactly 0.000                13

Every unfilled pair reads exactly zero, which is not a frequency. The mapping
file already warned about this — "a finding its rule base omits is generated
for nobody" — and the pericarditis row shows what writing them would have
cost: DDXPlus generates fever in 0 of 8623 pericarditis patients, against a
fixture value of 0.60 and a disease that genuinely causes fever. Thirteen
confidently sourced zeros would have been considerably worse than thirteen
labelled guesses.

Only ten of twenty-seven findings are mapped at all, and that ceiling is
structural rather than clerical: DDXPlus holds no examination signs, no
laboratory results and no imaging. Those are seventeen of the twenty-seven,
and they are where most of the remaining invented cells live. The route is
closed.

### Likelihood ratios invert

An earlier pass dismissed the JAMA Rational Clinical Examination series
because it reports likelihood ratios rather than raw frequencies. That was
recorded here as a negative result, and it was half wrong — which is the
argument for writing negative results down somewhere they can be re-examined.

Both ratios are functions of the same two unknowns, and the system inverts:

    LR+ = sens / (1 - spec)        spec = (LR+ - 1) / (LR+ - LR-)
    LR- = (1 - sens) / spec        sens = 1 - LR- * spec

Sensitivity in a cohort of *dyspnoeic emergency patients* is exactly
P(finding | heart failure) over this project's own population, so the review
of dyspnoea in the emergency department (Wang CS et al., JAMA
2005;294(15):1944-56) yields three cells directly:

    finding            LR+   LR-    recovered sens   was
    raised JVP         5.1   0.66            0.39    0.80  invented
    crackles           2.8   0.51            0.60    0.75  invented
    orthopnoea         2.2   0.65            0.50    0.755 DDXPlus

Only the sensitivity half is taken. The complement is P(finding | not heart
failure) pooled over the other seven diseases, and crackles and a raised JVP
are differentially caused by pneumonia and by cor pulmonale, so pooling them
would smear one disease's rate across the rest — the same objection that kept
PIOPED II's crackles row out of this knowledge base.

Orthopnoea overwrites a DDXPlus value through the existing precedence rule: a
frequency counted in real emergency patients outranks one counted in a
simulator. The two disagree by half again, the same gap already recorded for
pleuritic pain in pulmonary embolism, and it is reported rather than
reconciled.

Two caveats travel with the method, and are noted in the source comment as
well. A meta-analysis may pool LR+ and LR- over different subsets of studies,
so the recovered pair is close rather than exact; and the bands are the
reported confidence intervals pushed through the same inversion, which spans
the reported uncertainty without being a confidence interval in its own right.

### What it cost, which is the part worth reading

Coverage 34% to 35%, invented 101 to 99. The hard cases charged for it
immediately:

    arm            before                     after
    fixtures       10/10, 7 commits, 0 wrong  10/10, 8 commits, 0 wrong
    hard cases     4/5 top-1, 3 commits       4/5 top-1, 1 commit
    real cases     3/8, 0 wrong, rank 2.50    3/8, 0 wrong, rank 2.50
    calibration    ECE .298 Brier .102        ECE .301 Brier .103

Two hard cases stopped committing. fx-h01, the cardiac asthma, falls from 0.81
to 0.79 and escalates; fx-h05, the asthma, from 0.70 to 0.61. Both were being
answered *correctly* before.

The mechanism is not subtle. A raised JVP was invented at 0.80 and measured at
0.39, so the model had been more than twice as confident as the evidence
allows that a patient in pulmonary oedema has a raised JVP. Lowering three of
that disease's values also lowers the knowledge-base marginal those findings
are scored against, which is why a case about asthma moved too.

Kept. These are measurements replacing guesses, no wrong commit appeared
anywhere, and the fixture arm actually gained a commit. The precedent here is
consistent and was set when it cost something: the pericarditis ECG likelihood
was sourced *lower* than the invented value and kept anyway, and the guideline
workup costs a fixture case and was kept anyway. A model that is less decisive
because it stopped overstating its own evidence is behaving correctly, and the
two escalations are the abstention gate doing the thing the rest of this
document says it is for.

It is also the first time the hard cases have priced a change. The ten
development fixtures scored this as an improvement, seven commits to eight,
and would have reported nothing else.


### Two more inversions, and what five sourced cells did to one disease

The same review carries matched ratios for two further findings in this
vocabulary, and both corrections are large:

    finding                   LR+   LR-   recovered   was
    CXR venous congestion    12.0  0.48        0.54   0.85  invented
    leg oedema                2.3  0.64        0.50   0.999 DDXPlus

The radiograph figure is worth pausing on. A sensitivity of 0.54 says the film
is normal in nearly half of acute heart failure, which is the well-known
insensitivity of chest radiography in this setting — and precisely the sort of
clinical fact an invented 0.85 erases.

The leg oedema figure is worse than a bad guess. DDXPlus records leg swelling
in 7202 of 7205 simulated pulmonary oedema patients. A likelihood of 0.999 for
any clinical finding should have been suspicious on sight, and it survived
several passes through this file specifically because it was *sourced* — a
number with a citation attracts less scrutiny than one labelled invented,
which is a failure mode this project should have anticipated given how much
time it spends on the reverse case.

One finding was deliberately not taken. The same table gives "any abnormal
ECG" at LR+ 2.2 and LR- 0.64, inverting to 0.51. The concept here is
`exam:ecg_st_changes` — ST-segment change specifically, not any abnormality at
all. Writing 0.51 into it would be exactly the mis-mapping the DDXPlus mapping
file warns about: a confidently sourced wrong number, worse than the invented
one it replaced.

### The trend across both passes, stated plainly

    pass                     invented  coverage  ECE     Brier   overconf
    before Wang                   101       34%  0.298   0.102    +0.017
    three cells (JVP, crackles,
      orthopnoea)                  99       35%  0.301   0.103    +0.014
    five cells (+ CXR, oedema)     98       36%  0.316   0.107    -0.010

Coverage improved and ECE got worse, monotonically, across both passes. That
is the honest shape of it and it should not be smoothed over: replacing
invented numbers with measured ones has made this model's real-case
calibration slightly worse, not better. Two things argue for keeping it
anyway. The overconfidence figure moved from +0.017 to -0.010, which is closer
to calibrated in absolute terms and errs toward caution rather than away from
it; and no wrong commit appeared anywhere, on any arm, through either pass.

A construct note that belongs with these five. Wang's population is dyspnoeic
emergency patients with *heart failure*, and this knowledge base's disease is
*acute pulmonary oedema*, which SCOPE.md defines as the acute decompensation
of heart failure. Those are close enough to map — the study population is an
acute presentation to an emergency department with breathlessness, which is
this project's presentation exactly — but the study arm includes decompensated
patients less florid than frank pulmonary oedema, so these sensitivities are
if anything conservative for the extreme end of the disease.

### Three test claims moved, two of them for the better

**The weakened arm lost fx-001.** It now fails only on fx-009, down from two.
The pneumonia read as COPD had been losing partly because pulmonary oedema
claimed crackles at an invented 0.75 against a measured 0.60.

**The evidence-fit claim reversed for the third time, and then got a proper
test bed.** That test has now recorded, in sequence, that evidence fit cannot
separate failures from successes, that it can, that it cannot, and that it
can. Every one of those measurements was taken on a deliberately weakened arm
carrying exactly one failure, and a separation demonstrated over one failure
is a property of that failure. The test's own docstring said so and then
asserted the separation anyway, which is why it kept flipping.

It is now measured where a real failure exists under the *shipped*
configuration: fx-h04 in the hard case set. The result is stronger than the
original claim rather than merely consistent with it —

    fx-h01 0.2103   fx-h02 0.2493   fx-h03 0.1921   fx-h05 0.1480   (correct)
    fx-h04 0.2832                                                    (WRONG)

— the masquerade is the *best-explained case in the set*. A threshold on
evidence fit placed anywhere would reject correct answers before it rejected
this one. That is what a masquerade is, and it is why evidence fit cannot be
used as a safety check.

**And the hard cases priced this change too**, differently from the last one:
committed-correct went one to two while the fixture arm went eight back to
seven. The two sets disagree about whether this pass was an improvement, which
is the first time that has happened and is a better problem than the one
before it, when only one set could speak at all.


### D-dimer, a confirmation, and a defect recorded rather than fixed

Sensitivity is P(finding | disease) directly, so a diagnostic-accuracy
meta-analysis needs no inversion at all. The turbidimetric D-dimer review
(DARE NBK70277, nine studies, n=1901, emergency department) puts sensitivity
at 93% (95% CI 89–96) against an invented 0.95. That is the second time
sourcing has confirmed a guess rather than overturned one, and worth recording
alongside the times it did not.

The specificity is the uncomfortable half, and it exposes a defect this pass
deliberately does not fix. At 51%, P(raised D-dimer | not pulmonary embolism)
is 0.49 across the arm of suspected patients who turned out not to have one.
This knowledge base gives those same seven diseases 0.10 to 0.35, averaging
about 0.22:

    community_acquired_pneumonia   0.35     acute_pulmonary_oedema  0.30
    copd_exacerbation              0.25     acute_coronary_syndrome 0.20
    pericarditis                   0.15     panic_attack            0.10
                                            measured pooled:        0.49

The rivals are collectively understated by roughly a factor of two, which
makes D-dimer a stronger discriminator inside this model than it is in a real
emergency department. That is a measured defect in six numbers, and it is
being left in place.

The reason is the one that rejected the blanket backoff and the completed
grid, and it has not changed: 0.49 is a pooled figure over a mix of diseases,
and writing it into all seven would assert that a panic attack raises a
D-dimer as often as a pneumonia does. Both policies of the form "write one
value everywhere" have now been measured in this project and both were
rejected on evidence. Recording the defect is the honest option; papering over
it with a number nobody chose is the option that would make the coverage
percentage look better and the model worse.

A caveat on the one figure that was taken. The pooled studies use cutoffs from
190 to 500 microg/L across six turbidimetric assays, which is the same
heterogeneity that blocked the natriuretic-peptide column. It is tolerable
here only because a single number is being taken from a single pooled
analysis, rather than a column being assembled from sources that disagree
about what the test is.

### Where the sourcing effort now stands, column by column

Four passes have taken the invented count from 102 to 97 and coverage from 27%
to 37%. What remains is not a matter of effort, and the reasons differ per
column, which is the useful form for anyone picking this up:

    column                    cells   why it is stuck
    lab:raised_wcc              0/8   no diagnostic-accuracy literature treats
                                      the white count as a test for these
                                      diagnoses; it is a severity marker
    lab:raised_d_dimer          1/7   rivals need a by-diagnosis breakdown;
                                      only a pooled 0.49 exists
    lab:raised_bnp              0/6   BNP against NT-proBNP against
                                      age-specific cutoffs: no common scale
    exam:hypoxia                0/6   reported as a continuous PaO2 or
                                      saturation, rarely dichotomised
    exam:tachycardia            2/8   differentially caused by every rival;
                                      pooling smears it
    imaging:ctpa_filling_defect 0/5   near-definitional for PE, and no study
                                      reports it for the other seven

The single source shape that would break this open is unchanged from the
earlier taxonomy: a cohort of undifferentiated acute dyspnoea or chest pain
reporting each finding *by final diagnosis*. Four separate searches have not
found one. Until one exists, sourcing here is one cell per literature search,
and the remaining cells are the ones no literature was written to answer.


## Two more real patients, and the metrics they made worse

The real-case set was eight, one per modelled diagnosis, and its own docstring
called it too small to support any claim. It is now ten.

The selection rule was fixed before anything was run: second cases go to the
commonest emergency causes — pulmonary embolism and pneumonia — chosen on
epidemiology rather than on where the model happens to struggle. That
constraint matters more here than it does for sourcing, because a real-case
set is exactly the kind of thing one could quietly curate into a flattering
result, and nothing in the code would notice.

    pmc-11753817  pulmonary embolism, presenting as pneumonia: fever 39.4C,
                  productive cough, white count 11.2 -- and a NEGATIVE
                  D-dimer at 0.39 mg/L FEU against a stated 0-0.49. The
                  CTPA is what settles it.
    pmc-4775775   pneumonia in an 85-year-old, afebrile at 36.2C with a
                  white count of 5,000. Two of the findings this knowledge
                  base leans on hardest for pneumonia are absent.

Both turned out hard. That was not the criterion and is worth saying plainly.

### What they did to the numbers

    set          n    committed correct   wrong   mean rank   ECE     Brier
    eight        8                    3       0        2.50   0.317   0.107
    ten         10                    3       0        3.00   0.345   0.129

Every metric except the one that matters got worse, and the one that matters
did not move: **still zero wrong commits**. Both new cases escalate rather
than guessing, which is the behaviour the gate exists for.

The earlier mean rank of 2.50 was partly a property of the set. Adding two
genuinely hard presentations moved it to 3.00, and that is a better estimate
of the same quantity rather than a regression in the model — nothing about
the system changed between those two rows.

### The second case says something specific about an invented column

pmc-4775775 ranks its true diagnosis **seventh of eight**. The model barely
considers pneumonia in a patient who has a productive cough, hypoxia and a
basal infiltrate on the radiograph.

The likely reason is a finding recorded as explicitly *absent*: her white
count is 5,000, so `lab:raised_wcc` is False, and this knowledge base gives
pneumonia 0.80 for a raised white count. An absent finding that a disease
claims at 0.80 is a heavy penalty, and `lab:raised_wcc` is one of the columns
where **all eight cells are invented** — the column the sourcing survey
identified as having no diagnostic-accuracy literature behind it at all,
because the white count is used as a severity marker rather than as a test
for pneumonia.

So the entirely-invented column is not inert. It is load-bearing enough to
push a real pneumonia to seventh place, on a value nobody has ever measured.
That is a concrete instance of the general claim this project has been making
about its invented numbers since the ablation study, and it is the first time
a real patient has demonstrated it rather than a fixture.

It is stated as a hypothesis rather than acted on. Lowering pneumonia's white
count likelihood because one real case would benefit is precisely the tuning
this project refuses; the honest form is that the column is unsourceable, it
is doing real work, and one real patient now shows what that costs.


### Narrowing a claim: the BNP column is blocked, the BNP cell is not

An earlier entry recorded natriuretic peptide as unsourceable and moved on.
That was stated too broadly, and the correction is worth making precisely
because the original reasoning was right.

The objection was never that no figure exists. It was that assembling a
*column* means pairing a heart-failure value measured on BNP > 100 pg/mL with
a COPD value measured on an age-specific NT-proBNP threshold — different
analytes on different scales — so the column would state a comparison its
sources do not support. That still holds, and the other five cells stay
invented.

It does not apply to one cell from one study at one stated cutoff. Wang's
review gives BNP at 100 pg/mL a sensitivity of 93.5% and specificity of 52.9%,
and those cross-check against the negative likelihood ratio of 0.11 quoted in
the same review: (1 − 0.935) / 0.529 = 0.123 recovers it. The Breathing Not
Properly cohort gives 90% and 76% at the same cutoff independently, so the
band spans both.

`("acute_pulmonary_oedema", "lab:raised_bnp")` is now measured at 0.93.
Invented 97 to 96; coverage stays at 37%; behaviour unchanged on every arm.

The general lesson is about how a negative result gets recorded. "This source
is unusable" and "this column cannot be assembled" are different claims, and
writing the stronger one closed a door that was only partly shut. This is the
second time in two days that re-examining a recorded negative has produced
cells — the first being the likelihood-ratio inversion, dismissed because the
series reports ratios rather than frequencies.

### Real cases are a biased sampling frame, and the rejection rate shows it

The real-case set went from eight to ten. Getting there meant looking at five
candidates and rejecting three, which is worth recording because the reasons
are structural rather than bad luck:

    PMC10332670   inferior STEMI whose confirmed cause is coronary vasospasm
                  rather than atherosclerotic ACS, and only four findings
                  documented in this vocabulary
    PMC4495460    a clinicopathological case spanning years, whose final
                  pathology is decompensated hypertensive heart disease *and*
                  terminal bronchopneumonia -- two labels, not one
    PMC3841695    unilateral pulmonary oedema masquerading as pneumonia,
                  full text behind a verification page

Two of the three failed on the same thing: **a case report is published
because it is unusual**, and unusual frequently means the diagnosis is
atypical, dual, or outside the modelled eight. The sampling frame is selected
for exactly the property that makes a gold label hard to assign. That is a
limitation of using case reports as a real-patient substitute at all, and it
caps how far this set can grow without a different source — a registry, a
chart review, or the physician-curated cases the brief intended.

Three rejected against two accepted also says the accepted ones were not
cherry-picked for kindness: both are hard, and both were kept because their
labels are unambiguous, not because the model does well on them. It does not.


## A faithfully transcribed number that was still wrong, and what fixing it cost

Found by reading a demo transcript after the evidence-panel fix, not by a
failing test. The transcript showed, under the supporting evidence for
pneumonia:

    dyspnoea_at_rest absent; P(dyspnoea_at_rest |
    community_acquired_pneumonia) = 0.05; likelihood ratio 1.79

A pneumonia patient breathless at rest only 5% of the time is clinically
absurd in an emergency department, and 0.05 was the lowest value in that
column — below pericarditis at 0.30 and panic attack at 0.60.

The transcription was correct. The Merck chapter says dyspnoea in pneumonia
"usually is mild and exertional and is rarely present at rest", and the fixed
rubric maps "rare" to 0.05. **The error was the population.** That chapter
describes pneumonia at every severity, most of it managed at home. This
knowledge base's population is people who came to an emergency department
*because* they were breathless, and inside that population a pneumonia patient
is breathless at rest most of the time.

This is a different failure mode from the two the project already documents. A
mis-transcription (the pericarditis troponin, "almost always" read for
"often") is caught by re-reading the source. A population mismatch survives
re-reading the source, because the source says exactly what it was recorded as
saying.

The cost was not hypothetical. On pmc-4775775, a real 85-year-old with a
confirmed pneumonia, the single strongest argument the model made *against*
the correct diagnosis was that she was breathless at rest — and it ranked
pneumonia fifth of eight on a patient with a productive cough and a basal
infiltrate. A sourced number attracts less scrutiny than an invented one,
which is how a value thirteen times too small survived several audits with a
citation attached. That is the second time this week the same mechanism has
hidden a bad number, the first being DDXPlus's leg swelling at 0.999.

Replaced from a population-matched cohort: 954 acutely admitted patients, 265
with expert-panel-confirmed CAP, dyspnoea in 171 of them — 0.67. The rule was
fixed before looking: take whatever a population-matched source reports. One
definitional gap remains and is stated in the source comment — the study
records "dyspnoea", not "dyspnoea at rest" — and it is far smaller than the
thirteenfold gap it replaces.

### What it moved

    arm                     before                      after
    fixtures (top-1)        10/10, 7 commits, 0 wrong   10/10, 5 commits, 0 wrong
    hard cases              4/5,   2 commits, 1 WRONG   5/5,   3 commits, 0 wrong
    real cases (n=10)       3 correct, 0 wrong          3 correct, 0 wrong
    mean real rank          3.00                        3.10
    ECE / Brier             0.345 / 0.129               0.337 / 0.120
    overconfidence          +0.083                      +0.166
    pmc-4775775 rank        7th of 8                    3rd of 8

**The wrong commit is gone.** fx-h04, the pneumonia in a COPD patient
committed as a COPD exacerbation at 70%, is now answered correctly. It was the
only wrong commit anywhere in this project's test sets.

**And fx-009 stopped committing.** The pulmonary embolism that presents as a
pneumonia now escalates at 52% instead of committing at 65%. The loop still
ranks it first. That is the correct direction: fx-009's patient genuinely
looks like a pneumonia, and a knowledge base that says so is more accurate
than one that did not. The case moved from a confident right answer to an
uncertain right answer.

Net on the property this project actually claims: **one wrong commit removed,
zero introduced, on every set.** Correct commits fell, which is the trade
calibrated abstention is supposed to make.

The overconfidence figure moved the wrong way, +0.083 to +0.166, and that is
not explained by anything measured here. It is the same unexplained residual
recorded when the completed grid was rejected, and it belongs on the list of
things this project cannot account for rather than in a footnote.

### Five test claims moved, and one mechanism argument collapsed

- **The hard-case test now pins zero wrong commits** instead of one. The
  instrument found a defect and then priced its repair, which is what it was
  built for.
- **The evidence-fit claim reversed a fourth time**, and back to its original
  form. With no failing case left under the shipped configuration it is
  measured on a weakened arm again, where fx-009 fits at 0.554 inside a
  correct range of 0.124 to 0.601. No threshold separates them.
- **SCOPE.md's table lost a claim.** It said rest dyspnoea lowers pneumonia;
  the sourced value gives a likelihood ratio of 1.14, which discriminates in
  neither direction, so the claim is dropped rather than reversed.
- **fx-009 through the shipped loop escalates**, asserted precisely rather
  than deleted.
- **And the correlation argument inverted.** Fifth measurement, fifth answer:

        arm                      fixtures  mean cost
        plain                       9/10        15.6
        + correlation               9/10        20.7
        + decisive tests           10/10        20.1
        + both                     10/10        23.3

  Correlation weighting now buys nothing on this arm and costs a third more
  budget; the decisive-test rule is what repairs fx-009. This project spent
  more effort defending correlation weighting than any other mechanism, on
  the strength of measurements that have now reversed twice more since. The
  durable claim is the one the docstring has carried since the second
  reversal: a mechanism's value is a property of the numbers underneath it,
  and there is no answer to "does correlation help" independent of them.


## Auditing every narrative value for the population error, and what it settled

The pneumonia rest-dyspnoea defect was found by accident. Thirteen other
values come from the same source by the same rubric, so the question was
whether it had siblings. The check is cheap and repeatable: **does the cited
chapter section describe patients as they present acutely to an emergency
department, or the disease in general?**

    result of the audit
    -------------------
    11 correctly scoped     acute PE, acute pericarditis, panic disorder,
                            pneumonia diagnosis, acute pulmonary edema
     1 clear mismatch       acute_pulmonary_oedema / dyspnoea_at_rest
     2 milder, left alone   acute_pulmonary_oedema / exam:tachycardia
                            copd_exacerbation / exam:reduced_breath_sounds

One value I had suspected turned out fine on inspection: pericarditis'
dyspnoea comes from "acute pericarditis tends to cause chest pain and a
pericardial rub, sometimes with dyspnea" — right phrase, right population.

**The clear mismatch had its own correction sitting in the same file.**
`acute_pulmonary_oedema / dyspnoea_at_rest` was drawn from Ch. 211's general
"Symptoms and Signs" section, which describes heart failure as a chronic
condition: "the most common symptoms are dyspnea ... and fatigue", mapped to
"common" and 0.60. The disease modelled here is the acute decompensation, and
the same chapter's *acute pulmonary edema* section — already cited two entries
away for frothy sputum — lists "severe dyspnea" first among the defining
findings, reserving "sometimes" for the sputum in the same sentence. Re-scoped
to "characteristic", 0.85.

The two left alone are recorded rather than adjusted. Tachycardia in acute
pulmonary oedema is probably higher than the chronic figure, and reduced
breath sounds in a COPD exacerbation probably at least the stable-disease
rate, but neither chapter offers a correctly scoped sentence and inventing one
is what this file exists to prevent.

### What the re-scoping moved

    arm                  before                     after
    fixtures (top-1)     10/10, 5 commits, 0 wrong   9/10, 5 commits, 0 wrong
    hard cases            5/5,  3 commits, 0 wrong   5/5,  3 commits, 0 wrong
    real cases (n=10)     3 correct, 0 wrong          3 correct, 0 wrong
    ECE / Brier          0.337 / 0.120               0.329 / 0.114
    overconfidence       +0.166                      +0.158

Calibration improved on every measure. The cost is fx-010, an acute coronary
syndrome that now sits six points behind an acute pulmonary oedema — 0.43
against 0.49 — and escalates rather than committing. A near-tie the gate
declines, on the set already documented as spent.

### The six reversals were a property of the instrument, not the mechanism

This is the part worth carrying forward. The fixture arms after the
re-scoping:

    arm                      fixtures  mean cost
    plain                       9/10        15.9
    + correlation               8/10        19.8
    + decisive tests           10/10        20.6
    + both                       9/10       21.8

Correlation weighting now *costs* a fixture case. Across this project it has
been measured as worth one case, worth nothing, worth two, worth nothing, and
now worth minus one — six answers, every one of them taken on the same ten
cases. Those ten are saturated: the shipped configuration ranks nine or ten of
them whatever the mechanisms are set to, so the count was a coin toss dressed
as a measurement, and DESIGN.md has been reasoning from it for weeks.

Asked of the real patients instead, the question has never been close:

    correlated=False   real: 3 correct, 2 WRONG commits, Brier 0.200, oc +0.218
    correlated=True    real: 3 correct, 0 wrong commits,  Brier 0.114, oc +0.158

**Two wrong commits on real patients against none.** Calibrated abstention is
this project's claim, and correlation weighting is what protects it on the
only instrument able to see the difference. It stays on by default, and this
is the first time that default has had evidence behind it rather than a
fixture count.

The durable lesson is not about correlation. A measurement that reverses six
times is telling you the instrument is spent, and it took six reversals and a
purpose-built replacement set before anyone read it that way.


## Three open items, worked through: one was a measurement error, one is real, one stays open

### The overconfidence was the metric, not the model

+0.158 on the real cases had been carried in this document for weeks as an
unexplained residual, and reported twice as a cost of sourcing work. It was
never decomposed. Doing that takes one pass:

    subset             mean confidence   top-1 accuracy   overconfidence
    committed (n=3)              0.715            1.000           -0.285
    escalated (n=7)              0.348            0.000           +0.348
    all (n=10)                   0.458            0.300           +0.158

Every point of the headline figure comes from cases the gate **declined to
answer**. On those the system says, in effect, "I am 35% sure and I am not
committing" — and the metric scores that as overconfidence because the
top-ranked hypothesis happened to be wrong. It penalises precisely the
behaviour the abstention gate exists to produce.

On the cases it actually answers the system is **under**confident by 0.285,
which is the safe direction and the opposite of what the headline implied.

`CalibrationMetrics` now carries `committed_n`, `committed_mean_confidence`
and `committed_accuracy`, with a `committed_overconfidence` property. The
full-coverage figure is kept rather than replaced, because it is a real
quantity and suppressing it would be the more convenient kind of dishonesty,
but the docstrings now say which to read first and why.

This one is worth a note on process. The number was reported accurately every
time and interpreted wrongly every time, because nobody asked what it was
averaging over. A statistic that is correct and misleading is harder to catch
than one that is wrong.

### The D-dimer rivals really are understated, confirmed independently

The entry above records the pooled defect: specificity of 51% means
P(raised D-dimer | not pulmonary embolism) is 0.49 across the rivals, while
this knowledge base gave them values averaging 0.22. It was left unfixed
because a pooled figure cannot be apportioned across seven diseases.

A per-disease figure has no such problem. In 148 patients admitted with an
acute COPD exacerbation and investigated for pulmonary embolism, 92 had it
excluded, and 53 of those still had a D-dimer above the conventional 0.5
threshold — **0.58 against an invented 0.25**. Coverage 37% to 38%, ECE 0.329
to 0.324, Brier 0.114 to 0.110, every arm otherwise unchanged.

That is a second, independent confirmation of the twofold understatement,
from a different study design than the pooled specificity. The remaining five
rivals stay invented and stay wrong in the same direction; each needs its own
population-matched study.

One candidate was rejected on the way. A different AECOPD study reports 41 of
93 non-PE patients above its threshold, but that threshold is 990 µg/L,
optimised inside the study — not commensurable with the 190–500 range behind
the pulmonary embolism cell, and mixing them would rebuild the exact problem
that blocked the natriuretic-peptide column.

And the source needs reading carefully rather than quoting. It says
"fifty-three patients (36%) who did not have PE had higher than normal
D-dimer levels", where 36% is 53 of the full 148 rather than of the 92 without
embolism. The quantity this cell needs is the latter, 57.6%. The paper also
prints its units as pg/mL where the values are plainly mg/L.

### The white count stays unsourced, and the reason is now specific

`lab:raised_wcc` is the last column with all eight cells invented. The earlier
claim was that no diagnostic-accuracy literature treats the white count as a
test for these diagnoses. That still holds, and it is now backed by a
candidate examined and rejected rather than by assumption.

The population-matched pneumonia cohort used for rest dyspnoea does report
white counts, in two forms, and neither is this concept:

    leucocytes <3.5 or >8.8 x10^9/L    CAP 214/265 (80.8%)   non-CAP 456/689 (66.2%)
    neutrophils >7.5 x10^9/L           CAP 187/265 (71.1%)   non-CAP 362/689 (53.2%)

The first is bidirectional — it counts leukopenia, a different finding and in
sepsis a severity marker pointing the other way. The second is neutrophils
rather than a total white count. Writing either into `lab:raised_wcc` is the
same construct swap this file refused when it declined to map "any abnormal
ECG" onto `exam:ecg_st_changes`, and refusing it in one place and allowing it
in another would make the rule worthless.

A further search for a unidirectional leukocytosis rate in a matched
population returned prognostic studies and severity scores, which is the
literature behaving exactly as the original claim predicted: the white count
is measured to grade illness, not to identify it.

So this column is blocked for a reason about medicine rather than about
effort, and the honest position is that eight numbers no one has measured are
doing real work — including, as pmc-4775775 showed, burying a real pneumonia.


### Two more D-dimer rivals attempted, measured, and not taken

The COPD rival above worked because its source reports a counted proportion
at the conventional threshold in a matched population. The remaining five have
no such source, and two attempts at deriving them from reported quartiles were
made and rejected.

Two studies give D-dimer by final diagnosis as medians with interquartile
ranges. Where the 0.5 cutoff falls among the quartiles is distribution-free,
and by that alone the invented values are wrong:

    acute coronary syndrome, chest-symptom ED (n=1039)
        median 0.40, IQR 0.27-0.80   ->  0.25 to 0.50 above the cutoff
    acute coronary syndrome, all-ED D-dimer ordered (n=552)
        median 0.57, IQR 0.32-1.22   ->  0.50 to 0.75 above
    heart failure, chest-symptom ED (n=451)
        median 1.70, IQR 0.90-3.10   ->  at least 0.75 above
    respiratory infection, all-ED (n=1767)
        median 0.76, IQR 0.40-1.47   ->  0.50 to 0.75 above

The cells hold 0.20, 0.30 and 0.35. Every one is outside every bound. This is
now the third independent line of evidence that the column understates the
rivals, after the pooled specificity and the counted COPD proportion.

They were written, measured, and reverted. Three reasons, none of which is the
result:

**The two acute coronary syndrome sources do not overlap.** 0.25-0.50 against
0.50-0.75, on a quantity that has one true value. That is not a wide band, it
is two studies disagreeing — most plausibly because the second population is
patients in whom a clinician ordered a D-dimer, which selects for suspected
thrombosis. A number cannot be called measured while its sources contradict
each other, and picking the one that suits the model is the failure this file
exists to prevent.

**A point estimate needs a distributional model.** Every other value here is a
counted proportion or exact algebra on reported statistics — the
likelihood-ratio inversion is two equations in two unknowns and returns the
sensitivity the study would have printed. Fitting a log-normal to three
quantiles and integrating its tail is a different kind of act, and the
provenance tiers have no label that distinguishes it from a counted
frequency. Introducing that silently would degrade what MEASURED means.

**A half-sourced column can be worse than a uniformly invented one.** Writing
these would have left pulmonary embolism at 0.93, heart failure at 0.91 and
acute coronary syndrome near 0.4, beside pneumonia, pericarditis and panic
attack still at 0.35, 0.15 and 0.10 — values the same evidence says are too
low. The contrast between the sourced and unsourced halves would be an
artefact of which cells happened to have literature, not a fact about
patients. It is a general hazard of incremental sourcing that this project
has not had to face before, because previous passes either filled whole rows
(PIOPED, Wang) or single cells whose neighbours were already sourced.

The behaviour, measured before reverting and recorded as corroboration rather
than as the reason:

    arm            with the two cells      without
    fixtures        10/10, 7 commits        9/10, 5 commits
    hard cases      4/5,  2 commits         5/5,  3 commits
    real cases      2 correct, 0 wrong      3 correct, 0 wrong
    ECE / Brier     0.358 / 0.135           0.324 / 0.110

The spent fixture set improved and both instruments this project trusts got
worse, which is the pattern that should be expected when a change adds
artefactual contrast rather than information. The real patient lost was
pmc-4565285, a pulmonary embolism with a positive CTPA, which fell behind
acute pulmonary oedema once D-dimer stopped separating the two — an entirely
correct consequence of the numbers written, and a demonstration that they were
the wrong numbers to write on their own.

What stands: the rivals are understated, on three independent lines of
evidence, and five of the seven cells in this column remain invented and
remain wrong in the same direction. Fixing them needs sources that report
counted proportions at the conventional threshold in this presentation, of
which exactly one has been found.


## Two invented numbers wrong in opposite directions, and why fixing either alone made things worse

The entry above reverted a D-dimer correction because it cost a real patient
its answer. That revert was right and the explanation given for it was
incomplete. Pulling the thread found the actual defect, one column over.

pmc-4565285 is a pulmonary embolism with a pro-BNP of 11,479 and a positive
CTPA. Raising acute pulmonary oedema's D-dimer from an indefensible 0.30
flipped it to oedema, and the reason was not the D-dimer at all:

    P(raised BNP | acute pulmonary oedema)   0.93   measured (Wang 2005)
    P(raised BNP | pulmonary embolism)       0.20   invented

A raised natriuretic peptide argued nearly five to one for the wrong
diagnosis. The understated D-dimer values had been quietly cancelling that,
and correcting one of them exposed the other.

**0.20 asserts the opposite of the mechanism.** Acute embolism obstructs the
pulmonary circulation and strains the right ventricle; a strained ventricle
releases natriuretic peptide. In 63 consecutive emergency patients with acute
pulmonary embolism — with the haemodynamically unstable *excluded*, so if
anything this understates — 39 had NT-proBNP at or above 350 ng/l. **0.62.**

### The threshold assumption, stated rather than buried

The oedema cell is BNP > 100 pg/mL; this one is NT-proBNP >= 350 ng/l.
Different analytes. They are taken as answering the same question — did the
natriuretic peptide come back raised — because each is the standard clinical
decision threshold for its own assay in acute dyspnoea. That is weaker than a
same-assay comparison and a reader should know it.

It is narrower than the case this project refused earlier, and the distinction
is the point rather than an excuse. The rejected COPD figure rested on an
"age-specific NT-proBNP" threshold that varies per patient and was never
stated numerically; there was no scale to compare against at all. Here both
thresholds are single fixed published numbers.

That is the **second** time the blanket claim "this column cannot be
assembled" has had to be narrowed. The first narrowing produced the oedema
cell, this one produced the embolism cell, and both times the original claim
had been reached in one pass and written as though settled. A negative result
recorded too confidently costs more than one recorded tentatively, because
nobody re-opens it.

### What it moved

    arm                  before                      after
    fixtures (top-1)      9/10, 5 commits, 0 wrong    8/10, 5 commits, 0 wrong
    hard cases            5/5,  3 commits, 0 wrong    5/5,  3 commits, 0 wrong
    real cases (n=10)     3 correct, 0 wrong          3 correct, 0 wrong
    ECE / Brier          0.324 / 0.110               0.317 / 0.107
    committed overconf   -0.300                      -0.275
    coverage                38%                         39%

Both trusted instruments improved or held; the saturated fixture set lost a
case. That is the pattern this project now expects and reads the same way
every time.

**The case it lost is fx-009**, which is worth stating plainly because this
project has spent more effort on fx-009 than on any other case. Pulmonary
embolism there has fallen to third. The cause is a correct number: fx-009's
BNP is *normal*, and with 62 of every 100 acute embolisms raising it, a normal
result argues against embolism. The case was hand-written by this author with
that finding absent, which is a perfectly realistic embolism — 38% of them —
and now a harder one. The loop still declines to commit rather than committing
to something wrong.

### The general finding

Two invented numbers, wrong in opposite directions, can look like a working
model. The D-dimer column understated every rival; the BNP column understated
pulmonary embolism. Each error partly hid the other, and the first correction
attempted looked like a regression because it removed one side of a
cancellation.

That has a practical consequence for how the remaining invented numbers should
be approached. A cell-by-cell sourcing pass measured against behaviour will
reject correct changes whenever the cell it corrects is paired with an
uncorrected error elsewhere. The only defence is what happened here: when a
correction makes results worse, find the mechanism before reverting. The
revert in the previous entry stands, but it was accepted one step too early,
and the entry recorded a structural explanation — half-sourced columns — that
was true in general and not the operative cause in that instance.


### The PIOPED pass read half its own table

A mechanism audit of the remaining invented cells -- looking for values that
assert the opposite of known physiology, the way P(raised BNP | pulmonary
embolism) at 0.20 did -- turned up two that needed no new literature at all.

The PIOPED II pass took the *no-PE* arm of Tables 3 and 6 and applied it to
the seven rival diseases. That was the point of it: a study of patients
investigated for embolism and found not to have one describes exactly this
differential's other candidates. Both tables also report the PE arm. Those
columns were read, quoted in this file's own citation snippet, and never
applied to pulmonary embolism itself.

    finding             was                   PIOPED II, PE arm
    recent_immobility   0.61  DDXPlus         48 of 192   (0.25)
    calf_tenderness     0.40  invented        90 of 192   (0.47)

The immobility correction follows a precedent set twice already: a frequency
counted in real patients outranks one counted in a simulator. DDXPlus
generates immobility in 61% of its embolism patients against 25% observed in
192 real ones -- the same gap as pleuritic pain at 71% against 33%, resolved
the same way.

Coverage holds at 39% and invented falls 94 to 93. Both trusted instruments
are unchanged: real cases 3 correct with 0 wrong, hard cases 5 of 5 with 3
commits and 0 wrong. Calibration moved slightly the wrong way, ECE 0.317 to
0.325 and Brier 0.107 to 0.111.

It did move fx-009's evidence set, and in the direction the original story
predicts. Correlation now lifts pulmonary embolism from third to *first* on
that frozen set, where before it went fourth to second. Absent
venous-thromboembolism findings penalise embolism less once its own rates are
real -- the exact mirror of the first PIOPED pass, which stopped the rivals
being penalised for those same absences and left the disease the study is
about untouched.

**How it survived is the transferable part.** That pass was careful about
which arm it was reading, wrote a long comment justifying the pooling across
rivals, and never asked whether the table it had open said anything about the
disease in the middle of the differential. A sourcing pass aimed at one part
of a table can leave the rest of it unread, and the resulting gap is invisible
afterwards because the citation is present and correct.


### The troponin ordering violation, and the fixture arms retired as evidence

The same mechanism audit turned up a second ordering error. The grid had
P(raised troponin | pulmonary embolism) at an invented 0.25, sitting *below*
the sourced 0.32 for COPD exacerbation — asserting that a COPD exacerbation
raises troponin more often than a pulmonary embolism does. The mechanism says
the opposite, and it is the same one that made the natriuretic-peptide value
wrong: acute embolism strains the right ventricle, and a strained ventricle
releases both peptides and troponin.

In 220 consecutive patients admitted with acute pulmonary embolism, unselected
by haemodynamic status, 116 had a positive troponin. **0.53.** The band spans
that and a second cohort at 90 of 233 (0.39). The source used two assays and
admits that "the timing and indication, and the choice between 2 assays were
not explained", which is quoted rather than smoothed over — the same
assay-generation caveat the COPD troponin entry already carries.

Coverage 39% to 40%, invented 93 to 92. Hard cases unchanged at 5 of 5 with 3
commits and 0 wrong. Real cases held at 0 wrong commits and lost one
commitment: pmc-4565285, a pulmonary embolism whose troponin is normal, falls
from committing at 0.78 to escalating at 0.62 with the right diagnosis still
ranked first. Three points under the threshold, on a patient who really does
have a normal troponin — which 47 of every 100 pulmonary embolisms do. The
gate declining there is the gate working.

### The fixture arms are now retired as evidence

The correlation test no longer asserts any of them:

    plain            9/10   cost 13.7
    + correlation    8/10   cost 17.9
    + decisive       9/10   cost 18.3
    + both           8/10   cost 21.5

That measurement has returned a different answer eight times — worth one case,
nothing, two, nothing, minus one, and now minus one with the decisive rule no
longer recovering it. Every one was taken on the same ten cases, which the
shipped configuration ranks eight to ten of whatever the mechanisms are set
to. Pinning any of those counts pins noise, and pinning them is what kept this
question open for weeks. They stay in the test as a comment.

What the test asserts instead is the measurement that has held or widened
through every sourcing pass:

    correlated=False   real: 2 correct, 3 WRONG commits, Brier 0.234
    correlated=True    real: 2 correct, 0 wrong commits,  Brier 0.116

One of those two measurements is an instrument and the other is a coin, and it
took this project far too long to say so.

### Where the mechanism audit stands

Three ordering or mechanism violations found by reading the grid against
physiology rather than by any failing test, all three in the same disease:

    P(raised BNP | PE)          0.20 invented  ->  0.62 measured
    P(recent immobility | PE)   0.61 simulated ->  0.25 measured
    P(raised troponin | PE)     0.25 invented  ->  0.53 measured

Pulmonary embolism was the worst-described disease in a knowledge base built
around not missing it. Every one of the three understated a consequence of
right ventricular strain or overstated a risk factor, and each was invisible
to the test suite because tests check outcomes and these were inputs.

One suspect remains and could not be sourced. P(hypoxia | acute pulmonary
oedema) is invented at 0.40, below COPD at 0.55 and pulmonary embolism at
0.60, when alveolar flooding impairing gas exchange is the mechanism that
defines the disease. Every cohort found for it enrols patients *by* an oxygen
saturation threshold — SpO2 below 90% is the standard entry criterion for
acute cardiogenic pulmonary oedema trials — so the literature measures the
severity of selected patients rather than the prevalence of hypoxaemia among
unselected ones. It is recorded here as a suspected violation that the
available literature is structurally unable to settle.


### A whole invented column, from the accuracy half of a paper already cited here

`imaging:ctpa_filling_defect` was one of three columns with no sourced cell at
all, and it turned out the source was already open. PIOPED II published twice:
the clinical-characteristics paper this file has been quoting for symptom and
sign tables, and a companion paper giving the test's own accuracy against a
composite reference standard. Sensitivity 83%, specificity 96%, in the same
824 patients — 192 with embolism and 632 without, the exact denominators
already quoted here, which is a useful check that it is one cohort.

Sensitivity *is* P(filling defect | pulmonary embolism); one minus specificity
*is* P(filling defect | no pulmonary embolism). No inversion, no distributional
model. The two quantities the column needs are what an accuracy study reports.

    P(defect | PE)        0.95 invented   ->  0.83
    P(defect | not PE)    0.01-0.03       ->  0.04   (four rivals)

**The first is a safety correction rather than bookkeeping.** At 0.95 this
model treated a negative CTPA as near-conclusive against pulmonary embolism.
Measured sensitivity is 0.83, so roughly one embolism in six is missed by the
scan and a negative result should leave far more residual probability than the
knowledge base allowed. A number that made the system too willing to rule out
a red-flag diagnosis is the worst direction an error can take here, and it sat
in a fully invented column from the beginning.

Pooling the four rivals at one value is legitimate for the reason it was
legitimate for immobility and calf tenderness and is not for crackles: a
false-positive filling defect is a property of the scan, not something
pneumonia produces more often than a panic attack does.

Coverage 40% to 43%, invented 92 to 87 — the largest single move since the
PIOPED clinical tables. ECE improves sharply, 0.334 to 0.246; Brier worsens,
0.116 to 0.151. Every arm is otherwise unchanged: fixtures 8 of 10 with 5
commits and 0 wrong, hard cases 5 of 5 with 3 commits and 0 wrong, real cases
2 correct with 0 wrong.

It also moved fx-009's evidence set the other way for once. Correlation now
lifts pulmonary embolism from fourth to third where it had reached first,
because the single strongest piece of evidence for that diagnosis is weaker
than the knowledge base used to claim. The mechanism is unchanged and still
worth more than fivefold; there is simply less to lift with. That is the
correct consequence of a more honest number about a scan.

### The pattern across three passes

Two of the three columns with no sourced cell are now closed or partly closed,
and both closed from sources that were already in the file or already
searched. The white count stays open for a reason about the literature rather
than about effort, and hypoxia stays open because every cohort selects
patients by an oxygen threshold.

The transferable lesson is the same one the PIOPED tables taught two passes
ago: **a paper cited for one thing may answer a second question nobody
thought to ask it.** That pass read half a table; this one had been reading
half a study. In both cases the citation was present and correct, which is
precisely what makes the gap invisible on review.


### A withdrawn mapping, and a "finding" that was a bug

The ordering audit flagged one more column, and this time the fix lowered
coverage rather than raising it.

`exertional_chest_pain` read acute pulmonary oedema at 0.77 and acute coronary
syndrome at 0.36 — both sourced from DDXPlus. Exertional chest pain is the
cardinal ischaemic symptom, so an ordering that puts heart failure above acute
coronary syndrome is not a subtle discrepancy.

The DDXPlus question behind it is **E_218: "Do you have symptoms that are
increased with physical exertion but alleviated with rest?"** *Symptoms*, not
chest pain. Heart-failure patients answer yes because their breathlessness
does exactly that. `ddxplus_mapping.py` filed the pairing under "confident:
the two phrasings ask the same question", with the note "worse on exertion,
relieved by rest" — which drops the word that matters.

That file's own docstring, written before the mapping was made, states the
failure precisely:

> Get that wrong and the result is a *confidently sourced wrong number*, which
> is worse than the invented one it replaced — an invented number is labelled
> invented and nobody trusts it, while a mis-mapped one arrives with 1.3M
> patients behind it.

**And the project wrote the consequence up as a discovery.** SCOPE.md recorded
that "exertional chest pain no longer raises acute coronary syndrome" as one
of four claims the sourced numbers had contradicted, attributing it to
DDXPlus's generating model. It was not a fact about the generating model. It
was a fact about a mapping, and it sat in the scope document as evidence of
the project's own rigour for weeks.

The invented values it displaced were 0.80 for acute coronary syndrome and
0.20 for acute pulmonary oedema — the clinically sensible ordering, which the
mapping inverted.

    coverage        43%  ->  42%     (two cells returned to invented)
    ECE           0.246  ->  0.240
    Brier         0.151  ->  0.147

Every arm otherwise unchanged: fixtures 8 of 10 with 5 commits and 0 wrong,
hard cases 5 of 5 with 3 commits and 0 wrong, real cases 2 correct with 0
wrong. Removing a sourced number improved calibration slightly, which is the
corroboration rather than the reason.

### What the ordering audit is actually good for

Three passes of reading the grid against physiology have now found five
errors, none of them by a failing test:

    P(raised BNP | PE)              0.20 invented   -> 0.62 measured
    P(recent immobility | PE)       0.61 simulated  -> 0.25 measured
    P(raised troponin | PE)         0.25 invented   -> 0.53 measured
    P(defect | PE)                  0.95 invented   -> 0.83 measured
    P(exertional pain | oedema/ACS) 0.77 / 0.36     -> withdrawn

The technique is cheap and needs no literature: sort each column, ask whether
the ordering matches what the diseases do, and look hardest where a sourced
value sits below an invented one or a rival outranks the disease that owns the
finding. It has a much better hit rate than searching for new sources, and
every hit so far has been invisible to a test suite that checks outputs.

Two candidates it raised and cleared on inspection are worth recording so
nobody re-opens them. `leg_swelling` puts pulmonary embolism at 0.17 below
COPD at 0.20, which looks wrong until the sources are read: Miniati counted
*unilateral* limb swelling in 63 of 360 embolisms, while COPD's value is
bilateral oedema from cor pulmonale. Different findings sharing one concept —
a limitation of the vocabulary rather than an error in the numbers. And
`raised_troponin` puts acute pulmonary oedema at 0.30, which may be low given
how often acute heart failure raises troponin, but nothing available settles
it.

One suspect stands unresolved: P(hypoxia | acute pulmonary oedema) at 0.40,
below COPD and pulmonary embolism, when alveolar flooding impairing gas
exchange is the mechanism that defines the disease. Every cohort found for it
enrols patients *by* an oxygen-saturation threshold, so the literature
measures severity among selected patients rather than prevalence among
unselected ones.


## The ordering audit, made executable

Everything in this project's test suite checks outputs. The knowledge base is
inputs, and for most of its life nothing looked at it at all: 89 of 153
likelihoods are invented, and the only scrutiny they had received was whichever
ones happened to change a case.

Reading each column sorted, against what the diseases actually do, found five
errors in three passes:

    P(raised BNP | PE)            0.20, below every rival that mattered
    P(recent immobility | PE)     0.61, a simulator value 2.4x the truth
    P(raised troponin | PE)       0.25, below the sourced 0.32 for COPD
    P(defect | PE)                0.95, overstating a scan that misses 1 in 6
    P(exertional pain | oedema)   0.77, above ACS, from a mis-mapped question

Not one moved a fixture case enough to fail anything. Four were in pulmonary
embolism, in a knowledge base built around not missing it. A sixth and seventh
came from the same technique applied to the DDXPlus mappings rather than the
grid: two questions asking about "symptoms" pointed at concepts naming a
specific symptom.

That reading is now `ORDERING_CLAIMS` in `scripts/plausibility_check.py`:
twenty claims over sixty-five rival comparisons, each of the form *this
disease must score strictly above these rivals for this finding, because ...*
The reasons are deliberately textbook-level — a clinician can accept or reject
each in one sentence, which is the closest this project gets to the review it
cannot have.

**They constrain ordering, not magnitude.** A non-clinician can honestly
assert that consolidation is commoner in pneumonia than in panic attack. He
cannot honestly assert that it is 0.95 rather than 0.85, and the existing
`CLAIMS` check already covers direction against a marginal. Ordering is the
band of judgement in between, and it is where every one of the five errors
lived.

### Stating the claim you believe, and letting it fail

Two comparisons fail, both the same cell: acute pulmonary oedema's hypoxia at
an invented 0.40, below COPD at 0.55 and pulmonary embolism at 0.60, when
alveolar flooding impairing gas exchange is the mechanism that defines the
disease.

The first draft of the claim list quietly asserted only that oedema beats
*panic attack* on hypoxia — which is true, passes, and hides the defect inside
the check built to find it. The claim is now stated in the form believed
correct, fails, and the failure is listed in `KNOWN_ORDERING_VIOLATIONS` with
the reason it is tolerated: unsourceable rather than unexamined, because every
cohort found enrols patients *by* an oxygen-saturation threshold and so
measures severity among selected patients rather than prevalence among
unselected ones.

A check that only contains the assertions which pass measures nothing. This
one is allowed to be red in a specified place.

### What the guard actually guarantees

`test_the_grid_still_orders_each_finding_the_way_the_diseases_do` asserts the
set of failures equals the known list exactly, in both directions. A new
violation stops the build; a repaired one also stops the build, so the
exception list cannot quietly outlive the defect it documents. Verified by
planting a violation — panic attack's white count raised above pneumonia's —
and confirming the failure message names the cell.

What it does not do is validate the numbers. Sixty-five comparisons over a
153-cell grid leaves most of it unconstrained, and a value can satisfy every
ordering claim while being twice what it should be. This catches the class of
error that has actually occurred here five times; it does not catch the class
nobody has found yet.


## Forty-seven invented numbers the audit was never counting

dxagent.provenance tracked 153 likelihoods and 8 disease priors, and this
project quoted coverage over those as its headline honesty metric. It was not
counting the rest of the model.

     5  correlation weights     datasets/fixtures.py
    27  acquisition costs       datasets/fixtures.py
     5  gate thresholds         gate.py
     3  loop budget limits      agent.py
     7  selector constants      actions.py
    --
    47  all invented, none tracked

None of them is a likelihood, so none was tracked, and their absence from
every reported figure was not a decision anyone made.

SCOPE.md already states why this matters, about the disease priors, which had
exactly this problem until they were pulled into the report:

> They were invented and untracked for most of this project life, which was
> the more dangerous state -- an invented number the audit cannot name reads
> as an absence of a problem.

The hole was closed for the priors and nobody asked whether it existed
elsewhere. It did, five times over.

**The correlation weights are the case that should have been obvious.** Five
judgement calls between 0.70 and 0.85 that are the difference between zero and
three wrong commits on the ten real patients -- the single most load-bearing
mechanism measured anywhere in this project, and the only defence of the claim
that the system does not commit wrongly. Nothing recorded that its parameters
were invented, and a reader of the coverage figure would have had no way to
find out.

They are reported separately rather than folded into the likelihood
percentage, for the same reason the priors are. A correlation weight and a
P(finding | disease) answer different questions, and averaging them produces
a number that is easier to quote and means less.

The counts are read from the classes themselves rather than written down, so
adding a threshold or a cost moves the total and fails the test. A new
invented parameter cannot enter the model silently, which is the property
that was missing rather than the number 47.

### What this does not fix

Counting a number is not sourcing it, and none of the 47 has a citation. Two
of the groups are arguably not the same kind of object as a likelihood at all
-- a gate threshold is a policy choice about how much risk to accept, and
there is no study that reports the correct value of min_confidence. The
honest position is that they are design decisions rather than measurements,
which is a defensible thing to be, but it has to be said out loud rather than
achieved by leaving them out of the count.

The correlation weights are not in that category. They are claims about how
much two findings overlap in real patients, which is a measurable quantity
that nobody here has measured.


## Measuring the correlation weights, and finding they barely matter

The previous entry counted five correlation weights among the 47 invented
parameters and called them the worst case: judgement calls between 0.70 and
0.85 that are the difference between zero and three wrong commits on the real
patients, with nothing recording that they were invented. Unlike a gate
threshold, `rho` is a genuinely measurable quantity — `redundancy_weights`
uses it as a correlation coefficient between two findings — so it was
measured.

DDXPlus carries 200,091 patients across the eight modelled diseases. Three
within-group pairs have both findings mapped to evidence codes, so the phi
coefficient can be computed directly:

    pair                                  invented    measured
    leg_swelling / recent_immobility          0.85       0.495
    fever / productive_cough                  0.75       0.551
    wheeze_subjective / smoking_history       0.70       0.079

Every one is overstated, the last by a factor of nine. And the pairs the model
treats as independent are not:

    fever / palpitations                      0.00      -0.134
    pleuritic_pain / smoking_history          0.00      -0.162
    productive_cough / leg_swelling           0.00      -0.242
    wheeze_subjective / recent_immobility     0.00      -0.135

`redundancy_weights` takes the absolute value, so a correlation of -0.24 would
damp exactly as much as +0.24. The structure is overstated where it is
declared and set to zero where it is not, and one undeclared pair is three
times more correlated than a declared one.

### And it makes no difference

Substituting the three measured values for the invented ones:

    arm                    invented weights        measured weights
    real cases             2 correct, 0 wrong      2 correct, 0 wrong
    hard cases             5/5, 0 wrong            5/5, 0 wrong
    fixtures               8/10                    8/10
    ECE / Brier            0.240 / 0.147           0.237 / 0.146

Nothing moves. That is worth more than the correction would have been,
because it changes what the earlier warning was about. Correlation weighting
*as a mechanism* is the most load-bearing thing measured in this project —
turning it off produces three wrong commits on real patients where there were
none. The *values* of its weights, across the range from 0.079 to 0.85, are
close to irrelevant.

So the five invented weights are not the hazard the previous entry implied.
The hazard is the binary decision to damp correlated findings at all, and that
decision is supported by the strongest measurement here.

### Not adopted, and the reason is the measurement itself

The three measured values are not written into the knowledge base.

Adopting them would buy nothing except a better provenance count — the
behaviour is identical, so the only thing that would change is that five
invented parameters become three. Changing a number to improve how the audit
reads, when the number demonstrably does not affect any output, is the
metric-gaming this project refuses everywhere else.

Two further reasons that would matter if the first did not. DDXPlus is a
simulator, so these correlations describe its generating rules rather than
patients, and a correlation between findings is exactly the kind of structure
a rule-based generator imposes rather than observes. And each group weight
covers every pair in the group — group one has four members and therefore six
pairs, of which one is measured — so writing the measured value in would
assert five unmeasured pairs share it.

What is recorded instead: the declared weights are roughly twice the simulated
correlations, one is nine times too high, four undeclared pairs are as
correlated as the declared ones, and none of it changes an output. The next
person to touch this should know the numbers are wrong and that it does not
currently matter, rather than discovering the first half alone.


## Two claims the reports were making that the machinery was not

### The temperature was never fitted, and the header said it was

`TemperatureScaler` refuses to fit below thirty labelled cases and explains
why in its own comment: an early run fitted T=0.5 from two samples,
*sharpening* an already-overconfident posterior. The refusal is correct and
the machinery is honest.

The report was not. It printed

    temperature (fitted)          1.00

unconditionally, which reads as *a fit was performed and found the posterior
already calibrated*. On the fixture split the calibration fraction is three
cases against a floor of thirty, so no fit is ever attempted and 1.00 is the
untouched default — the opposite of what the line implied. It now reads

    temperature                   1.00  (NOT fitted: 3 calibration cases,
                                         needs 30 -- this is the default,
                                         not a finding)

Same shape as the unsourced disclosure that used to appear under "supporting
evidence": correct behaviour, wrong label, and a reader with no way to tell.
That is three of these now, all found by reading output rather than by a test.

**The underlying problem is not fixable here.** Thirty labelled cases do not
exist. Ten hand-extracted case reports is a third of the floor, the fixtures
are invented, and the synthetic corpus is generated from the knowledge base
being calibrated. Fitting on DDXPlus was considered and rejected: only eight
of twenty-seven concepts map to it, so its cases carry a third of the evidence
a real one does, and a temperature fitted on that distribution would not
transfer. What was fixable is the claim.

### Three columns were unsourceable partly because they were undefined

`exam:hypoxia` has been the standing red mark in the ordering audit: acute
pulmonary oedema at an invented 0.40, below COPD and pulmonary embolism, when
alveolar flooding impairing gas exchange is what defines the disease. It was
recorded as unsourceable because every cohort enrols patients *by* an oxygen
threshold.

That was true and incomplete. The concept also never said what it meant. Its
vocabulary entry is the HPO term "Hypoxemia" and nothing else, while every
study that could source it reports a threshold. The only numeric threshold
anywhere in this project is in `guidelines.py`, where the PERC rule is encoded
as `SaO2 < 95%` — a definition the knowledge base was using without stating.

The same gap explains two other blocked columns, and reading them back the
symptom was visible each time:

    lab:raised_wcc    a cohort figure was rejected for being bidirectional,
                      "leucocytes <3.5 or >8.8", which counts leukopenia --
                      but the concept never said it meant the raised side.
    lab:raised_bnp    could not be assembled across BNP > 100 pg/mL and an
                      age-specific NT-proBNP threshold, because the concept
                      fixed no scale to compare them on.

`OPERATIONAL_DEFINITIONS` in `vocabulary.py` now states the intended threshold
for all seven threshold-dependent concepts. They are conventions rather than
measurements — a clinician would recognise each as the ordinary reporting
threshold for its test — and writing them down does three things the silence
did not: gives a future sourcing pass a target to match, makes an existing
cell auditable for whether its study used a comparable cutoff, and turns
"unsourceable" into a claim about the literature rather than one that hides an
undefined concept.

Three sourced cells already deviate, and `deviations_from_operational_defini-
tions` lists them rather than smoothing them over: pulmonary embolism's
troponin came from a cohort using two assays at once, COPD's from a
conventional troponin I at 0.017 microg/L, and pulmonary embolism's
natriuretic peptide from NT-proBNP at 350 rather than 300. The citation
snippets carry each study's cutoff precisely so that comparison is possible.

**Defining the concept was necessary and not sufficient.** The hypoxia cell is
still invented and still fails the ordering claim: a target threshold does not
conjure a cohort that reports prevalence rather than enrolling on it. What
changed is that the blockage is now one thing — no unselected cohort reports
saturation by final diagnosis — instead of two, one of which was ours.

A test pins it: any threshold-dependent concept without a definition fails,
so a new one cannot enter the vocabulary the way these three did.


### Why the hypoxia cell cannot be sourced, established rather than assumed

The operational definition above set the threshold at SaO2 below 95%, which is
looser than the 90% earlier searches had assumed, so the search was repeated
against sources previously dismissed. It failed again, and this time the
failure is explained rather than reported.

There is no cohort reporting oxygen saturation by final diagnosis in
undifferentiated dyspnoea. What the search turns up instead is the reason:
**saturation is part of how the florid presentation is identified.** Trials of
acute cardiogenic pulmonary oedema enrol on SpO2 below 90%. The AHEAD
registry, 4,153 patients with 748 classified as pulmonary oedema, defines the
syndrome as *severe respiratory distress, with crackles over the lungs and
orthopnea with O2 saturation usually <90% prior treatment* -- and that figure
appears only in the classification criterion, never as an observed baseline
measurement afterwards.

Reading it as P(hypoxia | acute pulmonary oedema) would be circular in exactly
the way this project already refuses for panic attack troponin: patients are
in the group partly *because* of the finding being counted. The AHEAD wording
is tempting because usually maps cleanly onto the narrative rubric at 0.75,
which would have satisfied the failing ordering claim on the first try. It was
checked, and the check is what disqualified it.

Two things follow. The blockage is now a property of the concept rather than
an accident of what has been published: sourcing this cell needs a cohort
where pulmonary oedema is diagnosed by other means -- natriuretic peptide,
echocardiography -- with saturation reported afterwards. And the invented 0.40
is wrong in a second way nobody had noticed: if hypoxia is quasi-definitional
for the severe presentation, the value belongs nearer the 0.95 that
consolidation carries for pneumonia than to the middle of a column. The
ordering claim understates the error it catches.

The claim stays red. The note beside it now says all of this, so the next
person does not spend the same six searches.


## Eight more real patients, chosen by a rule, and the first wrong commit

Ten real cases was the weakest number in the project. A second pass added
one per diagnosis under a rule written down before any full text was read:
one title-restricted PMC query per condition, candidates taken in order,
four acceptance criteria, every rejection logged with its letter. Twenty-seven
rejections, most of them mimics or in-patient events that were never a
presentation. All eight were added before any touched the model.

    second pass (n=8)     0 correct, 1 WRONG, 7 escalated (truth first in 4)
    all real (n=18)       2 correct, 1 wrong, 15 escalated

The wrong one: a 23-year-old woman, a week of pleuritic pain, tachycardic,
white count 13.0, a pulmonary infiltrate, CT negative for embolism —
committed as pneumonia at 78%. Pericarditis, on tissue, after a large
pericardial effusion was drained. The vocabulary had no concept for an
effusion, cardiomegaly or PR depression, so pneumonia explained everything
the model could see and the gate's sixth condition was satisfied. The
confident error the gate was always documented as unable to catch, on a
real patient. Kept, per the rule.

Three tests pinned zero wrong commits on the real cases. Every comparison
they encode held in direction at n=18 (shipped 1 wrong against 4, 4 and 2
for the rejected alternatives); the pins moved to the measured counts. Two
extraction rulings recorded for the next pass: a CT with no infiltrate is
not a chest-radiograph negative, and "otherwise unremarkable" is not a
negative for any specific sign. The second withdrew a case that had been
accepted.

## The pericardial columns, and what a concept costs

Effusion and PR depression added as concepts, two of the four ESC
diagnostic criteria, with the pericarditis cells sourced from StatPearls
(effusion through the rubric's "often"; PR depression at the chapter's own
"more than half" for ECG change rather than the rubric's 0.85 for
"characteristic", since a component cannot outrun the whole). Cardiomegaly
not added: no frequency verifiable, and a cell invented to fix one known
case is tuning.

A concept only one disease lists is inert: the backoff for every other
disease is the marginal over the *describers*, so effusion at 0.55 for
pericarditis alone would have been 0.55 for pneumonia too. Each concept
therefore carries seven invented rival cells. The same defect had been
sitting on `exam:friction_rub` all along — pericarditis was its only
describer, a sourced 0.60 was every rival's backoff, and the sign the
hard-case docstring says separates myopericarditis from infarction had
never moved anything. Seven more cells, at the values the off-by-default
grid completion had already argued for.

    likelihoods           153 -> 176, invented 89 -> 110, coverage 42% -> 38%
    parameters outside    47 -> 50 (two costs, one correlation pairing)
    ordering claims       20 -> 23, comparisons 65 -> 86
    fx-h02 (myopericarditis)   escalated 79% -> committed correctly 96%
    the real pericarditis      still committed as pneumonia, 78%

With both findings handed to the reasoner, pericarditis 2% -> 15%. Not
enough against an infiltrate weighted 18:1 — and in the loop neither
finding was ever asked for. Vocabulary was necessary and not sufficient.

## Unrecorded findings were discounting the recorded ones

The redundancy weighting was computed over every finding in the state,
including those asked and never recorded. An unknown answer contributes a
likelihood of exactly 1.0 — nothing — yet counted as a correlated partner:
a present consolidation was discounted for a fever nobody measured, and PR
depression was weighted 0.56 because the ST-segment question had come back
unrecorded. Only findings with a known polarity now take part.

    fixtures, hard, held-out      unchanged
    real mean rank of truth       3.22 -> 2.50, one more correct commit
    the real pericarditis         78% -> 93% pneumonia

The wrong commit got worse because the wrong answer's evidence stopped being
discounted too. Same arithmetic in both directions; both kept.

## The workup the mechanism was missing, written after the case

Pericarditis sat at 4% when the loop decided, and a myopic selector does not
spend a turn on a diagnosis it has dismissed — the blind spot the PE and ACS
workups exist for, with no such rule for pericarditis. ESC 2015 makes ECG
and echocardiography Class I in suspected pericarditis; a third
presentation-triggered workup, armed by pleuritic pain or a rub, asks for
both. On the case the ECG returned PR depression at turn nine, pericarditis
went live, and the selector ordered the echocardiogram itself two turns
later.

    real (n=18)           4 correct, 0 wrong, 14 escalated
    other sets            unchanged in every verdict
    cost, rule alone      fixtures 21.7 -> 23.2, hard 15.6 -> 18.0, real 17.6 -> 17.5

The first cost figure written for this rule was +33%; it compared all
workups off against all on and charged the PE and ACS D-dimers and CT
angiograms to the pericarditis rule. Measured alone: +7%, and without it
the fixtures carry a wrong commit (fx-009, an embolism committed as
pneumonia at 68%) that the rule's ECG prevents by keeping the loop asking.
A staged version — ECG first, echo only on a positive — saved nothing and
was not kept.

Written after seeing the case it fixes, which is when to be most
suspicious. What keeps it on the right side: it transcribes a Class I
recommendation, reads the presentation and never the posterior, is neither
a threshold nor a cell value, moves no other verdict, and with it off the
case is wrong again (a test runs both directions).

**Tested on a pericarditis it had not seen.** Same query, same criteria,
continued from candidate 4; candidate 5 accepted and added before running.
Positional pain is not recorded as pleuritic under the module's own rule,
and a prolonged PR interval is not PR depression, so the rule could arm only
on the rub. The loop asked for the rub on its own at turn 11, the rule
ordered the ECG at turn 12 instead of the selector's turn 20, and the other
required items were findings the report never had. Escalated, pericarditis
first at 60%, identical with the rule off. Fired as designed, changed
nothing; neither validation nor refutation. Real cases now nineteen:
4 correct, 0 wrong, 15 escalated.

## One pneumonia sourcing pass, and the correlation claim it took down

Fifteen of eighteen real cases escalated on the pneumonia / oedema / COPD
tie, and the cells doing that work were invented. The cohort already cited
for pneumonia's rest dyspnoea (PMC11141191, 265 confirmed among 954
admitted) has a Table 1 nobody had read further.

    productive cough      0.85 (DDXPlus) -> 0.55   a cohort replaces the simulator
    leg swelling          0.05 -> 0.04
    smoking history       unlisted -> 0.74         first cell outside COPD/asthma
    hypoxia               0.45 -> 0.61             study cutoff 96%, ours 95%; a deviation

The hypoxia column was called structurally unsourceable because cohorts
enrol on saturation; this one enrolled on suspected infection, so for
pneumonia it is not circular. Refused with reasons: two bidirectional
cutoffs, an auscultation finding broader than crackles, and a
measured-at-admission fever of 29% against a cell defined as measured or
reported. COPD: the only open-access cohort gives medians and quartiles;
the file refuses quartile point estimates, so the bounds went into the
audit (`QUARTILE_BOUNDS`) and both invented COPD cells sit inside them.

    commits               unchanged, 4 correct 0 wrong
    held-out              ECE 0.335 -> 0.286, Brier 0.159 -> 0.170
    real mean rank        2.33 -> 2.50 (the second pneumonia lost first place)
    likelihoods           177, 108 invented, 39%

And the claim above that "without correlation weighting the real patients
draw three wrong commits against none, and that gap has held through every
sourcing pass" stopped being true here. The unweighted configuration now
commits nothing wrong on the real cases either (5 correct, 0 wrong against
the shipped 4, 0): the three were being driven by the invented
productive-cough cell. Re-measured three ways on every set — off, declared,
measured phi — the justification moved to the fixtures: without weighting
the loop commits on every held-out case (coverage 100% against 43%) and
gets two of the ten wrong against none. That effect does not depend on the
magnitudes: every weight set uniformly to 0.10 removes both wrong commits
too, keeps eight correct against six, and gives a better held-out Brier
(0.141 against 0.170). Not acted on — choosing a magnitude from a sweep over
ten cases is fitting the knowledge base to its benchmark — and recorded as
the thing a larger case set should decide. The measured phi values change
no verdict and stay unadopted.

## What a commit now says it did not look at

No posterior-conditioned criterion can investigate what the posterior has
dismissed, and the pericarditis case needed two findings together, so no
single-test criterion would have fired. What can be general is disclosure:
the ordering claims say which findings define each disease, and every
commit now reports the rivals one of whose defining findings was observed
present while the rest were never asked — a partial signature, read from
the findings and the asked set, never the posterior. On the old vocabulary
the pericarditis commit would have read *pericarditis: pleuritic pain
present; effusion, PR depression, rub never asked*. Eleven of fourteen
commits carry such a rival; it orders nothing and blocks nothing.
`ORDERING_CLAIMS` moved into the package so one list serves the audit and
the loop.

The one action that follows without reading the posterior — ask a *cheap*
unexamined defining finding, history or bedside, never a test — was
implemented and measured at ceilings of 0.2 and 1.0 on every set. No
verdict changed; mean rank moved 1.30 -> 1.20 and 2.42 -> 2.37; cost up one
to two units; Brier within noise. Off by default. The parameter pin caught
the ceiling as the fifty-first invented number, and it stays counted.

## One column at a time: tachycardia, and what a cohort table sources on the way

Six of eight cells invented in the finding the real cases record most often.
One search per disease for an open-access cohort reporting heart rate above
100 as a count:

    pneumonia     0.60 -> 0.55   231/420   Ebrahimzadeh 2015, radiographic CAP, >=100
    COPD          0.45 -> 0.35   82/238    García-Sanz 2012, ED exacerbations, >100
    infarction    0.45 -> 0.23   347/1510  Lan 2025, MIMIC-III (intensive care), >=100
    asthma, panic               not found; invented, attempt recorded

Seven other cells from the same papers: pneumonia fever 0.69 (simulator) ->
0.68 (measured, agrees); asthma dyspnoea 0.65 -> 0.91 and wheeze 0.87 ->
0.71 (Schnyder 2022, 160 adults); pericarditis fever 0.60 (narrative) ->
0.59 (count); pericarditis troponin 0.55 (narrative "often") -> 0.16 (55 of
351, Ceriani 2026). Recorded and not adopted: an effusion rate of 79% and an
ST-elevation count of 34.5% from a 93%-recurrent referral cohort; a second
pneumonia cohort's sputum at 84% against the 55% already sourced, which now
sets the band rather than the value. A pericarditis white-count bound joined
the audit.

The troponin correction moved a direction claim: SCOPE.md had "raised
troponin raises pericarditis" from the Merck sentence; at 16% against a
differential where infarction and embolism claim it, it lowers. The audit
caught the table being stale; the claim moved, not the number.

    fixtures       6 -> 7 commits, 0 wrong
    hard cases     5/5, 4 -> 5 commits, 0 wrong
    real (n=19)    4 -> 6 correct, 0 wrong (the SLE pericarditis, the second pneumonia)
    held-out       ECE 0.286 -> 0.260, coverage 43% -> 57%
    likelihoods    177, 108 -> 103 invented, 42%

Three fixture-level pins moved with it and were re-pinned with the reasoning:
the weighted weakened arm carries fx-009 again; the decisive-test arm now
loses fx-005; the unweighted shipped arm commits one wrong on the fixtures
rather than two, and one on the real cases rather than none. Sourced cells
change what every arm does, and the pins record it rather than resist it.

## Six more cells: what a further search did not find

After tachycardia and crackles, six invented cells remained the real cases
touch most: fever for ACS, oedema, COPD, asthma, panic; crackles for the
same five minus oedema plus pericarditis. One search per gap.

Two near-misses, declined: COPD temperature (García-Sanz 2012, already
cited) reports 28.0% above 37C -- a degree below this project's 38C
threshold, a different claim rather than a rounding gap. Oedema
temperature (Ross 2024, 2,246-patient Canadian AHF registry) is a mean and
SD, 36.2 +/- 0.7C, not a threshold count -- converting it needs the
distributional assumption already refused for D-dimer.

The other four searches returned nothing: no cohort reporting crackles or
fever as a discrete finding for ACS, asthma, pericarditis or panic attack.
All six stay invented; the searches and reasons are recorded in
fixtures.py so the next attempt does not repeat them.

## The cell the wrong commit needed, taken in the direction that hurts

pmc-13070269 (ACS, fever, pleuritic pain, raised troponin, crackles
absent) misread as pneumonia pointed at P(raised troponin | pneumonia),
invented at 0.08. Sourced: 140 of 491 confirmed CAP patients (Simsek
2026), troponin I >= 19.8 ng/L, the assay's own 99th-percentile URL,
matching this project's threshold exactly -- 28.5%, undersold more than
three-fold. Mechanism stated in the paper: type 2 myocardial injury from
systemic stress, not coronary disease.

Taken even though it makes the case worse (troponin-present now
discriminates ACS from pneumonia less; the wrong commit moves 78% -> 85%),
because declining to source a number based on what it does to one case is
the tuning this project refuses. 177 likelihoods, 102 invented, 42%.

Re-measured correlation weighting the same day: real cases now tie at one
wrong commit either way, on different patients (unweighted misreads the
pericarditis this project's own workup was built for; weighted misreads
this ACS). The fixture effect is unchanged. Held-out ECE flipped which arm
looks better, on seven cases -- reported, not trusted; Brier still favours
weighted.

## The tie collapsed further, and the mechanism behind it

Searched for a joint ACS-vs-pneumonia troponin discrimination study (the
open question above). Nothing found: PMC13557950 is the closest hit and
studies T2MI prediction inside an ACS-suspected cohort, not a comparison
against pneumonia. Declined.

Re-ran both arms on all 18 real cases directly rather than re-measuring
aggregates: pmc-5841117 now escalates correctly on BOTH arms (the
pericarditis workup's ECG/effusion questions settle it once asked,
independent of correlation), and pmc-13070269 is now the wrong commit on
BOTH arms too, at nearly the same confidence (85.3% unweighted, 84.6%
weighted). "Different patients" was true for exactly one sourcing pass and
is no longer true.

Decomposed the log-score by finding to see why weighting can't move this
one. Four of the ten findings sit in a correlation group (fever+crackles
in "consolidation", ecg+troponin in "ischaemia") and weighting lifts both
candidates by nearly the same amount (ACS +1.35 nats, pneumonia +1.32) --
the margin barely moves (3.80 -> 3.77). The finding that actually decides
the case, pleuritic pain (-2.30 nats against ACS), belongs to no
correlation group in `_CORRELATION_GROUPS` and is never discounted in
either arm. The mechanism only ever acts on facets of the two pictures it
was built to describe; a standalone discriminator outside both is immune
to it. This is also why the troponin sourcing pass moved the wrong
commit's confidence but never its verdict: troponin is inside a group, so
its value is a lever on magnitude in both arms, not on which candidate
wins -- pleuritic pain already decided that.

## Three open items: one fixed, one declined, one not a bug

Item 5 (no systematic engine-parity check) is now fixed:
`test_the_two_engines_agree_on_every_case_under_every_configuration` runs
both engines against all 37 fixture+real cases under every `LoopLimits`
config the project actually constructs (shipped, real-case, counted,
decisive) and both knowledge bases -- 118 pairs, zero mismatches. The
general version of the one-case regression test the recursion-limit bug
left behind.

Item 3 (102 invented cells): searched BNP for pneumonia and COPD, since
oedema's is already sourced and BNP is the textbook cardiac-vs-pulmonary
discriminator for exactly this triad. Found real proportion-elevated data
for COPD (PMC5223538: Lee 88 pg/mL->39%, Gariani 500 pg/mL->30%) -- but
neither matches the 100 pg/mL threshold oedema's cell already uses, so
taking either builds the mismatched-threshold column this file's own
comment already refuses. Declined correctly, not left unchecked. Pneumonia
search (PMC7073979) found only prognostic (mortality) BNP studies, nothing
with a threshold at all.

Item 4 (11/19 escalate): not a bug to fix. Even a successful BNP source
would only help cases where BNP was actually recorded in the source
report; the gate declining on a silent record is it working, not a defect.
Moves only with more of item 3's sourcing work, cell by cell.

## The UMLS licence was free, and the two blocked concepts are resolved

Standing item ("Two concepts have no UMLS CUI, waiting on a licence") was
never actually blocked on anything but signup. User already had a UTS
account; a key from it unblocked `build_umls_map.py` immediately.

First run: 35/37 resolved, `exam:ecg_pr_depression` unresolved,
`imaging:pericardial_effusion` resolved to the wrong sense (C0349077,
"noninflammatory" subtype -- pericarditis effusions are typically
inflammatory). Checked with `--suggest`, not accepted on the printed name
alone: both concepts have a correct generic CUI (C0031039, C0429068) that
sits in MTH rather than SNOMEDCT_US, so the SNOMEDCT_US-restricted search
either skipped to a wrong subtype or found nothing. Pinned both in
VERIFIED_CUI, same mechanism as the three pins already there. Second run:
36/37 (only the genuinely-absent CTPA filling-defect concept left).
`Vocabulary.umls_coverage`: 26/38 -> 28/38, live from the data file, no
code change needed. SCOPE.md's concept count was stale by two findings
independent of this (27 -> 29) and corrected at the same time.

## The wrong commit's real lever was pleuritic pain, not troponin

Mechanistic trace said pleuritic_pain (invented 0.10, ACS) carries the
decisive -2.30 nat penalty, untouched by correlation weighting. Checked:
Hess EP et al 2012 (Ann Emerg Med, n=2718 ED chest-pain, 30-day cardiac
events) gives sensitivity 6.5%, lower than invented, taken anyway. Effect
measured before writing anything: pmc-13070269 worse (85%->86% unweighted,
84.6%->85.1% weighted, predicted). Unanticipated: unweighted real-case
correct count 7->8 -- pmc-4672113 (true ACS, pleuritic pain absent) now
commits, because a lower P(present|ACS) raises P(absent|ACS) and this
patient has the absent version. Same cell, opposite effect on two real
patients, both in the direction the number implies. 177 likelihoods, 101
invented, 43%. No wrong-commit-count pin moved; only confidences did.

## No single-cell fix exists for most of the 11 escalating real cases

Asked the leverage question directly instead of continuing to search
columns blind: perturbed all 101 invented cells to 0.02/0.98 each,
one at a time, single-pass proposal + gate, across all 11 currently-
escalating real cases (same method as sensitivity.py, pointed at the
commit/escalate boundary on real cases instead of fixture top-1 identity).

8/11 have zero leverage points -- no single cell at either extreme
crosses the gate. Of the 3 that do: pmc-5841117 and pmc-10993079 only
have leverage cells that flip TOWARD pneumonia, which is wrong for both
(true pericarditis, true COPD) -- pneumonia wins by a margin robust to
any one cell, so sourcing these would only manufacture confident wrong
answers. pmc-4775775's one correct-direction leverage cell
(pneumonia lab:raised_wcc -> 0.02) needs an implausible value no citation
would support (leukocytosis is common in pneumonia, not 2%).

Revises the "moves with more sourcing, cell by cell" claim two entries
up: that was optimism, not measurement. This is a ceiling on single-cell
sourcing, not a backlog of citations waiting to be found. No fixtures.py
change, no test re-pin -- pure analysis, not a knowledge-base edit.

## Sourcing continues anyway: lab:raised_wcc, pneumonia cell

Not a lever for the 11 escalations (measured above) but still worth
sourcing for general accuracy -- most-invented column (8/8 diseases), a
commonly recorded real-world finding. Four searches found only mean/SD
leukocyte counts (same non-proportion refusal as temperature, D-dimer);
a fifth (Furer 2011, PMC6549842, "Absence of leukocytosis in
bacteraemic pneumococcal pneumonia") reported the proportion directly:
74.4% of adults raised (21% overall normal, 25.6% of adults). Taken at
0.744 against the invented 0.80 -- a small correction. No effect on any
wrong-commit count (confirms the perturbation finding: this cell has no
leverage on the escalating cases either). 177 likelihoods, 100 invented,
44%. 174 tests pass, coverage figures + PDFs updated.

## lab:raised_wcc, continued: ACS sourced, oedema declined, two blocked by an outage

ACS: Yeh YT et al 2016 (Medicine (Baltimore) 95(7):e2857, PMC4998652,
n=796 STEMI on primary PCI) reports 306 leukocytosis (>=12,000/uL,
38.4%) directly -- a clean stratified cohort, unlike pneumonia's. Taken
at 0.384 against the invented 0.25, deviation noted (12,000 vs this
project's ~11,000; STEMI-only population, not any ACS) and added to
`deviations_from_operational_definitions` (6 -> 7).

Oedema declined: KorAHF registry (PMC5449528) gives leukocytosis only
as a mortality-regression odds ratio (1.6) and WCC only as mean +/- SD
-- neither recovers a base rate, same refusal as pneumonia's other
declined searches. COPD and PE not reached: NCBI's E-utilities started
returning HTTP 500s and timing out mid-pass, a service outage, not a
finding -- queued for next time rather than declared searched.

Also patched a gap in the pneumonia WCC entry from the previous commit:
its note didn't flag that the source's numeric threshold for "normal
WBC count" isn't stated in the accessible abstract, unlike this ACS
cell which states its threshold explicitly and can be checked against
this project's definition. 177 likelihoods, 99 invented, 44%
(unrounded fraction moved, rounded percentage didn't). 174 tests pass.

## pmc-13070269: checked whether any cell can at least de-confidence it

Same perturbation method, weaker bar: not "commit correctly", just
"escalate instead of confidently wrong". 9 cells (PE hypoxia/WCC, ACS ST
changes, oedema fever/WCC, asthma WCC/troponin, pericarditis
tachycardia/WCC) cross that bar at an extreme -- none flip to correct.
All 9 need implausible values (PE hypoxia at 2% against a classically
~50% real rate, ACS ST changes at 2% against a hallmark-finding rate,
etc.) -- same probe-not-candidate shape as the escalating-case analysis.
Closes item 1: no single sourced number moves this case either
direction. Pure analysis, no fixtures.py change.

## Local-model reproducibility: CPU-only tested and rejected as a fix

The write-up's explanation ("a model split across CPU/GPU does not
produce bit-identical logits") was never itself tested -- checked now.
Same prompt (pmc-13070269's real LLMProposer render), repeated 4x within
one warm session: byte-identical every time on the default GPU+CPU
split, and separately byte-identical every time forced fully onto CPU
(num_gpu=0). CPU-only is also 6-7x slower (85-144s/call vs 15-20s) and
fixes nothing. One divergence did occur, exactly once, on the first call
right after switching num_gpu settings -- a reload artifact, not a
steady-state compute-split one. Matches the original observation better
too: the five runs that gave 28.6%-71.4% were separate script launches
(fresh model loads), not repeated calls to one warm process.

Confirming fully needs several full-baseline runs as separate processes,
~1hr+ unattended -- not done; the component is already a documented,
honest substitute for the mandated model, and that much runtime isn't
worth it for a secondary path. WRITEUP.md corrected to state what was
actually tested rather than repeat an untested explanation.

## lab:raised_wcc, resumed after the outage: COPD and PE both declined

NCBI's search recovered; resumed rather than left queued. COPD: 4
searches (VIRAE study PMC3379868, troponin-in-COPD PMC2718858,
mortality-predictor PMC7259855, stable-phenotype PMC7127861), all
mean/median +/- SD or IQR, one even uses a >=15x10^9/L threshold in a
regression without ever stating the N crossing it. PE: 2 searches
(PMC12942291's biomarker-outcome tables, general queries), same shape --
medians and per-unit odds ratios, never a presentation proportion.
Both declined on the same basis as oedema. Column: 2/8 sourced
(pneumonia, ACS), 3/8 declined (oedema, COPD, PE) for the identical
reason, 3/8 (asthma, panic, pericarditis) not yet attempted. 177
likelihoods, 99 invented, 44% (unchanged -- no new sourced cell this
pass). 174 tests pass, comment-only fixtures.py change.
