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
