# dxagent: an agentic diagnostic reasoner for acute dyspnoea and chest pain

Mohamed Aziz Ben Rejeb — DeepShift AI Summer Internship 2026, Project 1 (Healthcare)

This document is the project write-up. It describes what was built, what was
measured, and what is not delivered. Every figure in it comes from a live run
of the code in this repository, and where a number here disagrees with a number
in an earlier document, this one is current.

The single most important sentence comes first, because burying it would be the
error this whole project is organised against: **this system must not be used to
make or influence a clinical decision about any real patient.** Not because of
an unfinished feature — because 93 of its 153 likelihoods are invented, no
clinician has reviewed any part of it, and its accuracy has never been measured
on a real patient in a way that would support a claim.

---

## 1. What was built, and why one presentation

**Target: acute undifferentiated dyspnoea and chest pain in the adult**, over
eight competing causes: pulmonary embolism, community-acquired pneumonia, acute
coronary syndrome, acute pulmonary oedema, COPD exacerbation, asthma
exacerbation, pericarditis, and panic attack. Three are red flags — pulmonary
embolism, acute coronary syndrome, acute pulmonary oedema — and the system
treats them asymmetrically throughout.

The brief suggests three to five presentations. I covered one, deliberately.
The suggested examples (hyponatraemia, anaemia, unexplained dyspnoea) are
separate diagnostic problems with disjoint evidence sets; three of them covered
thinly produces a knowledge base too shallow to discriminate *within* any one of
them. One presentation with eight genuinely competing causes exercises the same
machinery — sequential evidence gathering, ranking, calibrated abstention —
against a differential where the candidates actually fight each other. Asthma
against COPD is a real clinical task; asthma against hyponatraemia is not.

One revision from my original list: congestive heart failure was replaced by
acute pulmonary oedema. CHF is chronic and was the only non-acute entity in an
otherwise acute differential, and it has no counterpart in the evaluation
dataset while acute pulmonary oedema does.

The vocabulary is 27 findings, each carrying an HPO identifier and, for 34 of
35 concepts, a UMLS CUI and SNOMED CT code. Eight diseases against 27 findings
is the 153-cell likelihood grid that the rest of this document keeps returning
to, because that grid is simultaneously the system's engine and its principal
weakness.

## 2. Architecture

Four layers, and a loop.

**Knowledge.** `knowledge.py` defines the KB interface and an in-memory
likelihood-table backend; `datasets/fixtures.py` is the knowledge base itself.
`guidelines.py` encodes Wells, PERC, CURB-65 and HEART with citations.
`merck.py` holds Merck Manual sentences as a single source cited two ways — as
derived likelihoods and as retrievable passages. `ontology.py` exposes findings
and causes as a traversable graph with supports/contradicts edges.
`provenance.py` records where every number came from.

**Reasoning.** `belief.py` holds the proposers. The default is
`BayesianProposer`: naive Bayes in log space, with a correlation-weighting step
so that several facets of one clinical picture do not multiply into several
independent pieces of evidence. `actions.py` chooses what to ask next.
`gate.py` does temperature scaling, conformal prediction, and the abstention
decision.

**Orchestration.** `agent.py` is the reference loop. `graph.py` is the same
loop as a LangGraph state machine — four nodes, propose → calibrate → decide →
observe, with a conditional edge back. They are kept equivalent by a test that
runs both over every case and asserts identical verdicts, and by a set of
shared helpers (`choose_action`, `still_outstanding`, `name_the_missing_workup`)
that exist specifically so the two cannot drift apart in the details.

Four nodes rather than the seven or nine a bigger diagram would show. The node
count is not the design; the loop has exactly the states it has, and splitting
`calibrate` into three nodes to make the picture busier would add diagram, not
capability.

**Evaluation.** `evaluation/` computes ranking, calibration, selective
prediction and DDx recall. `baselines.py` holds the two comparisons the brief
requires.

### The loop

Five steps per turn, and the loop repeats until the gate commits, escalates, or
a budget is exhausted:

1. **Propose** — rank the eight causes given the findings so far
   (`belief.BayesianProposer`).
2. **Calibrate** — temperature-scale the posterior (`gate.TemperatureScaler`).
3. **Evaluate** — apply the six abstention conditions (`gate.AbstentionGate`).
4. **Choose an action** — if not ready to decide, pick the next question or
   test by information gain per unit cost (`actions.InformationGainSelector`).
5. **Ask** — put it to the case oracle and record the answer
   (`environment.py`).

The selector scores a candidate as `gain × red_flag_weight / (cost + 1.5)`. The
cost offset stops a free question from scoring infinitely better than a cheap
one; the red-flag weight is the first of several places the system refuses to
treat all errors as equivalent. A second rule ranks *rule-out* actions by
discrimination, with a two-tier ordering that prefers a decisively
discriminating test over a marginally better-scoring cheap question.

### The abstention gate

The gate commits only if all six hold:

1. Calibrated top-1 probability ≥ 0.65.
2. Top-two margin ≥ 0.15 — this catches contested posteriors that a probability
   threshold alone waves through.
3. The top hypothesis is grounded in the knowledge base, with a citation.
4. **No red-flag diagnosis retains more than 0.10.**
5. Proposer disagreement below threshold, when two proposers are running.
6. The evidence is explained by something in the knowledge base.

Condition 4 is the one worth dwelling on, because it is a decision rather than a
default. The gate will not commit to a *more likely* diagnosis while a
time-critical one holds meaningful probability. Missing a panic attack and
missing a pulmonary embolism are not the same error, and no single confidence
threshold can express that difference. The same asymmetry drives
`guidelines.py`'s presentation-triggered workup: a D-dimer for possible PE, and
troponin plus ECG for possible ACS, fire on the *presentation* and never on the
model's current belief — because the diagnosis the model has already dismissed
is precisely the one it will never choose to investigate. That rule costs a
fixture case. It was kept anyway.

Failing any condition produces an abstention carrying a stated reason and a
named unresolved question, not a silent low-confidence answer.

## 3. The knowledge base and its provenance

**93 of 153 likelihoods are invented.** That is the number, stated on its own
line, because it is the most important fact about this project.

The remaining 60 carry a citation: 12 counted from the DDXPlus simulator, 35
from published cohorts, and 14 converted from Merck Manual narrative phrases.
The eight disease priors are separately sourced, 8 of 8, and reported on their
own line rather than folded in — merging them would let a sourced likelihood
table hide unsourced priors inside one flattering percentage.

`provenance.py` records three tiers:

- **MEASURED** — a frequency counted in a named cohort or dataset.
- **NARRATIVE** — a phrase from a reference text converted through a *fixed*
  rubric: "always" → 0.95, "often" → 0.55, "not typical" → 0.05, and so on. The
  rubric is deliberately coarse and is never re-tuned to produce a preferred
  answer. When the Merck Manual's "fever can occur in PE" mapped to 0.30 and
  landed almost exactly on the knowledge base's own marginal — making fever
  neutral for PE rather than the mild negative I expected — that was recorded,
  not adjusted.
- **INVENTED** — chosen by me, a non-clinician, to make the model behave
  sensibly.

The tiers exist so that the third category has a name and a count. An invented
number that the audit cannot identify is far more dangerous than one it can,
because it reads as an absence of a problem.

### Why most of the rest cannot be sourced

This is worth one worked example, because it explains the shape of the whole
problem better than a table would.

Consider P(raised troponin | panic attack) and P(exertional chest pain | panic
attack). No cohort reports either, and the reason is structural rather than
incidental: panic attack in this differential is a diagnosis of exclusion. A
study of panic attack patients presenting with chest pain is a study of patients
in whom cardiac causes were *already ruled out* — so the observed troponin rate
in that population is near zero by construction of the cohort, not by biology.
Using it would encode the exclusion criterion as if it were a finding. The
honest value is a low structural prior chosen to make the model behave, and
labelling it MEASURED because a paper contains a number would be exactly the
misconduct the tiers exist to prevent.

Generalise that: a textbook chapter describes what a disease presents with —
five to eight notable findings. The knowledge base is a full grid, so it
contains P(raised BNP | asthma exacerbation) and P(CTPA filling defect |
pneumonia). Nobody writes those sentences, because nobody needs them until they
build a grid like this one.

### Both bulk routes are proven exhausted, not assumed

- **DDXPlus** — the extractor was run over the full release. It reproduced only
  the pairs already sourced. Fifteen numbers, and no more available.
- **Merck** — two complete passes over all eight relevant chapters. About thirty
  candidate sentences, of which 14 were usable; most of the rest were false
  matches, drug adverse effects, or a neighbouring disease bleeding in.
- **UMLS relations** — probed and found unusable. For these eight conditions the
  relations are overwhelmingly translations, ICD crosswalks and MedDRA
  groupings, and the one clinically meaningful label mixes symptoms, risk
  factors and treatment complications without marking which is which.

### The priors, and the twentyfold band

The disease priors were invented for most of this project's life, and worse,
were invisible to the audit — `provenance` covered P(finding | disease) only.
They are now derived from StatPearls' *symptom-side* articles, which report
presentation-conditional aetiology: of patients arriving with dyspnoea, or with
chest pain, what fraction have each cause. That was the reframing that unlocked
them; disease-side literature cannot answer the question at all.

The derivation rests on three stated assumptions, and the two source
presentations disagree **twentyfold** on acute coronary syndrome — 3% of
dyspnoea presentations against 63% of chest pain presentations. This project's
presentation is both. Every prior therefore carries a band spanning that
disagreement, and the point estimate is a stated 50/50 blend rather than a
measurement. A single confident prior there would have been the dishonest
option.

DDXPlus cannot supply priors: its own paper states that per-pathology
generation rates were capped into a 10–100% band to avoid a dataset "dominated
by only a few pathologies". Its patient counts are a deliberate rebalancing
artefact, and presenting them as prevalence would launder a dataset-construction
decision as epidemiology.

## 4. What was measured rather than asserted

This section is the one I would most want read, because it is where the project
differs from a system that was tuned until it looked good.

### Sourcing the rivals, and a claim I had to take back

Every sourcing pass but the last asked the same question of a source: *what does
this disease look like?* A likelihood ratio needs the other half too — what the
**rivals** look like — and I had been answering that half by judgement
throughout, including in the two values this project complained loudest about.
Recent immobility sat at 0.61 and calf tenderness at 0.40 for the seven non-PE
diseases, both inherited from pulmonary embolism's own marginal. That amounts to
asserting that a pneumonia patient usually has a swollen calf.

PIOPED II (Stein PD et al., *Am J Med* 2007;120(10):871-9) answers it directly,
because of how the study was built: it enrolled patients *investigated* for
suspected pulmonary embolism and reports the arm in whom it was excluded. That
arm is exactly the population a differential reasons over. Table 3 gives
immobilisation in 121 of 632 (19%); Table 6 gives calf or thigh signs of DVT in
146 of 632 (23%).

| finding | invented | PIOPED II, no-PE arm |
|---|---|---|
| recent immobility | 0.61 | **0.19** |
| calf tenderness | 0.40 | **0.23** |

Fourteen cells moved from invented to measured. And it cost me a documented
claim, which is the part worth recording. DESIGN.md had stated that correlation
weighting plus Merck sourcing together repaired fx-009 — a pulmonary embolism
masquerading as pneumonia — and a test asserted that the correlated proposer put
PE on top of that evidence set. Both were true. Part of the reason was not
defensible: the rivals were being penalised for the *absence* of
venous-thromboembolism findings at rates nobody had chosen, so a large share of
PE's winning margin came from an invented number. With measured values that
support disappears.

What survives is the mechanism, which was always the actual claim:

| fx-009 evidence set | PE rank | PE probability |
|---|---|---|
| without correlation | 4 | 0.039 |
| with correlation | 2 | 0.276 |

Four correlated negatives are still not four independent penalties, and
weighting them is still worth a sevenfold lift on identical evidence. The case
itself is still answered correctly, because fx-009 is a case rather than a
frozen evidence set and the shipped loop goes on to gather what that snapshot
excludes. The test was narrowed to assert exactly that and no more.

**The general lesson, which this project kept relearning:** an invented number
does not announce itself by making results worse. This one made a result
*better*, propped up a documented claim, and was wrong by two to three times. It
surfaced only because the sourcing was done without checking first whether it
would help.

### Correlation weighting changed verdict four times

Whether correlation weighting "works" turned out to be a property of the numbers
underneath it, not a fixed fact about the mechanism. Across successive states of
the knowledge base it was worth about one fixture case, then nothing, then two
cases, then — after the PIOPED pass — the difference between one failure and two
on the deliberately weakened arm. The durable claim is not "correlation
weighting is worth *n* cases"; it is that a mechanism's value is measured
against a specific knowledge base and expires when that base changes.

### An overconfidence claim I made and retracted

I wrote in DESIGN.md, and in a commit message, that correlation weighting was
"the only thing between this model and unusable overconfidence." The evidence
was a case clearing a deliberately unclearable 99%/98% gate on the uncorrelated
knowledge base. That was the wrong evidence: the case in question is answered
*correctly*, and fixture calibration actually improved without the mechanism.
The claim was corrected in place rather than quietly deleted. The real evidence
for the mechanism is real-case calibration, which is a weaker and more specific
statement.

### A transcription error that inverted a diagnosis

I recorded the Merck Manual as saying troponin is "almost always elevated" in
pericarditis. Through the fixed rubric that became 0.95 — making a raised
troponin *more* suggestive of pericarditis than of acute coronary syndrome,
which is clinically backwards. Checking the source found "often" (30–50%), and
the sentence continues that troponin "cannot discriminate between pericarditis,
acute infarction, and pulmonary embolism." Corrected to 0.55. The rubric worked
exactly as designed; the error was upstream of it, in my reading, which is
precisely the class of error a fixed rubric cannot catch.

### A promising source that did not survive checking

The JAMA *Rational Clinical Examination* series looked like a bulk route. It
was not: it reports likelihood ratios and adjusted odds ratios rather than the
raw disease-conditional frequencies this knowledge base needs. That was
confirmed against the primary paper (Maisel 2002, *NEJM*) rather than inferred
from the review, and abandoned for a stated reason instead of quietly dropped.

## 5. Evaluation, and which numbers to distrust

### The two required baselines

Both are handed the **complete** patient record; the agentic loop must ask for
what it wants and pay per question.

| system | top-1 | top-5 | MRR | evidence seen |
|---|---|---|---|---|
| retrieval-only (no inference) | 71.4% | 100% | 0.857 | 27.0 findings |
| single-pass (KB, no loop) | 100% | 100% | 1.000 | 27.0 findings |
| **agentic loop** | 100% | 100% | 1.000 | **10.1 findings** |

The evidence column is the result, not the accuracy column. The loop matches
single-pass top-1 while seeing 38% of the findings. Comparing accuracy without
that column is the complete-profile flattery that MEDDxAgent criticises. And a
retrieval-only baseline landing within one case of the loop is a finding about
the benchmark's difficulty, not a compliment to the retriever.

Selective prediction, with the gate active: coverage 71.4%, selective accuracy
100%, full-coverage accuracy 100%, **accuracy gained +0.0%**, abstention
precision **0.0%**, AURC 0.000.

**Read that as a criticism of the fixture set, not a defence of the gate.**
The gate abstains on two cases it would have answered correctly, and buys
nothing for it, because the underlying model now gets all seven right and
there is no error left to decline. A test set the system saturates cannot
measure a mechanism whose whole purpose is knowing when to stop. The evidence
that the gate does something is elsewhere: zero wrong commits on the eight
real cases, and +10.9% accuracy gained at 52.9% abstention precision on the
much larger DDXPlus runs recorded in `DESIGN.md`.

Calibration on this split is ECE 0.303, Brier 0.097, mean confidence 69.7%
against 100% accuracy — that is **underconfidence of 30 points**, the
opposite failure to the one this project spends most of its time guarding
against, and another symptom of a saturated set rather than a property to
celebrate.

### The fixture set was spent, so it was replaced

The figures above have a consequence worth stating rather than hiding: a test
set the system aces cannot measure anything. Every mechanism in this project
was argued for as being worth some number of fixture cases, and that
measurement had stopped working.

`build_hard_cases` is five named diagnostic traps written to replace it —
cardiac asthma, myopericarditis, silent ischaemia in a diabetic, pneumonia in
a COPD patient, and asthma in someone who never smoked. They are kept separate
from the original ten so that every measurement already recorded keeps its
denominator. Each was written from the clinical picture *before* the model was
run on any of them, and none was kept or dropped on the basis of the result;
generating cases and keeping the failures would measure only a willingness to
search.

It found something on the first run. Four of five rank first, three commit
correctly, one escalates correctly at 61% — and **one commits wrongly at 70%
confidence**: a pneumonia in a COPD patient, read as a COPD exacerbation, with
consolidation visible on the chest radiograph. That is the first wrong commit
on any fixture set in this project.

The cause turned out to be a correlation group that damps the decisive
imaging test together with the three soft symptoms it exists to overrule —
exactly what `DESIGN.md` argued must not be done for D-dimer and CTPA, and
then did anyway for consolidation. Ungrouping it improves the fixtures, the
real-case ranking, ECE and Brier, costs overconfidence, and **still does not
fix the case that exposed it**. It is measured and recorded, not adopted; the
trade is mixed and one case is not a reason.

### The numbers that are not results

**10/10 on the fixture cases is a smoke test.** Ten cases I wrote by hand during
development. It demonstrates the machinery runs; it demonstrates nothing about
accuracy.

**20/20 on the synthetic corpus is worse than uninformative.** Those cases were
generated *from the knowledge base the agent reasons over*, so the score
measures self-consistency. Every generated id carries a `synthetic-` prefix
specifically so that one reaching a metrics table is visible rather than
depending on someone having remembered.

**DDXPlus agreement is bounded by what DDXPlus is.** It is a rule-based
simulator, so a naive-Bayes model recovers it partly by construction. Where a
real cohort disagreed with the simulator — pleuritic pain in PE, 71% simulated
against 33% observed — the disagreement is reported rather than reconciled.

### The real cases, and the honest result

Eight patients hand-extracted from open-access PMC case reports: real, never
seen by the knowledge base, no clinician review of the extraction.

- **3 committed and correct, 0 committed and wrong**, 7 escalated.
- Mean rank of the true diagnosis: 3.10 of 8.
- ECE 0.317, Brier 0.107.
- **Overconfidence −0.285 on the cases it answers**; +0.144 counting the ones
  it declined.

That last line needs its decomposition, because the headline figure is
misleading and this project reported it for weeks without noticing:

| subset | mean confidence | top-1 accuracy | overconfidence |
|---|---|---|---|
| committed (n=3) | 0.715 | 1.000 | **−0.285** |
| escalated (n=7) | 0.348 | 0.000 | +0.348 |

Every point of the positive figure comes from cases the gate **declined to
answer**, where the system says "I am 35% sure and I am not committing" and
is then scored as overconfident because its top-ranked hypothesis was wrong.
It penalises exactly the behaviour the gate exists to produce. On what it
actually answers, the system is *under*confident — the safe direction. A
statistic that is correct and misleading is harder to catch than one that is
wrong.

**Correlation weighting is what keeps the second line at zero.** Turned off,
the same ten patients produce **two wrong commits**, Brier 0.200 and
overconfidence +0.218. That measurement matters because the mechanism's value
had been assessed six times on the ten development fixtures and reversed every
time — worth one case, nothing, two, nothing, minus one. Those ten are
saturated, so the count was a coin toss dressed as a measurement. A number
that reverses six times is telling you the instrument is spent, and it took a
purpose-built replacement set to read it that way.

The set grew from eight to ten after the sourcing work, under a rule fixed
before anything was run: second cases go to the commonest emergency causes,
chosen on epidemiology rather than on where the model struggles. Both new
patients turned out hard — a pulmonary embolism presenting as pneumonia with
a *negative* D-dimer, and an afebrile pneumonia in an 85-year-old with a
normal white count. Every metric except one got worse, and the one that
matters did not move: still zero wrong commits, with both new cases
escalating rather than guessing. The earlier 2.50 was partly a property of the
smaller set.

Getting from eight to ten meant looking at five candidates and rejecting
three, and the reasons are structural. Two failed because **a case report is
published because it is unusual** — one was an infarct caused by coronary
vasospasm rather than atherosclerosis, another a years-long case whose final
pathology was decompensated heart disease *and* terminal bronchopneumonia.
The sampling frame is selected for exactly the property that makes a gold
label hard to assign, which caps how far this set can grow without a registry,
a chart review, or the physician-curated cases the brief intended.

One of them said something specific, and following it up found the worst
number in the knowledge base. The afebrile pneumonia ranked *seventh of
eight*, and the single strongest argument the model made against the correct
diagnosis was that she was **breathless at rest** — because the knowledge
base put P(rest dyspnoea | pneumonia) at **0.05**.

That value was transcribed correctly. The Merck chapter says dyspnoea in
pneumonia "is rarely present at rest", and the rubric maps "rare" to 0.05.
**The error was the population**: that chapter covers pneumonia at every
severity, most of it treated at home, while this system's population is people
who came to an emergency department *because* they were breathless. A cohort
of 954 acutely admitted patients records dyspnoea in 171 of the 265 with
confirmed pneumonia — 0.67, thirteen times higher.

This is a third failure mode, distinct from the two already documented. A
mis-transcription is caught by re-reading the source; a population mismatch
survives re-reading, because the source says exactly what it was recorded as
saying. And a sourced number attracts less scrutiny than an invented one,
which is how it survived several audits with a citation attached.

Correcting it **removed the only wrong commit in any of this project's test
sets** (fx-h04, a pneumonia read as COPD at 70%) and moved that real patient
from seventh to third. It also cost two fixture commits and turned fx-009 from
a confident right answer into an uncertain one — it now escalates at 52%
with pulmonary embolism still ranked first. Net on the property this project
claims: one wrong commit removed, none introduced. That is the trade
calibrated abstention is supposed to make.

The old point still stands too. The afebrile pneumonia ranks seventh of
eight, largely because her white count is normal and this knowledge base gives
pneumonia 0.80 for a raised one — a value in the single column where all
eight cells are invented and no diagnostic-accuracy literature exists, because
the white count is a severity marker rather than a test. The invented column
is not inert; it is load-bearing enough to bury a real pneumonia. It is
recorded rather than adjusted, since lowering it because one case would
benefit is the tuning this project refuses.

Zero wrong commits is the property that matters, because calibrated abstention
is this project's actual thesis. Three of eight is not a good accuracy number
and is not offered as one — n=10, hand-picked for clarity. It is reported this
small on purpose rather than not reported at all.

Every remaining escalation now names what it sought and could not get, e.g.
*"Pulmonary embolism exclusion could not be completed — sought but not
available: lab:raised_d_dimer."* That followed a cost-accounting fix: the loop
had been charging turns and budget for findings the source report never
recorded, so one case spent 98% of its budget on questions whose answers were
simply absent from the document. A finding a report never mentions means no test
was performed — it should cost neither a turn nor money. With that corrected, no
case stops for running out of money, and every escalation is now a statement
about the evidence rather than about the harness.

### A configuration inconsistency found while writing this, and fixed

`scripts/run_eval.py` built its knowledge base **without** correlation
weighting, while `consult.py`, `demo.py`, `eval_real_cases.py` and the web UI
all build it **with**. For most of this project's life the headline metrics
therefore described a configuration nobody ran. It understated the system
rather than flattering it, which is presumably why it survived unnoticed: a
number that is too low does not prompt anyone to check it.

It is now fixed, and the figures above are the corrected ones. The change is
worth stating in full because it moved the two numbers a reader is most
likely to quote:

| | before | after |
|---|---|---|
| loop top-1 (held-out) | 85.7% | **100%** |
| single-pass baseline | 85.7% | **100%** |
| MRR | 0.929 | 1.000 |
| Brier | 0.103 | 0.097 |
| abstention precision | 50.0% | **0.0%** |
| accuracy gained by the gate | +14.3% | **+0.0%** |

The first three lines look like an improvement and are not one — nothing
about the system changed, only which configuration was measured. The last two
are the honest cost: with the model now correct on all seven cases, the gate
has nothing left to decline and its apparent value on this set drops to zero.

**One correction to my own claim.** I first wrote that three metric-producing
scripts shared this bug. Checking each: `sensitivity.py` has an explicit
`--correlated` flag whose results are reported both ways in `SCOPE.md`, which
is a deliberate design rather than an oversight; and `plausibility_check.py`
produces byte-identical output either way, because the evidence table is built
from likelihood ratios against the marginal, which correlation weighting does
not touch. Only `run_eval.py` was actually wrong.

### The ablation, and what it says about the invented numbers

`sensitivity.py --ablate` deletes likelihoods outright rather than perturbing
them:

| knowledge base | fixture cases correct |
|---|---|
| complete | 9/10 |
| 93 invented deleted | **1/10** |
| 60 sourced deleted, invented kept | 7/10 |

(The 9/10 baseline rather than 10/10 is the same configuration inconsistency
described just above: `sensitivity.py` also builds without correlation
weighting. The ablation contrast is unaffected — all three rows share it.)

Perturbing any single invented number individually changes almost nothing.
Deleting them collectively destroys the system. They are individually
insensitive and collectively load-bearing, and the sensitivity list is a
sourcing priority, not a list of the numbers that matter. Whatever else is true
of this knowledge base, it is not mostly padding.

## 6. Responsible AI: privacy, safety, risk

### Privacy

**No record in this repository describes a real person.** DDXPlus patients are
simulated; the fixtures are invented; the PMC case reports are already published
open-access. That is a consequence of a scoping decision rather than of a
control, and I should not take credit for it as a privacy stance — MIMIC-IV was
skipped for *schedule* reasons (credentialing takes weeks), and the fact that
this also avoided an entire class of obligations was a side effect.

**What leaves the machine:** two opt-in paths. The LLM proposer and the
synthesis script send prompt text to a third-party API; both are off by default,
and the default Bayesian proposer is local arithmetic that makes no network
calls. If this were ever pointed at real patients, the LLM path would send
patient findings to a third party — a decision a deployment would have to make
deliberately, with a data processing agreement. Credentials are read from the
environment only, and `.gitignore` covers `.env`, `*.key`, `.gemini_key` and
`.anthropic_key`.

**PHI screening is a tripwire, not de-identification.** A regex layer catches
six classes of structured identifier; an optional spaCy NER layer catches a
person's name in prose, which regex cannot do by construction. A real clinical
de-identification pass would catch more. `screen()` reports
`ner_screening_active` on every run, so a checkout without spaCy shows a
degraded screen rather than a silently clean one.

### Safety

The six-condition gate is described in §2. What matters here is what it does
**not** catch: **confident errors where the wrong diagnosis explains the
evidence well.** Condition 6 asks whether the findings are explained by
*something* — in a masquerade, they are. I measured whether any threshold on
evidence fit separates these failures from successes; none does, and that
remained true after several rounds of sourcing. This is a stated limitation, not
a backlog item.

Two more: calibration is fitted on ten fixture cases (the scaler refuses to fit
below 30 samples, which prevents a nonsense fit but does not create data), and
conformal coverage guarantees assume exchangeability, which a curated fixture
set does not satisfy.

One category error is explicitly blocked: CURB-65 scores the severity of
pneumonia *already diagnosed*, so retrieving "confusion scores 1" as evidence
*for* pneumonia would be a category error — and semantic similarity commits it
happily, because the passage is full of pneumonia words. Rules are tagged by
kind and severity passages are excluded from grounding by default.

### Risk

The dominant risk is the knowledge base, covered in §3 and measured rather than
asserted. Two risks specific to the evaluation deserve naming: the synthetic
corpus measures self-consistency (§5), and DDXPlus is structurally incapable of
showing the value of at least three of this project's improvements, because a
rule-based simulator rewards the model that shares its assumptions.

**Automation bias** is a real risk of this specific output format. A ranked
differential with citations is more persuasive than a bare label, and therefore
more dangerous when wrong. Three choices push against it: contradicting evidence
is displayed alongside supporting evidence for every hypothesis; abstentions
state a reason and a named unresolved question rather than a shrug; and every
citation resolves to something a reader can check — including, explicitly, "this
number is a knowledge-base estimate."

The ontology graph carries the same discipline: 64 of its 84 edges rest on
invented numbers, so every edge carries its provenance tier and the DOT export
draws invented edges dashed. A dense, confident diagram built from guesses
misleads more efficiently than the guesses did alone.

**The graph also showed something the likelihood table did not:** at the default
threshold, **asthma exacerbation has no supporting edges at all.** Every finding
it carries is claimed at least as strongly by something else, almost always
COPD. The system still reaches the diagnosis — absent findings and the shape of
the whole posterior carry information no per-edge threshold can see — but no
single finding argues *for* asthma above this population's base rate. That is
this project's own stated justification for including asthma, now visible as an
empty row.

## 7. What is not delivered

**Blocked, permanently, on a dependency I did not have:**

- **No collaborating clinician.** Four deliverables depend on one: sign-off on
  the scope document, the physician-in-the-loop review cycles, clinician-graded
  evidence chains, and headline metrics on physician-curated cases. Substitutes
  exist for three (published decision rules for which findings support which
  cause, likelihood ratios for strength, DDXPlus gold labels for diagnoses).
  **Grading the evidence chain has no substitute and is not delivered.**
- **The mandated model never ran live.** No API credit existed for the project's
  duration. The LLM layer has been a swappable Protocol since week 1 for exactly
  this reason, and the default model id points at the current Opus-tier release
  — but that default has never been called. Every number in this project comes
  from the Bayesian proposer and a knowledge-base fallback, never from the
  mandated model.
- **AgentClinic never ran.** It needs its own API key plus a fork to accept this
  agent in place of its own — not a configuration change, and not attempted.
- **UMLS relations** — probed, found unusable, documented above.

**Bounded by measurement rather than blocked:** the 93 remaining invented
likelihoods. Both bulk routes are proven exhausted, and the PIOPED II pass
demonstrates that individual cells can still be recovered when a study reports
the right arm.

How far that goes is now itself measured, and the answer is less encouraging
than it first looked. Sourceability depends on the *kind* of finding rather
than on how much literature exists about it:

| kind of finding | cells | sourceable |
|---|---|---|
| binary observation, universal definition | ~20 | yes, one study at a time |
| continuous biomarker, assay-dependent cutoff | ~27 | only from one cohort covering every disease |
| differentially caused by the rivals | ~26 | only from a by-diagnosis cohort |
| genuinely unpublished | ~28 | no |

Natriuretic peptide was attempted and abandoned on the second row: a usable
figure exists for COPD exacerbation (104 of 167, 62%) but rests on an
age-specific NT-proBNP threshold, while a heart-failure figure would rest on
BNP > 100 pg/mL. Pooling them would manufacture a comparison the sources do
not support. That claim was later narrowed, and the narrowing matters: the
*column* cannot be assembled, but a single cell from a single study at a
single cutoff can be, so the heart-failure cell is now measured at 0.93 and
the other five stay invented. "This source is unusable" and "this column
cannot be assembled" are different claims, and writing the stronger one had
closed a door that was only partly shut. Chest radiography was attempted and worked on the first row,
because consolidation is a binary observation that different studies define
the same way — and it produced this project's first sourcing result that
*confirmed* an invented value (0.20 held by judgement, 0.18 measured) rather
than overturning it.

**DDXPlus is now closed rather than presumed closed.** All eighty pairs its
mapping can reach were computed, whether the cell existed or not. Fifteen it
can source, and thirteen unfilled pairs that read *exactly* 0.000 — which is
the simulator omitting a finding from its rule base, not a frequency. Writing
them would have asserted fever in 0% of pericarditis patients.

**And one dismissal turned out to be half wrong.** The JAMA Rational Clinical
Examination series reports likelihood ratios rather than frequencies, which is
why an earlier pass rejected it. But LR+ and LR- are two functions of the same
two unknowns and the system inverts, and sensitivity in a cohort of dyspnoeic
emergency patients is exactly P(finding | heart failure) over this project's
population. Three cells followed, and one of them matters: a raised JVP was
invented at **0.80** against a recovered **0.39**.

Five cells followed in total, and two more corrections were as large: the
chest radiograph shows venous congestion in only **0.54** of acute heart
failure, not the invented 0.85 — radiography is famously insensitive here
— and leg swelling, which DDXPlus recorded in **7202 of 7205** simulated
patients, is really 0.50. A likelihood of 0.999 for any clinical finding
should have been suspicious on sight; it survived several passes precisely
because it carried a citation, and a sourced number attracts less scrutiny
than one labelled invented.

**Coverage 34% to 36%, and ECE got worse at every step** — 0.298, 0.301,
0.316. That is the honest shape and it should not be smoothed over: replacing
invented numbers with measured ones made this model's real-case calibration
slightly worse. Two things argue for keeping it anyway. Overconfidence moved
+0.017 to -0.010, closer to calibrated and erring toward caution; and no wrong
commit appeared anywhere, on any arm, through either pass.

That correction cost something, and the new hard cases were what charged for
it. Two of them stopped committing — both cases the model had been getting
*right* — because it had been more than twice as confident as the evidence
allows that a patient in pulmonary oedema has a raised JVP. It was kept: these
are measurements replacing guesses, no wrong commit appeared anywhere, and a
model that is less decisive because it stopped overstating its evidence is
behaving correctly. The ten old fixtures scored the same change as an
improvement and would have reported nothing else.

The realistic ceiling is therefore nearer 36-40% than the 45-50% that a count
of the available literature suggests. Four sourcing passes have since taken
coverage from 27% to **37%**, which is inside that range, and `DESIGN.md`
records what each column is now stuck on: no accuracy literature treats the
white count as a diagnostic test, natriuretic peptides have no common scale,
hypoxia is reported continuously rather than dichotomised, and the findings
every rival causes need a by-diagnosis breakdown nobody has published.

One measured defect is recorded there rather than fixed, and it is worth
knowing about. A D-dimer meta-analysis puts specificity at 51%, so
P(raised D-dimer | not pulmonary embolism) is 0.49; this knowledge base gives
those seven diseases values averaging 0.22. They are collectively understated
about twofold, which makes D-dimer a stronger discriminator here than it is in
a real emergency department. It is not fixed because 0.49 is a pooled figure,
and writing it into all seven would assert that a panic attack raises a
D-dimer as often as a pneumonia does — the same "one value everywhere"
policy this project has now measured and rejected twice.

**A note on how to read every result here.** They describe a Bayesian reasoner
over a partly-invented knowledge base, calibrated and gated, evaluated against
two non-LLM baselines and eight real case reports. They say nothing about how an
LLM proposer would perform on the same cases, because one has never been run.

---

## Closing

The engineering is complete and the clinical content is not validated. That is
the honest one-line description, and everything above is an elaboration of it.

If there is one thing in this project I would defend as more than a student
exercise, it is not the architecture — it is that the knowledge base knows which
of its own numbers are guesses, reports the count on every run, and has a
regression test that fails when the documentation drifts away from the live
figure. Three separate times, sourcing a number overturned something I believed;
twice it made a result worse, and those results were kept and written down. A
system that cannot tell you which of its beliefs are invented cannot be
corrected, and a project that only records the corrections that flattered it has
not really recorded anything.
