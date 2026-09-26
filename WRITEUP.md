# dxagent: an agentic diagnostic reasoner for acute dyspnoea and chest pain

Mohamed Aziz Ben Rejeb — DeepShift AI Summer Internship 2026, Project 1 (Healthcare)

This document is the project write-up. It describes what was built, what was
measured, and what is not delivered. Every figure in it comes from a live run
of the code in this repository, and where a number here disagrees with a number
in an earlier document, this one is current.

The single most important sentence comes first, because burying it would be the
error this whole project is organised against: **this system must not be used to
make or influence a clinical decision about any real patient.** Not because of
an unfinished feature — because 99 of its 177 likelihoods are invented, no
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

The vocabulary is 29 findings, 26 of them carrying a UMLS CUI and SNOMED CT
code alongside the HPO identifier. Eight diseases against 29 findings is the
176-cell likelihood grid that the rest of this document keeps returning to,
because that grid is simultaneously the system's engine and its principal
weakness. It was 27 findings and 153 cells until the real-case second pass
(§5) showed the vocabulary could not see a pericardial effusion or PR
depression; the two concepts and their 23 new cells are accounted for there.

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

**99 of 177 likelihoods are invented.** That is the number, stated on its own
line, because it is the most important fact about this project. It was 89 of
153 until the pericardial columns were added (§5): two concepts the real cases
showed were missing, two sourced cells for pericarditis, and 21 invented rival
cells without which the new concepts would discriminate nothing. Coverage went
*down*, from 42% to 38%, and that is the honest direction — the count now
includes claims the model needs to make. A pneumonia sourcing pass (§5) then
took it to 39%, a pass on the tachycardia column (§5) to 41%, and one cell recovered from the same search (§5) to 42%.

The remaining 73 carry a citation: 7 counted from the DDXPlus simulator, 55
from published cohorts and StatPearls, and 11 converted from Merck Manual
narrative phrases. Both simulator and narrative counts fall as real cohorts
replace them.
The eight disease priors are separately sourced, 8 of 8, and reported on their
own line rather than folded in — merging them would let a sourced likelihood
table hide unsourced priors inside one flattering percentage.

**And the likelihood table is not the whole model.** Fifty-one further
numbers are invented and, until they were counted, appeared in no coverage
figure this project quoted: six correlation weights, twenty-nine acquisition
costs, five gate thresholds, four loop budget limits, seven selector
constants. They decide which questions get asked and when the system commits.

The worst case looked like the correlation weights — five judgement calls
between 0.70 and 0.85 that are the difference between **zero and three wrong
commits** on the ten real patients. Unlike a gate threshold, those are a
measurable quantity, so they were measured against 200,091 DDXPlus patients:
0.85 against 0.495, 0.75 against 0.551, 0.70 against **0.079**. All overstated,
the last ninefold, and four pairs the model calls independent measure up to
0.24.

Substituting the measured values changes nothing — same commits, same wrong
count, ECE 0.240 against 0.237. So the mechanism is load-bearing and its
numbers are not, which is a more useful thing to know than the correction
would have been. They are left alone: changing a number that affects no output
in order to improve how the audit reads is the metric-gaming this project
refuses elsewhere. SCOPE.md already states why an untracked invented
number is worse than a tracked one, about the disease priors, which had the
same problem until they were pulled into the audit: *an invented number the
audit cannot name reads as an absence of a problem.* That hole was closed for
the priors and nobody asked whether it existed elsewhere. It did, five times
over.

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

It expired once more while this document was being finished. The one
measurement that had never reversed — that without the weighting the real
patients drew three wrong commits, and with it none — reversed after a
pneumonia sourcing pass; §5 has the figures. The mechanism stays, for the
reason it was added. The claim that it was load-bearing does not.

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

### Nothing was checking the numbers themselves

Every test in this project checks outputs. The knowledge base is inputs, and
for most of its life nothing looked at it: 99 of 177 likelihoods are invented,
and the only scrutiny they got was whichever ones happened to change a case.

Reading each column sorted, against what the diseases actually do, found five
errors in three passes — a natriuretic peptide at 0.20 for pulmonary
embolism when right-ventricular strain is what releases it; an immobility rate
from a simulator at 2.4 times the measured one; a troponin below the sourced
value for COPD; a CT angiogram claimed at 0.95 when it misses one embolism in
six; and an exertional-chest-pain value taken from a DDXPlus question that
asks about *symptoms*, not chest pain. **Not one moved a fixture case enough
to fail anything, and four were in pulmonary embolism** — in a system built
around not missing it.

That reading is now executable: twenty ordering claims over sixty-five rival
comparisons, each of the form *this disease must score above these rivals for
this finding, because ...*, with reasons a clinician could accept or reject in
one sentence. They constrain ordering rather than magnitude, because ordering
is what a non-clinician can assert honestly.

Two comparisons fail and are listed as tolerated, both the same cell — acute
pulmonary oedema's hypoxia, which should lead its column and does not. The
claim is stated in the form believed correct and allowed to fail, because an
earlier draft asserted only the half that passes and hid the defect inside the
check built to find it.

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

**The headline this section should lead with is not the coverage figure.**
Over a sustained sourcing run, correct answers on the real patients fell from
3 to **2**, and commitments from 3 to 2 of 10. Wrong commits stayed at **0**
throughout and the hard cases held at 5 of 5. Every individual step replaced
an invented number with a measured one and was justified on its own; the
aggregate is a system that answers fewer real patients than it did.

That is the honest shape of what sourcing did here, and it is the opposite of
the story a rising coverage percentage tells. Two readings are available and
both are stated rather than chosen between: the model became less decisive
because it stopped overstating evidence it never had — several corrections
were of the form "this test is less conclusive than we claimed", including a
CT angiogram that misses one embolism in six — or ten hand-picked patients
are too few for a two-case difference to mean anything. The second reading is
almost certainly also true, which is itself a finding about the evaluation.

- **2 committed and correct, 0 committed and wrong**, 8 escalated.
- Mean rank of the true diagnosis: 3.10 of 8.
- ECE 0.246, Brier 0.151.
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

Zero wrong commits was the property that mattered at n=10, because calibrated
abstention is this project's actual thesis. Three of eight is not a good
accuracy number and was not offered as one — hand-picked for clarity, reported
this small on purpose rather than not reported at all. The second pass below
took the count to eighteen and the property away.

Every remaining escalation now names what it sought and could not get, e.g.
*"Pulmonary embolism exclusion could not be completed — sought but not
available: lab:raised_d_dimer."* That followed a cost-accounting fix: the loop
had been charging turns and budget for findings the source report never
recorded, so one case spent 98% of its budget on questions whose answers were
simply absent from the document. A finding a report never mentions means no test
was performed — it should cost neither a turn nor money. With that corrected, no
case stops for running out of money, and every escalation is now a statement
about the evidence rather than about the harness.

### Eight more real patients, and the first wrong commit

Ten was the weakest number in the project, so a second pass added one case
per diagnosis under a rule written down before any full text was read: one
title-restricted PMC query per condition, candidates taken **in order**, each
accepted or rejected on four stated criteria, every rejection logged with its
letter, nothing kept or discarded on how the model did. The log is in
`real_cases.py` — twenty-seven rejections, most of them mimics ("X masquerading
as acute coronary syndrome") or in-patient events that were never a
presentation. The whole pass ran before any of the eight touched the model.

The result on the eight new cases: **0 correct commits, 1 wrong commit,
7 escalations**, with the true diagnosis ranked first in four of the seven.
Across all eighteen: 2 correct, **1 wrong**, 15 escalations.

The wrong one is a 23-year-old woman with a week of pleuritic chest pain,
tachycardia, a white count of 13.0, a "pulmonary infiltrate" on chest imaging,
and a CT that excluded embolism. The model committed to pneumonia at 78%. Her
diagnosis was pericarditis, confirmed on tissue after a large pericardial
effusion was drained through a surgical window. The knowledge base has no
concept for a pericardial effusion, cardiomegaly, or PR depression — the
three findings that decide the case — so pneumonia explained everything the
model could see, and the gate's sixth condition, which asks whether the
findings are explained by *something*, was satisfied. This is exactly the
confident error §6 says the gate cannot catch, stated there in the abstract;
here it is on a real patient. Nothing about the extraction is
strained: "infiltrate" was mapped to consolidation the same way the first
pass mapped "basal infiltrate", and the report says what it says.

The rule was to keep whatever came, and this is kept. It is also the most
useful data point in the set: the first-pass claim of zero wrong commits was
a statement about ten patients whose deciding findings happened to be in the
vocabulary, and one patient whose deciding finding is not was enough to end
it. Ten cases could not show that. Eighteen did.

Two of the eight are hard on purpose because the rule chose them, not because
anyone did: a pulmonary embolism with no dyspnoea, no tachycardia, no hypoxia,
a normal ECG and a normal radiograph (pericarditis ranked first; embolism
third), and an NSTEMI with no chest discomfort at all (correctly ranked first,
escalated at 53% with embolism at 11%). Neither was selected for being
difficult. The rule took the second and ninth results of two queries.

### The pericardial columns: what the wrong commit bought, and what it did not

The wrong commit was a vocabulary gap, and that part was fixable. Two
concepts were added — `imaging:pericardial_effusion` and
`exam:ecg_pr_depression`, two of the four ESC diagnostic criteria for the
disease — with the pericarditis cells sourced from StatPearls (effusion
through the rubric's "often"; PR depression at the chapter's own "more than
half" figure for ECG change rather than the rubric's 0.85 for
"characteristic", because a component cannot outrun the whole). Cardiomegaly,
the third finding that decided the case, was **not** added: no frequency for
it could be verified, and a cell invented to fix one known case is the tuning
this project refuses.

The part that was not free: a concept only one disease lists is inert. The
backoff for every other disease is the marginal over the diseases that
*describe* the finding, so effusion at 0.55 for pericarditis alone would have
been 0.55 for pneumonia too and discriminated nothing. Each new concept
therefore needed seven invented rival cells. And the same defect turned out
to have been sitting on `exam:friction_rub` all along — pericarditis was its
only describer, so a sourced 0.60 was every rival's backoff and a rub present
moved nothing. The one sign the hard-case docstring says separates
myopericarditis from infarction had never done so. Seven more invented cells,
at the values the off-by-default grid completion had already argued for.

Net: 176 likelihoods, 110 invented, coverage down from 42% to 38%. Three new
ordering claims pin pericarditis as the leader of all three columns. The rub
cells paid off where the hard cases said they should: fx-h02, the
myopericarditis written to test exactly that sign, went from escalating at 79%
after twelve turns with acute coronary syndrome still at 10% to committing
correctly at 96% in five. Hard cases are 5 of 5 with four commits and none
wrong; the fixture set is unchanged in accuracy with one more commit. The
parameter count outside the table went from 47 to 50 (two costs, one
correlation pairing for the two readings of one ECG). All of it counted.

**And the case is still wrong.** Two measurements, both kept. With the two
findings handed to the reasoner directly, pneumonia falls from 0.94 to 0.78
and pericarditis rises from 0.02 to 0.15 — real movement, not enough: the
pulmonary infiltrate is weighted 18:1 for pneumonia, and on the findings this
vocabulary can express the patient genuinely looks like one. And in the loop
itself neither finding is ever asked for: the selector committed at turn
twelve with pericarditis at 4%, and a myopic information-gain selector does
not spend a turn on a diagnosis it has already dismissed. That is the blind
spot the presentation-triggered workup exists for, and there is no such rule
for pericarditis. Vocabulary was necessary and not sufficient, and the
honest sentence in the deep dive that read "the fix is vocabulary, not
tuning" was half right.

Two more fixes followed, measured one at a time, and between them the case
commits correctly at 92%.

The first was a defect, not a policy. The redundancy weighting was computed
over every finding in the state, including those asked for and never
recorded. An unknown answer contributes a likelihood of exactly 1.0 to the
product — nothing — yet it counted as a correlated partner, so a present
consolidation was discounted for a fever nobody had measured, and PR
depression was weighted 0.56 because the ST-segment question had come back
unrecorded. Only findings with a known polarity now take part. Measured on
every set: fixtures, hard cases and held-out split unchanged; real-case mean
rank of the truth 3.22 → 2.50, one more correct commit, no new wrong one —
and the pericarditis went from 78% pneumonia to **93%** pneumonia, because
the evidence for the wrong answer stopped being discounted too. Same
arithmetic in both directions, both kept.

The second was the workup the mechanism was missing. ESC 2015 makes ECG and
transthoracic echocardiography Class I in every case of suspected acute
pericarditis; pleuritic pain or a rub is how that suspicion is expressed in
this vocabulary. A third presentation-triggered workup, the same shape as
the PE and ACS ones, now asks for both. On this case the ECG came back at
turn nine with PR depression, pericarditis went live, and the selector
ordered the echocardiogram itself two turns later. With the rule off the
case is wrong again, and a test pins both directions.

Real cases now: **4 correct, 0 wrong, 14 escalated** of 18. The other sets
did not move.

This rule was written after seeing the case it fixes, which is exactly when
to be most suspicious, so what keeps it on the right side of that line is
stated rather than assumed: it transcribes a Class I recommendation, it
reads the presentation and never the posterior, it is not a threshold or a
cell value, and its effect on every other set is small and measured — and
the first measurement of it here was wrong. An earlier draft of this
paragraph put the rule's cost at +33% on the fixtures; that figure compared
*all* workups off against all on, and charged the PE and ACS workups'
D-dimers and CT angiograms to the pericarditis rule. Measured alone: the
fixtures go from 21.7 to 23.2 (+7%), the hard cases from 15.6 to 18.0, the
real cases from 17.6 to 17.5. And without it the fixtures carry a wrong
commit — fx-009, an embolism committed as pneumonia at 68% — that the
rule's ECG prevents by keeping the loop asking. Incidental, and still a
prevented wrong commit. A staged version (ECG on pleuritic pain, echo only
on a positive) was measured and saved nothing, so the flat guideline
transcription stays. It is a clinical policy decision awaiting a clinician.
The real test is the next pericarditis chosen by rule, not this one.

**That test was run.** The same query, the same four criteria, continued
from where the second pass stopped; the first acceptance was candidate 5,
a 23-year-old man with acute precordial pain, fever, tachycardia, ST
elevation V2–V5, a pericardial friction rub and a normal troponin — acute
pericarditis on cardiac MRI, with the lupus behind it found afterwards. Two
extraction rulings went against the case before it was run: the pain is
positional and never called pleuritic, so under the module's own rule
`pleuritic_pain` is not recorded and the workup cannot arm on it; and a
prolonged PR interval is not PR depression. The rule could arm only on the
rub, and only if the loop asked for it.

It did — at turn 11, on its own — and the rule then ordered the ECG at
turn 12 rather than the turn 20 the selector would have reached on its own.
The other two required items were an echocardiogram the report says was
never performed and a PR finding it does not contain. Verdict with the
rule: escalated, pericarditis first at 60%, under the 65% floor. Verdict
without it: identical. A correct abstention either way; the rule fired as
designed and changed nothing. That is the honest result of the test — not
a validation, not a refutation — and it stays in the set. Real cases are
now nineteen: 4 correct, 0 wrong, 15 escalated.

### What a commit now says it did not look at

The pericarditis fix was a rule for one disease. The blind spot behind it is
general: a diagnosis the loop dismisses early is one it never investigates,
and every posterior-conditioned criterion this project tried — four of them,
documented in `actions.py` — failed for exactly that reason. The case also
needed *two* findings in combination, so no single-test criterion would have
fired. Writing a presentation-triggered workup per disease would be the
mechanical answer, and each one is clinical content, not engineering.

What can be made general is disclosure. The ordering claims already say
which findings define each disease, so every commit now reports the rivals
one of whose defining findings was observed present while the rest were
never asked for — a partial signature the loop walked past. It reads the
findings and the asked set, never the posterior; it orders nothing and
blocks nothing. On the real pericarditis, at the moment it was committed as
pneumonia, the packet would have read: *pericarditis — pleuritic pain
present; effusion, PR depression, friction rub never asked.* That sentence
did not exist when it was needed, and a reader with it could have made the
call the loop could not.

Measured on the fourteen commits across all three sets, eleven carry at
least one such rival. That is not a defect count; an eight-question loop
cannot cover twenty-three defining findings, and most of those rivals are
correctly dismissed. It is what the loop did not look at, stated where it
can be argued with. The field is on every `CaseOutcome`, in the JSON export,
the evaluation scripts, the consultation transcript and the web UI; the
claims moved from the audit script into the package so one list serves both.

One action does follow from the disclosure without reading the posterior:
before committing, ask any unexamined defining finding of a partly-present
picture that is *cheap* — a history question or a bedside sign, never a
test. It is implemented (`LoopLimits.diligence_cost`) and it was measured
at two ceilings on every set. It changed no verdict anywhere. Mean rank of
the truth moved 1.30 → 1.20 on the fixtures and 2.42 → 2.37 on the real
cases, cost rose by one to two units, held-out Brier moved within noise. A
mechanism that changes no decision is not adopted for a small rank
movement, and a ceiling is one more invented number — the parameter pin
caught it as the fifty-first. It ships off, kept so the measurement can be
repeated. The disclosure turned out to be the part that matters; acting on
it did not.

### Separating the three dyspnoea diseases: one sourcing pass, and a claim it took down

Fifteen of eighteen real cases escalated because the knowledge base could
not separate pneumonia, pulmonary oedema and COPD on a thin record. The
cells doing that work are mostly invented, so the honest move was to source
them, and the cohort already cited for pneumonia's rest dyspnoea — 265
confirmed pneumonias among 954 acutely admitted patients, PMC11141191 —
reports the rest of its Table 1. Four cells taken: **productive cough**
0.85 → 0.55 (a real cohort replacing the simulator, as Miniati did for
embolism), **leg swelling** 0.05 → 0.04, **smoking history** 0.74 (the
first cell for that concept outside COPD and asthma), and **hypoxia** 0.45 →
0.61 — the column called structurally unsourceable because cohorts enrol on
saturation, except that this one enrolled on suspected infection, so for
pneumonia it is not circular; one point off the stated threshold, listed
in the deviations. Four figures refused with the reason recorded: two
bidirectional cutoffs, one auscultation finding broader than crackles, and
a measured-at-admission fever of 29% that conflicts with a cell defined as
measured *or* reported and is written down as a conflict rather than over
the value.

For COPD the only open-access cohort found reports medians and quartiles,
and this file already refuses to turn a quartile into a point estimate. The
bound itself is distribution-free, so it went into the audit instead:
`QUARTILE_BOUNDS` in `plausibility_check.py` now checks that the invented
COPD tachycardia and white-count cells sit inside the [0.25, 0.50] that 437
hospitalised exacerbations imply. Both do. Pulmonary oedema got nothing new.

Measured on every set: commits unchanged at 4 correct, 0 wrong; held-out
ECE 0.335 → 0.286, Brier 0.159 → 0.170; real-case mean rank 2.33 → 2.50,
because the second pneumonia lost her first place — a pneumonia that
expectorates only half the time is a weaker explanation of a productive
cough. 177 likelihoods, 108 invented, coverage 39% at that point.

### One column at a time: tachycardia

The finding the real cases record most often (17 of 19) had six of eight
cells invented. One search per disease for an open-access cohort reporting
heart rate above 100 as a count, taken at the threshold or with the
deviation listed:

    pneumonia     0.60 -> 0.55   231/420, radiographic pneumonia (Ebrahimzadeh 2015), >=100
    COPD          0.45 -> 0.35   82/238, ED exacerbations (García-Sanz 2012), >100
    infarction    0.45 -> 0.23   347/1510, MIMIC-III (Lan 2025), >=100 -- an intensive-care
                                 population, stated as such
    asthma, panic               not found; left invented, the attempt recorded

The papers found for their heart rate sourced or bounded seven other cells
on the way, which is the normal shape of this work — a cohort table is a
row, not a cell: pneumonia fever from the simulator's 0.69 to a measured
0.68 that agrees; asthma dyspnoea 0.65 → 0.91 and wheeze from the
simulator's 0.87 → 0.71 (Schnyder 2022, 160 adult presentations);
pericarditis fever from a Merck phrase at 0.60 to a count of 0.59, and
pericarditis **troponin from "often elevated" at 0.55 to 55 of 351, 0.16**
(Ceriani 2026) — the narrative off by a factor of three, the same shape as
the rest-dyspnoea correction. A white-count bound for pericarditis joined
the audit; an effusion figure of 79% in that referral cohort (93% recurrent
disease) and an ST-elevation count of 34.5% were recorded as notes, not
adopted, for population and concept mismatch. And a second pneumonia cohort
put sputum at 84% against the 55% taken two days earlier from the
expert-confirmed cohort: two studies, two definitions, no way to prefer one
without a judgement, so the value stays with the stricter diagnosis and the
band now spans both.

The troponin correction tripped the direction audit: SCOPE.md's table had
listed raised troponin under "raises pericarditis", a claim written from the
Merck sentence. Against a differential where infarction and embolism claim
troponin hard, 16% lowers it — the clinically right way round — and the
claim moved, not the number.

Measured on every set, cells sourced without looking at any case: fixtures
6 → 7 commits, none wrong; hard cases 5 of 5 with 5 commits, none wrong;
real patients **4 → 6 correct, 0 wrong**, the SLE pericarditis and the
second pneumonia now committing; held-out ECE 0.286 → 0.260, coverage 43%
→ 57%. Three fixture-level pins moved with it and are re-pinned with the
reasoning. 177 likelihoods, 99 invented, coverage 44%.

### One cell at a time: crackles

The search for tachycardia cohorts happened to surface one, PLOS ONE 2026 (PMC12962476), a straight cohort of 380 adults hospitalized with diagnosed community-acquired pneumonia — not a case-control design, the population match this project prefers over the pneumonia cohort already in use. Its Table 1 reports crackles present in 161 of 380 (42.4%), replacing an invented 0.80 that, measured against an actual admission cohort, turns out to have been closer to a textbook-severity figure than an observed one. The same table's Fever row (56.1%) disagrees with the cohort already sourced for that cell (67.9%, measured temperature); rather than pick one, the value stays with the measured-threshold study and the band widens to cover the reported-symptom one, the same treatment given to the productive-cough conflict.

This one cell cost the project its clean run. Absent crackles had been arguing hard against pneumonia at the invented 0.80; at a measured 0.42 it barely argues at all, and one real case — an ACS presenting with fever, pleuritic pain and a raised troponin, crackles absent — now reads as pneumonia at 78% instead of escalating. The finding is kept for the same reason every other correction in this project has been kept: it replaces a guess with a measurement, and a knowledge base that changes what it commits to when a cell moves is doing exactly what it is supposed to do. Real patients: 6 correct, **1 wrong**, one fewer than the sourced ideal and the honest number.

Fixing this case properly is not a cell fix — pericarditis's own friction rub, ECG and effusion cells are all sourced and none of them apply to an ACS differential; nothing about *this* rival is wrong. What would help is a cell this project does not have: P(raised troponin | ACS) versus P(raised troponin | pneumonia with cardiac strain), which needs a study built around exactly that confusion rather than one more general cohort table. Recorded as the next open question rather than patched around.

One structural bug surfaced on the way, unrelated to any cell value: `GraphAgent`'s LangGraph recursion limit was fixed at `4 * max_turns + 10`, sized for a configuration where every turn counts toward `max_turns`. The real-case configuration turns that off — an unknown answer is free — so the reference loop can legitimately spend one step per remaining vocabulary concept, far more than `max_turns`, before it runs out of candidates. One real case (a pulmonary oedema, committed correctly by the reference loop) hit that fixed ceiling and raised `GraphRecursionError` in the graph engine instead of committing — silently, because nothing exercises the two engines against the real cases under that configuration. The limit now scales to the vocabulary size when uninformative turns are free, and a regression test pins both engines agreeing on that case.

### Six more cells searched for, and not found

After tachycardia and crackles, six invented cells remained the real cases
ask about most: fever for acute coronary syndrome, pulmonary oedema, COPD,
asthma and panic attack, and crackles for the same five minus oedema plus
pericarditis. One search per gap. Two near-misses, both declined: COPD's
temperature is reported (García-Sanz 2012, already cited for tachycardia) at
28.0% above 37°C — a full degree below this project's 38°C threshold, which
is a different clinical claim, not a rounding difference, and widening the
definition to fit the data available is the fitting this project refuses.
Pulmonary oedema's temperature is reported by a 2,246-patient Canadian
registry (Ross 2024) as a mean and standard deviation, 36.2 ± 0.7°C — not a
threshold count, and converting it to one needs the distributional
assumption already refused for the D-dimer columns. The other four searches
returned nothing: no cohort reporting crackles or fever as a discrete finding
for acute coronary syndrome, asthma, pericarditis or panic attack turned up.
All six stay invented, with the search and the reason recorded in
`fixtures.py` so the next attempt starts from a different angle — a named
sign's likelihood ratio from a rational-clinical-examination review, rather
than one more disease-cohort table these five apparently do not report
crackles or fever in.

### The cell pmc-13070269 actually needed, taken in the direction that hurts

The wrong commit itself pointed at a number: an ACS with fever, pleuritic
pain and a raised troponin, misread as pneumonia, is a case where
P(raised troponin | pneumonia) is doing real work, and it was invented at
0.08. A search for troponin in confirmed CAP found a clean match — a
491-patient retrospective cohort (Şimşek 2026), troponin I at or above
19.8 ng/L, which the paper states is the assay's own 99th-percentile
upper reference limit, matching this project's threshold with no
deviation to record. 140 of 491, **28.5%**, against the invented 0.08 —
undersold by more than three-fold. The mechanism is stated in the paper:
pneumonia raises troponin through systemic stress and type 2 myocardial
injury, not coronary disease, and a cohort excluding recent acute cardiac
events and alternative diagnoses is measuring exactly that.

Sourcing it makes pmc-13070269 worse, not better. With troponin more
common in pneumonia than the invented cell claimed, a raised troponin
discriminates ACS from pneumonia *less*, and the wrong commit moves from
78% to **85%**. Taken anyway. The alternative — declining to source a
number because of what it will do to one case — is the tuning this
project exists to refuse, stated plainly in the comment beside the cell
in `fixtures.py` rather than left for a reader to notice on their own.
177 likelihoods, 100 invented, 44%.

What this means for the wrong commit: it is not a knowledge-base gap
anymore, at least not in the cells this pass could reach. Pericarditis's
signs don't apply to an ACS differential and were never the issue.
What would actually separate these two diagnoses is a study built around
the specific confusion — troponin in confirmed ACS against troponin in
pneumonia with cardiac strain, as a joint comparison rather than two
separate cohort tables — and general searches for either disease alone do
not turn that up. Recorded as the open question, not as something the next
search on this pattern is likely to close.

**Searched for directly, and not found.** Three targeted PubMed/NCBI
queries for a study comparing troponin between confirmed ACS and confirmed
pneumonia patients turned up nothing built for that comparison. The closest
candidate, PMC13557950 ("An easy-to-use model for screening type 2
myocardial infarction in geriatric patients with acute coronary syndrome"),
studies T2MI prediction *within* an already-ACS-suspected cohort — baseline
characteristics and a regression model for its training/validation split —
not a discrimination study against a pneumonia comparator. Rejected on that
basis rather than adopted on the strength of its title. The open question
above stands as stated; nothing found narrows it.

**And this pass took down a claim §4 relied on.** The case for correlation
weighting rested on real patients: "three wrong commits against none, and
that gap has held or widened through every sourcing pass." After this pass
the unweighted configuration commits nothing wrong on the eighteen real
cases either — 5 correct, 0 wrong against the shipped 4 correct, 0 wrong.
The three wrong commits it used to produce were being driven by an
invented pneumonia cell, and the sourcing that corrected it removed the
gap that had justified the mechanism. The test that pinned
three-against-zero now pins zero-against-zero and says why.

Re-measured on every set, three ways — weighting off, the declared
weights, and the measured phi values from 200,091 DDXPlus patients — the
mechanism's justification moved rather than vanished. On the **fixtures**,
switching it off makes the loop commit on every held-out case (coverage
100% against 43%) and get two of the ten wrong against none: four facets of
one clinical picture counted as four independent findings sharpen the
posterior straight past the gate. That is what the weighting does, and it
is visible on the set where the records are complete enough for redundancy
to accumulate. The measured values change no verdict anywhere and move the
held-out Brier from 0.170 to 0.160, third-decimal noise on seven cases;
they stay unadopted for the reason recorded in `fixtures.py`. Correlation
weighting stays on, for the gate's calibration on the fixtures and as the
principled correction — not for a real-patient gap that the only
instrument able to see it now says is not there.

Two more things about that justification, because it now rests on ten
invented cases and should be looked at hard. First, no third instrument
exists: the hard cases show no difference either way, the real records are
too thin for redundancy to accumulate, and DDXPlus samples evidence
independently given the pathology, so it has no dependence to correct and
the earlier DESIGN.md measurement showing correction makes it slightly
*worse* is the expected result. Second, the fixture effect does not depend
on the invented magnitudes. Every weight set uniformly to 0.10, 0.30, 0.50
or 0.95 removes both wrong commits just as the declared 0.70–0.85 do; what
the magnitude changes is how much the loop still commits — at 0.10 the
fixtures keep eight correct commits against six, with a better held-out
Brier (0.141 against 0.170) and the real cases five correct against four,
none wrong anywhere. That is a real observation and it is not acted on:
choosing a magnitude from a sweep over ten cases is fitting the knowledge
base to its own benchmark, and the measured phi values — the one non-tuned
alternative — agree with the declared ones on every verdict. So: the
mechanism is robust to its values; its values are still invented; and the
lighter setting is recorded here as the thing a larger case set should
decide.

**Re-measured again after tachycardia and crackles were sourced, and the
real-case story changed shape a second time.** It was no longer "weighting
prevents a wrong commit": both arms committed exactly one real patient
wrong, on different patients — unweighted misread pmc-5841117 (the
positional, sit-forward-relieved pericarditis this project's own
pericarditis workup was built around) as acute coronary syndrome; weighted
misread pmc-13070269, the ACS the crackles correction exposed (§5), as
pneumonia. That was true for one sourcing pass. The very next one (the
troponin cell above) closed the gap the rest of the way: pmc-5841117 now
escalates correctly on *both* arms — the pericarditis workup's own ECG and
effusion questions settle it once asked, independent of correlation — and
pmc-13070269 is now the wrong commit on both arms, at nearly the same
confidence (85.3% unweighted, 84.6% weighted — weighted is, if anything,
marginally more cautious about it). Two sourcing passes turned "different
patients, different directions" into "the same patient, the same
direction, on both arms."

The reason correlation weighting cannot rescue pmc-13070269 is mechanical
and worth stating precisely, because it explains why the story kept
reversing instead of settling. Decomposing the log-score by finding: four
of this patient's ten findings sit inside a fixed correlation group
(fever and crackles in "consolidation"; ECG changes and troponin in
"ischaemia"), and weighting them at 0.571 lifts acute coronary syndrome's
score by 1.35 nats and pneumonia's by 1.32 — almost exactly the same
amount, because both diseases have roughly the same number of "surprising"
values sitting inside those two groups for this specific finding-set. The
margin between them (3.80 nats unweighted, 3.77 weighted) barely moves.
What actually decides the case is **pleuritic pain**, present here and
carrying a lone −2.30 nat penalty against acute coronary syndrome that no
group touches — it is correlated with nothing in `_CORRELATION_GROUPS`, so
it is never discounted, in either arm. Correlation weighting only ever
acts on facets of the two clinical pictures it was built to describe; a
standalone discriminating finding outside both of them is invisible to it
and can dominate the outcome by itself. That is also why sourcing troponin
(78% → 85%, above) moved the wrong commit's *confidence* on the weighted
arm without changing which case fails or flipping either arm's verdict:
troponin sits inside "ischaemia," so its value passes through at full
weight unweighted and at 0.571 weighted — a lever on magnitude, not on
which candidate wins, because pleuritic pain already decided that.

The fixture story is unchanged and still clean: unweighted commits one
fixture wrong (fx-009) that weighted does not, at the cost of more turns
and a lower coverage. That is the effect this section's fixture-values
paragraph above describes, and it still holds as measured.

Held-out calibration on the same seven cases flipped which arm looks
better on ECE (weighted 0.251, unweighted 0.135; Brier still favours
weighted, 0.174 against 0.192) — worth reporting and not worth trusting:
seven cases give ECE almost no bins to be calibrated within, and a flip
driven by which handful of cases sit in which confidence bucket is noise
dressed as a finding. Brier, which does not bin, still prefers the
weighted arm and is the more honest number to read here.

Net position, restated rather than left implicit: correlation weighting
stays on for the fixtures, where its effect is real, sized, and
reproduced across three measurements now. Its case for the real patients
was never more than one anecdote wide, and the anecdote is now gone
entirely — both arms fail the same patient the same way, because the one
finding that decides this case sits outside every correlation group the
project has defined. The mechanism is kept for what it demonstrably does
on the fixtures, not for a real-patient argument that has run out of
patients to point to.

### Three open items, tried: one fixed, one searched and declined, one not a bug

A standing "what's missing" list carried three items forward. Tried in
order:

**A general engine-parity check now exists.** The recursion-limit bug two
sessions ago was found by accident — the one combination that triggered it
(real cases, the real-case budget flags) had never been run through both
engines before. `test_the_two_engines_agree_on_every_case_under_every_configuration`
is the systematic version: every case this project has (fixtures and real,
37 total), every `LoopLimits` configuration the project actually
constructs somewhere (the bare default, the real-case flags
`eval_real_cases.py` and the web UI use, and the two flags exercised in the
correlation test), and both knowledge bases — 118 case/configuration pairs
in one test. Zero mismatches the day this was written. If a future change
to either engine produces one, this is the test that reports it instead of
a silent `GraphRecursionError` found by accident later.

**BNP was searched for pneumonia and COPD, and correctly declined.** It
looked like the strongest remaining lead: the project already has oedema's
`lab:raised_bnp` sourced from Wang 2005 at a BNP > 100 pg/mL threshold, and
BNP is the textbook discriminator between cardiac and pulmonary dyspnoea —
exactly the pneumonia/oedema/COPD confusion behind most of the escalations
below. A systematic review of natriuretic peptides in COPD (PMC5223538)
turned up real, threshold-matched proportions for two AECOPD cohorts (Lee,
BNP > 88 pg/mL, 39% elevated; Gariani, BNP > 500 pg/mL, 30% elevated) — not
the mean/SD-only data this project has declined before, an actual
proportion at a stated cutoff. Neither cutoff is 100 pg/mL. Taking either
would build precisely the mismatched-threshold column this project's own
comment already refuses: a heart-failure value measured on one assay at
one cutoff sitting in the same table as a COPD value measured on a
different one. Declined on that basis, not on the basis that nothing was
found. A parallel search for pneumonia (PMC7073979) found only prognostic
studies — natriuretic peptides predicting mortality, no proportion
elevated at any threshold — so pneumonia's cell has nothing to decline, it
simply was not there.

**The UMLS licence turned out not to be a blocker.** It was listed as
"waiting on a licence" because nobody had submitted one — a UTS account is
free, and once a key existed `build_umls_map.py` resolved 35 of 37
concepts on the first run. The two that had been sitting unresolved for a
whole "structural, unfixable" category of the gap list — `exam:ecg_pr_depression`
and `imaging:pericardial_effusion` — turned out to need a second pass, not
a bigger one. The first attempt at `imaging:pericardial_effusion` resolved
to `C0349077`, "Pericardial effusion - noninflammatory" — a specific
subtype, and the wrong one: pericarditis effusions are typically
inflammatory. `exam:ecg_pr_depression` resolved to nothing at all. Checking
with `--suggest` rather than accepting the printed name showed why: the
correct generic concept for each (`C0031039` "Pericardial effusion",
`C0429068` "PR depression") lives only in UMLS's MTH source, not
SNOMEDCT_US, and the script restricts every search to SNOMEDCT_US to avoid
exactly the kind of cross-language noise `umls_probe.py` found. That
restriction is right for every other concept and wrong for these two.
Pinned both by hand, the same mechanism already used for three other
concepts search could not reach cleanly. Second run: 36 of 37, the one
remaining gap (`imaging:ctpa_filling_defect`) genuinely absent from UMLS
rather than unresolved. Coverage moved from 26/38 to 28/38, reported live
by `Vocabulary.umls_coverage` with no code change. SCOPE.md's concept
count was independently stale by two findings and corrected at the same
time.

**11 of 19 real cases still escalate, and that is not something to fix by
finding more cells.** It looked, going in, like item 3's BNP search might
double as a fix for this — the pneumonia/oedema/COPD triad on thin records
is exactly what a discriminating lab value would help. It didn't pan out,
for the reason above. But the honest second half of this item is that even
a successful BNP source would not have "fixed" the escalation count in any
simple sense: real case reports often don't mention BNP at all, so a
sourced cell only helps the subset of cases where the finding was actually
recorded. Declining to guess when the record is silent is the abstention
gate doing its job, not a defect the gate should be tuned away from. What
would actually move this number is more of the same sourcing work item 3
represents — cell by cell, as cohorts turn up — not a mechanism change.

### The wrong commit's real lever was never troponin, and sourcing it made the case worse on purpose

The mechanistic trace above (§"Re-measured again...") found that pleuritic
pain, not troponin, decides pmc-13070269: a −2.30 nat penalty against ACS
that no correlation group touches, invented at 0.10 and never checked. So
it was checked. Hess EP et al. (Ann Emerg Med 2012;59(2):115-25) followed
2,718 emergency-department chest-pain patients for 30-day cardiac events
at three academic centres — sensitivity of pleuritic-quality pain for a
true cardiac event was **6.5%**, lower than the invented 0.10, not higher.
Two independent reviews cited in the same paper agree only in direction
(pleuritic pain argues against ACS), at metrics — an LR and an odds ratio
— that don't convert cleanly to a sensitivity, so the band stays narrow
around Hess's own figure rather than stretched to cover numbers that
aren't the same quantity.

Taken anyway, same rule as the troponin cell: pmc-13070269 moves from 85%
to 86% pneumonia unweighted, 84.6% to 85.1% weighted. Worse, not better,
predicted before running the measurement and correct.

One thing the prediction did not anticipate: the unweighted real-case
count went from 7 correct to **8**. pmc-4672113, a true ACS with pleuritic
pain absent, previously escalated; a lower P(pleuritic pain present | ACS)
means a higher P(pleuritic pain absent | ACS) — 0.935 against the old
0.90 — so the same absence now argues *more* strongly for the correct
diagnosis on a different patient. Sourcing this cell helped and hurt on
two different real cases simultaneously, in the direction the mechanism
actually implies both times. Nothing here was chosen for that effect; it
is what checking an untested assumption produced. 177 likelihoods, 99
invented, 44%. 174 tests pass; no wrong-commit-count assertion needed
re-pinning because the count did not move, only which case's confidence
and which other case newly commits.

### There is no single-cell fix for most of the 11 escalating real cases, and this is measured now, not assumed

That cell was found by a mechanistic trace, not a search of a whole
column, so the same method was pointed at the standing "11 of 19 real
cases escalate" item to ask a sharper question than "source more cells":
*which* invented cells, if any, are actually holding any of these 11 back
from a commit? Every one of the 101 invented likelihoods was perturbed to
each extreme (0.02 and 0.98) in turn, one at a time, and re-run through a
single-pass proposal and the gate on all 11 currently-escalating cases —
the same perturbation method `sensitivity.py` already uses on the
fixtures, pointed at the real cases and at the commit/escalate boundary
instead of top-1 identity.

**Eight of the eleven have no leverage point at all.** No single invented
cell, moved to either extreme, crosses the gate's threshold. Their
uncertainty is not one bad number sitting in the way; it is genuinely
multi-factorial, spread thin enough across many findings that no single
citation could resolve it even in principle. This is a stronger, measured
version of "not a bug" — not just an inference from the gate's design, a
direct test that no single sourced cell, however extreme, would change
the outcome.

Of the three that do have a leverage point, two are worse than useless.
pmc-5841117 (true pericarditis) and pmc-10993079 (true COPD) each have a
handful of cells that cross the threshold — and every single one of them
crosses it *toward pneumonia*, which is wrong both times. Pneumonia wins
by a margin robust to any one cell's extreme; sourcing any of these
would not fix the case, it would convert an honest escalation into a
confident wrong answer, the exact failure mode correlation weighting and
the abstention gate both exist to prevent. Not pursued, for that reason.

The third, pmc-4775775 (true pneumonia), is the one case where the
leverage cell points the right way: pneumonia's own `lab:raised_wcc`,
pushed toward 0.02, commits it correctly. It is not a real lead either —
0.02 says leukocytosis is present in 2% of pneumonia, and it is a common
finding (60-80% in the cohorts already sourced for other diseases in this
file); no citation would plausibly support a value that extreme. The
perturbation sweep's edges are a probe, not a candidate value, and this
result shows the probe finding nothing rather than finding something.

The revision this makes to the record: "11 of 19 real cases still
escalate... moves only with more of item 3's sourcing work, cell by
cell" (above) was more optimistic than this measurement supports. More
sourcing can still narrow individual invented values and may eventually
move one of these cases by changing several cells' worth of evidence at
once, but there is no single citation waiting to be found for any of the
eight cases with no leverage point, and the other three would either make
things worse or need a value nothing would corroborate. The honest
description of this gap is a ceiling, not a backlog.

### The same question asked of pmc-13070269 itself, with a weaker bar

The perturbation analysis above covered the 11 escalating cases; it never
covered the one case that's worse than escalating — pmc-13070269, which
commits confidently to the wrong disease. Worth a separate pass, and with
a deliberately weaker bar than "commit correctly": does any single
invented cell, at either extreme, at least turn the wrong commit into an
honest escalation?

Nine do. `pulmonary_embolism`'s hypoxia and WCC, `acute_coronary_syndrome`'s
ST changes, `acute_pulmonary_oedema`'s fever and WCC, `asthma_exacerbation`'s
WCC and troponin, `pericarditis`'s tachycardia and WCC — each, pushed to
0.02 or 0.98, dilutes pneumonia's share of the posterior enough to drop
below the gate's confidence threshold. None flip the case to a *correct*
commit; the best any of them does is convert a confident wrong answer into
an honest "I don't know."

None is real. Every one needs the same kind of value the escalating-case
analysis already ruled out: PE causing hypoxia in 2% of patients (PE
classically causes it in something like half), ACS showing ST changes in
2% (a hallmark finding, not a rare one), pulmonary oedema presenting with
fever in 98% (fever is not a feature of cardiogenic oedema at all), and so
on for the rest — extremes with no study behind them, probe values, not
candidates. Checked so the record can say it was tried rather than assumed
impossible: even the weaker bar of "escalate instead of confidently wrong"
has no legitimate single-cell fix. This closes item 1 completely; nothing
further from a single sourced number can move this case in either
direction.

### The most-invented column, worked one cell at a time

`lab:raised_wcc` is invented for all 8 diseases, the single largest
remaining gap and one of the most commonly recorded findings in any real
report — not a lever for the 11 escalating cases (the perturbation
analysis above already ruled that out), but still worth having for
general accuracy, the same reasoning as every other sourcing pass in this
project. Pneumonia's cell (Furer 2011, PMC6549842) came after four dead
searches that returned only mean/SD leukocyte counts; ACS's came clean
on the first hit for that disease — Yeh YT et al. 2016 (Medicine
(Baltimore) 95(7):e2857, PMC4998652), 306 of 796 STEMI patients on
primary PCI with leukocytosis at admission (38.4%), against the invented
0.25. Oedema was checked and declined (the Korean Acute Heart Failure
registry gives leukocytosis only as a mortality odds ratio and WCC only
as mean ± SD, the same non-proportion shape already refused elsewhere).
COPD and pulmonary embolism were queued at that point — not
searched-and-declined, just not reached before NCBI's E-utilities started
returning HTTP 500s and timeouts partway through the pass, a service
outage rather than a finding.

Resumed once the outage cleared, and both declined for the same reason as
oedema. Four more COPD searches (a microbiology cohort, a troponin-in-COPD
study, a mortality-predictor cohort, a stable-phenotype study) and two
more for pulmonary embolism (a biomarker-outcome cohort, general
leukocytosis queries) turned up nothing but medians, means, and per-unit
odds ratios — never a proportion crossing a stated threshold. One COPD
paper uses a `>=15×10⁹/L` cutoff inside a mortality regression and still
never states how many patients crossed it, an odds ratio standing in for
a base rate the same way oedema's did.

The last three closed the column rather than left it open. Panic (one
relevant hit, hematologic indices in panic disorder) gives a mean ± SD in
the wrong population besides — chronic panic disorder between attacks,
not the acute presentation this concept means. Pericarditis's one
relevant hit, a 2025 review of inflammatory biomarkers in pericarditis,
is weaker than a mean ± SD: it says WCC "adds supportive yet nonspecific
information" with no number attached at all. Asthma turned up nothing
relevant in any search — COPD, paediatric, and exacerbation-prediction
cohorts, none reporting WCC for adult acute presentations.

`lab:raised_wcc` closes at 2 of 8 sourced (pneumonia, ACS) and 6 of 8
declined (oedema, COPD, pulmonary embolism, panic, pericarditis, asthma)
for what is now one structural reason rather than six separate ones:
across roughly twenty cohorts and reviews checked for this column, raised
WCC was reported as a mean, a median, a per-unit odds ratio, or a bare
qualitative remark in all but two. A threshold-crossing proportion — the
one shape this project can use without a distributional assumption — is
the exception in this literature, not the norm. Further searching this
specific column is not expected to find a seventh. 177 likelihoods, 99
invented, 44%.

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
| 89 invented deleted | **1/10** |
| 64 sourced deleted, invented kept | 7/10 |

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

Two more. **Calibration is not fitted at all** — a claim this document
itself used to get wrong, saying it was "fitted on ten fixture cases". The
scaler refuses below 30 samples, correctly, and the calibration split is
*three*, so the temperature is the untouched default of 1.00. The evaluation
header compounded it by printing `temperature (fitted) 1.00`, which reads as a
fit that found the posterior already calibrated; it now states that no fit
happened and why. Thirty labelled cases do not exist here: ten hand-extracted
case reports is a third of the floor, the fixtures are invented, and the
synthetic corpus is generated from the knowledge base being calibrated.

And conformal coverage guarantees assume exchangeability, which a curated
fixture set does not satisfy.

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
  — but that default has never been called. What *has* run, late and as a
  documented substitute, is a local model: `OpenAICompatibleLLM` in `llm.py`
  talks to Ollama with the standard library, no key, nothing leaving the
  machine, and `run_eval.py --llm` now runs the brief's single-pass baseline
  with it. On the held-out seven with `llama3.2:3b` handed the complete
  record: top-1 42.9%, top-5 71.4%, MRR 0.571 — below the retrieval-only
  baseline (57.1%) and the loop (71.4% on 41% of the evidence). A 3B model
  is a lower bound on what the mandated model would do, the run is not
  reproducible in the way the rest of this project is (a local model's
  output varies with build and hardware), and the row is reported for what
  it is: the first time that baseline held a model rather than a stand-in.
  Run a second time an hour later, same model, same seven cases, temperature
  0 requested: **28.6%**. Then, after adding a fixed seed to every request,
  three more runs back to back: 57.1%, 71.4%, 71.4%, with different picks
  each time. Five runs of one row on one set of seven cases: **28.6% to
  71.4%.** The seed is not the cause. What the split does was checked
  directly, not assumed: the same prompt (the full pmc-13070269 record,
  the actual `LLMProposer` render, not a toy) sent to the model repeatedly
  *within one warm session* came back byte-identical every time on the
  default GPU+CPU split and, separately, byte-identical every time forced
  fully onto CPU (`num_gpu=0`) — four runs each, temperature 0, seed 0. So
  the compute split is not producing non-bit-identical logits on repeated
  calls to an already-loaded model, which is what the original explanation
  claimed. Forcing CPU-only is also six to seven times slower (85-144s per
  call against 15-20s) and does not fix anything, so it is not adopted.
  One run in that same test did diverge, and only once: the first call
  immediately after switching from the default split to `num_gpu=0`,
  before the next three settled back to identical. That points at
  something tied to a *reload* -- how the model is placed in memory when a
  fresh process or a changed setting loads it, not the arithmetic of a
  steady GPU+CPU split -- which matches the five original runs better: each
  was a separate launch of the evaluation script, a fresh model load every
  time, not repeated calls to one warm process. Confirming that fully would
  mean running the whole baseline script several times as separate
  processes, upwards of an hour of unattended runtime for a component this
  project already reports honestly as non-reproducible; not done, because
  the added confidence is not worth an hour against a path that is a
  documented substitute, not the mandated model. What is fixed is the
  claim: the cause is a reload effect, not the split itself, and the
  record no longer states an explanation that a direct test contradicts.
  `run_eval.py --llm-repeats N` still runs the row N times and prints the
  spread, and no single-run LLM figure in this document is a result.

  Measuring the model's disagreement with the posterior at full evidence on
  thirteen cases: median total variation 0.37, against a gate threshold of
  0.40 that was set before any second proposer existed. So the fifth
  condition fires on roughly a third of decisions with this model — which is
  the coverage drop below, explained — and it is not adjusted, because
  choosing it now would be tuning to thirteen cases. The measurement also
  found a defect: three of the thirteen read exactly 0.00, which was not
  agreement but the model returning an unusable ranking, the proposer
  falling back to Bayes, and the consensus comparing Bayes with Bayes. The
  gate saw perfect agreement on turns with no second opinion. A fallback
  turn now reports NaN, which the gate treats as "no consensus proposer this
  turn" — neither a block nor an agreement — and the fallback count is
  exposed.

  **The job a model should have had — extraction — was built and measured,
  and the answer is no, not with these models.** `dxagent/extraction.py`
  puts the extraction rules from `real_cases.py` in front of a model and
  enforces the enforceable part mechanically: every finding must carry a
  passage found verbatim in the source or it is dropped, the discussion
  section is cut so a literature review's percentages are never seen, and a
  finding reported with two polarities across chunks is a conflict for the
  human. `scripts/extract_case.py` turns a PMC id into a `Case(...)` draft
  with every quote beside its finding and every rejection with its reason.
  Measured on the SLE pericarditis against the hand extraction of seven
  findings:

      llama3.2:3b   3 recovered, 2 missed, 2 wrong polarity, 17 invented, 23 rejected by quote
      qwen2.5:7b    3 recovered, 4 missed, 0 wrong polarity,  6 invented,  2 rejected by quote

  The quote check works: every fabricated sentence ("Fever was not
  reported") was caught. What it cannot catch is a real sentence quoted for
  a finding it does not mention. The 3B model read silence as absence
  seventeen times, each time citing an unrelated line; the 7B model did it
  less and worse — *raised BNP: present* and *raised D-dimer: present*,
  quoting "elevated inflammatory markers, including erythrocyte
  sedimentation rate", tests the report never ran. That is the
  confidently-wrong data point the extraction rules exist to prevent, with
  a genuine citation attached. Reviewing nine proposals to keep three is not
  less work than reading the report. The path stays, tested, as a draft
  generator with mandatory review; it does not replace the hand extraction,
  and no case in the set came from it.

  The same model *inside* the loop (`--llm-in-loop`, one call per turn, about
  an hour on this hardware) was also run once on the held-out seven, as a
  consensus proposer beside the Bayesian one:

      top-1 / MRR          71.4% / 0.833   (Bayesian loop alone: 71.4% / 0.833)
      ECE / Brier          0.274 / 0.167   (0.286 / 0.170)
      gate coverage        28.6%           (42.9%), selective accuracy 100% both
      evidence seen        12.1 of 27      (11.1)

  Same ranking, calibration a hundredth better, and the gate commits on
  fewer cases — the proposer-disagreement condition finally has a second
  proposer to disagree with. The hard cases say the same thing more
  sharply: 5 of 5 ranked first either way, but the consensus commits on 2
  (both correct) where the Bayesian loop commits on 4, the myopericarditis
  escalating at 95% and the silent ischaemia at 88% because the model's own
  ranking disagreed with the posterior enough to trip condition 5. Not a
  reproducible number for the same reason as the baseline, and not an
  argument for the model: on this evidence a 3B model in the loop changes
  what the loop *declines*, not what it gets right — and it declines things
  it had right.
- **AgentClinic never ran.** It needs its own API key plus a fork to accept this
  agent in place of its own — not a configuration change, and not attempted.
- **UMLS relations** — probed, found unusable, documented above.

**Bounded by measurement rather than blocked:** the 89 remaining invented
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
