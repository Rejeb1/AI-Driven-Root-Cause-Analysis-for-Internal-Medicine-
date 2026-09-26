# Responsible AI: privacy, safety, risk

The brief requires every project to address privacy, safety and risk. This
document does that, and it is written to be checkable: every claim below names
the file that implements it, and every limitation is stated as a limitation
rather than as future work.

The single most important sentence in it is this one. **This system must not be
used to make or influence a clinical decision about any real patient.** Not
because of an unfinished feature, but because 58% of the numbers it reasons
with are invented (`dxagent.provenance`), no clinician has reviewed any part of
it, and its accuracy has never been measured on a real patient. Nothing further
in this document softens that.

---

## 1. Privacy

### No real patient data is used anywhere in this project

This is the strongest privacy property the project has, and it is a
consequence of a decision rather than of a control:

| Data | What it is | Origin |
|---|---|---|
| `data/ddxplus/` | Simulated patients from a rule-based generator | Public research release |
| `datasets/fixtures.py` | Ten cases written by hand for development | Invented |
| `data/vocabulary.json` | Finding names with HPO / UMLS / SNOMED ids | Public terminologies |
| `cases.json` | Model-generated vignettes | Synthesised, gitignored |

No record in this repository describes a real person. DDXPlus patients are
simulated — the loader's own module docstring says so, and the project treats
that as a *methodological* problem (a rule-based simulator flatters a
naive-Bayes model) rather than as a privacy benefit it can take credit for.

**MIMIC-IV was considered and not pursued.** It is real, de-identified ICU
data behind a credentialing process. The stated reason in `SCOPE.md` is
schedule — credentialing takes weeks and would have cleared after the point of
use. The honest addition here is that avoiding it also avoided the entire class
of obligations that come with holding real patient records, and the project
should not present a scheduling decision as a privacy stance.

### What leaves the machine

Two code paths make network calls, both opt-in and both off by default:

- `AnthropicLLM` and `GeminiLLM` (`llm.py`) send prompt text to a third-party
  API. They are used by `LLMProposer` (only with `--llm`) and by
  `scripts/synthesize.py` (only with an explicit provider flag). The default
  proposer is `BayesianProposer`, which is local arithmetic and makes no calls.
- `scripts/build_umls_map.py` queries the UMLS API for concept identifiers. It
  sends finding *names* — "pleuritic pain" — never case data.

**If this system were ever pointed at real patients, the LLM paths would send
patient findings to a third party.** That is a design decision a deployment
would have to make deliberately, with a data processing agreement, and it is
not a decision this project has made. The Bayesian path exists partly so that
the system is fully functional without it.

Credentials are read from the environment, never from a file the repository
tracks; `.gitignore` covers `.env`, `*.key`, `.gemini_key` and
`.anthropic_key` so that a key written down for convenience during a run cannot
be committed by accident.

### PHI screening on generated text, and what it is not

The screen has two layers now, run together by `synthesis.screen`.
`synthesis.phi_scan` matches six classes of structured identifier by regular
expression: dates, long digit runs, phone numbers, emails, record numbers
(MRN/NHS/SSN) and UK postcodes. `synthesis.phi_scan_ner` adds a small spaCy
NER model (`en_core_web_sm`) that reads the prose itself and flags a person's
name — the specific gap a regex cannot close by construction.
`phi_scan("the patient, John Smith, reports chest pain")` still returns
nothing, and a test still asserts that, because it is still true of that one
function; `phi_scan_ner` on the same sentence returns `John Smith`, and a
second test asserts that.

**Still a tripwire, not full de-identification, and honest about the new
limits too.** A real clinical de-identification pass (scispaCy, Presidio,
tuned for clinical text specifically) would catch more than a small
general-purpose NER model does — an unusual name, a misspelling, a name
folded into an odd construction can all still slip past it, and it can also
over-flag an ordinary word as a name. `screen()`'s report carries
`ner_screening_active` on every run precisely so a checkout without spaCy
installed shows a degraded screen rather than a silently clean one — a clean
`phi_detail` on a run where that flag is `False` means *no structured pattern
matched*, nothing about names, exactly as before this layer existed.

---

## 2. Safety

Safety here means: the system should decline rather than guess, and it should
not be equally relaxed about all errors.

### It refuses to commit unless six conditions hold

`AbstentionGate` (`gate.py`) requires all of:

1. Calibrated top-1 probability ≥ `min_confidence` (0.65)
2. Top-two margin ≥ `min_margin` (0.15) — catches contested posteriors a
   probability threshold alone would wave through
3. The top hypothesis is grounded in the knowledge base, with a citation
4. No red-flag diagnosis retains more than `red_flag_tolerance` (0.10)
5. Proposer disagreement below `max_disagreement`, when two proposers are used
6. The evidence is explained by *something* in the knowledge base

Failing any one produces an abstention with a stated reason and a named
unresolved question, not a silent low-confidence answer.

### Errors are treated asymmetrically, on purpose

Condition 4 is the one that matters most. Three of the eight conditions are
time-critical — pulmonary embolism, acute coronary syndrome, acute pulmonary
oedema — and the gate will not commit to a *different, more likely* diagnosis
while one of them holds meaningful probability. Missing a benign diagnosis and
missing a time-critical one are not the same error, and a single confidence
threshold cannot express the difference.

The same asymmetry appears in evidence gathering. `guidelines.py` encodes two
presentation-triggered workups (D-dimer for possible PE; troponin and ECG for
possible ACS) that fire on the *presentation*, never on the model's current
belief — because a diagnosis the model has already dismissed is precisely the
one it will never choose to investigate. `DESIGN.md` records that this rule
costs one fixture case, and that it was kept anyway: ordering a D-dimer for a
hypoxic patient with pleuritic pain is correct whatever it costs ten invented
cases.

### A category error the retrieval layer is stopped from making

CURB-65 scores the severity of pneumonia *already diagnosed*. Retrieving
"confusion scores 1" as evidence *for* pneumonia would be a category error, and
semantic similarity commits it happily because the text is full of pneumonia
words. `guidelines.RuleKind` tags every rule, and `GuidelineIndex.ground`
excludes severity passages by default. There is a test for it.

### What the safety machinery does not catch

Stated plainly, because a safety section that only lists what works is
marketing:

- **Confident errors where the wrong diagnosis explains the evidence well.**
  Condition 6 reads whether the findings are explained by *something*; in a
  masquerade they are. `DESIGN.md` records the measurement showing that no
  threshold on evidence fit separates these failures from successes, and that
  this remains true after several rounds of sourcing.
- **Calibration is not fitted at all**, and this document previously said it
  was fitted on ten fixture cases. `TemperatureScaler` refuses below 30
  samples, correctly, and the calibration split is three, so the temperature
  is the untouched default of 1.00. The evaluation header used to print
  "temperature (fitted) 1.00", which reads as a fit that found the posterior
  already calibrated; it now states that no fit happened and how far short
  the sample is. Thirty labelled cases do not exist here, so every
  calibration number in this project should be read as indicative and as
  uncorrected.
- **Conformal coverage guarantees assume exchangeability**, which a curated
  fixture set does not satisfy.

---

## 3. Risk

### The dominant risk is the knowledge base, and it is measured

101 of 177 likelihoods are invented, and so are 51 further model parameters
that no coverage figure used to count: six correlation weights, twenty-nine
acquisition costs, five gate thresholds, three loop budget limits and seven
selector constants. They decide which questions get asked and when the loop
commits. `dxagent.provenance.parameter_report` names and counts them, and
reports them separately from the likelihoods rather than folded in, because
averaging two different quantities produces a figure that is easier to quote
and means less. The correlation weights are the ones to worry about: five
judgement calls. They were the difference between zero and three wrong
commits on the eighteen real patients until a pneumonia sourcing pass removed
the three; WRITEUP.md §5 records that the gap is gone and why the weighting
stays on regardless.

The eight disease priors are no longer among the untracked — they are derived from published presentation-conditional
aetiology, with bands wide enough to span the disagreement between the two
sources.
`dxagent.provenance` tracks every likelihood *and every prior* as measured,
narrative-derived or invented, and reports both splits on every run of
`scripts/sensitivity.py`.

Two measurements bound what that means:

- `sensitivity.py` perturbs each likelihood and reports which change a
  diagnosis. **This is local sensitivity and must not be read as "the rest do
  not matter"** — an earlier version of the script invited exactly that
  misreading.
- `sensitivity.py --ablate` deletes likelihoods outright. Removing the invented
  ones drops the fixture set from 10/10 to 1/10. They are individually
  insensitive and collectively load-bearing.

**The eight disease priors are derived rather than measured**, and reported
separately: `8 disease priors: 8 sourced, 0 invented`. They were invented and
untracked for most of this project's life, which was the more dangerous state
— an invented number the audit cannot name reads as an absence of a problem.

"Sourced" is doing careful work here. The values come from published
aetiology of dyspnoea and chest-pain presentations, but reaching a per-disease
prior from them needs three stated assumptions, and the two sources disagree
twentyfold on acute coronary syndrome. Every prior therefore carries a band
spanning that disagreement, and the point estimate is a stated 50/50 blend
rather than a measurement.

They cannot be sourced from anything already in the project. DDXPlus's own
paper states that per-pathology generation rates were capped into a 10–100%
band to avoid a dataset "dominated by only a few pathologies", so its patient
counts are a deliberate rebalancing artefact; presenting them as prevalence
would launder a dataset-construction decision as epidemiology, which is worse
than an acknowledged guess. Real priors need population-incidence figures for
emergency presentations of each condition.

### Risk of the evaluation flattering itself

Three separate mechanisms exist because a system evaluated against its own
assumptions reports self-consistency as accuracy:

- **Synthetic cases carry a `synthetic-` id prefix.** Cases generated from the
  knowledge base the agent reasons over cannot measure that agent — the current
  corpus scores 20/20, which is an artefact and is labelled as one in the
  README, in `DESIGN.md` and in the module docstring. The prefix makes one
  reaching a metrics table visible in the per-case output rather than resting
  on someone having remembered.
- **DDXPlus is a rule-based simulator**, so a naive-Bayes model recovers it by
  construction. `DESIGN.md` records three distinct improvements whose value
  DDXPlus is structurally incapable of showing.
- **Numbers sourced from DDXPlus are labelled as measured from a generating
  model**, not from patients. Where a real cohort disagreed with the simulator
  — pleuritic pain in PE, 71% simulated against 33% observed — the
  disagreement is reported rather than reconciled.

### The mandated model and evaluation harness never ran

Every number in this project comes from the Bayesian proposer and the
knowledge-base fallback the single-pass baseline uses in place of an LLM —
never from the Claude Opus 4.x-class model the brief mandates, and never
through AgentClinic, the evaluation harness the brief names. One exception,
added late: the single-pass baseline has now been run once with a local
3B model through Ollama (WRITEUP.md §7), chosen over a hosted free tier
precisely because nothing leaves the machine, so the privacy claims above
hold with it in place. It is a substitute, not the mandated model, and the
run says so in its own header. Both are
missing for reasons that would not resolve with more time on this machine:
no API credit existed for the mandated model at any point in the project, and
AgentClinic needs its own key plus a code fork to accept this agent rather
than its own. See `SCOPE.md` §4 for the detail.

The consequence for how to read every result in this document and in
`DESIGN.md`: they describe a Bayesian reasoner over a partly-invented
knowledge base, calibrated and gated, evaluated against two non-LLM
baselines. They say nothing about how an LLM proposer would perform on the
same cases, because one has never been run.

### Automation bias

The system's output is a ranked differential with citations, which is more
persuasive than a bare label and therefore more dangerous when wrong. Three
choices push against that: contradicting evidence is shown alongside supporting
evidence for every hypothesis; abstentions state a reason and a named
unresolved question rather than a shrug; and every citation resolves to a
source a reader can check — a decision rule, a paper, a Merck Manual sentence,
or an explicit statement that the number is a knowledge-base estimate.

### No clinical validation of any kind

No clinician has reviewed the scope document, the likelihood tables, the
evidence chains, or a single output. The brief's design for this is the
physician loop in weeks 5–6; it was not available for this project.
`SCOPE.md` lists what that costs and what was substituted — published decision
rules with citations for which findings support which cause, likelihood ratios
for strength, DDXPlus gold labels for diagnoses — and records that grading the
evidence chain has **no substitute and is not delivered**.

`synthesis.review_sample` selects the batch a clinician should spot-check, at
random rather than worst-first, because reviewing the cases the pipeline
already doubts measures the checker instead of the corpus. That sample has
never been reviewed.

---

## 4. If this were ever to be deployed

Not a roadmap — a list of what would have to be true first, none of which is
true now:

1. Every likelihood and prior sourced or clinician-supplied, with the
   provenance report showing no invented tier.
2. Prospective validation on real presentations, with headline metrics from
   physician-adjudicated cases rather than from DDXPlus agreement.
3. Calibration refitted on hundreds of cases, not ten.
4. Real de-identification (clinical NER) in place of the regex tripwire, if any
   real note text is processed.
5. A data processing agreement covering any LLM path, or the LLM path disabled.
6. A named clinician accountable for the knowledge base, and a route for
   clinicians to contest an output.
7. Monitoring for distribution shift, since a knowledge base fixed at build
   time silently decays.

Until then the honest description is the one in the README: a working research
prototype whose engineering is complete and whose clinical content is not
validated.
