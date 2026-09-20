# Scope and reasoning schema

Project 1 deliverable, Step 1. Draft — the clinical judgements below need a
physician's sign-off, which is not currently available; see *Constraints*.

## 1. Target presentation

**Acute undifferentiated dyspnoea and chest pain in the adult.**

One presentation rather than the brief's three-to-five, deliberately. The
suggested examples (hyponatraemia, anaemia, unexplained dyspnoea) are separate
diagnostic problems with disjoint evidence sets; covering three of them thinly
would produce a knowledge base too shallow to discriminate within any of them.
A single presentation with eight competing causes exercises the same machinery
— sequential evidence gathering, ranking, abstention — against a differential
where the candidates genuinely compete.

## 2. Candidate root causes

Eight, all acute:

| Cause | Why it is in scope |
|---|---|
| Pulmonary embolism | Time-critical, commonly missed, has published rules (Wells, PERC) |
| Community-acquired pneumonia | The most common competitor, published severity rule (CURB-65) |
| Acute coronary syndrome | Time-critical, published rule (HEART) |
| Acute pulmonary oedema | The acute decompensation of heart failure |
| Acute COPD exacerbation | Overlaps heavily on dyspnoea and wheeze |
| Acute asthma exacerbation | Distinguishing it from COPD is a real clinical task |
| Pericarditis | Shares pleuritic pain with PE |
| Panic attack | The benign cause that must not be reached by exclusion alone |

Three are red flags — pulmonary embolism, acute coronary syndrome, acute
pulmonary oedema — and the system treats them asymmetrically: it will not
commit to a benign diagnosis while one of them retains meaningful probability.

**A revision from the original list.** Congestive heart failure was replaced by
acute pulmonary oedema. CHF is a chronic condition and was the only
non-acute entity in an otherwise acute differential; it also has no counterpart
in the evaluation dataset, whereas acute pulmonary oedema does.

## 3. Reasoning schema

For each cause, the knowledge base records:

- **P(finding | cause)** for every finding in the vocabulary — the likelihood
  table that ranks the differential.
- **Supporting and contradicting evidence**, computed as a likelihood ratio
  against the knowledge-base-wide marginal. A ratio above one supports, below
  one argues against, and the magnitude is the strength. Both polarities count:
  an expected finding that is *absent* argues against a diagnosis as much as an
  unexpected one present.
- **Correlation between findings**, so that several facets of one clinical
  picture do not multiply into several independent pieces of evidence.
- **Acquisition cost** per finding, so the agent can weigh a question against a
  test.

Findings use HPO identifiers (`data/vocabulary.json`) so every concept has a
stable, citable id.

### Discriminating evidence, by cause

The evidence that most moves each diagnosis, as currently encoded:

| Cause | Raises it | Lowers it |
|---|---|---|
| Pulmonary embolism | CTPA filling defect, raised D-dimer, sudden onset, rest dyspnoea, hypoxia | orthopnoea, pulmonary oedema on CXR, consolidation on CXR |
| Pneumonia | consolidation on CXR, fever, raised WCC, productive cough, crackles | CTPA filling defect, leg swelling |
| Acute coronary syndrome | raised troponin, sudden onset | productive cough, consolidation on CXR, fever |
| Acute pulmonary oedema | pulmonary oedema on CXR, raised BNP, orthopnoea, leg swelling, raised JVP | fever, raised WCC, consolidation on CXR |
| COPD exacerbation | productive cough, rest dyspnoea, smoking history, hypoxia | pulmonary oedema on CXR, sudden onset, raised JVP |
| Asthma exacerbation | rest dyspnoea, sudden onset | pulmonary oedema on CXR, raised BNP, crackles |
| Pericarditis | raised troponin, fever, pleuritic pain | consolidation on CXR, crackles, productive cough |
| Panic attack | sudden onset, palpitations, tachycardia | CTPA filling defect, hypoxia, productive cough |

**This table means something narrower than it looks, and the difference
matters.** "Raises it" here is *relative to the other seven candidates*, not
relative to the general population — the likelihood ratio is computed against
this knowledge base's own marginal. A finding can be strongly associated with a
disease and still fail to raise it here, because a rival claims it harder.

The table is generated from the knowledge base rather than written from
intuition, and `scripts/plausibility_check.py` fails if the two drift apart.
An earlier hand-written version made four claims the sourced numbers
contradicted, all four for that reason:

- **Pleuritic pain no longer raises pulmonary embolism.** A real cohort puts it
  at 33% in PE (Miniati 2012), while pericarditis reaches 69% and pneumonia 53%.
  It is a *pleuritic-pain* finding, not a PE finding, once the competitors are
  sourced too.
- **Exertional chest pain: a claim made here and since retracted.** This
  document used to record that exertional chest pain no longer raises acute
  coronary syndrome, because DDXPlus scored it at 77% in acute pulmonary
  oedema against 36% in ACS, and presented that as something the sourced
  numbers had revealed. It was a mapping error. The DDXPlus question behind
  it, E_218, asks "Do you have symptoms that are increased with physical
  exertion but alleviated with rest?" — *symptoms*, not chest pain, which is
  why heart-failure patients answer yes. The mapping was withdrawn, the two
  cells returned to the invented values it had displaced (0.80 for ACS and
  0.20 for pulmonary oedema, the clinically sensible ordering), and coverage
  fell from 43% to 42% as a result. A confidently sourced wrong number is
  worse than an invented one, and this is what that looks like.
- **Wheeze barely separates asthma from COPD** (87% against 91%), which is
  precisely what this document's own COPD row predicts when it calls
  distinguishing them "a real clinical task".
- **Fever is now neutral for pulmonary embolism**, not lowering. The Merck
  Manual says fever "can occur" in PE; the fixed rubric maps that to 0.30,
  which lands on the knowledge base's marginal almost exactly. The rubric is
  deliberately coarse and is not re-tuned to produce a preferred answer, so
  this is recorded rather than adjusted.
- **Rest dyspnoea no longer lowers pneumonia**, and the reason is a sourcing
  error rather than a surprise about medicine. The Merck chapter says
  dyspnoea in pneumonia "usually is mild and exertional and is rarely present
  at rest", the rubric mapped "rare" to 0.05, and the transcription was
  faithful — but that chapter describes pneumonia at every severity, most
  of it managed at home, while this document's population is people who came
  to an emergency department *because* they were breathless. A cohort of 954
  acutely admitted patients records dyspnoea in 171 of the 265 with confirmed
  pneumonia, so the value is 0.67 and not 0.05. The old number was arguing
  against the correct diagnosis in real patients; see DESIGN.md for what
  correcting it cost.

**Most of these numbers are still invented, and which ones are not is
recorded.** 39% of the 177 likelihoods carry a citation — 9 from DDXPlus, 13
from the Merck Manual's narrative text, 47 from published cohorts and
StatPearls. The fraction fell from 42% when two pericardial concepts were
added with invented rival cells, and recovered a point on a pneumonia
sourcing pass; see WRITEUP.md §5.
`scripts/sensitivity.py` reports the split and `dxagent.provenance` tracks it
per number. See *Constraints*.

### Mandatory workup

Two checks are triggered by the presentation rather than by the model's
current belief, because a diagnosis the model has already dismissed is exactly
the one it will never choose to investigate:

- **PE exclusion** — pleuritic pain, rest dyspnoea, hypoxia or sudden onset
  requires a D-dimer.
- **ACS exclusion** — chest pain of possible cardiac character requires a
  troponin and an ECG.

## 4. Constraints

**No collaborating clinician was available.** Four deliverables depend on one
and cannot be met: sign-off on this document, the physician-in-the-loop review
cycles, clinician-graded evidence chains, and headline metrics reported on
physician-curated cases. Substitutes:

| Was to come from the clinician | Substitute |
|---|---|
| Which findings support which cause | Published decision rules — Wells, PERC, CURB-65, HEART, encoded with citations |
| How strongly | Likelihood ratios against the KB marginal |
| Gold diagnoses | DDXPlus, reported as *agreement with the DDXPlus differential* |
| Which cases are genuinely uncertain | Gold-differential confidence in the true diagnosis, as an abstention proxy |
| Grading the evidence chain | **No substitute.** This capability is not delivered. |

**The mandated ontology layer, delivered in halves.** Section 6 specifies UMLS,
SNOMED CT and RxNorm for "authoritative concepts and relations".

*Concepts* — delivered, 34 of 35. `scripts/build_umls_map.py` resolves 26 of
the 27 findings and all 8 conditions to a UMLS CUI and a SNOMED CT code,
restricted to SNOMEDCT_US and to the semantic types appropriate to each kind of
concept. Every concept carries them alongside its HPO identifier, and
`Vocabulary.umls_coverage` reports how many are bound rather than assuming all
are. HPO is kept as well rather than replaced: it supplies the hierarchy this
vocabulary already uses, and SNOMED CT is what a clinical system would
exchange.

The one unresolved finding is `imaging:ctpa_filling_defect`, and it is absent
rather than unexamined: UMLS has filling-defect concepts for the ureter, kidney
and bladder but none naming the pulmonary artery, and the generic concept would
not say which vessel while the disease concept would code the conclusion as
though it were the sign supporting it. Two further resolutions are
approximations recorded as such in the script — `Musculoskeletal immobility` is
not the Wells criterion (immobilisation ≥3 days, which SNOMED has no single
concept for), and `Leukocytosis` is a disorder concept standing in for a
laboratory result.

*Relations* — not delivered, and the reason is measured. `scripts/umls_probe.py`
found that for these eight conditions UMLS relations are overwhelmingly
translations, ICD crosswalks and MedDRA groupings, and its one clinically
meaningful label mixes symptoms, risk factors and treatment complications
without marking which is which. Supports/contradicts edges are therefore
hand-encoded from Wells, PERC, CURB-65 and HEART with citations.

Those edges are now also a graph. `dxagent.ontology` builds one whose nodes are
findings and causes and whose edges are labelled supports/contradicts, with
`scripts/build_ontology.py` to query it, export JSON, or emit Graphviz DOT. It
adds no clinical knowledge: every edge is the same likelihood ratio
`InMemoryKnowledgeBase.evidence_split` already computes on demand, so the graph
and the reasoner cannot disagree — there is a test asserting exactly that. What
it adds is a form that can be walked and drawn rather than only executed.

Two qualifications travel with it. **It is association, not causation**, and
the brief's word "causal" would be wrong: an edge says a finding is more
expected under a cause than in the population this knowledge base describes.
Commit `0624bdf` exists specifically to stop risk factors resolving to the
disorders they cause, and calling this graph causal would license inferences
it cannot support. **And most edges rest on invented numbers** — 64 of 84 at
the default threshold — so every edge carries its provenance tier, the summary
reports the split, and the DOT export draws invented edges dashed. A dense,
confident diagram built from guesses misleads more efficiently than the guesses
did on their own.

The graph immediately showed something the likelihood table did not: at the
default threshold **asthma exacerbation has no supporting edges at all**. Every
finding it carries is claimed at least as strongly by something else, almost
always COPD. The system still reaches the diagnosis — absent findings and the
shape of the whole posterior carry information a per-edge threshold cannot see
— but no single finding argues *for* asthma above this population's base rate.
That is the content of this document's own justification for including asthma,
which calls separating it from COPD "a real clinical task", now visible as an
empty row.

*RxNorm* — not applicable. It names drugs and their ingredients; the vocabulary
here holds symptoms, signs, laboratory results and imaging findings, and
nothing in scope reasons about medication. That is a scope fact rather than a
substitution.

**The likelihood tables are still mostly invented — 108 of 177.** UMLS was
probed and found unusable for this purpose — mostly translations and billing
crosswalks, and its one clinical relation mixes symptoms, risk factors and
treatment complications without distinguishing them. HPO supplies concepts but
no disease-to-finding edges outside rare Mendelian disease. Sourcing has
therefore proceeded one number at a time, from three sources of descending
strength: 41 frequencies from published cohorts (Miniati 2012, Zègre-Hemsey
2018, Noorain 2016, StatPearls), 10 counted in the DDXPlus simulator, and 13
converted from
Merck Manual narrative phrases through the fixed rubric in
`dxagent.provenance`.

`scripts/sensitivity.py` reports which likelihoods actually change a diagnosis
when perturbed, and that count is not stable — it was 30 before correlation
weighting and the mandatory-workup fix, and a current run gives 3 with
correlation on, 1 with it off. Rerun it before choosing what to source next
rather than trusting a figure written down here.

That list is a sourcing priority and nothing more. It is not a list of the
numbers that matter, and the distinction is easy to get backwards:
`--ablate` deletes the invented likelihoods outright and lets the knowledge
base fall back to its own marginals, and the fixture set drops from 10/10 to
1/10. The invented numbers are individually insensitive and collectively
load-bearing. Whatever else is said about this knowledge base, it cannot be
said that most of it is padding.

**Most of the remaining 89 are not sourceable from a reference text at all.**
A textbook chapter describes what a disease presents with — five to eight
notable findings. The knowledge base is a full grid of 8 diseases against every
finding in the vocabulary, so it contains cells like P(raised BNP | asthma
exacerbation) and P(CTPA filling defect | pneumonia). No chapter on asthma
states how often BNP is raised in asthma, because no one writes that sentence.
A scan of all eight relevant Merck chapters for the unsourced cells returns
roughly thirty candidate sentences, and most of those are false matches — drug
adverse effects, a different disease bleeding in from a neighbouring chapter.
The realistic remaining yield from this source is single figures. The honest
description of the rest is that they are structural priors chosen to make the
model behave sensibly, not measurements, and calling them anything else would
be the misconduct this project's provenance tiers exist to prevent.

**The disease priors are now derived from published aetiology, with wide
bands.** They were invented for most of this project. Until recently they were also invisible to the audit —
`dxagent.provenance` covered `P(finding | disease)` only, so a reader saw
a sourced percentage with no hint that the priors were not part of it at
all. They now carry the same provenance tier as the likelihoods and are
reported on their own line (`8 disease priors: 8 sourced, 0 invented`),
counted separately rather than folded in, because merging them would let a
sourced likelihood table hide eight unsourced priors inside one flattering
percentage.

DDXPlus cannot supply them: its own paper states that generation rates were
capped into a 10–100% band to avoid a dataset "dominated by only a few
pathologies", so its patient counts are a rebalancing artefact rather than
epidemiology. What does supply them is StatPearls' symptom-side articles,
which give presentation-conditional aetiology — of patients arriving with
dyspnoea, or with chest pain, what fraction have each cause. See `_PRIORS` in
`datasets/fixtures.py` for the derivation and the three assumptions it rests
on, and DESIGN.md for what it changed.

The bands are wide on purpose. The two source presentations disagree twentyfold
on acute coronary syndrome — 3% of dyspnoea, 63% of chest pain, renormalised —
and this project's presentation is both. A single confident prior there would
be the dishonest option.

**The Merck Manual is cited, not indexed wholesale.** Nine sentences were read
from the 19th edition, converted to likelihoods, and also embedded into the
retrieval corpus so a hypothesis can be grounded in the sentence rather than
only in the number derived from it (`dxagent.merck` is the single source both
read from). Whole chapters were deliberately not chunked and indexed: that
would mean embedding and serving pages of a purchased, copyrighted textbook
rather than citing specific claims from it. Section 4 of the brief asks for a
searchable knowledge base built from that text; what is delivered is a
searchable corpus of 46 passages — 32 from the four encoded decision rules,
14 from the manual — which is narrower than the brief's wording and is a
scope decision rather than an unfinished task.

**MIMIC-IV was not pursued.** Credentialing takes weeks and would clear after
the point of use.

**AgentClinic and the mandated model were not exercised.** Section 6 names
AgentClinic as the evaluation harness and a Claude Opus 4.x-class model as the
required stack; neither has run against this system, for two different
reasons rather than one shared excuse. AgentClinic needs an OpenAI or
Replicate API key to drive its own doctor/patient simulation loop, and using
it properly means forking its code to accept this project's agent in place of
its own — not a configuration change, and not attempted. The mandated model
has not run live at all: no API credit was available for the project's
duration. A local substitute has (Ollama, `llama3.2:3b`, through an
OpenAI-compatible adapter that needs no key), for the single-pass baseline
only; WRITEUP.md §7 carries the numbers and the caveats. The LLM layer
(`dxagent.llm.LLMClient`) has been a swappable
Protocol since week 1 for exactly this reason — `AnthropicLLM`'s default
model id now points at the current Opus-tier release as the honest current
equivalent of what the brief specifies, but that default has never actually
been called. The brief's required single-pass baseline is meant to be an LLM
prompted directly on the case; without a key it falls back to a single-pass
KB reasoner instead and says so in its own output rather than silently
reusing the label, but every result in this project still comes from the
Bayesian proposer and that KB fallback, never from the mandated model itself.

## 5. What "done" means for this presentation

- A ranked differential over the eight causes, each hypothesis carrying
  supporting and contradicting evidence with citations.
- Calibrated confidence, reported as ECE and Brier, with a coverage–risk curve.
- An abstain/escalate pathway that defers with a stated reason and a named
  unresolved question.
- Evaluation against both required baselines — a single-pass model and a
  retrieval-only ranker.

All four are implemented. Their *quality* is bounded by the invented
likelihoods and the invented priors, which together are the project's principal
limitation and are measured rather than asserted: `dxagent.provenance` reports
the likelihood split per number, and the priors are flagged above as the part
that measurement does not yet reach.
