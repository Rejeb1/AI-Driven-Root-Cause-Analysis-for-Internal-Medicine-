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
| Pulmonary embolism | sudden onset, pleuritic pain, hypoxia, raised D-dimer, CTPA filling defect | productive cough, fever |
| Pneumonia | fever, productive cough, crackles, raised WCC, consolidation on CXR | sudden onset, orthopnoea |
| Acute coronary syndrome | exertional chest pain, ECG ST changes, raised troponin | fever, pleuritic pain |
| Acute pulmonary oedema | orthopnoea, raised JVP, raised BNP, pulmonary oedema on CXR | fever |
| COPD exacerbation | smoking history, wheeze, reduced breath sounds | sudden onset |
| Asthma exacerbation | wheeze, reversible symptoms | smoking history, fever |
| Pericarditis | pleuritic pain, friction rub | raised BNP, orthopnoea |
| Panic attack | palpitations, absence of hypoxia | hypoxia, ECG changes, raised troponin |

**Most of these are invented, and which ones are not is recorded.** 24% of
the 135 likelihoods now carry a citation — 15 from DDXPlus, 14 from the Merck
Manual's narrative text, 4 from a published cohort. `scripts/sensitivity.py`
reports the split and `dxagent.provenance` tracks it per number. See
*Constraints*.

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

Those edges are also not a graph. Section 4 asks for an ontology whose nodes
are findings and causes and whose edges are labelled supports/contradicts;
what exists computes that relation on demand — `InMemoryKnowledgeBase.
evidence_split` scores each observed finding by likelihood ratio against the
knowledge-base marginal and returns the two lists with citations — rather than
storing it as a traversable structure. The reasoning is auditable and every
edge is cited, which is what the requirement is for, but there is no object a
reader can walk or draw, and the word *causal* would be wrong for it in any
case: what is encoded is measured association, and commit `0624bdf` exists
specifically to stop risk factors resolving to the disorders they cause.

*RxNorm* — not applicable. It names drugs and their ingredients; the vocabulary
here holds symptoms, signs, laboratory results and imaging findings, and
nothing in scope reasons about medication. That is a scope fact rather than a
substitution.

**The likelihood tables are still mostly invented — 102 of 135.** UMLS was
probed and found unusable for this purpose — mostly translations and billing
crosswalks, and its one clinical relation mixes symptoms, risk factors and
treatment complications without distinguishing them. HPO supplies concepts but
no disease-to-finding edges outside rare Mendelian disease. Sourcing has
therefore proceeded one number at a time, from three sources of descending
strength: 4 frequencies from a published cohort (Miniati 2012), 15 counted in
the DDXPlus simulator, and 9 converted from Merck Manual narrative phrases
through the fixed rubric in `dxagent.provenance`.

`scripts/sensitivity.py` reports which likelihoods actually change a diagnosis
when perturbed, and that count is not stable — it was 30 before correlation
weighting and the mandatory-workup fix, and a current run gives 4 with
correlation on, 16 with it off. Rerun it before choosing what to source next
rather than trusting a figure written down here.

That list is a sourcing priority and nothing more. It is not a list of the
numbers that matter, and the distinction is easy to get backwards:
`--ablate` deletes the invented likelihoods outright and lets the knowledge
base fall back to its own marginals, and the fixture set drops from 9/10 to
3/10. The invented numbers are individually insensitive and collectively
load-bearing. Whatever else is said about this knowledge base, it cannot be
said that most of it is padding.

**Most of the remaining 102 are not sourceable from a reference text at all.**
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

**The disease priors are invented and are not even tracked.** All eight
prevalences are guesses, and unlike the likelihoods they have no provenance
tier: `dxagent.provenance` covers `P(finding | disease)` only. An invented
number that the audit cannot report as invented is worse than one it can, and
this is the clearest remaining instance of it in the project.

**The Merck Manual is cited, not indexed wholesale.** Nine sentences were read
from the 19th edition, converted to likelihoods, and also embedded into the
retrieval corpus so a hypothesis can be grounded in the sentence rather than
only in the number derived from it (`dxagent.merck` is the single source both
read from). Whole chapters were deliberately not chunked and indexed: that
would mean embedding and serving pages of a purchased, copyrighted textbook
rather than citing specific claims from it. Section 4 of the brief asks for a
searchable knowledge base built from that text; what is delivered is a
searchable corpus of 41 passages — 32 from the four encoded decision rules,
9 from the manual — which is narrower than the brief's wording and is a scope
decision rather than an unfinished task.

**MIMIC-IV was not pursued.** Credentialing takes weeks and would clear after
the point of use.

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
