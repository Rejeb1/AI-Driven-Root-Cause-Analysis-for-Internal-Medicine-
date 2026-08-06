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

**These are invented.** See *Constraints*.

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

**The likelihood tables are invented.** UMLS was probed and its relations found
unusable for this purpose — mostly translations and billing crosswalks, and its
one clinical relation mixes symptoms, risk factors and treatment complications
without distinguishing them. HPO supplies concepts but no disease-to-finding
edges outside rare Mendelian disease. `scripts/sensitivity.py` identifies which
30 of the 135 likelihoods actually change a diagnosis; those are the ones worth
sourcing first.

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
likelihoods, which is the project's principal limitation and is measured rather
than asserted.
