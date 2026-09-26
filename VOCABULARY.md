# Findings vocabulary (HPO concept layer)

Deliverable for the KB step: a citable, versioned identifier for every clinical
finding the system reasons over. Built from the Human Phenotype Ontology,
release `hp/releases/2026-06-23`.

```bash
python scripts/build_vocabulary.py          # build, verify, freeze
python scripts/build_vocabulary.py --refresh  # re-download hp.obo
```

Output: `data/vocabulary.json` — pin this in version control.

## What HPO does and does not provide

| | Status |
|---|---|
| **Nodes** — findings with stable ids, synonyms, hierarchy | Works. 31/38 concepts mapped, including labs and ECG. **28/38 now carry a UMLS CUI.** The two added after the real-case second pass — `imaging:pericardial_effusion` (HP:0001698) and `exam:ecg_pr_depression` (no HPO term) — were the last ones blocked on a key; both resolved once one was available, and both needed a manual pin rather than the plain search: the generic concept for each (`C0031039` "Pericardial effusion", `C0429068` "PR depression") is rooted in MTH, not SNOMEDCT_US, so the SNOMEDCT_US-restricted search either skipped to a wrong specific subtype (a noninflammatory-effusion concept, the wrong clinical sense for a pericarditis workup) or found nothing at all. `--suggest` surfaced the generic concept both times; pinned in `VERIFIED_CUI` in `build_umls_map.py`, same mechanism as the three pins already there. |
| **Edges** — disease→finding relations | **Unusable for this project.** |

The edge problem is not a coverage gap to be patched; it is a domain mismatch.
The HPO annotation file distributes rare Mendelian disease only (OMIM, Orphanet,
DECIPHER — 12,956 diseases). Measured against the 2026-06-23 release:

- pulmonary embolism: **0** entries
- myocardial infarction / acute coronary syndrome: **0**
- angina: **0**
- urinary tract infection: **0**
- "pneumonia": 7 entries, all rare variants (e.g. lymphoid interstitial pneumonia)
- "diabetes mellitus": 31 entries, all neonatal or monogenic forms
- "atrial fibrillation": 17 entries, all familial forms

There is also no contradicts layer: 727 of 285,598 annotations carry the `NOT`
qualifier, a quarter of one percent.

Disease→finding edges must therefore come from elsewhere and be sourced
individually. That remains the open problem in this project.

## Coverage

| Modality | Mapped |
|---|---|
| history | 15/19 |
| exam | 7/7 |
| ECG | 3/3 |
| lab | 4/4 |
| imaging | 1/3 |

Six concepts are deliberately unmapped rather than approximated:

- `smoking_history`, `recent_immobility` — exposures, not phenotypes. HPO is
  the wrong ontology; these need their own vocabulary.
- `sudden_onset` — temporal qualifier; belongs in an onset field.
- `calf_tenderness` — nearest HPO term is morphological, not a tenderness sign.
- `imaging:cxr_consolidation`, `imaging:ctpa_filling_defect` — radiological
  descriptors; RadLex is the right source.

A wrong normalisation is worse than a missing one: it silently merges findings
that must stay distinct, and nothing downstream can detect it.

## Two traps, both hit during construction

**1. Adjacent HPO ids are often antonyms.** This table initially mapped
`productive_cough` to HP:0031246, which is *Nonproductive cough*. Coverage
reported 100%, the id format was valid, and the posterior would have been
corrupted on every case with a cough. It was caught only by printing every
resolved label and reading them. The build script now prints all labels and
flags negation prefixes; `test_mapped_names_are_not_negations` guards the class.

**2. Substring matching produces confident nonsense.** Searching `rale` returns
`P mitrale`, an ECG finding. Mapping is therefore an explicit curated table with
a per-row note, never fuzzy matching — which is also what makes it auditable,
since a reviewer can disagree with one row rather than the whole table.

## Known soft spots

Flagged rather than silently accepted:

- `exertional_chest_pain` → generic *Chest pain*. HPO has no exertional
  qualifier, so the exertional character is **lost** and must be carried in
  `Finding.raw`. This is real information loss on a clinically important
  distinction.
- `lab:raised_bnp` → *Abnormal circulating BNP concentration*. No
  raised-specific child exists, so direction is lost.
- `exam:ecg_st_changes` → parent *Abnormal ST segment*. Elevation (HP:0012251)
  and depression (HP:0012250) are distinct children carrying different
  implications and should be separated once the case representation allows.
- `exam:reduced_breath_sounds` → parent *Abnormal breath sound*; no specific
  reduced/diminished term exists.

## Why this is a stand-in

HPO is filling the slot UMLS/SNOMED was meant to occupy. It is a reasonable
substitute for findings, but it is not a clinical terminology and it will not
cover diagnoses, procedures, or drugs. The `Concept` record is shaped to take a
second identifier per concept, so adding UMLS CUIs was an additive change, not
a rewrite -- which is what actually happened once a UTS key was available: a
free licence, not the blocker the "critical path" framing above implied. See
the nodes row above for what changed and how.
