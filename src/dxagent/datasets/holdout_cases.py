"""A second, separate set of real patients, selected by a rule fixed in advance.

Why a second set
----------------
The 25 cases in ``real_cases`` have been looked at through dozens of
sourcing passes; every design choice since has been made with their results
in view, so they are no longer a clean test. They are also too few to fit
anything: the temperature scaler needs 30 independent labelled cases and the
gate thresholds have never been fitted at all, because fitting them on the
only real cases would be tuning on the test set.

This set exists to fix both, and it is kept apart from ``REAL_CASES`` on
purpose -- never merged into it, never used by any test that pins a
behaviour, never used to choose a likelihood.

The rule, committed before any search was run
---------------------------------------------
Frozen model: the knowledge base and gate as of the commit that added this
file with an empty ``HOLDOUT_CASES``. Any later change to either is reported
against both the frozen run and the new one, never silently replacing the
first.

Search, one per diagnosis, in PubMed, relevance order, each ANDed with
``case reports[pt] AND free full text[sb] AND english[la] AND 2015:2026[dp]``:

- pulmonary_embolism: ``"pulmonary embolism"[ti]``
- acute_coronary_syndrome: ``("myocardial infarction"[ti] OR "acute coronary syndrome"[ti])``
- community_acquired_pneumonia: ``("community-acquired pneumonia"[ti] OR "pneumonia"[ti])``
- acute_pulmonary_oedema: ``("pulmonary edema"[ti] OR "pulmonary oedema"[ti] OR "acute heart failure"[ti])``
- copd_exacerbation: ``("COPD"[ti] OR "chronic obstructive"[ti]) AND exacerbation[tiab]``
- asthma_exacerbation: ``asthma[ti] AND (exacerbation[tiab] OR "status asthmaticus"[tiab] OR "acute severe"[tiab])``
- pericarditis: ``pericarditis[ti]``
- panic_attack: ``panic[ti]``

Walk each list in order, at most the first 40 results, and take the first
four that satisfy all of:

1. an adult (18 or over) with a full text retrievable from PubMed Central;
2. the confirmed final diagnosis is the searched one of the eight, and the
   presentation includes breathlessness, chest pain or palpitations;
3. not mixed: no concurrent acute diagnosis outside the eight treated as
   co-primary (sepsis from another source, COVID-19, cancer presenting at
   the same time, trauma, poisoning), and not a variant that makes the label
   a judgement call (spontaneous coronary artery dissection, vasospasm,
   Takotsubo; non-cardiogenic oedema; septic, tumour, fat or amniotic
   embolism; purulent, tuberculous or post-operative pericarditis);
4. at least six findings in this project's vocabulary stated explicitly,
   extracted under the same discipline as ``real_cases`` (explicit statements
   only; silence stays unknown; close-but-not-quite is left out);
5. not already in ``REAL_CASES`` and not a candidate that set records as
   rejected.

Every candidate walked past is recorded below with its reason. A diagnosis
that cannot fill four within 40 results is recorded short, not topped up
from another diagnosis -- COPD and asthma are expected to fall short, for the
reasons ``real_cases`` already records.

What the walk found (searched 2026-09-29, ranks are PubMed relevance)
-------------------------------------------------------------------
21 cases, not 32: the rule was followed and the shortfall recorded, not
topped up. Taken: PE 4, ACS 3, pneumonia 1, oedema 4, COPD 4, asthma 4,
pericarditis 0, panic 1. The set is below 30, so calibration stays unfitted
(see "Use" below). The pattern in the shortfalls is itself a finding: case
reports are written about the unusual, so the commonest forms of pneumonia
and pericarditis are almost never reported alone.

- PE, taken ranks 1, 7, 13, 27 (first patient of two). Skipped: 2 trauma;
  3 concurrent cancer; 4 and 5 fewer than six findings; 6 inpatient during
  an ulcerative colitis flare; 8 IgA vasculitis co-primary; 9 presented
  with arm pain; 10 paediatric; 11, 14, 18, 20 COVID-19; 12 mixed with CMV
  pericarditis; 15 no PMC full text; 16 not a PE; 17 myeloma presenting at
  the same time; 19 tumour; 21 chronic thromboembolic; 22 new
  cardiomyopathy as co-primary; 23 peri-operative; 24 foreign-body
  embolism; 25 hydatid; 26 near-syncope without breathlessness, chest pain
  or palpitations.
- ACS, taken ranks 6, 14, 21; 3 of 4 within 40. Skipped: 1, 3, 5, 7, 20
  mimics, not ACS; 2 thyrotoxicosis co-primary; 4 five findings at
  presentation (its pericarditis came the next day); 8, 27, 30, 33, 35, 37
  fewer than six findings; 9 no ACS at the index visit (troponin negative,
  stents patent); 10 embolic from valve thrombosis; 11 diabetic
  ketoacidosis; 12, 26, 36, 38 no PMC full text; 13 presented with
  pharyngeal pain; 15, 31 abstract only; 16 embolic (Takayasu); 17
  post-operative, presented with delirium; 18 neonate; 19 tumour embolism;
  22 antiphospholipid syndrome; 23, 40 toxic (cannabis, energy drink); 24
  pathology teaching case; 25 a late sequela; 28 non-atherosclerotic
  vasculopathy; 29 purulent pericarditis; 32 COVID-19; 34 chemotherapy
  toxicity; 39 five findings and embolic-versus-thrombotic unresolved.
- Pneumonia, taken rank 29; 1 of 4 within 40. Skipped: 2 pulmonary
  actinomycosis (chronic granulomatous, not acute CAP); 3 COPD with a
  hospital-course infection; 6 tuberculosis; 17 no breathlessness or
  chest pain (gastrointestinal onset); 23, 35, 39 fewer than six findings;
  28 denied breathlessness and chest pain; 34 vasculitis with HIV; every
  other rank non-infectious (eosinophilic, organising, interstitial,
  lipoid, hypersensitivity), opportunistic in AIDS or immunosuppression,
  COVID-19, paediatric or veterinary, or without a PMC full text.
- Oedema, taken ranks 13, 18, 25, 29. Skipped: 2 aged 14; 10
  electrophysiology report with no clinical presentation; 19 asymptomatic;
  5, 21 drug-induced; every other rank before 29 non-cardiogenic
  (re-expansion, negative-pressure, neurogenic, high-altitude,
  peri-operative, toxin) or without a PMC full text.
- COPD, taken ranks 12, 15, 16, 22. Skipped: 1, 17, 18, 19 already
  rejected in ``real_cases``; 2 hypercapnic cerebral oedema co-primary; 3
  an image report with fewer than six findings; 4 neurological
  presentation weeks after discharge; 5 foreign-body aspiration; 6, 8, 20
  not case reports; 7 image report, no text; 9 no PMC full text; 10 stable
  COPD found at a routine visit; 11 presented with syncope; 13 myxoma
  mimic; 14 pneumonia co-primary; 21 PRES.
- Asthma, taken ranks 2, 24, 25, 39. Skipped: 1 hypereosinophilia with
  infection; 3 tracheal stenosis; 4, 5 drug reaction co-primary; 6
  broncholithiasis; 7 already in ``real_cases``; 8 alveolar haemorrhage; 9
  Takotsubo; 10, 19 abstract only; 11 tracheobronchomalacia; 12
  hereditary haemorrhagic telangiectasia; 13, 14, 20, 21 no PMC full text;
  15 pneumomediastinum co-primary; 16 not asthma (the oedema case taken
  above); 17 dermatology drug reaction; 18, 28, 38 COVID-19; 22 lung
  herniation; 23, 29 under 18; 26, 30, 35, 37 chronic management; 27
  ketoacidosis; 31 already rejected in ``real_cases``; 32 group data only;
  33 cocaine; 34 concurrent urinary infection; 36 vocal cord dysfunction.
- Pericarditis, none of 40. Every rank is tuberculous, constrictive,
  purulent, cholesterol, IgG4, rheumatoid, vasculitic, uraemic, malignant,
  COVID-19-related, post-traumatic, a mimic, or a review; the two acute
  idiopathic-type candidates (11, minoxidil-associated; 39, relapsing)
  record fewer than six findings at any one presentation.
- Panic, taken rank 20; 1 of 4 within 40. Skipped: 1, 3, 14, 26, 28, 32,
  39 no PMC full text; 2, 6, 7, 10, 16, 17, 19, 23, 27, 30, 31, 33, 40
  psychiatric or outpatient course without an acute presentation of this
  kind; 5 and 24 a co-primary consequence named in the title
  (rhabdomyolysis, mediastinal emphysema) -- the same treatment given to
  asthma's rank 15; 8 functional neurological disorder; 9, 12, 13, 15, 21,
  22, 34, 36 an organic mimic; 11 already rejected in ``real_cases``; 18
  final diagnosis PE; 25, 35 already in ``real_cases``; 29 final diagnosis
  HOCM; 37 toxic; 38 five findings.

Amendment 1 -- a second wave and a fitting plan, committed before its search
------------------------------------------------------------------------------
Wave 1 fell short in four diagnoses. Wave 2 fills only those, with new
title queries (same filters, same inclusion rules 1-5, 40 results each, in
relevance order), skipping anything already in either set or already walked
past in wave 1, and taking at most the shortfall:

- pericarditis, 4: ``("acute pericarditis"[ti] OR "idiopathic pericarditis"[ti] OR "viral pericarditis"[ti])``
- community_acquired_pneumonia, 3: ``("pneumococcal"[ti] OR "Streptococcus pneumoniae"[ti] OR "Legionella"[ti] OR "Mycoplasma pneumoniae"[ti] OR "lobar pneumonia"[ti]) AND pneumonia[tiab]``
- panic_attack, 3: ``"panic attack"[tiab] AND ("emergency"[tiab] OR "chest pain"[tiab] OR palpitations[tiab])``
- acute_coronary_syndrome, 1: ``("NSTEMI"[ti] OR "non-ST-elevation"[ti] OR "STEMI"[ti] OR "ST-elevation myocardial infarction"[ti])``

The frozen model stays the one named above; wave-2 cases get their own first
run, reported apart from wave 1 as well as pooled.

If the pooled set reaches 30, two things are fitted on it -- never on the
original 25 -- and then evaluated on the original 25:

1. Temperature. ``TemperatureScaler.fit`` on (the loop's final differential,
   the true label) for every holdout case, on its own grid. Adopted as the
   shipped default only if it lowers the negative log-likelihood of the true
   label on the original 25.
2. The confidence threshold. On the holdout, the lowest ``min_confidence``
   in {0.50, 0.55, ..., 0.80} with zero wrong commits (the shipped 0.65 if
   none is lower). Adopted only if it adds no wrong commit on the original 25.

What wave 2 found (searched 2026-09-29): 7 more, pooled 28 -- still under
30, so nothing is fitted and the fitting plan below does not run.

- Pericarditis, taken ranks 5, 22, 25, 38. Skipped: 1 an MI; 2, 15 after
  procedures for cancer; 3, 17, 18, 28 mimics or dissection; 4, 12, 13,
  23, 35 no PMC full text; 6 leukaemia differentiation syndrome; 7
  bronchogenic cyst; 8 after catheter ablation; 9 drug with ulcerative
  colitis; 10 gunshot; 11, 21, 36 COVID-19; 14 concurrent liver injury on
  immunosuppression; 16 chemotherapy; 19 pacing-lead perforation; 20
  filter strut; 24 anorexia with pneumomediastinum; 26 Salmonella
  aneurysm; 27, 33 presented as cholecystitis; 29 attributed to a first
  e-cigarette (exposure-attributed, as ACS ranks 23 and 40 in wave 1); 30
  after stem-cell transplant; 31 oesophageal perforation; 32 vaccine
  myopericarditis; 34 atrial flutter co-primary; 37 too thin at any one
  presentation.
- Pneumonia, taken ranks 6, 23; 2 of 3 within 40. Skipped: 5 ARDS with a
  possible second (fungal) infection, fewer than six findings; 18 walked
  past in wave 1; 25 no breathing symptoms; 26 lymphoma on chemotherapy;
  31 kidney transplant on immunosuppression; 32 hairy-cell leukaemia; 33
  tachypnoea and hypoxia but no breathlessness reported; every other rank
  not a pneumonia presenting to hospital (meningitis, joint, eye, skin,
  endocarditis, vaccine), COVID-19, paediatric, nosocomial,
  immunosuppressed, or with a co-primary complication.
- Panic, none: the query returned 10 results, of which 1, 6, 9 were already
  handled in wave 1, 2 is hyperventilation-induced hypophosphataemia (the
  co-primary rule, as wave 1 rank 5), and the rest are not panic attacks
  (seizure, Wilson's disease, vasospasm, cannabinoid intoxication, thyroid
  tumour, dystonia).
- ACS, taken rank 9. Skipped: 1, 3, 5, 6, 7, 8 mimics; 2 three findings;
  4 carbon monoxide poisoning.

Not fitted, on purpose: the red-flag tolerance (10%), which decides how much
residual probability of a time-critical cause is acceptable. That is a
clinical policy with a cost on both sides, not a parameter to optimise
against 30 case reports.

Use, fixed in the same commit
-----------------------------
First, the frozen model runs on this set once and the result is reported as
it comes out: a fresh test, the first one this project has had since its
real cases stopped being unseen. Second, and only if the set reaches 30
independent cases, it becomes the calibration set: the temperature is
fitted here and evaluated on the original 25, which were never used to fit
it. Below 30, calibration stays unfitted and that is reported.
"""

from __future__ import annotations

from ..environment import Case

HOLDOUT_CASES: tuple[Case, ...] = (
    # ---- pulmonary embolism ------------------------------------------------
    # PMC12702526 -- bilateral PE presenting with complete heart block.
    Case(
        case_id="ho-12702526",
        presenting_complaint=(
            "35-year-old man, three days of presyncope with intermittent "
            "atypical chest pain and breathlessness"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "smoking_history": True,  # "tobacco use for approximately 10 years"
            # "did not report any overt syncope or palpitations"
            "palpitations": False,
            "exam:tachycardia": False,  # heart rate 28 (complete heart block)
            "exam:hypoxia": False,  # "oximetry of 98% on room air"
            "lab:raised_d_dimer": True,  # 0.643 mg/L, reference < 0.5
            # CT-PE protocol: segmental "filling defect" on both sides
            "imaging:ctpa_filling_defect": True,
            # "high-sensitivity cardiac troponin I and NT-pro-brain
            # natriuretic peptide were also normal"
            "lab:raised_troponin": False,
            "lab:raised_bnp": False,
            "lab:raised_wcc": False,  # "complete blood count ... normal"
        },
        vocabulary=frozenset(),
    ),
    # PMC7249768 -- PE with hyperhomocysteinaemia in a young man.
    Case(
        case_id="ho-7249768",
        presenting_complaint=(
            "24-year-old man, three weeks of worsening bilateral pleuritic "
            "chest pain, exertional breathlessness, one episode of haemoptysis"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "pleuritic_pain": True,
            "sudden_onset": False,  # "three weeks of worsening"
            "smoking_history": True,  # "daily tobacco ... use"
            "exam:hypoxia": True,  # "92% oxygen saturation on room air"
            # "The remainder of his exam, including lung examination, was
            # unremarkable". Heart rate 98 but the ECG "showed sinus
            # tachycardia": contradictory, so tachycardia is left out.
            "exam:crackles": False,
            "lab:raised_troponin": False,  # "a normal troponin"
            "lab:raised_wcc": False,  # CBC: "no other abnormalities"
            "imaging:cxr_consolidation": False,  # "chest x-ray showed no abnormalities"
            "imaging:cxr_pulmonary_oedema": False,
            "lab:raised_d_dimer": True,  # 1,371 ng/mL
            # BNP 288 pg/mL is labelled "(normal range)" by the report and is
            # above this project's 100 pg/mL: contradictory, left out. The
            # angiography type is not stated, so no CTPA finding either.
        },
        vocabulary=frozenset(),
    ),
    # PMC8302121 -- massive PE with cardiogenic shock; known myeloma on
    # lenalidomide (a treated background condition, not presenting now).
    Case(
        case_id="ho-8302121",
        presenting_complaint=(
            "72-year-old woman on lenalidomide for myeloma, 36-48 hours of "
            "rapidly worsening breathlessness, now at rest, no chest pain"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "dyspnoea_at_rest": True,  # "significant dyspnea even at rest"
            "pleuritic_pain": False,  # "She had no significant chest pain"
            "exertional_chest_pain": False,
            "exam:tachycardia": True,  # 118 beats/min
            "exam:hypoxia": True,  # "oxygen saturation in the low 80s"
            "exam:crackles": False,  # "lung fields were clear to auscultation"
            # "Chest radiography showed no acute cardiopulmonary disease"
            "imaging:cxr_consolidation": False,
            "imaging:cxr_pulmonary_oedema": False,
            "lab:raised_troponin": True,  # "mild troponin T elevation, 0.19 ng/ml"
        },
        vocabulary=frozenset(),
    ),
    # PMC4966210, case 1 -- PE in a high-altitude native soldier.
    Case(
        case_id="ho-4966210",
        presenting_complaint=(
            "29-year-old soldier native to high altitude, 12 hours of left "
            "pleuritic chest pain, breathlessness and one episode of haemoptysis"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "pleuritic_pain": True,
            "dyspnoea_at_rest": True,  # "breathlessness ... since 12 h"
            # "He denies the history of fever, ... calf pain, swelling"
            "fever": False,
            "calf_tenderness": False,
            "leg_swelling": False,
            "smoking_history": True,  # "smoker with pack year of 8.5"
            "exam:tachycardia": False,  # pulse 72/min
            "exam:hypoxia": True,  # "oxygen saturation on room air 88%"
            "exam:crackles": True,  # "coarse crepts in the left infra-axillary area"
            "exam:ecg_st_changes": False,  # "no significant abnormality"
            # "consolidation in left lower zone"
            "imaging:cxr_consolidation": True,
            "lab:raised_wcc": False,  # "normal total leukocyte count"
        },
        vocabulary=frozenset(),
    ),
    # ---- acute coronary syndrome --------------------------------------------
    # PMC9733161 -- silent inferior MI presenting as breathlessness,
    # complicated by ventricular septal rupture.
    Case(
        case_id="ho-9733161",
        presenting_complaint=(
            "55-year-old man who smokes, three days of shortness of breath, "
            "no chest pain"
        ),
        diagnosis="acute_coronary_syndrome",
        features={
            "smoking_history": True,
            "dyspnoea_at_rest": True,  # "shortness of breath for the past 3 days"
            # "No associated symptoms of chest pain were reported"
            "pleuritic_pain": False,
            "exertional_chest_pain": False,
            "exam:hypoxia": False,  # "normal oxygen saturation on room air"
            # "Blood work showed elevated troponin and brain natriuretic peptide"
            "lab:raised_troponin": True,
            "lab:raised_bnp": True,
            "imaging:cxr_pulmonary_oedema": True,  # "chest X-ray showed pulmonary edema"
        },
        vocabulary=frozenset(),
    ),
    # PMC5067724 -- acute inferior MI with biventricular thrombi.
    Case(
        case_id="ho-5067724",
        presenting_complaint=(
            "34-year-old man without risk factors, three days of "
            "breathlessness and substernal chest pain"
        ),
        diagnosis="acute_coronary_syndrome",
        features={
            "dyspnoea_at_rest": True,  # "complaints of shortness of breath"
            "smoking_history": False,  # "no conventional risk factors such as ... smoking"
            "exam:tachycardia": False,  # pulse 66/min
            # "crepitations were heard over both lower lung fields"
            "exam:crackles": True,
            "leg_swelling": False,  # "edema feet were absent"
            "lab:raised_wcc": True,  # "neutrophilic leucocytosis"
            # "ST segment elevation in II, III and aVF"
            "exam:ecg_st_changes": True,
        },
        vocabulary=frozenset(),
    ),
    # PMC8498672 -- anterior STEMI in a smoker; incidental pleuropericardial
    # cyst.
    Case(
        case_id="ho-8498672",
        presenting_complaint="68-year-old man, chronic smoker, two days of chest pain",
        diagnosis="acute_coronary_syndrome",
        features={
            "smoking_history": True,
            "exam:tachycardia": False,  # pulse 79
            "exam:hypoxia": False,  # "96% on room air"
            "exam:friction_rub": False,  # "no murmurs or pericardial rub"
            "exam:crackles": False,  # "lung sounds were normal with no crackles"
            # "ST segment elevation with Q waves in the anterior leads"
            "exam:ecg_st_changes": True,
        },
        vocabulary=frozenset(),
    ),
    # ---- community-acquired pneumonia ---------------------------------------
    # PMC8093108 -- bacteraemic Pasteurella multocida pneumonia.
    Case(
        case_id="ho-8093108",
        presenting_complaint=(
            "79-year-old man, one day of haemoptysis, breathlessness, chest "
            "pain, subjective fever and diarrhoea"
        ),
        diagnosis="community_acquired_pneumonia",
        features={
            "dyspnoea_at_rest": True,
            "fever": True,  # "subjective fever" -- reported counts, per the definition
            "exam:hypoxia": True,  # 93% on room air
            "exam:tachycardia": True,  # 114 beats/minute
            # "diminished breath sounds at bases bilaterally"
            "exam:reduced_breath_sounds": True,
            "lab:raised_wcc": True,  # 15.9 K/mcL
            "imaging:cxr_consolidation": True,  # "right middle lobe infiltrate"
            # "CTA of the chest was negative for PE"
            "imaging:ctpa_filling_defect": False,
        },
        vocabulary=frozenset(),
    ),
    # ---- acute pulmonary oedema ----------------------------------------------
    # PMC12481139 -- pregnant woman with known asthma, first treated as an
    # asthma attack; the cause was severe rheumatic mitral stenosis.
    Case(
        case_id="ho-12481139",
        presenting_complaint=(
            "37-year-old woman, 13 weeks pregnant, known asthma, woke at "
            "night with severe breathlessness"
        ),
        diagnosis="acute_pulmonary_oedema",
        features={
            "dyspnoea_at_rest": True,  # "severe respiratory distress"
            "smoking_history": True,  # "a smoker"
            "exam:tachycardia": True,  # pulse 138
            "exam:hypoxia": True,  # "94% on 5 L of oxygen"
            "wheeze_subjective": True,  # "diffuse wheezing"
            "exam:crackles": True,  # "coarse bilateral basal crackles"
            "exam:raised_jvp": False,  # "jugular venous pressure was not raised"
            "leg_swelling": False,  # "no peripheral oedema"
        },
        vocabulary=frozenset(),
    ),
    # PMC8304534 -- decompensated heart failure from a coronary cameral fistula.
    Case(
        case_id="ho-8304534",
        presenting_complaint=(
            "40-year-old woman, two weeks of increasing breathlessness, "
            "orthopnoea, weight gain and ankle swelling"
        ),
        diagnosis="acute_pulmonary_oedema",
        features={
            "dyspnoea_at_rest": True,
            "orthopnoea": True,
            "leg_swelling": True,  # "+2 bilateral lower limb pitting edema"
            "sudden_onset": False,  # "2-week history of increasing dyspnea"
            "exam:tachycardia": False,  # 87 beats/min
            "fever": False,  # 97.6F
            "exam:hypoxia": False,  # "98% on room air"
            # "decreased breath sounds at both lung bases with crackles superiorly"
            "exam:reduced_breath_sounds": True,
            "exam:crackles": True,
            "lab:raised_bnp": True,  # NT-proBNP 3,110 pg/ml
            "exam:ecg_st_changes": False,  # "without ischemic changes"
            # "findings suggestive of pulmonary edema"
            "imaging:cxr_pulmonary_oedema": True,
        },
        vocabulary=frozenset(),
    ),
    # PMC12212617 -- flash pulmonary oedema, ischaemic cardiomyopathy.
    Case(
        case_id="ho-12212617",
        presenting_complaint=(
            "65-year-old man with hypertension, sudden breathlessness at rest "
            "with a feeling of drowning and a cough with white sputum"
        ),
        diagnosis="acute_pulmonary_oedema",
        features={
            "sudden_onset": True,  # "sudden-onset shortness of breath"
            "dyspnoea_at_rest": True,  # "He was resting at home when..."
            "productive_cough": True,  # "productive cough of whitish sputum"
            # "with no fever, chest pain, or leg swelling"
            "fever": False,
            "pleuritic_pain": False,
            "exertional_chest_pain": False,
            "leg_swelling": False,
            "imaging:cxr_pulmonary_oedema": True,
            "imaging:pericardial_effusion": True,  # "mild pericardial effusion"
        },
        vocabulary=frozenset(),
    ),
    # PMC10313490 -- recurrent pulmonary oedema from transient severe mitral
    # regurgitation; the first of her presentations.
    Case(
        case_id="ho-10313490",
        presenting_complaint=(
            "79-year-old woman, three days of substernal chest pain, "
            "sweating and breathlessness, worse lying flat"
        ),
        diagnosis="acute_pulmonary_oedema",
        features={
            "dyspnoea_at_rest": True,
            # "Her symptoms occurred when lying flat. She started to sleep upright"
            "orthopnoea": True,
            "exam:tachycardia": False,  # 77 beats/min
            "exam:crackles": False,  # "normal breath sounds"
            "leg_swelling": False,  # "lower extremities were without edema"
            "lab:raised_wcc": False,  # "complete blood count ... unremarkable"
        },
        vocabulary=frozenset(),
    ),
    # ---- COPD exacerbation -----------------------------------------------------
    # PMC11940554 -- end-stage COPD exacerbation with hypercapnia and a raised
    # D-dimer.
    Case(
        case_id="ho-11940554",
        presenting_complaint=(
            "76-year-old man with COPD on home oxygen, breathlessness and "
            "cough, saturation 77% on his usual oxygen"
        ),
        diagnosis="copd_exacerbation",
        features={
            "dyspnoea_at_rest": True,
            "exam:hypoxia": True,  # 77% on two litres
            "smoking_history": True,  # "two packs of cigarettes per day for 40 years"
            "fever": False,  # 99.4F
            "exam:tachycardia": False,  # 95 beats per minute
            "wheeze_subjective": True,  # "bilateral wheezing"
            "leg_swelling": False,  # "no signs of fluid overload, such as leg swelling"
            "lab:raised_d_dimer": True,  # 5.58 ug/mL
            "lab:raised_wcc": True,  # WBC 13.4
        },
        vocabulary=frozenset(),
    ),
    # PMC11042794 -- COPD exacerbation with severe pulmonary hypertension
    # (cor pulmonale), never previously tested.
    Case(
        case_id="ho-11042794",
        presenting_complaint=(
            "63-year-old man, long-standing breathlessness now worse, with two "
            "months of swelling of both legs"
        ),
        diagnosis="copd_exacerbation",
        features={
            "dyspnoea_at_rest": True,
            "leg_swelling": True,
            # "an occasional nonproductive cough at baseline"
            "productive_cough": False,
            "exam:hypoxia": True,  # 70% on room air
            "exam:tachycardia": True,  # 112 bpm
            "lab:raised_bnp": True,  # pro-BNP 5,153 pg/ml
            "lab:raised_troponin": True,  # 24 ng/L, normal < 15
            # He smoked marijuana daily and denied tobacco: smoking history
            # is left out rather than read either way.
        },
        vocabulary=frozenset(),
    ),
    # PMC10923375 -- COPD exacerbation with cor pulmonale; the report's
    # subject is a later herpes zoster eruption.
    Case(
        case_id="ho-10923375",
        presenting_complaint=(
            "65-year-old man, 30 years of smoking, chronic productive cough, "
            "breathlessness worse over 15 days, three months of leg swelling"
        ),
        diagnosis="copd_exacerbation",
        features={
            "productive_cough": True,  # "copious amounts of mucoid sputum"
            "smoking_history": True,
            "sudden_onset": False,  # "insidious in onset"
            # "Rest provided relief, while exertion aggravated the symptoms"
            "dyspnoea_at_rest": False,
            "orthopnoea": False,  # "no history of orthopnea"
            "leg_swelling": True,
            "fever": True,  # "mild intermittent fever" -- reported
            "pleuritic_pain": False,  # "There was no history of chest pain"
            "exertional_chest_pain": False,
        },
        vocabulary=frozenset(),
    ),
    # PMC5760872 -- COPD exacerbation with acute hypercapnic respiratory
    # failure managed with NIV-NAVA.
    Case(
        case_id="ho-5760872",
        presenting_complaint=(
            "56-year-old man with COPD, three days of breathlessness and "
            "productive cough, drowsy for one day"
        ),
        diagnosis="copd_exacerbation",
        features={
            "productive_cough": True,  # "cough with expectoration"
            # "no history of fever, chest pain, hemoptysis, orthopnea, pedal edema"
            "fever": False,
            "pleuritic_pain": False,
            "exertional_chest_pain": False,
            "orthopnoea": False,
            "leg_swelling": False,
            "smoking_history": True,  # "a reformed smoker"
        },
        vocabulary=frozenset(),
    ),
    # ---- asthma exacerbation ---------------------------------------------------
    # PMC11524714 -- life-threatening exacerbation of mild asthma.
    Case(
        case_id="ho-11524714",
        presenting_complaint=(
            "19-year-old woman with mild asthma, five days of breathlessness "
            "despite frequent salbutamol"
        ),
        diagnosis="asthma_exacerbation",
        features={
            "dyspnoea_at_rest": True,
            "fever": False,  # "She denied fever"
            "smoking_history": False,  # "She does not smoke or vape"
            "exam:tachycardia": False,  # "her vitals were normal"
            # "complete blood count were reassuring and showed no signs of infection"
            "lab:raised_wcc": False,
            "imaging:cxr_consolidation": False,  # "negative for pneumonia"
            "wheeze_subjective": True,  # "wheezing"
            "exam:reduced_breath_sounds": True,  # "poor aeration"
        },
        vocabulary=frozenset(),
    ),
    # PMC13218743 -- near-fatal asthma after seafood, rescued with VV-ECMO.
    Case(
        case_id="ho-13218743",
        presenting_complaint=(
            "19-year-old woman with asthma, sudden severe breathlessness after "
            "eating seafood"
        ),
        diagnosis="asthma_exacerbation",
        features={
            "sudden_onset": True,
            "dyspnoea_at_rest": True,
            "exam:tachycardia": True,  # "tachycardic"
            "exam:hypoxia": True,  # 64% on room air
            "wheeze_subjective": True,  # "bilateral diffuse wheezing"
            "exam:reduced_breath_sounds": True,  # "markedly reduced air entry"
            # "without evidence of focal consolidation"
            "imaging:cxr_consolidation": False,
        },
        vocabulary=frozenset(),
    ),
    # PMC13175005 -- severe asthma exacerbation managed with NIV.
    Case(
        case_id="ho-13175005",
        presenting_complaint=(
            "18-year-old woman with asthma, breathlessness not relieved by "
            "her inhaler"
        ),
        diagnosis="asthma_exacerbation",
        features={
            "dyspnoea_at_rest": True,
            "exam:hypoxia": True,  # SpO2 89%
            "exam:tachycardia": True,  # 122 beats per minute
            "fever": False,  # "afebrile on arrival"
            "wheeze_subjective": True,  # "wheezing throughout both lung fields"
            "exam:reduced_breath_sounds": True,  # "decreased air entry"
        },
        vocabulary=frozenset(),
    ),
    # PMC9477784 -- asthma exacerbation the day after a COVID-19 booster, in a
    # woman with Marfan syndrome and repaired aorta.
    Case(
        case_id="ho-9477784",
        presenting_complaint=(
            "55-year-old woman, sudden breathlessness the day after a vaccine "
            "booster, unable to lie flat"
        ),
        diagnosis="asthma_exacerbation",
        features={
            "sudden_onset": True,  # "sudden shortness of breath"
            "dyspnoea_at_rest": True,
            "exam:tachycardia": True,  # 112 beats/min
            "fever": False,  # 36.3C
            "exam:hypoxia": True,  # 94% on 10 l/min
            # "unable to lie down and remained in an orthopneic position"
            "orthopnoea": True,
            "wheeze_subjective": True,  # "wheezing in both lungs"
            # WBC 10,240/uL is called elevated by the report but is under this
            # project's ~11,000: left out rather than read either way.
        },
        vocabulary=frozenset(),
    ),
    # ---- panic attack ----------------------------------------------------------
    # PMC4858511 -- panic attacks ten years after heart transplantation.
    Case(
        case_id="ho-4858511",
        presenting_complaint=(
            "22-year-old woman, heart transplant at 12, sudden chest "
            "tightness, palpitations, breathlessness and sweating at night"
        ),
        diagnosis="panic_attack",
        features={
            "sudden_onset": True,  # "suddenly experienced"
            "dyspnoea_at_rest": True,
            "palpitations": True,
            # "cardiac enzymes, and D-dimers were all normal"
            "lab:raised_troponin": False,
            "lab:raised_d_dimer": False,
            "exam:tachycardia": True,  # "heart rate of 110 beats per minute"
        },
        vocabulary=frozenset(),
    ),
    # ======== wave 2 (amendment 1) ========
    # ---- pericarditis ----------------------------------------------------------
    # PMC7279685 -- recurrent pericarditis with human metapneumovirus, a month
    # after a first idiopathic episode; later tamponade.
    Case(
        case_id="ho-7279685",
        presenting_complaint=(
            "26-year-old man, three days of sharp left chest pain worse lying "
            "down and better sitting up, fever, dry cough, breathlessness"
        ),
        diagnosis="pericarditis",
        features={
            "fever": True,  # reported; afebrile on arrival -- reported counts
            "productive_cough": False,  # "dry cough"
            "dyspnoea_at_rest": True,  # "difficulty breathing"
            "exam:tachycardia": True,  # 115 bpm
            "exam:raised_jvp": False,  # "normal neck veins"
            "lab:raised_d_dimer": False,  # 359 ng/ml
            "lab:raised_troponin": False,  # < 0.012 ng/ml
            # WBC 11.0 sits on this project's ~11 cutoff: left out.
        },
        vocabulary=frozenset(),
    ),
    # PMC7489791 -- post-viral pericarditis with a Brugada phenocopy on ECG.
    Case(
        case_id="ho-7489791",
        presenting_complaint=(
            "35-year-old man, one day of substernal chest pain radiating to "
            "the back, worse on inspiration, two weeks after a cold"
        ),
        diagnosis="pericarditis",
        features={
            "pleuritic_pain": True,  # "worsening with inspiration"
            "exam:tachycardia": False,  # "vital signs were unremarkable"
            "lab:raised_wcc": False,  # "normal white blood counts"
            "imaging:cxr_consolidation": False,  # "no evidence of pneumonia"
            # "CT angiography was negative for pulmonary embolus"
            "imaging:ctpa_filling_defect": False,
            "lab:raised_troponin": False,  # two negative assays
            "exam:ecg_st_changes": True,  # "diffuse ST-PR discordance"
            "imaging:pericardial_effusion": False,  # echo "without ... effusion"
        },
        vocabulary=frozenset(),
    ),
    # PMC12240549 -- idiopathic pericarditis (first episode; recurred a year
    # later).
    Case(
        case_id="ho-12240549",
        presenting_complaint=(
            "20-year-old woman, two days of severe sharp substernal chest pain "
            "worse lying back and better leaning forward, chills"
        ),
        diagnosis="pericarditis",
        features={
            "exam:tachycardia": True,  # 124 beats per minute
            "fever": False,  # 98.5F
            "exam:hypoxia": False,  # 99% on room air
            "exam:ecg_st_changes": False,  # "no ST wave changes were observed"
            "lab:raised_d_dimer": True,  # "elevated ... D-dimer levels"
            "imaging:ctpa_filling_defect": False,  # "ruled out with chest CT"
            "imaging:cxr_consolidation": False,  # chest X-ray within normal limits
            "lab:raised_troponin": False,
            "lab:raised_wcc": False,  # 7.2
        },
        vocabulary=frozenset(),
    ),
    # PMC12874572 -- acute pericarditis with an incidental right ventricular
    # outflow tract aneurysm.
    Case(
        case_id="ho-12874572",
        presenting_complaint=(
            "32-year-old man, two weeks of sharp pleuritic anterior chest pain, "
            "worse lying flat, partly relieved sitting forward"
        ),
        diagnosis="pericarditis",
        features={
            "pleuritic_pain": True,
            # "denied ... dyspnea on exertion, palpitations, orthopnea, ... fever"
            "palpitations": False,
            "orthopnoea": False,
            "fever": False,
            "smoking_history": True,  # "10 pack-year smoking history"
            "exam:tachycardia": False,  # 82 bpm
            "exam:hypoxia": False,  # 98% on room air
            "exam:friction_rub": False,  # "There was no pericardial rub"
            "exam:raised_jvp": False,  # "no ... jugular venous distension"
            "lab:raised_wcc": True,  # 11,200/uL
        },
        vocabulary=frozenset(),
    ),
    # ---- community-acquired pneumonia -----------------------------------------
    # PMC9273171 -- Legionella pneumonia during the COVID-19 pandemic (COVID
    # tests negative).
    Case(
        case_id="ho-9273171",
        presenting_complaint=(
            "56-year-old man, four days of fever and shortness of breath, no "
            "cough or chest pain"
        ),
        diagnosis="community_acquired_pneumonia",
        features={
            "fever": True,  # 104.0F
            "dyspnoea_at_rest": True,
            # "There was no history of cough, chest pain"
            "productive_cough": False,
            "pleuritic_pain": False,
            "exertional_chest_pain": False,
            "smoking_history": False,  # "no history of smoking"
            "exam:tachycardia": True,  # 128
            "exam:hypoxia": True,  # 92% on room air
            # "diminished breath sounds in the left lower lobe"
            "exam:reduced_breath_sounds": True,
            "lab:raised_wcc": True,  # 11.9 k/uL
            "lab:raised_d_dimer": True,  # 2,112 ng/mL D-DU, reference 0-230
            # "ill-defined pulmonary infiltrate in the lower lobe"
            "imaging:cxr_consolidation": True,
        },
        vocabulary=frozenset(),
    ),
    # PMC11272502 -- severe Legionella longbeachae pneumonia from potting soil.
    Case(
        case_id="ho-11272502",
        presenting_complaint=(
            "70-year-old man, a keen gardener, high fever, cough and "
            "breathlessness after a few days of diarrhoea"
        ),
        diagnosis="community_acquired_pneumonia",
        features={
            "fever": True,  # 39.8C
            "dyspnoea_at_rest": True,
            "exam:hypoxia": True,  # SpO2 70% on 15 L/min
            "smoking_history": True,  # 20 pack-years
            "exam:tachycardia": False,  # 78 bpm, "relative bradycardia"
            "exam:crackles": True,  # "bilateral coarse crackles"
            # chest radiography: "ground-glass and infiltrative opacities"
            "imaging:cxr_consolidation": True,
        },
        vocabulary=frozenset(),
    ),
    # ---- acute coronary syndrome -------------------------------------------------
    # PMC8788021 -- anterior STEMI evolving into a de Winter pattern.
    Case(
        case_id="ho-8788021",
        presenting_complaint="34-year-old man who smokes, 49 minutes of chest pain",
        diagnosis="acute_coronary_syndrome",
        features={
            "smoking_history": True,  # 18 pack-years
            "fever": False,  # 36.5C
            "exam:tachycardia": False,  # 70 bpm
            "lab:raised_troponin": True,  # cTnT 47 ng/L, normal < 40
            "lab:raised_bnp": True,  # 124.5, normal < 100
            # first ECG "showed an acute anterior ... myocardial infarction"
            "exam:ecg_st_changes": True,
        },
        vocabulary=frozenset(),
    ),
)


__all__ = ["HOLDOUT_CASES"]
