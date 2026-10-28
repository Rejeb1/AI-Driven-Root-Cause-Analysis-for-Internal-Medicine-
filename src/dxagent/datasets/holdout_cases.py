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

Amendment 2 -- a third wave and a second frozen model, committed before
its search and before the model it will test exists
------------------------------------------------------------------------------
The 28 cases above have now been run and looked at, and the model has
changed since (two cells sourced after the internship report). Three design
changes are about to be made: an escalation that states what would resolve
it, ST elevation and ST depression as separate findings, and a language
model reading the presenting complaint. None of them may be judged on cases
already seen, so a third wave is fixed here first.

Search: continue every walk above, in the same relevance order with the
same filters, from rank 41 to rank 80 -- the eight wave-1 queries, plus the
four wave-2 queries for pericarditis, pneumonia, panic and ACS. Inclusion
rules 1-5 unchanged, plus: not already in either wave or walked past in
one. Take the first **three** per diagnosis that qualify (wave-2 query
first where a diagnosis has two lists, then the wave-1 list), at most 24 in
all; a diagnosis that cannot fill three within ranks 41-80 is recorded
short, not topped up.

Extraction, the same discipline as before, with one addition: where the
report says which way the ST segment moved, record ``ecg:st_elevation`` or
``ecg:st_depression`` as well as ``exam:ecg_st_changes``. The 53 cases
already extracted get the same addition, read from their reports for the
ECG sentence only, with every change listed in DESIGN.md.

Model v2 is the commit that adds the three design changes, made before this
wave is searched. Wave 3 is run once on v2 and once on the current model
(v1, commit 9c12b9f), and reported apart from waves 1-2 as well as pooled.
Nothing is tuned after the wave-3 run; a change prompted by it is a v3
and needs a fourth wave.

If the pooled holdout reaches 30, Amendment 1's fitting plan runs on it,
with v2, exactly as written there.

What wave 3 found (searched 2026-10-16, ranks are PubMed relevance)
-------------------------------------------------------------------
PE (wave-1 list): taken r43 PMC11343825, r44 PMC10932506, r49 PMC6624151.
  Skipped: r41 acute pancreatitis co-primary; r42 image report, no text; r45
  presented with leg pain, breathlessness began as an inpatient; r46, r47
  COVID-19; r48 tumour embolism.

ACS (wave-2 list first): taken w2 r63 PMC12539291, w2 r68 PMC12097248, w1
  r48 PMC12426847. Skipped, wave-2 list: r41, r42, r51, r59, r61, r62, r64,
  r72-r74 mimics; r43, r79 drug-induced; r44 cocaine with COVID-19; r45
  adolescent, no PMC; r46 RSV infection co-primary; r47 lung carcinoma; r48
  on chemotherapy for malignancy; r49, r54, r77 embolic; r50 carbon
  monoxide; r52 thyrotoxicosis co-primary; r53 non-atherosclerotic thrombus,
  cause unresolved (as wave-1 r39); r55, r65 no case text; r56 vasospasm;
  r57 presented as limb ischaemia; r58 four findings; r60 no PMC; r66
  adrenal crisis co-primary; r67 non-obstructive ectasia, no thrombus; r69
  hypertensive emergency with heart failure co-primary; r70 coronary-
  subclavian steal; r71 cannabis; r75 amniotic fluid embolism; r76 final
  diagnosis PE; r78 TTP co-primary; r80 attributed to hot-weather running
  (exposure-attributed, as wave-2 pericarditis r29). Wave-1 list: r41 no
  PMC; r42 drug-induced; r43 no case text; r44 myocarditis co-primary; r45
  mimic; r46 embolic; r47 no PMC.

Extraction note: "acute onset" read as sudden_onset, as real_cases already
  reads "acute shortness of breath"; "vital signs unremarkable" read as no
  tachycardia only.

ACS correction during the walk: w1 r48 PMC12426847 withdrawn -- its sixth
  finding relied on "otherwise unremarkable vital signs" as a negative,
  which real_cases.py's recorded ruling forbids; five findings without it.
  "acute onset" likewise NOT read as sudden_onset (close-but-not-quite); no
  accepted case depends on it.

ACS wave-1 list, continued: r48 five findings (above); r49 embolic
  (duplicate of wave-2 r49); r50, r59, r72, r73 no case text; r51 post-
  operative; r52, r56, r63, r74, r78 no PMC; r53 intracardiac mass; r54,
  r61, r70, r71, r77 mimics; r55 endomyocardial fibrosis co-primary; r57
  pseudoxanthoma elasticum vasculopathy; r58 gastric perforation co-primary;
  r60 not an ACS; r62 walked in wave 1; r64 Kounis (allergic vasospasm); r65
  oleander poisoning; r66 already in the holdout; r67 autopsy; r68 carbon
  monoxide; r69 tumour embolism; r75 presented with headache; r76 aged 9;
  r79 five findings; r80 simulated (telemetry artefact). ACS SHORT: 2 of 3
  within ranks 41-80, recorded, not topped up.

Pneumonia (wave-2 list first): taken r55 PMC11162253, r61 PMC9757645, r74
  PMC7581029. Skipped: r41, r44 not a pneumonia presentation (pancreatitis,
  rash); r42 myocarditis co-primary; r43, r48 children; r45, r50, r54, r58
  no PMC; r46 pericardial abscess; r47, r49, r60, r62 immunocompromised or
  cancer; r51 HUS and meningitis; r52 aortic aneurysm; r53 atrial flutter
  with cardiomyopathy co-primary; r56 autopsy; r57 no breathlessness, cough
  or chest pain; r59 rhabdomyolysis co-primary; r63 agranulocytosis co-
  primary; r64 empyema named in the title (co-primary consequence, as wave-1
  asthma r15); r65 five findings (its consolidation is on CT, which
  real_cases.py rules is not the chest-film concept); r66 COVID-19; r67
  child; r68 four findings; r69 no breathing symptoms; r70 no English text.

Oedema: taken r44 PMC6009803 (first patient of two), r59 PMC9826619.
  Skipped: r41, r47, r48, r50, r53, r56 drug-induced; r42, r51, r55, r61,
  r62, r65, r69, r75, r78, r80 non-cardiogenic; r43 giant cell myocarditis;
  r45 negative-pressure with Takotsubo; r46 serotonin syndrome; r49, r70 no
  PMC; r52, r67 veterinary; r54 no case text; r57 post-operative; r58
  subarachnoid haemorrhage; r60 after pacemaker insertion; r63 postpartum
  hypertension-related, mechanism a judgement call; r64 severe anaemia and
  pericardial effusion co-primary; r66 coronary spasm; r68 ex vivo lung; r71
  adenocarcinoma; r72, r76 pre-eclampsia; r73 carbon monoxide; r74 atrial
  myxoma; r77 SVT co-primary (as wave-2 pericarditis r34); r79
  thyrotoxicosis. OEDEMA SHORT: 2 of 3.

COPD (list has 66 results): none taken. Skipped: r41 presented with syncope;
  r42, r66 mimics; r43 tracheal stenosis; r44 already in the holdout; r45
  COVID-19; r46, r51, r54, r60, r65 not case reports of a presentation; r47
  foreign-body aspiration; r48 two findings; r49 Aspergillus co-primary; r50
  PRES; r52 hypercapnic cerebral oedema; r53 haematological malignancy; r55
  lung malignancy; r56 Listeria meningitis; r57 stable COPD at a routine
  visit; r58, r62, r63 no PMC; r59 methaemoglobinaemia; r61 myoclonus
  presentation; r64 HSV oesophagitis. COPD SHORT: 0 of 3.

Asthma: taken r59 PMC9949361, r68 PMC6010635. Skipped: r41, r57 already
  recorded in real_cases; r42 EGPA question; r43, r47, r64, r71, r78, r79
  under 18; r44, r55 mimics; r45 Pseudomonas infection co-primary; r46, r69,
  r72, r80 pneumothorax or pneumomediastinum; r48, r54 COVID-19; r49, r51,
  r66 chronic management; r50 HHT (as wave-1 r12); r52, r67, r77 fewer than
  six findings; r53 drug reaction co-primary; r56 tracheal tumour; r58 final
  diagnosis PE; r60, r63 Strongyloides hyperinfection; r61 influenza
  pneumonia co-primary; r62 drug liver injury; r65 ICU-acquired weakness
  named in the title; r70 foreign body; r73, r75 Takotsubo; r74 urinary
  infection co-primary (as wave-1 r34); r76 attributed to weather change
  (exposure-attributed, as ACS r80). ASTHMA SHORT: 2 of 3.

Extraction notes: exam wheeze -> wheeze_subjective and a wholly normal chest
  film -> both CXR concepts absent, as real_cases already does; "ECG
  unremarkable" not read as a specific negative. r68's pharyngitis is the
  named trigger ("an asthma attack due to acute pharyngitis"), not a co-
  primary diagnosis.

Pericarditis (wave-2 list first): taken w2 r65 PMC6439374, w2 r76
  PMC11089582. Skipped, wave-2 list: r41, r43, r47, r54 (no PMC), r60, r74,
  r75, r77, r78, r80 COVID-19; r42, r56 leukaemia; r44, r70 vaccine-related;
  r45, r50, r53, r68 mimics; r46, r52, r67 drug-induced; r48, r73 SLE; r49
  after stem-cell transplant; r51 neoplastic; r55 pre-eclampsia co-primary;
  r57, r72 after ablation; r58 post-pericardiotomy; r59 concurrent PE; r61
  bloodstream infection; r62, r71 presented as cholecystitis; r63 no PMC;
  r64 anorexia with pneumomediastinum; r66 differentiation syndrome; r69
  Salmonella aneurysm; r79 bronchogenic cyst. Wave-1 list: r41 rheumatoid;
  r42-44, r47, r56, r57, r64, r75 walked in waves 1-2; r45, r54, r69, r76,
  r78 purulent or bacterial; r46 Graves' co-primary; r48, r50, r53, r66,
  r67, r72 constrictive; r49, r58, r60, r63 tuberculous or mycobacterial;
  r51 peri-infarct; r52 familial Mediterranean fever (autoinflammatory, as
  rheumatoid and vasculitic); r55 mesothelioma; r59 parasitic; r61 no case
  text; r62 no PMC; r65 Guillain-Barre co-primary; r68, r74, r77 veterinary;
  r70 lymphoma; r71 transplant recipient; r73 asymptomatic overdose; r79
  drug-induced; r80 post-operative. PERICARDITIS SHORT: 2 of 3.

Extraction note: 37.8C is below this project's 38C, recorded as fever
  absent; "infiltrate" on a chest film read as cxr_consolidation, as in r55.

Panic (wave-2 list has 10 results, none at 41-80; wave-1 list has 57): none
  taken. Skipped: r41 walked (ventricular fibrillation); r42 autoimmune
  encephalitis; r43 cohort study; r44, r45, r46, r51, r52 psychiatric or
  outpatient course; r47 unrelated presentation; r48, r53, r54, r55, r57 no
  PMC; r49 hypercapnic COPD failure co-primary; r50 VF with Takotsubo; r56
  already in the holdout. PANIC SHORT: 0 of 3.

WAVE 3: 14 cases (PE 3, ACS 2, pneumonia 3, oedema 2, COPD 0, asthma 2,
  pericarditis 2, panic 0). Pooled holdout 42 >= 30.

Amendment 3 -- a fourth wave for model v3, committed after v3 is frozen and
before its search
------------------------------------------------------------------------------
Wave 3 has been run and looked at, and its wrong commit (ho-6010635) chose
which cells model v3 re-searched. v3 also adds two findings, haemoptysis and
syncope. None of that may be judged on waves 1-3, so a fourth wave is fixed
here, with model v3 already committed (49ac91e) and no wave-4 case searched.

Search: continue every walk above, same relevance order and filters, from
rank 81 to rank 120 -- the eight wave-1 queries and the four wave-2 queries.
Inclusion rules 1-5 unchanged, plus: not already in any wave or walked past
in one. Take the first **three** per diagnosis that qualify (wave-2 query
first where a diagnosis has two lists, then the wave-1 list), at most 24; a
diagnosis that cannot fill three is recorded short, not topped up. COPD's
list (66 results) and panic's (57 and 10) end before rank 81, so each gets one
new title query, walked from rank 1 to 40 under the same rules, skipping
anything handled in waves 1-3:

- copd_exacerbation: ``("COPD"[ti] OR "chronic obstructive"[ti] OR emphysema[ti]) AND ("respiratory failure"[tiab] OR hypercapnic[tiab] OR "acute exacerbation"[tiab])``
- panic_attack: ``("panic"[tiab] OR "anxiety attack"[tiab]) AND ("chest pain"[ti] OR palpitations[ti] OR dyspnea[ti] OR "shortness of breath"[ti] OR mimic*[ti])``

Extraction, the same discipline, with one addition: record ``haemoptysis``
and ``syncope`` where the report states them (presyncope is not syncope).

Run: wave 4 once on v3 and once on v2 (the model at 6fd37af), both at the
shipped real-case temperature T = 1.25 with the Bayesian gate alone; the
second reading (Gemini) is run as well where the API answers, and reported
apart. The comparison fixed now: wrong commits first, correct commits
second, v3 against v2, on wave 4 alone, then pooled. Nothing is tuned
after the run; a change it prompts is a v4 and needs a fifth wave. No
refitting: the temperature stays 1.25 whatever wave 4 shows.

What wave 4 found (searched 2026-10-28, ranks are PubMed relevance)
-------------------------------------------------------------------
Full text read through NCBI E-utilities, as in wave 3; "no case text" means
efetch returned no body. Multi-patient reports are judged on their first
patient, as wave 3 took PMC6009803. An ST statement that fills a parent and a
child cell counts once toward rule 4's six. Rank drift since wave 3: items
already in a set reappeared at oedema r84, r95, asthma r100, r105, r107,
pericarditis r81, r94, r102, r103 and ACS w2 r88, and were skipped as handled.

PE (wave-1 list): taken r90 PMC11047289, r103 PMC6392898, r107 PMC7008764.
  Skipped: r81 sarcoidosis co-primary; r82 vaccine-attributed; r83
  supplement-attributed (exposure-attributed, as ACS r80 in wave 3); r84 RV
  noncompaction with VT co-primary; r85, r86, r96, r100 no case text; r87
  severe asthma co-primary; r88 clozapine-attributed; r89, r95 COVID-19; r91
  emphysematous pancreatitis co-primary; r92 four findings (the seizure-like
  episode not read as syncope); r93 myxoma; r94 stroke co-primary; r97
  tumour embolism (choriocarcinoma); r98 pneumonia treated alongside for a
  week, co-primary; r99 leiomyomatosis (tumour embolism); r101 five findings;
  r102 amyloidosis; r104 CMV pericarditis co-primary; r105 mimic; r106 no PMC.

ACS (wave-2 list first): none taken. Skipped, wave-2 list: r81 thyrotoxicosis
  with normal coronaries; r82 Takayasu, teenager; r83, r89, r92, r96, r101,
  r107 no PMC; r84 sumitriptan; r85 seizure-associated, no plaque rupture; r86
  Cogan vasculitis; r87 subclavian steal (as wave-3 r70); r88 handled; r90 no
  presenting symptom stated; r91 stroke thrombolysis co-primary; r93 pacing
  complication; r94 trauma; r95 four findings; r97, r112 external compression;
  r98, r113 vasospasm; r99 endocarditis; r100 foreign body; r102, r105, r118,
  r119 no case text; r103 post-stroke thrombolysis; r104 Brugada mimic; r106
  five findings (ST elevation counted once); r108 final diagnosis PE; r109
  Kounis; r110 moyamoya; r111 three findings; r114 dobutamine with
  hyperthyroidism; r115, r116 COVID-19; r117 SCAD; r120 hypertensive
  emergency co-primary (as wave-3 r69). Wave-1 list: r81 antiphospholipid
  thrombosis; r82 marijuana (as wave-3 r71); r83, r87 vasculitic; r84, r118
  later mechanical complication, not the presentation; r85, r88, r100, r106,
  r113 mimics; r86 myocardial bridging; r89, r95, r98, r101, r103, r111,
  r114, r116, r119 no PMC; r90 embolic; r91 coronary aneurysm, first patient
  three findings; r92 concurrent PE; r93 adolescent; r94 histopathology
  teaching case; r96 four findings; r97, r109, r112 no case text; r99 first
  patient in diabetic ketoacidosis; r102 trauma; r104 fistula steal, troponin
  normal; r105 Takotsubo question; r107 mucormycosis after transplant; r108,
  r110 phaeochromocytoma/paraganglioma; r115 tacrolimus; r117 tracheal
  injury; r120 ketoacidosis co-primary. ACS SHORT: 0 of 3.

Pneumonia (wave-2 list first): none taken. Skipped, wave-2 list: r81, r91,
  r94, r107 mucocutaneous or haematological Mycoplasma syndromes; r82
  endometritis and sepsis co-primary; r83, r85, r92, r95, r118 children; r84
  surveillance report; r86, r102 after stem-cell transplant; r87 hernia mimic;
  r88 tachypnoea and hypoxia but no breathlessness reported (as wave-2 r33);
  r89, r98, r112 no case text; r90, r99, r113 co-primary complication
  (haemolysis, splenial lesion, heart failure); r93 case series of a region;
  r96, r97, r110, r116 COVID-19; r100 catatonia series; r101 pneumothorax
  co-primary; r103 vaccine eosinophilic pneumonia; r104 endocarditis with HIV;
  r105 dual bacteraemia; r106 long-term corticosteroids (immunosuppressed);
  r108, r114 nosocomial; r109 hairy-cell leukaemia; r111 no PMC; r115
  arthritis; r117 exposure study; r119, r120 mimic or later complication.
  Wave-1 list: r81, r83, r104, r112 eosinophilic pneumonia; r82 no
  breathlessness or chest pain, new diabetes; r84, r87, r90, r108 COVID-19;
  r85 outpatient course, later COVID-19; r86, r99, r107, r113, r115 organising
  or interstitial; r88, r95, r100, r106, r109, r116, r117 immunocompromised;
  r89 emphysema diagnosed at the same presentation (co-primary); r91
  varicella; r92 no case text; r93, r119 no PMC; r94 endobronchial lipoma;
  r96, r110 mimic; r97 ultrasound method report; r98 handled; r101 hiccups and
  syncope, no breathlessness or chest pain; r102 ventilator-associated; r103,
  r120 hydropneumothorax named in the title; r105 splenic rupture; r111
  veterinary; r114 parasitic; r118 mycobacterial. PNEUMONIA SHORT: 0 of 3.

Oedema: none taken. Skipped: r81, r96, r105, r107 re-expansion; r82, r88,
  r89, r90, r98, r99, r100, r103, r106, r108, r113, r116, r119 non-cardiogenic;
  r83, r94 phaeochromocytoma; r84, r95 handled; r85 veterinary; r86, r104
  post-operative; r87 after pacemaker insertion (as wave-3 r60); r91 LVAD;
  r92 paediatric; r93 final diagnosis PE, lymphoma on chemotherapy; r97
  antiphospholipid syndrome with lupus myocarditis; r101, r110, r112, r115,
  r120 no PMC; r102 child, high altitude; r109, r114 Takayasu; r111 after
  hyperbaric oxygen; r117 chronic pacing management; r118 immunotherapy for
  lung cancer. OEDEMA SHORT: 0 of 3.

Asthma: none taken. Skipped: r81, r85 EGPA; r82 poisoning co-primary; r83,
  r84, r91, r95, r99, r103, r108-r110, r116 chronic management; r86, r102
  COVID-19; r87 ketoacidosis co-primary; r88, r89, r90, r94, r111, r114, r117,
  r118 under 18; r92 alveolar haemorrhage; r93 nocardiosis; r96
  Strongyloides hyperinfection (as wave-3 r60); r97 infective focus
  co-primary, temperatures contradictory; r98 ABPA; r100, r105, r107 handled;
  r101 no case text; r104 eosinophilic pneumonia; r106 cohort study; r112,
  r113 mimics; r115 myasthenic crisis co-primary; r119 vaccine-attributed;
  r120 pneumorrhachis (as pneumomediastinum). ASTHMA SHORT: 0 of 3.

Pericarditis (wave-2 list, 103 results, then wave-1): none taken. Skipped,
  wave-2 list: r81, r94, r102, r103 handled; r82 presented as cholecystitis;
  r83 teenager after COVID-19; r84 Salmonella aneurysm; r85 after primary PCI;
  r86 anorexia with pneumomediastinum; r87 COVID-19; r88 leukaemia; r89 mimic;
  r90 bronchogenic cyst; r91 post-pericardiotomy; r92 SLE; r93 bloodstream
  infection; r95 concurrent active infections; r96 recurrent, chronic
  management; r97 children, no PMC; r98 mimic; r99 filter strut; r100 Graves'
  co-primary; r101 drug-induced. Wave-1 list: r81, r84, r88, r97, r100, r102,
  r104, r105, r110-r112, r115, r117, r118 constrictive; r82 gunshot; r83, r95,
  r99 COVID-19 or vaccine; r85 on ocrelizumab (immunosuppressed); r86, r90,
  r91, r98, r103, r107, r109 purulent or bacterial; r87, r120 veterinary;
  r89, r101, r114, r116 tuberculous; r92 bronchogenic cyst; r93 SLE; r94
  procedural; r96 monkeypox; r106 aortic dissection; r108 absent pericardium;
  r113 adolescent; r119 IgG4. PERICARDITIS SHORT: 0 of 3.

COPD (new query, 82 results, ranks 1-40): none taken. Skipped: r1, r11, r13,
  r23, r31 no PMC; r2, r8, r10, r30 combined fibrosis and emphysema; r3, r4,
  r6, r7, r12, r17, r21, r22 surgical or subcutaneous emphysema; r5, r27, r35
  fungal, nocardial or mycobacterial co-primary; r9, r16 chronic home
  ventilation; r14 first patient obtunded, no breathlessness reported; r15
  lung cancer; r18 veterinary; r19 PRES; r20 asthma with pneumothorax; r24
  final diagnosis "AECOPD by community-acquired pneumonia", two of the eight
  (as PE r98 above); r25 cohort study; r26 chronic PE; r28 pneumothorax; r29
  congenital lobar emphysema; r32 panic attacks triggering hypercapnia, two of
  the eight; r33 handled; r34 bullous emphysema; r36 mimic; r37, r38, r40
  COVID-19; r39 intraoperative. COPD SHORT: 0 of 3.

Panic (new query, 2 results): none taken. r1 final diagnosis borderline
  personality disorder, which the report distinguishes from panic disorder;
  r2 serotonergic mimic. PANIC SHORT: 0 of 3.

Extraction notes: "pinkish sputum" (r90) is not read as haemoptysis, and
  "worsened suddenly" after three weeks of exertional breathlessness is not
  read as sudden_onset (close-but-not-quite); r107's saturation was 98% and
  then 60% after a second syncope in the same visit, conflicting, so left out;
  r107's "physical exam including heart and lungs ... normal" read as lungs
  clear; laboratory values without a reference range left out, as before.

WAVE 4: 3 cases, all pulmonary embolism. 21 of the 24 places went unfilled,
  and the rule says record the shortfall rather than top it up. Three cases
  cannot say whether v3 is better than v2; they are a first look, not a
  verdict. Two of the three present with syncope, one of the two signs v3
  added -- the PE list's rank order, not a choice.

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
            # "did not report any overt syncope or palpitations" -- presyncope
            # is not syncope, so the syncope cell is absent
            "palpitations": False,
            "syncope": False,
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
            "haemoptysis": True,  # "one episode of haemoptysis"
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
            "haemoptysis": True,  # "one episode of haemoptysis"
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
            # "ST segment elevation in II, III and aVF and ST segment depression in
            # V4, V5, and V6" -- both stated
            "exam:ecg_st_elevation": True,
            "exam:ecg_st_depression": True,
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
            # "ST segment elevation with Q waves in the anterior leads"
            "exam:ecg_st_elevation": True,
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
            "haemoptysis": True,  # in the presenting complaint
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
            "haemoptysis": False,
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
            # "at least 1-mm ST elevation" (V2), with diffuse ST-PR discordance
            "exam:ecg_st_elevation": True,
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
            # "ST-segment elevation was noted in the lead V2-V4"
            "exam:ecg_st_elevation": True,
        },
        vocabulary=frozenset(),
    ),
)


# ---- wave 3 (Amendment 2) ---------------------------------------------------
# Searched 2026-10-16, after model v2 was frozen (commit b0af386) and before
# either model was run on any of these. Ranks 41-80 of the same queries, three
# per diagnosis at most. 14 found: PE 3, ACS 2, pneumonia 3, oedema 2, COPD 0,
# asthma 2, pericarditis 2, panic 0. Every candidate walked past is listed in
# WAVE3_WALK below with its reason.
WAVE3_CASES: tuple[Case, ...] = (
    # ---- pulmonary embolism (wave-1 list, ranks 43, 44, 49) ---------------
    # PMC11343825 -- saddle PE with thrombus in transit across a PFO.
    Case(
        case_id="ho-11343825",
        presenting_complaint=(
            "51-year-old man with sleep apnoea, one day of progressive "
            "breathlessness and palpitations"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "dyspnoea_at_rest": True,  # "progressive dyspnea"
            "palpitations": True,
            "exam:hypoxia": False,  # "oxygen saturation 95% on room air"
            "leg_swelling": True,  # "mild right lower extremity edema"
            "exam:tachycardia": True,  # "EKG showed sinus tachycardia"
            "exam:ecg_st_changes": False,  # "without signs of ischemia"
            "lab:raised_troponin": True,  # 0.05, reference 0-0.01
            "lab:raised_d_dimer": True,  # 10.13, reference < 0.50
            "imaging:ctpa_filling_defect": True,  # "saddle pulmonary embolus"
        },
        vocabulary=frozenset(),
    ),
    # PMC10932506 -- PE in rheumatoid arthritis on etanercept; emphysema and
    # bypass surgery six months earlier are background.
    Case(
        case_id="ho-10932506",
        presenting_complaint=(
            "67-year-old man with emphysema and coronary bypass six months "
            "ago, sharp left-sided chest pain"
        ),
        diagnosis="pulmonary_embolism",
        features={
            # "acute in onset" is not read as sudden_onset (close, not quite)
            "smoking_history": True,  # 12 pack-years, quit six months ago
            # "no change in his ambulatory status ... No recent long-distance travel"
            "recent_immobility": False,
            "exam:tachycardia": True,  # 110 per minute
            "exam:hypoxia": True,  # 86% on room air
            "exam:crackles": False,  # "No lung crackles or wheezing"
            "wheeze_subjective": False,
            "leg_swelling": False,  # "No pedal edema"
            "exam:raised_jvp": False,  # "or jugular venous distension"
            "lab:raised_wcc": False,  # "no leukocytosis"
            "lab:raised_bnp": False,  # "B-natriuretic peptide was not elevated"
            "imaging:ctpa_filling_defect": True,  # segmental and subsegmental PE
        },
        vocabulary=frozenset(),
    ),
    # PMC6624151 -- submassive PE after a long car journey.
    Case(
        case_id="ho-6624151",
        presenting_complaint=(
            "55-year-old man, pleuritic chest pain and breathlessness with a "
            "faint during a long car journey"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "syncope": True,  # "a faint" on the journey
            "pleuritic_pain": True,
            "dyspnoea_at_rest": True,  # "shortness of breath"
            # "a car trip from Texas to Cleveland" -- as real_cases records a
            # long flight
            "recent_immobility": True,
            "exam:tachycardia": True,  # pulse 119
            "exam:hypoxia": True,  # 93% on room air
            "fever": False,  # 36.6C
            "exam:crackles": False,  # "clear lungs"
            # "significant for ... ProBNP 3188pg/mL, and initial troponin 0.12"
            "lab:raised_bnp": True,
            "lab:raised_troponin": True,
            "imaging:ctpa_filling_defect": True,  # "multiple bilateral filling defects"
        },
        vocabulary=frozenset(),
    ),
    # ---- acute coronary syndrome (wave-2 list ranks 63, 68) ----------------
    # PMC12539291 -- STEMI from plaque erosion in a 23-year-old smoker.
    Case(
        case_id="ho-12539291",
        presenting_complaint=(
            "23-year-old man who smokes heavily, sudden chest tightness at "
            "work that did not ease with rest"
        ),
        diagnosis="acute_coronary_syndrome",
        features={
            "smoking_history": True,  # > 30 cigarettes a day
            "sudden_onset": True,  # "sudden-onset chest tightness"
            "exam:tachycardia": False,  # 65 bpm
            "exam:hypoxia": False,  # 100% on room air
            # "ST-segment elevations in leads I, aVL, and V2-4"
            "exam:ecg_st_changes": True,
            "exam:ecg_st_elevation": True,
            "lab:raised_wcc": True,  # "elevated white blood cell count of 13,600"
            # "without valve disease or effusion"
            "imaging:pericardial_effusion": False,
            # hs-troponin T 0.095 ng/mL is given without a reference: left out
        },
        vocabulary=frozenset(),
    ),
    # PMC12097248 -- STEMI at 37 weeks' gestation; thrombotic, no dissection.
    Case(
        case_id="ho-12097248",
        presenting_complaint=(
            "26-year-old woman, 37 weeks pregnant, mid-chest pain that came "
            "on while grocery shopping"
        ),
        diagnosis="acute_coronary_syndrome",
        features={
            "exam:tachycardia": False,  # 99 beats per minute
            "exam:hypoxia": False,  # 99% on room air
            # "ST elevation in leads II, III, V3, V6, and aVF with reciprocal
            # depressions in V1, V2, I and aVL" -- both stated, as ho-5067724
            "exam:ecg_st_changes": True,
            "exam:ecg_st_elevation": True,
            "exam:ecg_st_depression": True,
            "lab:raised_troponin": True,  # 0.07 and 6.04, reference <= 0.03
        },
        vocabulary=frozenset(),
    ),
    # ---- community-acquired pneumonia (wave-2 list ranks 55, 61, 74) -------
    # PMC11162253 -- Legionella pneumonia with ten days of fever.
    Case(
        case_id="ho-11162253",
        presenting_complaint=(
            "69-year-old man, ten days of progressive fever and breathlessness "
            "with a dry cough"
        ),
        diagnosis="community_acquired_pneumonia",
        features={
            "fever": True,  # 101.8F
            "dyspnoea_at_rest": True,  # "shortness of breath"
            "sudden_onset": False,  # "10-day history of progressive"
            "productive_cough": False,  # "a dry, nonproductive cough"
            "exam:hypoxia": True,  # 91%
            "exam:tachycardia": False,  # pulse 79
            "exam:crackles": True,  # "diffuse, bilateral crackles"
            # "alveolar infiltrates ... consistent with multifocal pneumonia"
            "imaging:cxr_consolidation": True,
        },
        vocabulary=frozenset(),
    ),
    # PMC9757645 -- Mycoplasma pneumonia with cavitated lesions.
    Case(
        case_id="ho-9757645",
        presenting_complaint=(
            "55-year-old man who smokes, one week of daily fever, pleuritic "
            "chest pain and a cough turning productive"
        ),
        diagnosis="community_acquired_pneumonia",
        features={
            "fever": True,  # daily fever, maximum 39C
            "pleuritic_pain": True,
            "productive_cough": True,  # "initially dry then productive"
            # "dyspnea after medium exertion" -- exertional, not at rest: left out
            "smoking_history": True,  # 40 pack-years
            "exam:tachycardia": False,  # "normocardic"
            "exam:hypoxia": False,  # 96% on room air
            # "No other abnormalities were noted, including on pulmonary
            # auscultation" -- names the lungs, so it is a negative
            "exam:crackles": False,
            "wheeze_subjective": False,
            "lab:raised_wcc": True,  # 16 x10^3, reference 4.8-10.8
            "exam:ecg_st_changes": False,  # "The electrocardiogram revealed no changes"
            "lab:raised_troponin": False,  # "high-sensitivity troponin I was normal"
            # the film shows a new hypertransparency; the cavitated densities
            # are on CT, which is not the chest-film concept
        },
        vocabulary=frozenset(),
    ),
    # PMC7581029 -- fatal Legionella pneumonia, new sequence type.
    Case(
        case_id="ho-7581029",
        presenting_complaint=(
            "59-year-old man who smokes, six days of high fever with "
            "progressive breathlessness and blood-streaked yellow sputum"
        ),
        diagnosis="community_acquired_pneumonia",
        features={
            "fever": True,  # maximum 39.6C (36.9C on admission after dexamethasone)
            "productive_cough": True,  # "yellow sputum"
            "dyspnoea_at_rest": True,  # "progressive dyspnea"
            "sudden_onset": False,  # progressive over six days
            "smoking_history": True,  # 30 pack-years
            "exam:tachycardia": True,  # 140
            "exam:hypoxia": True,  # SO2 78%
            "exam:crackles": True,  # "extensive moist rales in both lungs"
            "leg_swelling": True,  # "edema of both lower extremities"
            "lab:raised_wcc": False,  # 3.1 x10^9/L
            # consolidation is on CT only; proBNP 1271 and troponin 0.34 are
            # given without a reference or a flag: all left out
        },
        vocabulary=frozenset(),
    ),
    # ---- acute pulmonary oedema (wave-1 list ranks 44, 59) -----------------
    # PMC6009803, case 1 of 2 -- hypertensive cardiogenic pulmonary oedema.
    Case(
        case_id="ho-6009803",
        presenting_complaint=(
            "65-year-old woman with heart failure and hypertension, acute "
            "breathlessness and chest tightness"
        ),
        diagnosis="acute_pulmonary_oedema",
        features={
            "dyspnoea_at_rest": True,  # "acute dyspnea"
            "exam:tachycardia": True,  # 103 bpm
            "exam:hypoxia": True,  # 92% on a high-flow mask
            "fever": False,  # 36C
            "orthopnoea": True,
            "exam:crackles": True,  # "bilateral basal rales"
            # "Chest X-ray ... diagnostic of acute pulmonary edema"
            "imaging:cxr_pulmonary_oedema": True,
            # "nonspecific ST-T changes" is not ST change: left out
        },
        vocabulary=frozenset(),
    ),
    # PMC9826619 -- upper-lobe cardiogenic oedema from mitral chordal rupture;
    # serial COVID-19 PCR negative.
    Case(
        case_id="ho-9826619",
        presenting_complaint=(
            "77-year-old man, nine days of gradually worsening tiredness, "
            "breathlessness, cough and breathlessness lying flat"
        ),
        diagnosis="acute_pulmonary_oedema",
        features={
            "sudden_onset": False,  # "gradual onset"
            "dyspnoea_at_rest": True,
            "orthopnoea": True,
            # "He denied chest pain, palpitation, haemoptysis, cough with
            # expectoration, and fever" -- both chest-pain concepts absent, as
            # ho-8302121 reads "no significant chest pain"
            "pleuritic_pain": False,
            "haemoptysis": False,
            "exertional_chest_pain": False,
            "palpitations": False,
            "productive_cough": False,
            "fever": False,  # denied; 36.4C
            "smoking_history": False,  # "no history of smoking"
            "exam:tachycardia": True,  # 112
            "exam:hypoxia": True,  # "the patient was hypoxemic"
            "exam:crackles": True,  # "bilateral lung crackles"
            "exam:raised_jvp": False,  # "jugular venous pressure was 4 cm"
            "leg_swelling": False,  # "no leg oedema"
            "imaging:cxr_pulmonary_oedema": True,
            "exam:ecg_st_changes": False,  # "without ST-segment or T-wave changes"
            "imaging:ctpa_filling_defect": False,  # "no evidence of pulmonary embolism"
            "lab:raised_wcc": False,  # "white cell count was normal"
            "lab:raised_bnp": True,  # "his BNP was also high"
        },
        vocabulary=frozenset(),
    ),
    # ---- asthma exacerbation (wave-1 list ranks 59, 68) --------------------
    # PMC9949361 -- severe acute asthma after cold exposure, readmitted.
    Case(
        case_id="ho-9949361",
        presenting_complaint=(
            "38-year-old man with asthma, two days of progressive "
            "breathlessness after cold exposure, cough and audible wheeze"
        ),
        diagnosis="asthma_exacerbation",
        features={
            "dyspnoea_at_rest": True,  # unable to complete sentences
            "sudden_onset": False,  # "2-day history of progressive"
            "wheeze_subjective": True,  # "audible wheezes"
            "fever": False,  # "no associated fever"; 37.3C
            "exam:tachycardia": True,  # 145
            # "unremarkable for any pulmonary infiltrates, pneumothorax,
            # pleural effusion, or pathology" -- a wholly normal film
            "imaging:cxr_consolidation": False,
            "imaging:cxr_pulmonary_oedema": False,
            # "mild cough" and "no ... cough" both stated: left out
        },
        vocabulary=frozenset(),
    ),
    # PMC6010635 -- exacerbation after three weeks on a dummy inhaler,
    # triggered by acute pharyngitis.
    Case(
        case_id="ho-6010635",
        presenting_complaint=(
            "42-year-old woman with lifelong asthma, three weeks of dry cough "
            "and increasing breathlessness, now febrile"
        ),
        diagnosis="asthma_exacerbation",
        features={
            "productive_cough": False,  # "dry cough"
            "dyspnoea_at_rest": True,  # "increasing breathlessness"
            "sudden_onset": False,  # three weeks
            "exam:tachycardia": True,  # 140
            "exam:hypoxia": False,  # 98% on room air at admission
            "fever": True,  # 39C
            "wheeze_subjective": True,  # "bilateral poly-phonic wheezes"
            "lab:raised_wcc": True,  # 11.5, range 4.5-8
        },
        vocabulary=frozenset(),
    ),
    # ---- pericarditis (wave-2 list ranks 65, 76) ---------------------------
    # PMC6439374 -- recurrent idiopathic pericarditis with haemorrhagic
    # effusion; the readmission is the presentation extracted.
    Case(
        case_id="ho-6439374",
        presenting_complaint=(
            "22-year-old woman, pericarditis four months ago, three days of "
            "chest pain worse on breathing in and lying down"
        ),
        diagnosis="pericarditis",
        features={
            "pleuritic_pain": True,  # "inspiratory and worse on lying down"
            "exam:tachycardia": True,  # 115
            "fever": False,  # 37.8C, under this project's 38C
            "imaging:pericardial_effusion": True,  # 28 mm on echocardiography
            "lab:raised_troponin": False,  # < 0.012, reference < 0.08
            "lab:raised_wcc": False,  # 6810, reference 4000-11000
        },
        vocabulary=frozenset(),
    ),
    # PMC11089582 -- idiopathic acute (myo)pericarditis with tamponade after
    # a flu-like illness; renal cancer resected three years earlier.
    Case(
        case_id="ho-11089582",
        presenting_complaint=(
            "60-year-old man, days of fever, chills and sore throat, now "
            "chest pressure, breathlessness and lethargy"
        ),
        diagnosis="pericarditis",
        features={
            "fever": True,
            "dyspnoea_at_rest": True,  # "shortness of breath"
            "exam:tachycardia": True,  # heart rate in the 120s
            "exam:ecg_st_changes": True,  # "diffuse ST-elevation in all leads"
            "exam:ecg_st_elevation": True,
            "imaging:pericardial_effusion": True,  # "large pericardial effusion"
            # "bilateral lung infiltrate", read as consolidation as in ho-11162253
            "imaging:cxr_consolidation": True,
            "lab:raised_troponin": True,  # hs-troponin I 4,767 ng/L, reference 0-54
        },
        vocabulary=frozenset(),
    ),
)

HOLDOUT_CASES = HOLDOUT_CASES + WAVE3_CASES
WAVE3_IDS = frozenset(c.case_id for c in WAVE3_CASES)


# ---- wave 4 (Amendment 3) ---------------------------------------------------
# Empty in the commit that fixed the rule (c6e2363), as wave 1 was; filled in
# the next commit, before any model was run on these cases. PE 3, every other
# diagnosis 0 -- the walk and its reasons are in the module docstring.
WAVE4_CASES: tuple[Case, ...] = (
    # PMC11047289 -- massive PE with localised pulmonary oedema, a month after
    # rib fractures.
    Case(
        case_id="ho-11047289",
        presenting_complaint=(
            "47-year-old man, three weeks of worsening exertional "
            "breathlessness, chest pain and pinkish sputum, now breathless "
            "at rest"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "dyspnoea_at_rest": True,  # "it became noticeable at rest"
            "exam:tachycardia": True,  # pulse 126
            "exam:hypoxia": True,  # SpO2 81% on a reservoir mask at 15 L/min
            "fever": False,  # 37.7C, under 38C, as wave 3 recorded 37.8C
            "exam:crackles": True,  # "Crackles were auscultated in the left upper lung field"
            # "seated indoors for the whole day" for a month after rib
            # fractures; the report's own differential says "prolonged inactivity"
            "recent_immobility": True,
            "lab:raised_troponin": True,  # "elevated levels of troponin I"
            "lab:raised_bnp": True,  # NT-proBNP 5,200 pg/mL, "elevated"
            # chest radiograph: "consolidation in the left upper lung field"
            "imaging:cxr_consolidation": True,
            # contrast CT: "thrombotic occlusions were noted in the other
            # pulmonary artery branches"
            "imaging:ctpa_filling_defect": True,
        },
        vocabulary=frozenset(),
    ),
    # PMC6392898 -- high-risk PE in a previously healthy woman.
    Case(
        case_id="ho-6392898",
        presenting_complaint=(
            "55-year-old woman, 40 days of breathlessness and pleuritic chest "
            "pain, with brief syncope two days before admission"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "pleuritic_pain": True,
            "dyspnoea_at_rest": True,  # "complained of dyspnea"
            "syncope": True,  # "transitory (10 minutes) episodes of syncope"
            "fever": False,  # 36.0C
            "exam:tachycardia": False,  # pulse 98
            "exam:hypoxia": True,  # PaO2 52 mmHg on 5 L/min oxygen, cyanosed lips
            "exam:crackles": False,  # "no moist rales were heard"
            "exam:reduced_breath_sounds": True,  # "Breath sounds were diminished"
            "leg_swelling": False,  # "No edema was found in the lower extremities"
            "lab:raised_wcc": False,  # 6.82 x10^9/L
            # enhanced chest CT: "filling defects in the right and left main
            # pulmonary arteries"
            "imaging:ctpa_filling_defect": True,
            # troponin I 0.011, pro-BNP 3425 and D-dimer 2.47 are given
            # without a reference range: left out
        },
        vocabulary=frozenset(),
    ),
    # PMC7008764 -- PE presenting as recurrent syncope.
    Case(
        case_id="ho-7008764",
        presenting_complaint=(
            "60-year-old woman, syncope while walking after palpitations, mild "
            "chest discomfort and breathlessness; more than ten similar "
            "episodes over a year"
        ),
        diagnosis="pulmonary_embolism",
        features={
            "syncope": True,  # "lost consciousness for a minute"
            "palpitations": True,
            "dyspnoea_at_rest": True,  # "shortness of breath"
            "exam:tachycardia": False,  # heart rate 93 on initial evaluation
            # "physical exam including heart and lungs ... normal", read as
            # lungs clear
            "exam:crackles": False,
            "lab:raised_troponin": True,  # troponin T 0.015 ng/mL, normal < 0.010
            # CT angiography: "prominent bilateral pulmonary emboli"
            "imaging:ctpa_filling_defect": True,
            # SpO2 98% at first, 60% after a second syncope in the same
            # visit: conflicting, left out. "Non-specific ST and T wave
            # changes", later re-read as S1Q3T3: left out.
        },
        vocabulary=frozenset(),
    ),
)

HOLDOUT_CASES = HOLDOUT_CASES + WAVE4_CASES
WAVE4_IDS = frozenset(c.case_id for c in WAVE4_CASES)


__all__ = ["HOLDOUT_CASES", "WAVE3_CASES", "WAVE3_IDS", "WAVE4_CASES", "WAVE4_IDS"]
