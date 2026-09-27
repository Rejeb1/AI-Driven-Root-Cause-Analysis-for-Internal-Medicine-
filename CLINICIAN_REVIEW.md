# Clinician review sheet

**For:** a clinician with about an hour. **From:** the dxagent project
(DeepShift AI internship, 2026). **Why:** every number and rule below is
either invented by a non-clinician or transcribed from a guideline by
one. This sheet lists only the items where a clinician's judgement would
change what the system does. Nothing else in the project needs your time.

**What the system is.** A diagnostic assistant for adults presenting with
acute breathlessness or chest pain, choosing among eight causes: pulmonary
embolism, community-acquired pneumonia, acute coronary syndrome, acute
pulmonary oedema, COPD exacerbation, asthma exacerbation, pericarditis and
panic attack. It asks questions and orders tests one at a time, then
either commits to a diagnosis or escalates ("I am not confident; a
clinician should decide"). It weighs findings with a probability for each
finding in each disease; 89 of those 177 probabilities carry a citation
(78 counted in published patient cohorts, 11 converted from textbook
sentences), and 88 are still guesses.

**How to answer.** A range is fine ("somewhere between 20 and 40%"). "I
don't know" and "the question is wrong" are useful answers too.

---

## Part 1 — Five policy decisions (about 15 minutes)

Each is currently set one way. The question is whether that is acceptable.

**1.1 When should the system commit instead of escalating?**
It commits only when its top diagnosis reaches 65% probability, leads
the runner-up by 15 points, and no time-critical diagnosis (PE, ACS)
remains above 10%. Otherwise it escalates. On 20 real published cases it
commits 8 times (7 correct, 1 wrong) and escalates 12.
*Is escalating on 12 of 20 real cases too cautious, about right, or not
cautious enough? Are 65% / 15 points / 10% sensible thresholds?*

**1.2 Pulmonary embolism workup trigger.**
A D-dimer (then imaging if positive) is required before any commit
whenever the patient has *any* of: pleuritic pain, breathlessness at
rest, low oxygen saturation (under 95%), or sudden onset. This follows
the PERC rule's "patients in whom PE is being considered." It orders
D-dimers on many pneumonia and COPD patients.
*Is this trigger too broad, too narrow, or right?*

**1.3 Pericarditis workup trigger.**
An ECG, an echocardiogram and auscultation for a rub are required
whenever the patient has pleuritic pain, a rub, *or ST changes*. ST
changes were added recently because they are one of the four ESC
criteria; that made the system ask about 8% more questions, since it now
also fires on heart attacks and embolisms with ST changes.
*Is arming a pericarditis workup on ST changes reasonable in an ED
chest-pain population?*

**1.4 Should a positive CT pulmonary angiogram be near-decisive?**
The system treats a positive CTPA as strong but not overriding evidence
(it sees filling defects in 83% of PE and 4% of non-PE). In one real case
(Part 3, case B) a CTPA-confirmed PE was ranked 7th of 8, because fever,
a productive cough and a *normal* D-dimer each argued against PE and
together outweighed the scan.
*In practice, does a positive CTPA end the discussion, or would you also
weigh the symptoms against it?*

**1.5 Fill in the remaining guesses, or leave them neutral?**
For 55 finding–disease pairs nobody has characterised (e.g. "palpitations
in pulmonary embolism", "wheeze in pneumonia"), the system currently uses
the average across all eight diseases. The alternative is to write in a
best guess for each. Measured on the real cases, the effect has pointed a
different way on each of the last four changes to the numbers, so the
measurements cannot decide this on their own.
*Would you rather the system use a rough clinical guess for pairs like
these, or stay neutral where no one has measured anything?*

---

## Part 2 — Eight guessed numbers that change diagnoses (about 15 minutes)

Changing any one of these by 15 percentage points changes at least one
diagnosis on the test cases. They are the guesses that matter most. For
each: out of 100 adults presenting to an ED with this condition, how many
would have this finding?

| Condition | Finding | Current guess | Your estimate |
|---|---|---|---|
| Asthma exacerbation | fever (≥38 °C) | 15 | |
| Asthma exacerbation | productive cough | 40 (a study caps any cough at 64) | |
| Asthma exacerbation | heart rate > 100 | 45 | |
| Asthma exacerbation | SpO₂ < 95% on air | 35 | |
| Asthma exacerbation | crackles | 10 | |
| Asthma exacerbation | white cell count > 11 | 20 | |
| Acute pulmonary oedema | troponin above the 99th centile | 30 | |
| Community-acquired pneumonia | D-dimer > 500 µg/L FEU | 35 (a study bounds it between 25 and 50) | |

---

## Part 3 — Four real cases (about 20 minutes)

All from open-access PubMed Central case reports; findings were extracted
by a non-clinician. For each, two questions: **(a) is the extraction
right?** and **(b) what would you have concluded from these findings
alone?**

**A. PMC13070269 — the one wrong commit.** 32-year-old man, burning
pleuritic chest pain radiating to the left shoulder, breathless, fever
the day before. Recorded: fever, pleuritic pain, rest dyspnoea,
tachycardia, ST changes, raised troponin, raised white count; no hypoxia,
no crackles, non-smoker. **True diagnosis: ACS. System: pneumonia at 76%,
committed.** *Is pneumonia a defensible reading of these findings, or is
this a clear error?*

**B. PMC11753817 — PE ranked 7th of 8.** 36-year-old man, a day of
fever, cough and chest pain, then sudden breathlessness. Recorded: fever,
productive cough, rest dyspnoea, sudden onset, tachycardia, hypoxia,
raised white count, **positive CTPA**; D-dimer normal, no consolidation on
X-ray. **True diagnosis: PE. System: pneumonia first (43%), escalated.**
*See 1.4.*

**C. PMC4565285 — PE ranked first, not committed.** 78-year-old man,
seven days of progressive exertional breathlessness, no chest pain.
Recorded: crackles, hypoxia, raised D-dimer, raised BNP, positive CTPA; no
tachycardia, no chest pain of any kind. **System: PE first at 49%,
escalated** ("findings not well explained by any diagnosis"). *Was
declining to commit reasonable here?*

**D. PMC12393936 — very thin record.** 52-year-old man with childhood
asthma, progressive breathlessness needing oxygen. Recorded: rest
dyspnoea, wheeze, smoker; no fever, no tachycardia. **True diagnosis:
asthma. System: ACS first (32%), asthma 5th, escalated.** *With only
these findings, what would you want to know next?*

---

## Part 4 — Three numbers that no study can supply (about 5 minutes)

These are about panic attack, which is diagnosed by excluding other
causes. A normal troponin, a normal oxygen saturation and the absence of
exertional pain are part of how the diagnosis is reached, so no study can
count how often they are abnormal in panic attack. Only a clinical
estimate can fill them.

| Finding in a confirmed panic attack | Current guess | Your estimate |
|---|---|---|
| raised troponin | 3 | |
| SpO₂ < 95% | 3 | |
| exertional chest pain | 20 | |

And one number the project knows is wrong: **productive cough in
pulmonary embolism** is set at 20% from a textbook sentence ("less common
symptoms include cough"), but a 360-patient cohort found *any* new cough
in only 4% of PE patients. *What would you put it at?*

---

## What happens to your answers

Each answer is written into the knowledge base with a citation marking it
as a clinician's estimate (a separate tier from published measurements),
and the effect on all test cases is measured and reported, including any
that get worse. Nothing you say is tuned to make a particular case come
out right.
