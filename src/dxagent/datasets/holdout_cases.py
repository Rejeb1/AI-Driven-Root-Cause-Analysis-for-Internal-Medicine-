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

HOLDOUT_CASES: tuple[Case, ...] = ()


__all__ = ["HOLDOUT_CASES"]
