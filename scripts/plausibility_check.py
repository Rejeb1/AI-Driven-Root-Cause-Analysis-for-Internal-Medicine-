#!/usr/bin/env python3
"""Check the knowledge base against its own stated clinical claims.

    python scripts/plausibility_check.py

Sourcing answers "is this number backed by a citation". It does not answer
"is this number right", and for 102 of 135 likelihoods nobody has checked
that at all -- sourcing and correctness are different questions, and this
project has only been answering the first one.

There is no physician to ask the second question of. What there is: SCOPE.md
already states, in prose, which findings should raise and which should lower
each of the eight diagnoses -- the "Discriminating evidence" table, written
before most of the knowledge base was sourced. That table is a falsifiable
claim about direction, encoded here as executable checks: for every (disease,
finding) pair SCOPE.md calls "raises it", the finding's likelihood ratio
against the knowledge-base marginal must exceed 1; for "lowers it", it must
be below 1.

This is a real check and a narrow one. It catches a number pointing the wrong
way entirely -- which is the kind of error a citation search would also catch,
just slower, and the kind an invented guess is most likely to make by
accident. It says nothing about *magnitude*: a "raises it" finding at LR 1.01
passes here and may still be a bad number. It also only checks the 47 claims
SCOPE.md happens to make, not the other ~90 unsourced likelihoods with no
stated claim to check against. Treat a pass as "not obviously wrong", not as
"verified".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.schemas import Finding, Polarity  # noqa: E402

# Transcribed directly from SCOPE.md section 3, "Discriminating evidence, by
# cause". Kept here rather than parsed from the markdown so a change to one is
# a visible diff against the other, not a silent drift -- the whole point of
# this script is to catch the two disagreeing.
CLAIMS: tuple[tuple[str, str, str], ...] = (
    # (disease, concept, "raises" | "lowers")
    ("pulmonary_embolism", "sudden_onset", "raises"),
    ("pulmonary_embolism", "pleuritic_pain", "raises"),
    ("pulmonary_embolism", "exam:hypoxia", "raises"),
    ("pulmonary_embolism", "lab:raised_d_dimer", "raises"),
    ("pulmonary_embolism", "imaging:ctpa_filling_defect", "raises"),
    ("pulmonary_embolism", "productive_cough", "lowers"),
    ("pulmonary_embolism", "fever", "lowers"),
    ("community_acquired_pneumonia", "fever", "raises"),
    ("community_acquired_pneumonia", "productive_cough", "raises"),
    ("community_acquired_pneumonia", "exam:crackles", "raises"),
    ("community_acquired_pneumonia", "lab:raised_wcc", "raises"),
    ("community_acquired_pneumonia", "imaging:cxr_consolidation", "raises"),
    ("community_acquired_pneumonia", "sudden_onset", "lowers"),
    ("community_acquired_pneumonia", "orthopnoea", "lowers"),
    ("acute_coronary_syndrome", "exertional_chest_pain", "raises"),
    ("acute_coronary_syndrome", "exam:ecg_st_changes", "raises"),
    ("acute_coronary_syndrome", "lab:raised_troponin", "raises"),
    ("acute_coronary_syndrome", "fever", "lowers"),
    ("acute_coronary_syndrome", "pleuritic_pain", "lowers"),
    ("acute_pulmonary_oedema", "orthopnoea", "raises"),
    ("acute_pulmonary_oedema", "exam:raised_jvp", "raises"),
    ("acute_pulmonary_oedema", "lab:raised_bnp", "raises"),
    ("acute_pulmonary_oedema", "imaging:cxr_pulmonary_oedema", "raises"),
    ("acute_pulmonary_oedema", "fever", "lowers"),
    ("copd_exacerbation", "smoking_history", "raises"),
    ("copd_exacerbation", "wheeze_subjective", "raises"),
    ("copd_exacerbation", "exam:reduced_breath_sounds", "raises"),
    ("copd_exacerbation", "sudden_onset", "lowers"),
    ("asthma_exacerbation", "wheeze_subjective", "raises"),
    ("asthma_exacerbation", "smoking_history", "lowers"),
    ("asthma_exacerbation", "fever", "lowers"),
    ("pericarditis", "pleuritic_pain", "raises"),
    ("pericarditis", "exam:friction_rub", "raises"),
    ("pericarditis", "lab:raised_bnp", "lowers"),
    ("pericarditis", "orthopnoea", "lowers"),
    ("panic_attack", "palpitations", "raises"),
    ("panic_attack", "exam:hypoxia", "lowers"),
    ("panic_attack", "exam:ecg_st_changes", "lowers"),
    ("panic_attack", "lab:raised_troponin", "lowers"),
)


def main() -> int:
    kb = build_knowledge_base()

    checked = 0
    skipped_single = 0
    failures: list[tuple[str, str, str, float, str]] = []
    for label, concept, direction in CLAIMS:
        entry = kb.get(label)
        if entry is None or concept not in entry.features:
            continue
        # A concept only one disease characterises has no real marginal to
        # compare against: background() averages over the diseases that
        # describe it, and with one describer that average *is* the value
        # being tested. The ratio is then mechanically 1.0 regardless of what
        # the number is, and reporting that as a contradiction blames the
        # data for a property of the check. exam:friction_rub is the case
        # that surfaced this: pericarditis is its only describer in this KB.
        describers = sum(1 for d in kb.diseases() if concept in d.features)
        if describers <= 1:
            skipped_single += 1
            continue
        checked += 1
        finding = Finding(concept, Polarity.PRESENT)
        background = kb.background_likelihood(finding)
        ratio = entry.likelihood(finding, kb.background(concept)) / background
        ok = (ratio > 1.0) if direction == "raises" else (ratio < 1.0)
        if not ok:
            source = entry.sources.get(concept)
            tier = source.provenance.value if source else "invented"
            failures.append((label, concept, direction, ratio, tier))

    print(f"{checked} of {len(CLAIMS)} SCOPE.md claims are checkable against a real marginal")
    print(f"{skipped_single} skipped: the concept has only one describing disease, so the")
    print("  ratio would be forced to exactly 1.0 regardless of the value -- a property")
    print("  of the check, not a finding about the data")
    print(f"{len(failures)} contradict the document\n")

    if not failures:
        print("None of the checked claims are contradicted. This is not a")
        print("validation of the 90-odd likelihoods with no stated claim to")
        print("check against -- see the module docstring.")
        return 0

    print(f"{'disease':<30} {'concept':<28} {'says':<8} {'LR':>6}  tier")
    print("-" * 84)
    for label, concept, direction, ratio, tier in failures:
        print(f"{label:<30} {concept:<28} {direction:<8} {ratio:>6.2f}  {tier}")

    print(
        "\nEach row is a number pointing the direction SCOPE.md says it should\n"
        "not. That means one of two things is wrong: the number, or the claim\n"
        "written down before the number was sourced. Read the citation (if any)\n"
        "before assuming it is the number -- SCOPE.md's table predates most of\n"
        "the sourcing work and was never revisited against it."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
