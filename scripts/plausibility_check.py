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

What a pass proves, and how that changed
----------------------------------------
Read this before quoting the result anywhere.

On the first run this was a genuine validation. SCOPE.md's table had been
written by hand, from intuition, before most of the knowledge base was
sourced, so it was an *independent* claim: comparing it to the numbers could
find a number pointing the wrong way. It found four contradictions.

All four turned out to be the document's fault rather than the data's, for one
reason worth stating plainly: SCOPE.md was reasoning about each finding against
the general population, while the likelihood ratio here is against *this
knowledge base's own marginal*. Pleuritic pain genuinely suggests pulmonary
embolism in the world; it does not raise PE above the average of these
particular eight candidates, because pericarditis and pneumonia are sourced
higher for it. Both statements are true and they are not the same statement.

The table was then regenerated from the knowledge base, so the independence is
gone and so is the validation. **This script is now a regression test, not a
validation**: it detects the document and the code drifting apart, which is
worth having and is a much smaller claim. It can no longer tell you a number is
wrong, because the thing it compares against is now derived from that number.

Restoring the validation would need a claim written independently of the
knowledge base -- by a clinician, or transcribed from a reference text without
looking at the current values. That is exactly the physician-loop input the
project does not have.

Two further limits regardless. It says nothing about *magnitude*: a "raises"
finding at LR 1.01 passes and may still be badly wrong. And it only covers
findings the table names -- roughly a third of the knowledge base, weighted
toward the sourced ones, so the invented majority remains unchecked.
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
    ("pulmonary_embolism", "imaging:ctpa_filling_defect", "raises"),
    ("pulmonary_embolism", "lab:raised_d_dimer", "raises"),
    ("pulmonary_embolism", "sudden_onset", "raises"),
    ("pulmonary_embolism", "dyspnoea_at_rest", "raises"),
    ("pulmonary_embolism", "exam:hypoxia", "raises"),
    ("pulmonary_embolism", "orthopnoea", "lowers"),
    ("pulmonary_embolism", "imaging:cxr_pulmonary_oedema", "lowers"),
    ("pulmonary_embolism", "imaging:cxr_consolidation", "lowers"),
    ("community_acquired_pneumonia", "imaging:cxr_consolidation", "raises"),
    ("community_acquired_pneumonia", "fever", "raises"),
    ("community_acquired_pneumonia", "lab:raised_wcc", "raises"),
    ("community_acquired_pneumonia", "productive_cough", "raises"),
    ("community_acquired_pneumonia", "exam:crackles", "raises"),
    ("community_acquired_pneumonia", "dyspnoea_at_rest", "lowers"),
    ("community_acquired_pneumonia", "imaging:ctpa_filling_defect", "lowers"),
    ("community_acquired_pneumonia", "leg_swelling", "lowers"),
    ("acute_coronary_syndrome", "lab:raised_troponin", "raises"),
    ("acute_coronary_syndrome", "sudden_onset", "raises"),
    ("acute_coronary_syndrome", "productive_cough", "lowers"),
    ("acute_coronary_syndrome", "imaging:cxr_consolidation", "lowers"),
    ("acute_coronary_syndrome", "fever", "lowers"),
    ("acute_pulmonary_oedema", "imaging:cxr_pulmonary_oedema", "raises"),
    ("acute_pulmonary_oedema", "lab:raised_bnp", "raises"),
    ("acute_pulmonary_oedema", "orthopnoea", "raises"),
    ("acute_pulmonary_oedema", "leg_swelling", "raises"),
    ("acute_pulmonary_oedema", "exam:raised_jvp", "raises"),
    ("acute_pulmonary_oedema", "fever", "lowers"),
    ("acute_pulmonary_oedema", "lab:raised_wcc", "lowers"),
    ("acute_pulmonary_oedema", "imaging:cxr_consolidation", "lowers"),
    ("copd_exacerbation", "productive_cough", "raises"),
    ("copd_exacerbation", "dyspnoea_at_rest", "raises"),
    ("copd_exacerbation", "smoking_history", "raises"),
    ("copd_exacerbation", "exam:hypoxia", "raises"),
    ("copd_exacerbation", "imaging:cxr_pulmonary_oedema", "lowers"),
    ("copd_exacerbation", "sudden_onset", "lowers"),
    ("copd_exacerbation", "exam:raised_jvp", "lowers"),
    ("asthma_exacerbation", "dyspnoea_at_rest", "raises"),
    ("asthma_exacerbation", "sudden_onset", "raises"),
    ("asthma_exacerbation", "imaging:cxr_pulmonary_oedema", "lowers"),
    ("asthma_exacerbation", "lab:raised_bnp", "lowers"),
    ("asthma_exacerbation", "exam:crackles", "lowers"),
    ("pericarditis", "lab:raised_troponin", "raises"),
    ("pericarditis", "fever", "raises"),
    ("pericarditis", "pleuritic_pain", "raises"),
    ("pericarditis", "imaging:cxr_consolidation", "lowers"),
    ("pericarditis", "exam:crackles", "lowers"),
    ("pericarditis", "productive_cough", "lowers"),
    ("panic_attack", "sudden_onset", "raises"),
    ("panic_attack", "palpitations", "raises"),
    ("panic_attack", "exam:tachycardia", "raises"),
    ("panic_attack", "imaging:ctpa_filling_defect", "lowers"),
    ("panic_attack", "exam:hypoxia", "lowers"),
    ("panic_attack", "productive_cough", "lowers"),
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
