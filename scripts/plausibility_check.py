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
    # Was "lowers" while the Merck-derived 0.05 stood. That value came
    # from a chapter describing pneumonia at every severity, most of it
    # managed at home; in a cohort of acutely admitted patients the rate
    # is 0.67, and the claim is dropped rather than reversed because an
    # LR of 1.14 is not a discriminating finding in either direction.
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


# Ordering claims: which disease must lead a column, and over which rivals
# ---------------------------------------------------------------------------
# Now in dxagent.claims, because the loop reads them too (a commit discloses
# which rivals' defining findings it never examined). The audit is unchanged.
from dxagent.claims import ORDERING_CLAIMS  # noqa: E402

# Ordering claims this knowledge base currently fails, each with the reason it
# is tolerated. The list is expected to shrink and must never grow silently: a
# new violation means either a number moved the wrong way or a claim here is
# wrong, and both should stop a build.
KNOWN_ORDERING_VIOLATIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "exam:hypoxia",
        "acute_pulmonary_oedema",
        "copd_exacerbation",
        "Alveolar flooding impairing gas exchange is the mechanism that "
        "defines acute pulmonary oedema, so it should lead this column. It "
        "sits at an invented 0.40, below COPD at 0.55 and pulmonary embolism "
        "at 0.60. "
        "Unsourceable, and the reason is structural rather than a gap in the "
        "literature. Six searches found no cohort reporting oxygen "
        "saturation by final diagnosis in unselected dyspnoea, and what they "
        "found instead explains why: saturation is part of how the severe "
        "presentation is identified. Trials of acute cardiogenic pulmonary "
        "oedema enrol on SpO2 below 90%, and the AHEAD registry classifies "
        "the syndrome as severe respiratory distress with crackles and "
        "orthopnoea and oxygen saturation usually under 90% before "
        "treatment, a figure that appears only in the classification "
        "criterion and never as an observed measurement afterwards. "
        "Taking it would be circular in the way this project already refuses "
        "for panic attack troponin: the patients are in the group partly "
        "because of the finding being counted. Sourcing this cell needs a "
        "cohort of undifferentiated dyspnoea in which pulmonary oedema is "
        "diagnosed by other means, natriuretic peptide or echocardiography, "
        "with saturation reported afterwards. That study may exist; it has "
        "not been found. "
        "The entanglement also means the invented 0.40 is wrong in a second "
        "way: if hypoxia is quasi-definitional for the florid presentation, "
        "the value belongs nearer the 0.95 that consolidation carries for "
        "pneumonia than to the middle of a column.",
    ),
    (
        "exam:hypoxia",
        "acute_pulmonary_oedema",
        "pulmonary_embolism",
        "The same cell, against the other rival that outranks it.",
    ),
)


def ordering_failures(kb):
    """Every ordering claim the knowledge base violates.

    Yields (concept, leader, rival, leader_value, rival_value).
    """
    out = []
    for concept, leader, rivals, _why in ORDERING_CLAIMS:
        entry = kb.get(leader)
        if entry is None or concept not in entry.features:
            continue
        top = entry.features[concept]
        for rival in rivals:
            other = kb.get(rival)
            if other is None or concept not in other.features:
                continue
            if other.features[concept] >= top:
                out.append((concept, leader, rival, top, other.features[concept]))
    return out


# Distribution-free bounds on invented cells, from cohorts that report a
# median and interquartile range rather than a fraction above a threshold.
# Where the threshold falls among the quartiles bounds the fraction without
# any distributional assumption: above Q3 means fewer than a quarter exceed
# it, between the median and Q3 means between a quarter and a half. fixtures.py
# refuses to turn such a bound into a point estimate -- that would need a
# log-normal assumption no other cell depends on -- and that refusal stands.
# What a bound can do is check an invented value, the way an ordering claim
# does: constrain without asserting a magnitude.
#
#   (disease, concept, low, high, source, what the quartiles were)
QUARTILE_BOUNDS: tuple[tuple[str, str, float, float, str, str], ...] = (
    ("copd_exacerbation", "exam:tachycardia", 0.25, 0.50,
     "PMC6307152 Table 2, 437 hospitalised COPD exacerbations, admission",
     "heart rate median 91.5-95, IQR to 102.5-108 bpm; the 100 cutoff sits "
     "between the median and Q3 in both arms"),
    ("copd_exacerbation", "lab:raised_wcc", 0.25, 0.50,
     "PMC6307152 Table 2, 437 hospitalised COPD exacerbations, admission",
     "leucocytes median 10.1, IQR 7.7-14 x10^9/L; the 11 cutoff sits between "
     "the median and Q3"),
    # Not written, and recorded: the same table has admission temperature at
    # median 36.4 with Q3 37.0, so fewer than a quarter were febrile on
    # measurement. That bounds measured fever, and this project's fever
    # concept is measured OR reported. Not a bound on the cell.
)


def bound_failures(kb) -> list[tuple[str, str, float, float, float]]:
    """Invented or sourced cells outside a distribution-free bound."""
    out = []
    for label, concept, low, high, _src, _why in QUARTILE_BOUNDS:
        entry = kb.get(label)
        if entry is None or concept not in entry.features:
            continue
        value = entry.features[concept]
        if not (low <= value <= high):
            out.append((label, concept, value, low, high))
    return out


def report_bounds(kb) -> int:
    """Print the quartile-bound audit. Returns the number of failures."""
    failures = bound_failures(kb)
    print()
    print(f"{len(QUARTILE_BOUNDS)} distribution-free bounds from cohort quartiles")
    print("  a bound checks an invented value without replacing it -- see")
    print("  QUARTILE_BOUNDS for the sources and where each cutoff fell")
    for label, concept, low, high, _src, _why in QUARTILE_BOUNDS:
        entry = kb.get(label)
        value = entry.features.get(concept) if entry else None
        mark = "  " if value is not None and low <= value <= high else "!!"
        shown = f"{value:.2f}" if value is not None else "--"
        print(f"  {mark} {label:<28} {concept:<20} {shown} in [{low:.2f}, {high:.2f}]")
    print(f"{len(failures)} outside their bound")
    return len(failures)


def report_ordering(kb) -> int:
    """Print the ordering audit. Returns the number of unexpected failures."""
    import textwrap

    failures = ordering_failures(kb)
    seen = {(c, l, r) for c, l, r, _, _ in failures}
    known = {(c, l, r) for c, l, r, _w in KNOWN_ORDERING_VIOLATIONS}
    compared = sum(len(rivals) for _c, _l, rivals, _w in ORDERING_CLAIMS)

    print()
    print(f"{len(ORDERING_CLAIMS)} ordering claims, {compared} rival comparisons")
    print("  which disease must lead each column and over which rivals, with a")
    print("  textbook reason per claim -- see ORDERING_CLAIMS for the reasons")

    unexpected = seen - known
    print(f"{len(failures)} violated, {len(unexpected)} of them unexpected")
    print()

    if failures:
        print(f"{'concept':<28} {'should lead':<26} {'but below':<26}  values")
        print("-" * 96)
        for concept, leader, rival, top, other in failures:
            flag = "" if (concept, leader, rival) in known else "  <-- NEW"
            print(f"{concept:<28} {leader:<26} {rival:<26}  "
                  f"{top:.2f} vs {other:.2f}{flag}")
        print()
        for concept, leader, rival, why in KNOWN_ORDERING_VIOLATIONS:
            if (concept, leader, rival) in seen:
                print(f"known, and why it is tolerated -- {concept} / {leader}:")
                for line in textwrap.wrap(why, 74):
                    print(f"    {line}")
                break
    return len(unexpected)


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
        ratio = entry.likelihood(finding, kb.backoff(concept)) / background
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
        return report_ordering(kb) + report_bounds(kb)

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
    return 1 + report_ordering(kb) + report_bounds(kb)


if __name__ == "__main__":
    raise SystemExit(main())
