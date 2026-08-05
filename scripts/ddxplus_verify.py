#!/usr/bin/env python3
"""Verify a local DDXPlus release before building anything on top of it.

Answers, in order:

  1. Which release files are present, and can they be opened?
  2. Does the column layout match what ``DDXPlusLoader`` expects?
  3. Which pathologies does the release actually contain?
  4. Do this project's eight target conditions map onto them?
  5. Can evidence codes be decoded into readable names?
  6. Does the differential column parse, and is it usable as a gold label?

Usage
-----
    python scripts/ddxplus_verify.py --root path/to/ddxplus
    python scripts/ddxplus_verify.py --root path/to/ddxplus --rows 0   # scan all

Download: https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374
(CC-BY). The release ships as zipped CSVs plus two JSON metadata files; this
script reads either the .zip or an unzipped .csv without needing them extracted.

Nothing here writes to the release or to ``data/``. It reports and exits.

What this script deliberately does not do
-----------------------------------------
It does not judge whether DDXPlus is *suitable*. Three properties of the
release are fixed by how it was generated and no amount of verification changes
them -- they are printed as reminders at the end because they must be stated in
the write-up, not discovered later:

  * Cases whose true pathology fell outside the rule-based system's differential
    were discarded during generation (paper 3.3). The hard tail is absent.
  * Evidences are conditionally independent given (pathology, age, sex) apart
    from a declared hierarchy (paper eq. 3), so a naive-Bayes likelihood table
    is the generating model rather than an approximation of it.
  * Pathology rates were capped to [10%, 100%] for balance (paper 3.2), so
    observed class frequencies are not clinical prevalence.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from difflib import get_close_matches
from pathlib import Path

# The eight root causes in scope, as named in this project, mapped to the
# pathology string expected in the release. Where the mapping is not one-to-one
# it is recorded here rather than silently normalised, because the mismatch is
# a scoping decision that belongs in the write-up.
TARGETS: dict[str, dict] = {
    "pulmonary embolism": {"expect": ["Pulmonary embolism"], "note": ""},
    "pneumonia": {
        "expect": ["Pneumonia"],
        "note": "release does not distinguish community- from hospital-acquired",
    },
    "acute coronary syndrome": {
        "expect": ["Possible NSTEMI / STEMI", "Unstable angina"],
        "note": "one label here spans two release pathologies; score a hit on either",
    },
    "acute pulmonary edema": {
        "expect": ["Acute pulmonary edema"],
        "note": "stands in for decompensated heart failure; the release has no chronic CHF",
    },
    "copd exacerbation": {"expect": ["Acute COPD exacerbation / infection"], "note": ""},
    "asthma exacerbation": {
        "expect": ["Bronchospasm / acute asthma exacerbation"],
        "note": "",
    },
    "pericarditis": {"expect": ["Pericarditis"], "note": ""},
    "panic attack": {"expect": ["Panic attack"], "note": ""},
}

# Conditions in the release that share the acute dyspnoea / chest-pain
# presentation without being target causes. These are the near-misses the
# differential has to discriminate against; evaluating only over TARGETS throws
# them away and makes the task easier than it should be.
KNOWN_DISTRACTORS = [
    "Spontaneous pneumothorax",
    "Myocarditis",
    "Atrial fibrillation",
    "PSVT",
    "GERD",
    "Boerhaave",
    "Anaphylaxis",
    "Anemia",
    "Stable angina",
    "Pulmonary neoplasm",
    "Tuberculosis",
    "Bronchitis",
    "Bronchiectasis",
    "Guillain-Barré syndrome",
    "Larygospasm",  # release spells it this way
    "Epiglottitis",
    "Influenza",
    "URTI",
    "Sarcoidosis",
    "Chagas",
]

EXPECTED_COLUMNS = {
    "pathology": "PATHOLOGY",
    "evidences": "EVIDENCES",
    "initial_evidence": "INITIAL_EVIDENCE",
    "differential": "DIFFERENTIAL_DIAGNOSIS",
    "age": "AGE",
    "sex": "SEX",
}

PATIENT_STEMS = [
    "release_train_patients",
    "release_validate_patients",
    "release_test_patients",
]


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode 'Guillain-Barre'
    with its accent and raises mid-report. Reconfigure rather than strip the
    accents, since the release spells them that way and the strings have to
    match it exactly."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def rule(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def open_patient_file(root: Path, stem: str) -> tuple[io.TextIOBase, str] | None:
    """Return a text handle over a patient split, from .csv or .zip.

    Returns None if neither form is present, so a partial download reports as a
    missing split rather than an exception.
    """
    csv_path = root / f"{stem}.csv"
    if csv_path.exists():
        return csv_path.open(newline="", encoding="utf-8"), f"{stem}.csv"

    zip_path = root / f"{stem}.zip"
    if zip_path.exists():
        archive = zipfile.ZipFile(zip_path)
        entries = [i for i in archive.infolist() if not i.is_dir()]
        if not entries:
            print(f"  {zip_path.name}: archive is empty")
            return None
        # The published release stores the CSV with no file extension at all
        # (the entry is literally 'release_train_patients'), so selecting on
        # '.csv' finds nothing. Prefer an explicit .csv when one exists, and
        # otherwise fall back to the largest entry.
        csvs = [i for i in entries if i.filename.lower().endswith(".csv")]
        chosen = csvs[0] if csvs else max(entries, key=lambda i: i.file_size)
        handle = io.TextIOWrapper(archive.open(chosen), encoding="utf-8", newline="")
        return handle, f"{zip_path.name}::{chosen.filename}"
    return None


def check_files(root: Path) -> dict[str, str]:
    """Inventory the release directory."""
    rule("1. release files")
    found: dict[str, str] = {}

    for stem in PATIENT_STEMS:
        opened = open_patient_file(root, stem)
        if opened is None:
            print(f"  MISSING   {stem}.csv / .zip")
            continue
        handle, label = opened
        handle.close()
        size = (root / f"{stem}.csv") if (root / f"{stem}.csv").exists() else (root / f"{stem}.zip")
        print(f"  ok        {label}  ({size.stat().st_size / 1e6:.1f} MB)")
        found[stem] = label

    for name in ("release_conditions.json", "release_evidences.json"):
        path = root / name
        if path.exists():
            print(f"  ok        {name}  ({path.stat().st_size / 1e3:.0f} KB)")
            found[name] = name
        else:
            print(f"  MISSING   {name}")

    if not any(s in found for s in PATIENT_STEMS):
        print("\n  No patient split found. Is --root pointing at the release directory?")
    return found


def check_columns(root: Path, stem: str) -> list[str]:
    """Compare the header against what DDXPlusLoader assumes."""
    rule("2. column layout")
    opened = open_patient_file(root, stem)
    if opened is None:
        print("  no split to check")
        return []
    handle, label = opened
    with handle:
        header = next(csv.reader(handle), [])

    print(f"  reading {label}")
    print(f"  found: {header}\n")

    ok = True
    for attr, expected in EXPECTED_COLUMNS.items():
        if expected in header:
            print(f"  ok        {attr:<18} -> {expected}")
        else:
            near = get_close_matches(expected, header, n=1, cutoff=0.6)
            hint = f"  (closest: {near[0]})" if near else ""
            print(f"  MISMATCH  {attr:<18} -> {expected} not in header{hint}")
            ok = False

    if not ok:
        print(
            "\n  DDXPlusLoader raises on a mismatch rather than mis-parsing, so pass\n"
            "  a DDXPlusColumns(...) override at the call site instead of editing\n"
            "  the defaults."
        )
    return header


def scan_rows(root: Path, stem: str, limit: int) -> tuple[Counter, list[dict], int]:
    """Count pathologies and keep a few rows back for format checks."""
    opened = open_patient_file(root, stem)
    if opened is None:
        return Counter(), [], 0
    handle, _ = opened
    counts: Counter[str] = Counter()
    samples: list[dict] = []
    n = 0
    with handle:
        for row in csv.DictReader(handle):
            counts[row.get(EXPECTED_COLUMNS["pathology"], "")] += 1
            if len(samples) < 3:
                samples.append(row)
            n += 1
            if limit and n >= limit:
                break
    return counts, samples, n


def check_pathologies(counts: Counter, scanned: int, root: Path) -> None:
    rule("3. pathologies in the release")
    conditions_path = root / "release_conditions.json"
    declared: list[str] = []
    if conditions_path.exists():
        meta = json.loads(conditions_path.read_text(encoding="utf-8"))
        declared = sorted(meta.keys())
        print(f"  release_conditions.json declares {len(declared)} conditions")
    print(f"  {len(counts)} distinct pathologies seen across {scanned:,} scanned rows\n")

    if declared and len(declared) != len(counts):
        unseen = set(declared) - set(counts)
        if unseen:
            print(f"  declared but not in the scan ({len(unseen)}): {sorted(unseen)[:8]}")
            print("  (expected if --rows sampled only part of the split)\n")

    for name, n in counts.most_common():
        share = 100.0 * n / max(scanned, 1)
        print(f"  {name:<45} {n:>8,}  {share:5.2f}%")


def check_targets(counts: Counter) -> None:
    rule("4. do the eight target conditions map onto the release?")
    available = list(counts)
    resolved = 0

    for target, spec in TARGETS.items():
        hits = [p for p in spec["expect"] if p in counts]
        missing = [p for p in spec["expect"] if p not in counts]

        if hits and not missing:
            total = sum(counts[h] for h in hits)
            detail = " + ".join(f"{h} ({counts[h]:,})" for h in hits)
            print(f"  ok        {target:<26} -> {detail}")
            resolved += 1
        elif hits:
            print(f"  PARTIAL   {target:<26} -> found {hits}, missing {missing}")
            resolved += 1
        else:
            near = get_close_matches(spec["expect"][0], available, n=2, cutoff=0.5)
            hint = f"  closest in release: {near}" if near else ""
            print(f"  MISSING   {target:<26} -> {spec['expect']}{hint}")
        if spec["note"]:
            print(f"            note: {spec['note']}")

    print(f"\n  {resolved}/{len(TARGETS)} target conditions resolved")

    present = [d for d in KNOWN_DISTRACTORS if d in counts]
    rule("   near-miss conditions available as hard negatives")
    print(f"  {len(present)}/{len(KNOWN_DISTRACTORS)} expected distractors present:")
    for name in present:
        print(f"    {name:<32} {counts[name]:>8,}")
    absent = [d for d in KNOWN_DISTRACTORS if d not in counts]
    if absent:
        print(f"  not found (check spelling against section 3): {absent}")
    print(
        "\n  Evaluate the differential over the full pathology set, not just the\n"
        "  eight targets -- restricting to the targets removes exactly the\n"
        "  confusable cases the evidence-weighting is supposed to handle."
    )


def check_evidences(root: Path, samples: list[dict]) -> None:
    rule("5. can evidence codes be decoded?")
    path = root / "release_evidences.json"
    if not path.exists():
        print("  release_evidences.json missing -- evidence codes cannot be decoded.")
        print("  Without it, explanations cite codes like 'E_54_@_V_89', which is")
        print("  unreadable and defeats the transparency requirement.")
        return

    meta = json.loads(path.read_text(encoding="utf-8"))
    print(f"  release_evidences.json defines {len(meta)} evidences")

    antecedents = sum(1 for v in meta.values() if v.get("is_antecedent"))
    print(f"  {antecedents} flagged as antecedents, {len(meta) - antecedents} as symptoms")
    print("  (the antecedent flag is how risk factors are told apart from")
    print("   symptoms -- the distinction UMLS could not give us)\n")

    if not samples:
        return
    raw = samples[0].get(EXPECTED_COLUMNS["evidences"], "")
    codes = _parse_literal(raw)
    print(f"  decoding {min(len(codes), 6)} codes from the first scanned row:")
    for code in codes[:6]:
        base = code.split("_@_")[0]
        entry = meta.get(base)
        if entry is None:
            print(f"    {code:<24} -> NOT FOUND in release_evidences.json")
            continue
        kind = "antecedent" if entry.get("is_antecedent") else "symptom  "
        name = entry.get("question_en") or entry.get("name") or "?"
        print(f"    {code:<24} -> [{kind}] {name[:60]}")


def check_differential(samples: list[dict]) -> None:
    rule("6. is the differential column usable as a gold label?")
    if not samples:
        print("  no rows scanned")
        return
    raw = samples[0].get(EXPECTED_COLUMNS["differential"], "")
    if not raw:
        print(f"  {EXPECTED_COLUMNS['differential']} is empty or absent in the scanned row")
        return

    parsed = _parse_literal(raw)
    print(f"  parsed {len(parsed)} entries from the first scanned row")
    ok_shape = all(isinstance(p, (list, tuple)) and len(p) == 2 for p in parsed)
    print(f"  shape is [[pathology, probability], ...]: {ok_shape}")
    if ok_shape:
        total = sum(float(p[1]) for p in parsed)
        print(f"  probabilities sum to {total:.3f}")
        print("\n  top of that differential:")
        for name, prob in sorted(parsed, key=lambda p: -float(p[1]))[:5]:
            print(f"    {float(prob):.3f}  {name}")

    print(
        "\n  Note: DDXPlusLoader names this column but load_cases() never reads it,\n"
        "  and Case carries only `diagnosis: str` while metrics.py takes\n"
        "  truths: dict[str, str]. DDx recall and any abstention proxy keyed on\n"
        "  differential breadth need that plumbed through first."
    )


def _parse_literal(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def reminders() -> None:
    rule("properties of the release that verification cannot fix")
    print(
        "  These follow from how DDXPlus was generated. State them in the\n"
        "  write-up rather than letting a reviewer find them.\n\n"
        "  survivorship   Cases whose true pathology fell outside the rule-based\n"
        "                 system's differential were discarded (paper 3.3). The\n"
        "                 hard tail is gone, so abstention is measured on a\n"
        "                 population with the difficult cases already removed.\n\n"
        "  independence   Evidences are sampled independently given pathology,\n"
        "                 age and sex (paper eq. 3). A naive-Bayes likelihood\n"
        "                 table is therefore the generating model, not an\n"
        "                 approximation -- high scores measure recovery of the\n"
        "                 simulator, not clinical reasoning.\n\n"
        "  priors         Rates were capped to [10%, 100%] for class balance\n"
        "                 (paper 3.2). Observed frequencies are not prevalence;\n"
        "                 take priors from the literature instead.\n\n"
        "  negatives      The authors do not guarantee that relevant negative\n"
        "                 evidence was collected (conclusion). Absence in a row\n"
        "                 is an annotation artifact, so 'contradicts' edges must\n"
        "                 come from guideline text, not from this data.\n\n"
        "  Report results as agreement with the DDXPlus differential, which is\n"
        "  what they are, rather than as diagnostic accuracy."
    )


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="DDXPlus release directory")
    parser.add_argument(
        "--split",
        default="release_validate_patients",
        choices=PATIENT_STEMS,
        help="which split to scan (default: validate, the smallest)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200_000,
        help="rows to scan; 0 scans the whole split (default: 200000)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"--root {root} does not exist.")
        print("Download: https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374")
        return 1

    found = check_files(root)
    if not any(s in found for s in PATIENT_STEMS):
        return 1

    split = args.split if args.split in found else next(s for s in PATIENT_STEMS if s in found)
    if split != args.split:
        print(f"\n  {args.split} not present; scanning {split} instead")

    check_columns(root, split)
    counts, samples, scanned = scan_rows(root, split, args.rows)
    if not scanned:
        print("\nNo rows read. Check the column layout above.")
        return 1

    check_pathologies(counts, scanned, root)
    check_targets(counts)
    check_evidences(root, samples)
    check_differential(samples)
    reminders()
    return 0


if __name__ == "__main__":
    sys.exit(main())
