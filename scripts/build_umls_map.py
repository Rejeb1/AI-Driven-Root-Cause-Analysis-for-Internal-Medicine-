#!/usr/bin/env python3
"""Resolve every concept in the vocabulary to a UMLS CUI and SNOMED CT code.

    $env:UMLS_API_KEY="..."           # PowerShell
    python scripts/build_umls_map.py            # writes data/umls_concepts.json
    python scripts/build_umls_map.py --dry-run  # show the search terms only

Section 6 of the brief mandates UMLS, SNOMED CT and RxNorm for "authoritative
concepts and relations to discipline the causal reasoning". This delivers the
concepts half, which is the half UMLS is good at.

The relations half is not delivered, and the reason is measured rather than
assumed: ``scripts/umls_probe.py`` found that for these eight conditions UMLS
relations are overwhelmingly translations, ICD crosswalks and MedDRA
groupings, and that its one clinically meaningful label,
``clinically_associated_with``, mixes symptoms, risk factors and treatment
complications without marking which is which. Supports/contradicts edges are
therefore hand-encoded from published decision rules with citations; see
``guidelines.py``. Run the probe to reproduce that finding before accepting it.

RxNorm is not used because nothing in scope is a medication. The target
presentation is a differential over eight acute conditions and the vocabulary
holds symptoms, signs, laboratory results and imaging findings. RxNorm names
drugs and their ingredients, so it has nothing to contribute here -- that is a
scope fact, not a substitution.

Why the search terms are written out below
------------------------------------------
``exam:crackles`` is a vocabulary key, not something to send to a search API,
and turning it into "crackles" mechanically produces hits for the wrong sense
about as often as the right one. Each term is written by hand so that a wrong
resolution is a visible line in this file rather than a silent transformation,
and every result is printed with its preferred name so the mapping can be
checked against what UMLS thought you meant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent.datasets import build_knowledge_base  # noqa: E402

BASE = "https://uts-ws.nlm.nih.gov/rest"

# Semantic types worth accepting per kind of concept. Restricting the search is
# what stops "asthma" resolving to a questionnaire item and "troponin" to the
# protein rather than the test result.
#   T033 Finding          T184 Sign or Symptom      T046 Pathologic Function
#   T047 Disease/Syndrome T048 Mental/Behavioral    T034 Lab or Test Result
#   T060 Diagnostic Procedure                       T201 Clinical Attribute
_FINDING_TYPES = "T033,T184,T046,T201"
_RESULT_TYPES = "T034,T060,T033,T184"
_DISEASE_TYPES = "T047,T046,T048,T033"

# Vocabulary key -> (term to search, semantic types to accept).
# Checked by eye against the UMLS preferred name the script prints back.
SEARCH_TERMS: dict[str, tuple[str, str]] = {
    # history
    "fever": ("Fever", _FINDING_TYPES),
    "productive_cough": ("Productive cough", _FINDING_TYPES),
    "pleuritic_pain": ("Pleuritic pain", _FINDING_TYPES),
    "dyspnoea_at_rest": ("Dyspnea at rest", _FINDING_TYPES),
    "sudden_onset": ("Sudden onset", _FINDING_TYPES),
    "orthopnoea": ("Orthopnea", _FINDING_TYPES),
    "leg_swelling": ("Swelling of lower extremity", _FINDING_TYPES),
    "calf_tenderness": ("Calf tenderness", _FINDING_TYPES),
    "smoking_history": ("History of tobacco use", _FINDING_TYPES),
    "exertional_chest_pain": ("Exertional chest pain", _FINDING_TYPES),
    "palpitations": ("Palpitations", _FINDING_TYPES),
    "wheeze_subjective": ("Wheezing", _FINDING_TYPES),
    "recent_immobility": ("Immobility", _FINDING_TYPES),
    # examination
    "exam:crackles": ("Crackles", _FINDING_TYPES),
    "exam:tachycardia": ("Tachycardia", _FINDING_TYPES),
    "exam:hypoxia": ("Hypoxemia", _FINDING_TYPES),
    "exam:raised_jvp": ("Jugular venous distension", _FINDING_TYPES),
    "exam:reduced_breath_sounds": ("Decreased breath sounds", _FINDING_TYPES),
    "exam:friction_rub": ("Pericardial friction rub", _FINDING_TYPES),
    "exam:ecg_st_changes": ("ST segment changes", _RESULT_TYPES),
    # laboratory
    "lab:raised_wcc": ("Leukocytosis", _RESULT_TYPES),
    "lab:raised_d_dimer": ("Elevated D-dimer", _RESULT_TYPES),
    "lab:raised_troponin": ("Elevated troponin", _RESULT_TYPES),
    "lab:raised_bnp": ("Elevated brain natriuretic peptide", _RESULT_TYPES),
    # imaging
    "imaging:cxr_consolidation": ("Pulmonary consolidation", _RESULT_TYPES),
    "imaging:cxr_pulmonary_oedema": ("Pulmonary edema", _RESULT_TYPES),
    "imaging:ctpa_filling_defect": ("Pulmonary embolism", _RESULT_TYPES),
}

DISEASE_TERMS: dict[str, str] = {
    "pulmonary_embolism": "Pulmonary embolism",
    "community_acquired_pneumonia": "Community acquired pneumonia",
    "acute_coronary_syndrome": "Acute coronary syndrome",
    "acute_pulmonary_oedema": "Acute pulmonary edema",
    "copd_exacerbation": "Acute exacerbation of COPD",
    "asthma_exacerbation": "Acute asthma exacerbation",
    "pericarditis": "Pericarditis",
    "panic_attack": "Panic attack",
}


def request(path: str, api_key: str, **params) -> dict:
    params["apiKey"] = api_key
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise SystemExit(
                "401 Unauthorized. Check UMLS_API_KEY at "
                "https://uts.nlm.nih.gov -> My Profile."
            ) from exc
        if exc.code == 404:
            return {}
        raise


def resolve(term: str, semantic_types: str, api_key: str) -> dict | None:
    """Best-matching CUI for ``term``, restricted to SNOMED CT."""
    payload = request(
        "/search/current",
        api_key,
        string=term,
        sabs="SNOMEDCT_US",
        semanticTypes=semantic_types,
        pageSize=5,
    )
    for item in payload.get("result", {}).get("results", []):
        if item.get("ui") and item["ui"] != "NONE":
            return {"cui": item["ui"], "preferred_name": item.get("name", "")}
    return None


def snomed_code(cui: str, api_key: str) -> str:
    """The SNOMED CT identifier for a CUI, or empty when it has none.

    A CUI is a UMLS concept; the SNOMED code is what a clinical system would
    actually exchange, so both are recorded rather than treating them as
    interchangeable.
    """
    payload = request(
        f"/content/current/CUI/{cui}/atoms",
        api_key,
        sabs="SNOMEDCT_US",
        pageSize=5,
    )
    for atom in payload.get("result", []):
        code_url = atom.get("code") or ""
        if code_url:
            return code_url.rstrip("/").rsplit("/", 1)[-1]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("UMLS_API_KEY"))
    parser.add_argument("--out", default="data/umls_concepts.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    kb = build_knowledge_base()
    concepts = sorted({c for e in kb.diseases() for c in e.features})

    missing = [c for c in concepts if c not in SEARCH_TERMS]
    if missing:
        print(f"no search term written for: {missing}")
        print("Add one to SEARCH_TERMS rather than letting it resolve by accident.")
        return 1

    if args.dry_run:
        print(f"{len(concepts)} findings and {len(DISEASE_TERMS)} conditions\n")
        for concept in concepts:
            print(f"  {concept:<32} -> {SEARCH_TERMS[concept][0]!r}")
        for label, term in DISEASE_TERMS.items():
            print(f"  {label:<32} -> {term!r}")
        return 0

    if not args.api_key:
        print(
            'No API key. PowerShell:  $env:UMLS_API_KEY="your-key"\n'
            "Get one at https://uts.nlm.nih.gov -> sign in -> My Profile.\n"
            "Use --dry-run to see the search terms without a key."
        )
        return 1

    out: dict[str, dict] = {"findings": {}, "conditions": {}}
    unresolved: list[str] = []

    for kind, items in (
        ("findings", [(c, *SEARCH_TERMS[c]) for c in concepts]),
        ("conditions", [(k, v, _DISEASE_TYPES) for k, v in DISEASE_TERMS.items()]),
    ):
        print(f"\n{kind}")
        for key, term, types in items:
            found = resolve(term, types, args.api_key)
            if found is None:
                print(f"  UNRESOLVED  {key:<32} {term!r}")
                unresolved.append(key)
                continue
            code = snomed_code(found["cui"], args.api_key)
            out[kind][key] = {
                "search_term": term,
                "cui": found["cui"],
                "snomed_ct": code,
                "umls_preferred_name": found["preferred_name"],
            }
            # Printing the preferred name is the check: if it does not describe
            # the concept you meant, the search term is wrong, not the data.
            print(
                f"  {key:<32} {found['cui']:<10} "
                f"{('SCT ' + code) if code else 'no SCT code':<18} "
                f"{found['preferred_name']}"
            )
            time.sleep(0.15)  # be polite to NLM

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    resolved = len(out["findings"]) + len(out["conditions"])
    total = len(concepts) + len(DISEASE_TERMS)
    print(f"\nresolved {resolved}/{total}; wrote {path}")
    if unresolved:
        print(
            "unresolved: " + ", ".join(unresolved) + "\nFix the search term rather "
            "than accepting a near miss -- an approximately right CUI is worse "
            "than none, because it looks authoritative."
        )
    print(
        "\nCheck the preferred names above against what you meant. A CUI that "
        "resolves\nbut names something else is the failure mode this cannot "
        "detect for you."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
