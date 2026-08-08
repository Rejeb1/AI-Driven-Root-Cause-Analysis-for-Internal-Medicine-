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

That check is not optional, and a first pass through this script proves it. It
resolved without error, printed a SNOMED CT code for every hit, and five of
those hits were the wrong concept: ``smoking_history`` came back as the
patient's *mother's* smoking history, ``lab:raised_wcc`` as a cerebrospinal-
fluid-specific leukocytosis, ``imaging:cxr_pulmonary_oedema`` as a risk
assessment rather than a finding, and ``imaging:ctpa_filling_defect`` as a
diagnosis-suspicion code because the search term used the disease's name
instead of the imaging sign. All four had a CUI, a SNOMED code and a plausible-
looking preferred name. ``_WRONG_SENSE`` now filters known bad patterns
(maternal/paternal history, "risk of", CSF-specific, etc.) as a backstop, and
the search terms below were corrected -- but the filter is a backstop, not a
substitute for reading the printed output.
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
#   T080 Qualifier Value (needed for pure temporal/severity qualifiers, which
#        SNOMED often models as qualifier values rather than findings)
_FINDING_TYPES = "T033,T184,T046,T201"
_RESULT_TYPES = "T034,T060,T033,T184"
_DISEASE_TYPES = "T047,T046,T048,T033"
# Qualifiers only. Including T033 here let "Sudden onset" match the finding
# "Edema of extremity of sudden onset", which contains the phrase and is not
# the concept: a temporal qualifier has to resolve to a qualifier.
_QUALIFIER_TYPES = "T080"
# Risk factors and exposures. T046 Pathologic Function is deliberately absent:
# with it, "Immobilization" resolved to "Thromboembolism of vein due to
# prolonged immobilization" -- the disorder the exposure causes rather than the
# exposure. A risk factor must not resolve to its own consequence, or the
# knowledge base asserts the conclusion in the premise.
_EXPOSURE_TYPES = "T033,T184,T201,T080"

# Substrings that mark a UMLS/SNOMED hit as the wrong sense even though it
# matched the search string. Found the hard way: a first pass through this
# script resolved "smoking_history" to "Maternal history of harmful pattern of
# tobacco use" (the patient's mother, not the patient), "lab:raised_wcc" to a
# cerebrospinal-fluid-specific leukocytosis concept, and
# "imaging:cxr_pulmonary_oedema" to a risk-assessment concept rather than the
# finding itself. All three had SNOMED CT codes and preferred names, and none
# of that made them right. Matched case-insensitively against the preferred
# name; a candidate that matches is skipped rather than accepted with a
# warning, because a warning is easy to scroll past and a missing entry is not.
_WRONG_SENSE = (
    "maternal",
    "paternal",
    "family history",
    "risk of",
    "risk for",
    "csf:",
    "cerebrospinal",
    "suspected",
    "screening for",
    "exposure to",
)


def _wrong_sense(preferred_name: str) -> bool:
    lowered = preferred_name.lower()
    return any(marker in lowered for marker in _WRONG_SENSE)

# Vocabulary key -> (term to search, semantic types to accept).
# Checked by eye against the UMLS preferred name the script prints back.
SEARCH_TERMS: dict[str, tuple[str, str]] = {
    # history
    "fever": ("Fever", _FINDING_TYPES),
    "productive_cough": ("Productive cough", _FINDING_TYPES),
    "pleuritic_pain": ("Pleuritic pain", _FINDING_TYPES),
    "dyspnoea_at_rest": ("Dyspnea at rest", _FINDING_TYPES),
    "orthopnoea": ("Orthopnea", _FINDING_TYPES),
    "leg_swelling": ("Swelling of leg", _FINDING_TYPES),
    "calf_tenderness": ("Pain in calf", _FINDING_TYPES),
    # "History of tobacco use" ranked a maternal/antenatal SNOMED concept
    # first. "Cigarette smoker" is the patient's own status and is what the
    # fixture concept means -- a Wells/PERC risk factor, not a family history.
    "smoking_history": ("Cigarette smoker", _EXPOSURE_TYPES),
    "exertional_chest_pain": ("Exertional chest pain", _FINDING_TYPES),
    "palpitations": ("Palpitations", _FINDING_TYPES),
    "wheeze_subjective": ("Wheezing", _FINDING_TYPES),
    "recent_immobility": ("Immobile", _EXPOSURE_TYPES),
    # examination
    "exam:crackles": ("Crackles", _FINDING_TYPES),
    "exam:tachycardia": ("Tachycardia", _FINDING_TYPES),
    "exam:hypoxia": ("Hypoxemia", _FINDING_TYPES),
    "exam:raised_jvp": ("Jugular venous distension", _FINDING_TYPES),
    "exam:reduced_breath_sounds": ("Decreased breath sounds", _FINDING_TYPES),
    "exam:friction_rub": ("Pericardial friction rub", _FINDING_TYPES),
    "exam:ecg_st_changes": ("ST segment changes", _RESULT_TYPES),
    # laboratory
    # "Leukocytosis" first ranked a CSF-specific (cerebrospinal fluid) subtype,
    # a different body fluid entirely. Rewriting the query as "White blood cell
    # count raised" then matched nothing at all. "Leukocytosis" is the term UMLS
    # actually indexes, so the query stands and the wrong subtype is excluded by
    # _WRONG_SENSE instead -- filtering the result is the right lever here,
    # contorting the query was not.
    "lab:raised_wcc": ("Leukocytosis", _RESULT_TYPES),
    "lab:raised_d_dimer": ("D-dimer above reference range", _RESULT_TYPES),
    "lab:raised_troponin": ("Troponin increased", _RESULT_TYPES),
    "lab:raised_bnp": ("Brain natriuretic peptide increased", _RESULT_TYPES),
    # imaging
    "imaging:cxr_consolidation": ("Consolidation of lung", _RESULT_TYPES),
    # Searching "Pulmonary edema" under result types (T034/T060/T033/T184)
    # skipped the disorder concept and matched a risk-assessment one instead.
    # SNOMED does not separately model "CXR shows pulmonary edema" from the
    # disorder itself, so this deliberately reuses the disease semantic types.
    "imaging:cxr_pulmonary_oedema": ("Pulmonary edema", _DISEASE_TYPES),
    # The original term here was "Pulmonary embolism" -- the disease name, not
    # the imaging finding -- which is what matched a diagnosis-suspicion
    # concept instead of a CT finding. "Filling defect of pulmonary artery"
    # names the actual radiological sign a CTPA is read for.
    "imaging:ctpa_filling_defect": (
        "Filling defect of pulmonary artery", _RESULT_TYPES,
    ),
    # "Sudden onset" alone matched an unrelated, over-specific finding
    # ("Edema of extremity of sudden onset") because SNOMED models onset as a
    # qualifier value (T080), which the finding-restricted search excluded.
    "sudden_onset": ("Sudden onset", _QUALIFIER_TYPES),

}

DISEASE_TERMS: dict[str, str] = {
    "pulmonary_embolism": "Pulmonary embolism",
    "community_acquired_pneumonia": "Community acquired pneumonia",
    "acute_coronary_syndrome": "Acute coronary syndrome",
    "acute_pulmonary_oedema": "Acute pulmonary edema",
    "copd_exacerbation": "Acute exacerbation of COPD",
    # "Acute asthma exacerbation" resolved to the intrinsic-asthma-specific
    # subtype (excludes allergic/extrinsic asthma), narrower than the fixture
    # condition, which does not distinguish asthma phenotype.
    "asthma_exacerbation": "Exacerbation of asthma",
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
    """Best-matching CUI for ``term``, restricted to SNOMED CT.

    Skips candidates whose preferred name marks them as the wrong sense (see
    ``_WRONG_SENSE``) rather than taking the first hit unconditionally. The
    first version took the first hit, and that is what produced the maternal-
    history and CSF-leukocytosis mismatches: both ranked first for their
    search string and both were wrong.
    """
    payload = request(
        "/search/current",
        api_key,
        string=term,
        sabs="SNOMEDCT_US",
        semanticTypes=semantic_types,
        pageSize=10,
    )
    for item in payload.get("result", {}).get("results", []):
        name = item.get("name", "")
        if item.get("ui") and item["ui"] != "NONE" and not _wrong_sense(name):
            return {"cui": item["ui"], "preferred_name": name}
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


def suggest(term: str, api_key: str) -> None:
    """Print what UMLS actually holds for a term, unfiltered.

    For diagnosing a failure rather than guessing at it. When a search returns
    nothing, the question is whether the concept is absent from SNOMED or the
    query is simply not how UMLS words it -- and those need different fixes.
    Semantic types and the wrong-sense filter are both disabled here, so the
    output shows every candidate along with the type that would have excluded
    it.
    """
    payload = request(
        "/search/current", api_key, string=term, sabs="SNOMEDCT_US", pageSize=20
    )
    results = payload.get("result", {}).get("results", [])
    if not results or results[0].get("ui") == "NONE":
        print(f"  nothing in SNOMEDCT_US for {term!r}")
        # Retry across all of UMLS: a concept can exist without a SNOMED atom,
        # which is a different problem from the term being wrong.
        payload = request("/search/current", api_key, string=term, pageSize=10)
        results = payload.get("result", {}).get("results", [])
        if results and results[0].get("ui") != "NONE":
            print("  but present elsewhere in UMLS:")
        else:
            print("  and nothing anywhere in UMLS -- the term is wrong.")
            return

    for item in results:
        if item.get("ui") == "NONE":
            continue
        name = item.get("name", "")
        flag = "  <-- excluded by _WRONG_SENSE" if _wrong_sense(name) else ""
        print(f"  {item['ui']:<10} {item.get('rootSource', ''):<12} {name}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("UMLS_API_KEY"))
    parser.add_argument("--out", default="data/umls_concepts.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--suggest",
        metavar="TERM",
        help="show unfiltered UMLS candidates for a term, to diagnose a failure",
    )
    args = parser.parse_args()

    if args.suggest:
        if not args.api_key:
            print('Set UMLS_API_KEY first.')
            return 1
        suggest(args.suggest, args.api_key)
        return 0

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
