"""Findings vocabulary, backed by the Human Phenotype Ontology.

Scope, stated plainly because it is easy to over-claim here:

  HPO supplies the NODES -- a stable, citable identifier for each clinical
  finding, with synonyms and a subsumption hierarchy. This works well; the
  target presentations are covered, including laboratory and ECG findings.

  HPO does NOT supply the EDGES. Its disease-phenotype annotations cover rare
  Mendelian disease only (OMIM/Orphanet/DECIPHER); the 2026-06-23 release has
  no entry for pulmonary embolism, myocardial infarction, angina, or urinary
  tract infection. Of 285,598 annotations, 727 carry the NOT qualifier, so
  there is no usable 'contradicts' layer either. Disease-finding edges must
  come from elsewhere and be sourced individually.

This module is therefore the concept layer only. It stands in for UMLS/SNOMED
while that licence is pending, and it is deliberately shaped so that swapping
in UMLS later means adding a second identifier per concept, not rewriting call
sites.

Mapping policy
--------------
Local concept keys are mapped to HPO terms by an EXPLICIT CURATED TABLE, never
by string matching. Substring matching on this vocabulary is actively unsafe:
searching 'rale' against HPO returns 'P mitrale', an ECG finding. Every mapping
below is an assertion a human made and a reviewer can check, which is the whole
point of a vetted knowledge base.

Unmapped is a legitimate state. A concept with no HPO term is carried with
``hpo_id=None`` rather than forced onto an approximate parent, because a wrong
normalisation is far more damaging than a missing one -- it silently merges
findings that should stay distinct.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HPO_RELEASE = "hp/releases/2026-06-23"
HPO_SOURCE = (
    "https://github.com/obophenotype/human-phenotype-ontology/releases/"
    "latest/download/hp.obo"
)


@dataclass(frozen=True)
class Term:
    """One HPO term."""

    hpo_id: str
    name: str
    synonyms: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    definition: str = ""


@dataclass(frozen=True)
class Concept:
    """A finding in the local vocabulary, optionally bound to an HPO term."""

    key: str  # local stable key used throughout the codebase
    hpo_id: str | None
    hpo_name: str | None
    modality: str  # history | exam | lab | ecg | imaging
    note: str = ""
    # UMLS and SNOMED CT identifiers, from data/umls_concepts.json when it has
    # been built. Carried alongside the HPO id rather than replacing it: HPO is
    # a phenotype ontology with a hierarchy this vocabulary already uses, and
    # SNOMED CT is what a clinical system would exchange. They answer different
    # questions and collapsing them would lose one of the answers.
    cui: str | None = None
    snomed_ct: str | None = None
    umls_name: str | None = None

    @property
    def is_grounded(self) -> bool:
        return self.hpo_id is not None

    @property
    def is_umls_grounded(self) -> bool:
        """Whether section 6's mandated ontology layer covers this concept."""
        return self.cui is not None

    def __str__(self) -> str:  # pragma: no cover - display only
        ids = self.hpo_id or "UNMAPPED"
        if self.cui:
            ids += f" / {self.cui}"
        return f"{self.key} [{ids}]"


# ---------------------------------------------------------------------------
# Curated mapping.
#
# Each row: local key -> (HPO id, modality, note). Verified against the
# 2026-06-23 release. Notes record judgement calls so a reviewer can disagree
# with a specific decision rather than the table as a whole.
# ---------------------------------------------------------------------------

CURATED: dict[str, tuple[str | None, str, str]] = {
    # -- history ------------------------------------------------------------
    "fever": ("HP:0001945", "history", ""),
    "productive_cough": (
        "HP:0031245",
        "history",
        "'Productive cough'. NOTE: HP:0031246 is 'Nonproductive cough' -- the "
        "adjacent id is the exact opposite concept. An earlier version of this "
        "table had that inversion and it was caught only by printing every "
        "resolved term name. Adjacent HPO ids are frequently antonyms; never "
        "trust an id without rendering its label.",
    ),
    "cough": ("HP:0012735", "history", ""),
    "pleuritic_pain": ("HP:0033771", "history", "Pleuritic chest pain"),
    "exertional_chest_pain": (
        "HP:0100749",
        "history",
        "mapped to generic 'Chest pain'; HPO has no exertional qualifier, so "
        "the exertional character is LOST in normalisation and must be carried "
        "in Finding.raw. This is a real information loss, not a tidy mapping.",
    ),
    "dyspnoea_at_rest": (
        "HP:0002094",
        "history",
        "'Dyspnea'; the at-rest qualifier is not represented in HPO",
    ),
    "orthopnoea": ("HP:0012764", "history", "Orthopnea"),
    "leg_swelling": ("HP:0012398", "history", "Peripheral edema"),
    "calf_tenderness": (
        None,
        "history",
        "no adequate HPO term; 'Abnormal calf musculature morphology' "
        "HP:0001430 is morphological, not a tenderness sign. Left unmapped "
        "deliberately -- see module docstring.",
    ),
    "palpitations": ("HP:0001962", "history", ""),
    "wheeze_subjective": (
        "HP:0030828",
        "history",
        "'Wheezing'; HPO does not distinguish reported from auscultated",
    ),
    "sudden_onset": (
        None,
        "history",
        "temporal qualifier, not a phenotype; belongs in a separate onset "
        "field rather than the findings vocabulary",
    ),
    "recent_immobility": (
        None,
        "history",
        "risk factor / exposure, not a phenotype. HPO is the wrong ontology "
        "for this class; needs its own vocabulary.",
    ),
    "smoking_history": (
        None,
        "history",
        "behavioural exposure, not a phenotype. Same class as above.",
    ),
    "haemoptysis": ("HP:0002105", "history", "Hemoptysis"),
    "syncope": ("HP:0001279", "history", ""),
    "night_sweats": ("HP:0030166", "history", ""),
    "weight_loss": ("HP:0001824", "history", ""),
    "fatigue": ("HP:0012378", "history", ""),
    # -- examination --------------------------------------------------------
    "exam:crackles": ("HP:0030830", "exam", "Crackles"),
    "exam:reduced_breath_sounds": (
        "HP:0030829",
        "exam",
        "mapped to parent 'Abnormal breath sound'; HPO lacks a specific "
        "reduced/diminished term, so this is deliberately less specific",
    ),
    "exam:raised_jvp": ("HP:0030847", "exam", "Abnormal jugular venous pressure"),
    "exam:tachycardia": ("HP:0001649", "exam", ""),
    "exam:friction_rub": ("HP:0034788", "exam", "Pericardial friction rub"),
    "exam:hypoxia": (
        "HP:0012418",
        "exam",
        "'Hypoxemia' -- note HP:0005947 'Decreased sensitivity to hypoxemia' "
        "is a different concept and a naive search returns it first",
    ),
    "exam:deep_vein_thrombosis": ("HP:0002625", "exam", "Deep venous thrombosis"),
    # -- ECG ----------------------------------------------------------------
    "exam:ecg_st_changes": (
        "HP:0012249",
        "ecg",
        "parent 'Abnormal ST segment'; elevation HP:0012251 and depression "
        "HP:0012250 are distinct children and should be separated once the "
        "case representation supports it -- they carry different implications",
    ),
    "ecg:st_elevation": ("HP:0012251", "ecg", ""),
    "ecg:st_depression": ("HP:0012250", "ecg", ""),
    "exam:ecg_pr_depression": (
        None,
        "ecg",
        "PR-segment depression, one of the two ECG findings StatPearls calls "
        "the most characteristic of acute pericarditis. HPO has no PR-segment "
        "term. Added after a real pericarditis committed as pneumonia at 78% "
        "because nothing in this vocabulary could see what decided the case; "
        "kept under the exam: prefix so it classifies and costs like "
        "exam:ecg_st_changes.",
    ),
    # -- laboratory ---------------------------------------------------------
    "lab:raised_d_dimer": ("HP:0033106", "lab", "Elevated circulating D-dimer"),
    "lab:raised_troponin": (
        "HP:0410173",
        "lab",
        "troponin I; troponin T is a separate term HP:0410174. Assays differ "
        "and conflating them is a real-world error.",
    ),
    "lab:raised_bnp": (
        "HP:0031138",
        "lab",
        "'Abnormal circulating BNP concentration' -- parent term; verify "
        "whether a raised-specific child exists in this release",
    ),
    "lab:raised_wcc": (
        "HP:0001974",
        "lab",
        "'Increased total leukocyte count'. Chosen over HP:6001425 "
        "'Hyperleukocytosis', which denotes a far more extreme elevation "
        "(leukostasis territory) than a routine raised white count.",
    ),
    # -- imaging ------------------------------------------------------------
    "imaging:cxr_consolidation": (
        None,
        "imaging",
        "radiological finding; HPO coverage of imaging descriptors is thin. "
        "RadLex is the appropriate vocabulary here.",
    ),
    "imaging:cxr_pulmonary_oedema": ("HP:0100598", "imaging", "Pulmonary edema"),
    "imaging:pericardial_effusion": (
        "HP:0001698",
        "imaging",
        "Pericardial effusion, on echocardiography or CT. One of the four ESC "
        "diagnostic criteria for acute pericarditis and absent from this "
        "vocabulary until a real pericarditis with a large effusion was "
        "committed as pneumonia. Under imaging: because it is a separate test "
        "with a separate cost, not a chest-radiograph reading.",
    ),
    "imaging:ctpa_filling_defect": (
        None,
        "imaging",
        "radiological finding; see above",
    ),
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SYNONYM = re.compile(r'synonym: "([^"]+)"')


def parse_obo(path: str | Path) -> dict[str, Term]:
    """Parse hp.obo into id -> Term. Obsolete terms are dropped.

    A hand-rolled parser is used rather than a full OBO library because only
    five fields are needed and the format for those is stable. If richer
    axioms are ever required, switch to pronto rather than extending this.
    """
    terms: dict[str, Term] = {}
    current: dict | None = None

    def flush() -> None:
        if current and current.get("id") and not current.get("obsolete"):
            terms[current["id"]] = Term(
                hpo_id=current["id"],
                name=current.get("name", ""),
                synonyms=tuple(current.get("synonyms", [])),
                parents=tuple(current.get("parents", [])),
                definition=current.get("definition", ""),
            )

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line == "[Term]":
            flush()
            current = {"synonyms": [], "parents": []}
            continue
        if line.startswith("["):  # [Typedef] etc -- stop collecting
            flush()
            current = None
            continue
        if current is None:
            continue
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("name: "):
            current["name"] = line[6:]
        elif line.startswith("def: "):
            current["definition"] = line[5:]
        elif line.startswith("is_obsolete: true"):
            current["obsolete"] = True
        elif line.startswith("is_a: "):
            current["parents"].append(line[6:].split(" !")[0].strip())
        elif line.startswith("synonym: "):
            match = _SYNONYM.match(line)
            if match:
                current["synonyms"].append(match.group(1))
    flush()
    return terms


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@dataclass
class Vocabulary:
    """The findings vocabulary: local keys bound to HPO terms."""

    terms: dict[str, Term] = field(default_factory=dict)
    concepts: dict[str, Concept] = field(default_factory=dict)
    release: str = HPO_RELEASE

    @property
    def umls_coverage(self) -> tuple[int, int]:
        """(concepts carrying a CUI, total). Reportable, not assumed."""
        grounded = sum(1 for c in self.concepts.values() if c.is_umls_grounded)
        return grounded, len(self.concepts)

    @classmethod
    def build(
        cls,
        obo_path: str | Path,
        curated: dict | None = None,
        umls_path: str | Path | None = None,
    ) -> "Vocabulary":
        terms = parse_obo(obo_path)
        curated = curated or CURATED
        # Section 6's ontology layer, when it has been built. Absent, concepts
        # simply carry no CUI and `umls_coverage` reports 0 -- the vocabulary
        # stays usable rather than requiring a licence to load.
        umls: dict[str, dict] = {}
        if umls_path is not None and Path(umls_path).exists():
            umls = json.loads(
                Path(umls_path).read_text(encoding="utf-8")
            ).get("findings", {})
        concepts: dict[str, Concept] = {}
        for key, (hpo_id, modality, note) in curated.items():
            term = terms.get(hpo_id) if hpo_id else None
            if hpo_id and term is None:
                # The curated table references a term absent from this release.
                # Loudly unmapped rather than silently dropped: HPO obsoletes
                # and merges terms between releases, and a mapping table that
                # rots without warning is worse than no table.
                note = (note + " | " if note else "") + (
                    f"WARNING: {hpo_id} not found in release {HPO_RELEASE}"
                )
                hpo_id = None
            coded = umls.get(key, {})
            concepts[key] = Concept(
                key=key,
                hpo_id=hpo_id,
                hpo_name=term.name if term else None,
                modality=modality,
                note=note,
                cui=coded.get("cui"),
                snomed_ct=coded.get("snomed_ct") or None,
                umls_name=coded.get("umls_preferred_name"),
            )
        return cls(terms=terms, concepts=concepts)

    # -- lookup ------------------------------------------------------------

    def concept(self, key: str) -> Concept | None:
        return self.concepts.get(key)

    def term(self, hpo_id: str) -> Term | None:
        return self.terms.get(hpo_id)

    def ancestors(self, hpo_id: str, max_depth: int = 12) -> list[str]:
        """All ancestor ids, nearest first. Cycle-safe."""
        seen: list[str] = []
        visited: set[str] = {hpo_id}
        frontier = [hpo_id]
        depth = 0
        while frontier and depth < max_depth:
            nxt: list[str] = []
            for tid in frontier:
                term = self.terms.get(tid)
                if not term:
                    continue
                for parent in term.parents:
                    if parent not in visited:
                        visited.add(parent)
                        seen.append(parent)
                        nxt.append(parent)
            frontier = nxt
            depth += 1
        return seen

    def generalise(self, key: str, levels: int = 1) -> str | None:
        """Back off to an ancestor when a specific finding is unavailable.

        Useful when the KB characterises 'Chest pain' but the case records
        'Pleuritic chest pain'. Backing off loses specificity, so callers
        should prefer the exact term whenever it exists.
        """
        concept = self.concepts.get(key)
        if not concept or not concept.hpo_id:
            return None
        chain = self.ancestors(concept.hpo_id)
        # Skip the structural roots -- generalising to 'All' is meaningless.
        chain = [c for c in chain if c not in ("HP:0000001", "HP:0000118")]
        if not chain:
            return None
        return chain[min(levels, len(chain)) - 1]

    def lookup_exact(self, text: str) -> Term | None:
        """Exact name or synonym match, case-insensitive.

        Exact only, by design. Substring matching against this vocabulary
        produces confident nonsense: 'rale' matches 'P mitrale'.
        """
        needle = text.strip().lower()
        for term in self.terms.values():
            if term.name.lower() == needle:
                return term
            if any(s.lower() == needle for s in term.synonyms):
                return term
        return None

    # -- reporting ---------------------------------------------------------

    def coverage(self) -> dict[str, object]:
        mapped = [c for c in self.concepts.values() if c.is_grounded]
        unmapped = [c for c in self.concepts.values() if not c.is_grounded]
        by_modality: dict[str, list[int]] = {}
        for c in self.concepts.values():
            slot = by_modality.setdefault(c.modality, [0, 0])
            slot[1] += 1
            if c.is_grounded:
                slot[0] += 1
        return {
            "release": self.release,
            "hpo_terms": len(self.terms),
            "concepts": len(self.concepts),
            "mapped": len(mapped),
            "unmapped": [c.key for c in unmapped],
            "by_modality": {k: f"{v[0]}/{v[1]}" for k, v in by_modality.items()},
        }

    def to_json(self, path: str | Path) -> None:
        """Freeze the vocabulary so downstream runs are reproducible.

        Pinning matters: HPO releases monthly, and an unpinned vocabulary makes
        two evaluation runs silently incomparable.
        """
        payload = {
            "release": self.release,
            "source": HPO_SOURCE,
            "concepts": [
                {
                    "key": c.key,
                    "hpo_id": c.hpo_id,
                    "hpo_name": c.hpo_name,
                    "modality": c.modality,
                    "note": c.note,
                    "cui": c.cui,
                    "snomed_ct": c.snomed_ct,
                    "umls_name": c.umls_name,
                }
                for c in sorted(self.concepts.values(), key=lambda x: x.key)
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Vocabulary":
        """Load a frozen vocabulary. No hp.obo needed."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        concepts = {
            c["key"]: Concept(
                key=c["key"],
                hpo_id=c["hpo_id"],
                hpo_name=c["hpo_name"],
                modality=c["modality"],
                note=c.get("note", ""),
                cui=c.get("cui"),
                snomed_ct=c.get("snomed_ct"),
                umls_name=c.get("umls_name"),
            )
            for c in payload["concepts"]
        }
        return cls(terms={}, concepts=concepts, release=payload["release"])


# What the threshold-dependent concepts actually mean
# ---------------------------------------------------------------------------
# The map above gives every concept an HPO term, which names a qualitative
# state: "Hypoxemia", "Elevated circulating D-dimer", "Increased total
# leukocyte count". No threshold. Every study that could source one of these
# reports a threshold, so matching a source to a concept meant a judgement
# call every time, and those judgement calls are exactly where the sourcing
# effort kept stalling:
#
#   exam:hypoxia      unsourceable, partly because nothing said what it meant.
#                     The only threshold stated anywhere in this project is in
#                     guidelines.py, where the PERC rule is encoded with
#                     "SaO2 < 95%".
#   lab:raised_wcc    a cohort reported "leucocytes <3.5 or >8.8", which is
#                     bidirectional and counts leukopenia. Rejected -- but the
#                     concept never said it meant the raised side only.
#   lab:raised_bnp    BNP > 100 pg/mL against an age-specific NT-proBNP
#                     threshold: a column that could not be assembled because
#                     the concept did not fix a scale.
#   lab:raised_troponin  the vocabulary note already admits "assays differ"
#                     and stops there.
#
# So these are the intended operational definitions. They are conventions
# rather than measurements -- a clinician would recognise each as the ordinary
# reporting threshold for its test -- and writing them down does three things
# the previous silence did not: it gives a future sourcing pass a target to
# match against, it makes an existing sourced cell auditable for whether its
# study used a comparable cutoff, and it turns "unsourceable" into a claim
# about the literature rather than a claim that hides an undefined concept.
#
# Where a cell already in the knowledge base was sourced at a different
# threshold, that is recorded in the citation snippet rather than smoothed
# over; ``deviations_from_operational_definitions`` below lists them.
OPERATIONAL_DEFINITIONS: dict[str, str] = {
    "exam:hypoxia": (
        "Arterial oxygen saturation below 95% on room air, the threshold this "
        "project already encodes for the PERC rule in guidelines.py."
    ),
    "exam:tachycardia": "Heart rate above 100 beats per minute at rest.",
    "fever": (
        "Temperature at or above 38C, measured or reported. The ambiguity is "
        "real and is inherited from the sources: DDXPlus asks 'fever, either "
        "felt or measured', while cohort studies record a measured "
        "temperature, and the two differ by more than a rounding -- one real "
        "case here is a pneumonia reporting days of fever who is afebrile on "
        "arrival."
    ),
    "lab:raised_d_dimer": (
        "Above the assay's conventional venous-thromboembolism rule-out "
        "threshold, around 500 microg/L fibrinogen-equivalent units. Not an "
        "age-adjusted or study-optimised cutoff; one candidate source was "
        "rejected for using 990."
    ),
    "lab:raised_troponin": (
        "Above the assay's 99th-percentile upper reference limit. Generation "
        "matters and is not resolved by this definition: high-sensitivity "
        "assays detect elevations conventional ones miss, which is why the "
        "COPD and pulmonary embolism cells here carry wide bands rather than "
        "tight ones."
    ),
    "lab:raised_bnp": (
        "BNP above 100 pg/mL, or NT-proBNP above its acute-dyspnoea "
        "threshold of roughly 300 ng/L. Two analytes treated as answering one "
        "question -- did the natriuretic peptide come back raised -- which is "
        "an assumption, and the reason the rest of that column is still "
        "invented."
    ),
    "lab:raised_wcc": (
        "Total white cell count above the upper reference limit, around "
        "11 x10^9/L. Unidirectional: leukopenia is a different finding and in "
        "sepsis points the other way, which is why a bidirectional "
        "'<3.5 or >8.8' figure was rejected rather than used."
    ),
}


def deviations_from_operational_definitions(kb) -> list[tuple[str, str, str]]:
    """Sourced cells whose study used a threshold other than the intended one.

    Returns (disease, concept, note). This is a reading of the citation
    snippets rather than a parse: the snippets record the cutoff each study
    used, deliberately, so that this comparison is possible at all.
    """
    known = {
        ("pulmonary_embolism", "lab:raised_troponin"):
            "high-sensitivity troponin T above 14 ng/L or troponin I above "
            "0.5 ng/mL, two assays in one cohort",
        ("copd_exacerbation", "lab:raised_troponin"):
            "conventional troponin I at 0.017 microg/L",
        ("pulmonary_embolism", "lab:raised_bnp"):
            "NT-proBNP at 350 ng/L rather than the 300 named above",
        ("community_acquired_pneumonia", "exam:hypoxia"):
            "oxygen saturation below 96% rather than the 95% named above",
    }
    out = []
    for (label, concept), note in known.items():
        entry = kb.get(label)
        if entry is None or concept not in entry.features:
            continue
        source = entry.sources.get(concept)
        if source is not None and source.citation is not None:
            out.append((label, concept, note))
    return out


__all__ = [
    "CURATED",
    "Concept",
    "HPO_RELEASE",
    "HPO_SOURCE",
    "Term",
    "OPERATIONAL_DEFINITIONS",
    "Vocabulary",
    "deviations_from_operational_definitions",
    "parse_obo",
]
