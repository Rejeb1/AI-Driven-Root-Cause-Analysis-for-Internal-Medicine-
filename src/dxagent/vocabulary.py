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

    @property
    def is_grounded(self) -> bool:
        return self.hpo_id is not None

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.key} [{self.hpo_id or 'UNMAPPED'}]"


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

    @classmethod
    def build(
        cls, obo_path: str | Path, curated: dict | None = None
    ) -> "Vocabulary":
        terms = parse_obo(obo_path)
        curated = curated or CURATED
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
            concepts[key] = Concept(
                key=key,
                hpo_id=hpo_id,
                hpo_name=term.name if term else None,
                modality=modality,
                note=note,
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
            )
            for c in payload["concepts"]
        }
        return cls(terms={}, concepts=concepts, release=payload["release"])


__all__ = [
    "CURATED",
    "Concept",
    "HPO_RELEASE",
    "HPO_SOURCE",
    "Term",
    "Vocabulary",
    "parse_obo",
]
