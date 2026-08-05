"""DDXPlus adapter.

DDXPlus ships as CSVs of synthetic patients plus JSON metadata for conditions
and evidences. This module maps that release into ``Case`` and
``InMemoryKnowledgeBase`` so the same loop and the same metrics run over it
unchanged.

The dataset is not vendored here -- point ``root`` at a local copy, zips and
all; splits are read straight out of the published archives, so nothing needs
extracting first. The loader is written against the published column layout and
will raise a clear error rather than silently mis-parse if the layout differs;
if the release you have uses different column names, override them at the call
site instead of editing the defaults.

Two things worth knowing before drawing conclusions from DDXPlus results:

  * The patients are simulated from a rule-based model, so a system that
    recovers the generating structure scores well without necessarily doing
    clinical reasoning. Strong numbers here are a floor, not evidence.
  * The differential labels are produced by the same generative process, so
    "agreement with the DDXPlus differential" and "clinical correctness" are not
    the same quantity. Report them as the former.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..environment import Case
from ..knowledge import Citation, DiseaseEntry, InMemoryKnowledgeBase, register_costs


@dataclass
class DDXPlusColumns:
    """Column names in the patient CSV. Override if your release differs."""

    pathology: str = "PATHOLOGY"
    evidences: str = "EVIDENCES"
    initial_evidence: str = "INITIAL_EVIDENCE"
    differential: str = "DIFFERENTIAL_DIAGNOSIS"
    age: str = "AGE"
    sex: str = "SEX"


@dataclass
class DDXPlusLoader:
    """Loads a DDXPlus split from disk."""

    root: Path
    columns: DDXPlusColumns = field(default_factory=DDXPlusColumns)
    default_cost: float = 1.0
    _vocabulary: frozenset[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"DDXPlus root {self.root} not found. Download the release and "
                "pass its directory as root=."
            )

    def evidence_vocabulary(
        self, filename: str = "release_evidences.json"
    ) -> frozenset[str]:
        """Every concept a patient row can express, categoricals expanded.

        The CSV encodes a categorical or multi-choice answer as
        ``<code>_@_<value>``, so the concept space is not the 223 evidence codes
        but the ~970 code/value pairs they expand to. Reading it from the
        metadata rather than from the rows keeps the vocabulary identical no
        matter which split is loaded or how few rows of it -- a vocabulary
        inferred from a sample would make the same patient look closed-world
        for one run and open-world for another.
        """
        if self._vocabulary is not None:
            return self._vocabulary
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. It defines which concepts a patient record "
                "covers, and without it every unlisted evidence reads as "
                "'not recorded' rather than 'absent', so negative findings "
                "carry no information. Pass closed_world=False to accept that."
            )
        meta = json.loads(path.read_text(encoding="utf-8"))
        concepts: set[str] = set()
        for code, entry in meta.items():
            values = entry.get("possible-values") or []
            if entry.get("data_type") == "B" or not values:
                concepts.add(code)
            else:
                concepts.update(f"{code}_@_{value}" for value in values)
        self._vocabulary = frozenset(concepts)
        return self._vocabulary

    @contextmanager
    def _open_split(self, filename: str):
        """Open a patient split as text, from a plain .csv or the published .zip.

        The release ships each split as a zip whose single member carries no
        file extension at all, so neither the bare name nor a '.csv' suffix
        resolves against a fresh download, and selecting members by '.csv'
        finds nothing. Accept the stem, the .csv or the .zip, and work the rest
        out -- extracting 800 MB by hand first is not a reasonable prerequisite
        for loading the data.
        """
        stem = filename
        for suffix in (".csv", ".zip"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        csv_path = self.root / f"{stem}.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                yield handle, csv_path.name
            return

        zip_path = self.root / f"{stem}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                entries = [i for i in archive.infolist() if not i.is_dir()]
                if not entries:
                    raise ValueError(f"{zip_path} contains no files")
                named = [i for i in entries if i.filename.lower().endswith(".csv")]
                chosen = max(named or entries, key=lambda i: i.file_size)
                with archive.open(chosen) as raw:
                    yield (
                        io.TextIOWrapper(raw, encoding="utf-8", newline=""),
                        f"{zip_path.name}::{chosen.filename}",
                    )
            return

        raise FileNotFoundError(
            f"No split named {stem!r} in {self.root}. Looked for {stem}.csv and "
            f"{stem}.zip."
        )

    def load_cases(
        self, filename: str, limit: int | None = None, closed_world: bool = True
    ) -> list[Case]:
        vocabulary = self.evidence_vocabulary() if closed_world else frozenset()
        cases: list[Case] = []
        with self._open_split(filename) as (handle, label):
            reader = csv.DictReader(handle)
            self._check_columns(reader.fieldnames or [], label)
            for index, row in enumerate(reader):
                if limit is not None and index >= limit:
                    break
                evidences = _parse_list(row[self.columns.evidences])
                initial = row.get(self.columns.initial_evidence, "").strip()
                cases.append(
                    Case(
                        case_id=f"ddxplus-{index:06d}",
                        presenting_complaint=initial or "unspecified",
                        diagnosis=_slug(row[self.columns.pathology]),
                        # DDXPlus records positive evidences only, so the
                        # feature map is sparse and the closed-world convention
                        # is carried by ``vocabulary`` instead. Storing an
                        # explicit False for each of the ~970 absent concepts
                        # would be equivalent and cost roughly 128M dict
                        # entries across the training split.
                        features={e: True for e in evidences},
                        initial_findings=(initial,) if initial else (),
                        demographics={
                            "age": row.get(self.columns.age),
                            "sex": row.get(self.columns.sex),
                        },
                        vocabulary=vocabulary,
                        differential=_parse_differential(
                            row.get(self.columns.differential, "")
                        ),
                    )
                )
        return cases

    def _check_columns(self, found: list[str], label: str) -> None:
        required = {self.columns.pathology, self.columns.evidences}
        missing = required - set(found)
        if missing:
            raise ValueError(
                f"{label} is missing expected columns {sorted(missing)}. "
                f"Found: {sorted(found)}. Pass a DDXPlusColumns override."
            )

    def build_knowledge_base(
        self, cases: list[Case], smoothing: float = 1.0
    ) -> InMemoryKnowledgeBase:
        """Estimate a KB from labelled cases.

        This makes the KB an empirical summary of the training split rather than
        a vetted clinical source. That is fine for a benchmark baseline and
        wrong for the deployed system, where the KB must be independently
        sourced -- otherwise "grounded in the knowledge base" reduces to
        "consistent with the training data", which is not the claim the project
        intends to make. Laplace smoothing keeps unseen features from producing
        zero likelihoods.
        """
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[str, int] = defaultdict(int)
        vocabulary: set[str] = set()

        for case in cases:
            totals[case.diagnosis] += 1
            for concept, present in case.features.items():
                vocabulary.add(concept)
                if present:
                    counts[case.diagnosis][concept] += 1

        # A sparse feature map never mentions an absent concept, so the loop
        # above only sees concepts some patient reported. Seed the space from
        # the declared vocabulary as well, or a feature that is rare across the
        # whole split gets no likelihood entry and silently falls back to the
        # KB-wide marginal at inference time. Unioning the distinct vocabulary
        # objects rather than one per case keeps this O(vocabulary), not
        # O(cases x vocabulary).
        for declared in {case.vocabulary for case in cases}:
            vocabulary |= declared

        register_costs({c: self.default_cost for c in vocabulary})
        kb = InMemoryKnowledgeBase()
        total_cases = max(sum(totals.values()), 1)
        for label, n in totals.items():
            features = {
                concept: (counts[label][concept] + smoothing) / (n + 2 * smoothing)
                for concept in vocabulary
            }
            kb.add(
                DiseaseEntry(
                    label=label,
                    prevalence=n / total_cases,
                    features=features,
                    citations=(
                        Citation(
                            source_id="DDXPLUS",
                            locator=label,
                            snippet=f"empirical estimate from {n} training cases",
                        ),
                    ),
                    notes="likelihoods estimated from data, not clinically vetted",
                )
            )
        return kb

    def load_condition_metadata(self, filename: str = "release_conditions.json") -> dict:
        path = self.root / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def _parse_list(raw: str) -> list[str]:
    """DDXPlus stores lists as Python-literal strings in the CSV."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _parse_differential(raw: str) -> tuple[tuple[str, float], ...]:
    """Parse DIFFERENTIAL_DIAGNOSIS into (label, probability) pairs.

    Stored as a Python-literal list of ``[pathology, probability]``. Labels are
    slugged with the same function as ``PATHOLOGY`` so they line up with KB
    labels; skipping that leaves the gold differential and the predicted one
    speaking different vocabularies, and every recall number silently reads
    zero.

    Probabilities are renormalised. The release already normalises them, but
    the generation process removed out-of-scope pathologies from some rows
    (paper 3.3), so a handful do not sum to 1.

    Parsed here rather than through ``_parse_list``: that helper flattens every
    element with ``str(v)``, which is right for the evidence column but turns
    each ``[label, probability]`` pair into the string ``"['Anemia', 0.25]"``.
    Every pair then fails the shape check below and the gold differential comes
    back empty -- silently, since an absent gold differential is a legitimate
    state for other sources.
    """
    raw = (raw or "").strip()
    if not raw:
        return ()
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ()
    if not isinstance(parsed, (list, tuple)):
        return ()

    pairs: list[tuple[str, float]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        label, probability = item
        try:
            weight = float(probability)
        except (TypeError, ValueError):
            continue
        if weight > 0.0:
            pairs.append((_slug(str(label)), weight))
    if not pairs:
        return ()
    total = sum(weight for _, weight in pairs)
    if total > 0:
        pairs = [(label, weight / total) for label, weight in pairs]
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    return tuple(pairs)


def _slug(label: str) -> str:
    return "_".join(label.strip().lower().split())


__all__ = ["DDXPlusColumns", "DDXPlusLoader"]
