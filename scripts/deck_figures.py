# -*- coding: utf-8 -*-
"""Live figures for the two generated PDFs.

Both decks used to carry their numbers as string literals, and both went
stale the moment sourcing moved: they were still claiming "99 of 135
likelihoods invented" and "73% invented" in a page footer long after the
knowledge base had moved to 101 of 153 and 66%. Nothing executed those
strings, so nothing caught them -- the same failure the documentation drift
guard in ``tests/test_dxagent.py`` exists to prevent for the Markdown files,
reappearing in the one format nobody thinks to grep.

So the decks now ask the knowledge base at build time. A regenerated PDF
cannot disagree with the code it describes, and the numbers below are the
only place a figure is written down once.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dxagent import (  # noqa: E402
    AbstentionGate,
    DiagnosticAgent,
    LoopLimits,
    Verdict,
    provenance,
)
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_knowledge_base  # noqa: E402
from dxagent.datasets.fixtures import build_cases  # noqa: E402


@dataclass(frozen=True)
class Figures:
    total: int
    invented: int
    sourced: int
    coverage: int          # per cent of likelihoods carrying a citation
    invented_pct: int
    ddxplus: int
    merck: int
    cohorts: int
    priors_total: int
    priors_sourced: int
    fixtures_correct: int
    fixtures_total: int
    real_total: int
    real_committed_correct: int
    real_committed_wrong: int
    real_rows: tuple[tuple[str, str, str], ...]
    tests: int             # 0 when pytest could not be asked

    @property
    def invented_of_total(self) -> str:
        return f"{self.invented} of {self.total} likelihoods invented"

    @property
    def footer_claim(self) -> str:
        return f"the knowledge base is {self.invented_pct}% invented"


_PRETTY = {
    "pulmonary_embolism": "Pulmonary embolism",
    "community_acquired_pneumonia": "Community-acquired pneumonia",
    "acute_coronary_syndrome": "Acute coronary syndrome",
    "acute_pulmonary_oedema": "Acute pulmonary oedema",
    "copd_exacerbation": "COPD exacerbation",
    "asthma_exacerbation": "Asthma exacerbation",
    "pericarditis": "Pericarditis",
    "panic_attack": "Panic attack",
}


def _shorten(reason: str, limit: int = 78) -> str:
    """The most informative clause of an escalation reason.

    Escalation reasons are semicolon-joined and the first clause is almost
    always "no remaining action would meaningfully narrow the differential",
    which is true of nearly every escalation and therefore says nothing about
    the individual case. Prefer a clause that names what was sought and could
    not be had, or the specific threshold that was missed.
    """
    clauses = [c.strip() for c in reason.split(";") if c.strip()]
    if not clauses:
        return ""
    ranked = sorted(
        clauses,
        key=lambda c: (
            "not available" not in c,
            "below threshold" not in c,
            "not excluded" not in c,
        ),
    )
    best = ranked[0]
    if len(best) > limit:
        best = best[: limit - 1].rstrip() + "…"
    return best


def _count_tests() -> int:
    """How many tests the suite collects, or 0 if pytest cannot be asked.

    Written as a soft dependency on purpose: the decks should still build on
    a checkout without pytest, and a stat card that quietly disappears is
    better than one that confidently states a number nobody verified.
    """
    import re
    import subprocess

    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests")],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    match = re.search(r"(\d+) tests? collected", done.stdout)
    return int(match.group(1)) if match else 0


def collect(reason_limit: int = 78) -> Figures:
    """Live figures. ``reason_limit`` is how much of an escalation reason the
    caller has room for: the seven-page explainer prints these in a narrow
    column and needs them short, the deep dive gives them a column of their
    own and can show the concept that was sought, which is the informative
    half and the first thing an aggressive clip removes."""
    kb = build_knowledge_base(correlated=True)
    report = provenance.report(kb)

    citations: Counter[str] = Counter()
    for entry in kb.diseases():
        for concept in entry.features:
            source = entry.sources.get(concept)
            if source and source.citation:
                citations[source.citation.source_id] += 1
    ddxplus = citations.get("DDXPLUS", 0)
    merck = citations.get("MSD-19E", 0)
    cohorts = sum(v for k, v in citations.items() if k not in {"DDXPLUS", "MSD-19E"})

    gated = DiagnosticAgent(kb=kb, proposer=BayesianProposer(kb), gate=AbstentionGate(kb=kb))
    fixtures = build_cases()
    correct = sum(
        1 for case in fixtures if gated.run(case).differential.top.label == case.diagnosis
    )

    # The real-case arm turns off both budget flags, matching
    # scripts/eval_real_cases.py: a finding the source report never records
    # means no test was performed, so it costs neither a turn nor money.
    agent = DiagnosticAgent(
        kb=kb,
        proposer=BayesianProposer(kb),
        gate=AbstentionGate(kb=kb),
        limits=LoopLimits(
            uninformative_turns_still_count=False,
            unanswered_actions_still_cost=False,
        ),
    )
    rows: list[tuple[str, str, str]] = []
    committed_ok = committed_wrong = 0
    for case in REAL_CASES:
        outcome = agent.run(case)
        truth = _PRETTY.get(case.diagnosis, case.diagnosis)
        if outcome.verdict is Verdict.COMMITTED:
            if outcome.prediction == case.diagnosis:
                committed_ok += 1
                result = "Committed — correct"
            else:
                committed_wrong += 1
                result = f"Committed — wrong ({_PRETTY.get(outcome.prediction, outcome.prediction)})"
        else:
            rank = [h.label for h in outcome.differential.hypotheses].index(case.diagnosis) + 1
            reason = outcome.escalation.reason if outcome.escalation else ""
            result = f"Escalated — ranked {rank}; {_shorten(reason, reason_limit)}"
        rows.append((case.presenting_complaint, truth, result))

    sourced = report.measured + report.narrative
    return Figures(
        total=report.total,
        invented=report.invented,
        sourced=sourced,
        coverage=round(100 * sourced / report.total),
        invented_pct=round(100 * report.invented / report.total),
        ddxplus=ddxplus,
        merck=merck,
        cohorts=cohorts,
        priors_total=report.priors_total,
        priors_sourced=report.priors_sourced,
        fixtures_correct=correct,
        fixtures_total=len(fixtures),
        real_total=len(REAL_CASES),
        real_committed_correct=committed_ok,
        real_committed_wrong=committed_wrong,
        real_rows=tuple(rows),
        tests=_count_tests(),
    )


if __name__ == "__main__":
    f = collect()
    print(f)
