#!/usr/bin/env python3
"""Run the diagnostic loop over a case set and print the metrics report.

    python scripts/run_eval.py                       # synthetic fixtures
    python scripts/run_eval.py --trace               # per-turn reasoning trace
    python scripts/run_eval.py --noisy               # lossy history taking
    python scripts/run_eval.py --sweep               # threshold sweep
    python scripts/run_eval.py --ddxplus /path/to/release --limit 2000
    python scripts/run_eval.py --no-baselines        # skip the comparison

The report always includes the two baselines the brief requires, because an
accuracy figure printed without them is the easiest number in this project to
read as better than it is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits  # noqa: E402
from dxagent.belief import BayesianProposer, ConsensusProposer  # noqa: E402
from dxagent.datasets import build_cases, build_knowledge_base  # noqa: E402
from dxagent.evaluation import evaluate, split_cases  # noqa: E402


def load(args) -> tuple:
    if args.ddxplus:
        from dxagent.datasets.ddxplus import DDXPlusLoader

        loader = DDXPlusLoader(root=Path(args.ddxplus))
        train = loader.load_cases(args.train_file, limit=args.limit)
        test = loader.load_cases(args.test_file, limit=args.limit)
        return loader.build_knowledge_base(train), train, test

    kb = build_knowledge_base()
    cases = build_cases()
    calibration, evaluation = split_cases(cases, calibration_fraction=0.35)
    return kb, calibration, evaluation


def have(module: str) -> bool:
    from importlib.util import find_spec

    return find_spec(module) is not None


def build_proposer(kb, args):
    """Assemble the proposer stack the flags ask for.

    Grounding is opt-in rather than automatic despite being mandated: the first
    call downloads BGE-M3 (~2.3 GB) and indexing costs about a minute, and a
    default that does that to someone running the fixtures once is a bad
    default. The engine choice below is automatic because LangGraph is light
    and adds no runtime cost.
    """
    proposer = BayesianProposer(kb)

    if args.llm:
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  --llm given but ANTHROPIC_API_KEY is unset; skipping")
        else:
            from dxagent.belief import LLMProposer
            from dxagent.llm import AnthropicLLM

            # Consensus rather than replacement: belief.py exists to expose
            # disagreement between a transparent posterior and a model's
            # clinical intuition, and averaging them would destroy the signal
            # the gate reads.
            proposer = ConsensusProposer(
                primary=proposer, secondary=LLMProposer(kb, AnthropicLLM())
            )
            print("  proposer: consensus (Bayesian + LLM)")

    if args.grounded:
        if not (have("qdrant_client") and have("sentence_transformers")):
            print("  --grounded given but retrieval extras are not installed; skipping")
        else:
            from dxagent.retrieval import GroundedProposer, GuidelineIndex

            print("  building guideline index (first run downloads BGE-M3)...")
            index = GuidelineIndex().build()
            proposer = GroundedProposer(proposer, index)
            print(f"  proposer: retrieval-grounded over {len(index._passages)} passages")

    return proposer


def baseline_comparison(kb, test_cases, result) -> str:
    """The two comparisons the brief demands, run over the same cases.

    Printed with the agent rather than separately, because a headline number
    with no baseline beside it invites the reader to supply their own -- and
    on DDXPlus a ranker with no inference at all scores in the nineties, which
    is not what anyone assumes when they see an accuracy figure.
    """
    from dxagent.baselines import RetrievalOnlyBaseline, SinglePassBaseline
    from dxagent.evaluation import run_agent
    from dxagent.evaluation.metrics import (
        differential_metrics,
        ranking_metrics,
        selective_metrics,
    )

    truths = result.truths
    gold = {
        case.case_id: case.differential for case in test_cases if case.differential
    }

    rows = [
        ("retrieval-only (no inference)", RetrievalOnlyBaseline(kb)),
        ("single-pass (no loop)", SinglePassBaseline(kb)),
    ]

    lines = [
        "",
        "baselines (project brief section 9)",
        f"  {'system':<32}{'top-1':>8}{'top-5':>8}{'MRR':>8}{'DDx@5':>9}{'turns':>8}",
    ]

    def row(name: str, outcomes) -> str:
        ranking = ranking_metrics(outcomes, truths)
        selective = selective_metrics(outcomes, truths)
        ddx = differential_metrics(outcomes, gold) if gold else None
        recall = f"{ddx.recall_at_5:>8.1%}" if ddx and ddx.n else "     n/a"
        return (
            f"  {name:<32}{ranking.top1:>8.1%}{ranking.top5:>8.1%}"
            f"{ranking.mrr:>8.3f}{recall:>9}{selective.mean_turns:>8.1f}"
        )

    for name, baseline in rows:
        lines.append(row(name, run_agent(baseline, test_cases)))
    lines.append(row("agentic loop", result.outcomes))

    lines.append(
        "\n  A retrieval-only baseline that does no inference is the floor the "
        "loop\n  has to clear. Where it does not clear it by much, the benchmark "
        "is the\n  finding, not the system."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddxplus", help="path to a local DDXPlus release directory")
    parser.add_argument("--train-file", default="release_train_patients.csv")
    parser.add_argument("--test-file", default="release_test_patients.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trace", action="store_true", help="print per-turn traces")
    parser.add_argument("--noisy", action="store_true", help="use the lossy oracle")
    parser.add_argument("--sweep", action="store_true", help="sweep gate thresholds")
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-cost", type=float, default=40.0)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument(
        "--engine",
        choices=("auto", "loop", "graph"),
        default="auto",
        help="auto uses the LangGraph state machine when langgraph is installed",
    )
    parser.add_argument(
        "--grounded",
        action="store_true",
        help="ground hypotheses in retrieved guideline passages (downloads BGE-M3)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="add an LLM proposer alongside the Bayesian one (needs ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip the section 9 baseline comparison",
    )
    parser.add_argument("--json", help="write full results to this path")
    args = parser.parse_args()

    kb, calibration_cases, test_cases = load(args)
    print(
        f"knowledge base: {len(kb.diseases())} conditions | "
        f"calibration: {len(calibration_cases)} cases | "
        f"evaluation: {len(test_cases)} cases"
    )

    proposer = build_proposer(kb, args)

    use_graph = args.engine == "graph" or (args.engine == "auto" and have("langgraph"))
    print(f"  engine: {'LangGraph state machine' if use_graph else 'reference loop'}")

    def build_agent(min_confidence: float, min_margin: float):
        gate = AbstentionGate(
            kb=kb, min_confidence=min_confidence, min_margin=min_margin
        )
        limits = LoopLimits(max_turns=args.max_turns, max_cost=args.max_cost)
        if use_graph:
            from dxagent.graph import GraphAgent

            # No trace= : the graph's per-node state is the inspectable
            # artefact, and duplicating the loop's print-based trace would give
            # two accounts of the same turn that could drift apart.
            return GraphAgent(
                kb=kb, proposer=proposer, gate=gate, limits=limits
            )
        return DiagnosticAgent(
            kb=kb, proposer=proposer, gate=gate, limits=limits, trace=args.trace
        )

    if args.sweep:
        print("\nthreshold sweep")
        header = f"{'conf':>6} {'margin':>7} {'cover':>7} {'sel.acc':>8} {'abst.prec':>10} {'cost':>7}"
        print(header)
        print("-" * len(header))
        for min_confidence in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
            for min_margin in (0.00, 0.10, 0.20):
                result = evaluate(
                    build_agent(min_confidence, min_margin),
                    test_cases,
                    calibration_cases=calibration_cases,
                    noisy=args.noisy,
                    fit_calibration=not args.no_calibration,
                )
                s = result.selective
                sel = "n/a" if s.selective_accuracy != s.selective_accuracy else f"{s.selective_accuracy:.1%}"
                prec = "n/a" if s.abstention_precision != s.abstention_precision else f"{s.abstention_precision:.1%}"
                print(
                    f"{min_confidence:>6.2f} {min_margin:>7.2f} {s.coverage:>7.1%} "
                    f"{sel:>8} {prec:>10} {s.mean_cost:>7.2f}"
                )
        return 0

    if args.trace:
        print("\nper-case traces")
    result = evaluate(
        build_agent(args.min_confidence, args.min_margin),
        test_cases,
        calibration_cases=calibration_cases,
        noisy=args.noisy,
        fit_calibration=not args.no_calibration,
    )

    print("\n" + result.report())

    if not args.no_baselines:
        print(baseline_comparison(kb, test_cases, result))

    print("\nper-case detail")
    for outcome in result.outcomes:
        truth = result.truths[outcome.case_id]
        top = outcome.differential.top
        mark = "ok " if top.label == truth else "MISS"
        verdict = "commit " if not outcome.abstained else "escalate"
        print(
            f"  {outcome.case_id} {verdict} {mark} "
            f"p={top.probability:.2f} top1={top.label} truth={truth} "
            f"turns={len(outcome.steps)} cost={outcome.budget_spent:.1f}"
        )
        if outcome.escalation:
            print(f"      reason: {outcome.escalation.reason}")
        if outcome.steps:
            path = " -> ".join(s.action.target for s in outcome.steps)
            print(f"      evidence sought: {path}")

    if args.json:
        result.to_json(Path(args.json))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
