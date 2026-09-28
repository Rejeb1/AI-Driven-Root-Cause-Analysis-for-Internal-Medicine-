#!/usr/bin/env python3
"""A language model on the real cases: alone, and inside the loop.

    python scripts/llm_real_cases.py                    # single pass, Gemini, 3 runs
    python scripts/llm_real_cases.py --repeats 2 --no-complaint
    python scripts/llm_real_cases.py --in-loop          # the full system, both proposers

**Single pass** gives each case's complete extracted record to the model in
one prompt and scores its top-ranked diagnosis, beside the brief's other
baselines on the same real cases -- retrieval-only, the Bayesian proposer
with no loop, and the loop itself. Two things this is careful about,
because each would inflate the model's number silently:

- **Fallbacks are counted, not scored.** ``LLMProposer`` returns the
  Bayesian posterior when the model's answer cannot be parsed. Scoring that
  as the model's answer would credit it with Bayes' results.
- **``--no-complaint`` isolates the findings.** The model also reads the
  free-text presenting complaint, which the Bayesian proposer ignores, so
  the default comparison is not like for like. Run both.

It also reports the model's stated confidence against the accuracy it
earned -- the gap ``VerbalisedCalibrator`` exists to correct -- and whether
there are enough independent cases to fit that correction (there are not).

**In loop** runs the actual design: ``ConsensusProposer`` with the Bayesian
proposer primary and the model secondary, the gate escalating when they
disagree, on the real-case loop configuration. It is compared with the
Bayesian-only loop case by case.

Needs ``GEMINI_API_KEY`` (free tier) or ``--provider`` with its own
settings. Any model here is a documented substitute for the brief's
mandated one, and the published case reports may be in its training data:
it sees only the extracted findings, never the report or its identifier,
which reduces that risk without removing it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.baselines import RetrievalOnlyBaseline, _observe_everything  # noqa: E402
from dxagent.belief import BayesianProposer, ConsensusProposer, LLMProposer  # noqa: E402
from dxagent.datasets import REAL_CASES, build_knowledge_base  # noqa: E402
from dxagent.gate import VerbalisedCalibrator  # noqa: E402

REAL_LIMITS = dict(uninformative_turns_still_count=False, unanswered_actions_still_cost=False)


@dataclass
class SinglePass:
    """Per-case answers from repeated single-pass runs."""

    answers: dict[str, list[str]] = field(default_factory=dict)  # "FALLBACK" when parsing failed
    stated: list[tuple[str, float, bool]] = field(default_factory=list)  # case, confidence, correct

    @property
    def runs(self) -> int:
        return sum(len(a) for a in self.answers.values())

    def correct(self, cases) -> int:
        return sum(
            a == case.diagnosis for case in cases for a in self.answers[case.case_id]
        )

    @property
    def fallbacks(self) -> int:
        return sum(a.count("FALLBACK") for a in self.answers.values())

    @property
    def stable(self) -> int:
        return sum(len(set(a)) == 1 for a in self.answers.values())


def single_pass(kb, llm, cases, repeats: int = 1, withhold_complaint: bool = False) -> SinglePass:
    model = LLMProposer(kb=kb, llm=llm)
    result = SinglePass()
    for case in cases:
        findings = _observe_everything(case)
        complaint = "" if withhold_complaint else case.presenting_complaint
        answers = []
        for _ in range(repeats):
            model.last_verbalised_confidence = float("nan")
            differential = model.propose(findings, complaint)
            if model.last_was_fallback:
                answers.append("FALLBACK")
                continue
            answers.append(differential.top.label)
            if not math.isnan(model.last_verbalised_confidence):
                result.stated.append((
                    case.case_id,
                    model.last_verbalised_confidence,
                    differential.top.label == case.diagnosis,
                ))
        result.answers[case.case_id] = answers
    return result


def run_loop(kb, proposer, cases) -> dict[str, tuple[str, str]]:
    agent = DiagnosticAgent(
        kb=kb, proposer=proposer, gate=AbstentionGate(kb=kb), limits=LoopLimits(**REAL_LIMITS)
    )
    out = {}
    for case in cases:
        outcome = agent.run(case)
        if outcome.verdict is Verdict.COMMITTED:
            out[case.case_id] = (
                "correct" if outcome.prediction == case.diagnosis else "WRONG",
                outcome.prediction,
            )
        else:
            reason = outcome.escalation.reason if outcome.escalation else ""
            out[case.case_id] = ("escalated", reason[:90])
    return out


def tally(outcomes) -> str:
    c = Counter(v[0] for v in outcomes.values())
    return f"{c['correct']} correct / {c['WRONG']} wrong / {c['escalated']} escalated"


def report_single_pass(kb, llm, args) -> None:
    cases = REAL_CASES
    bayes = BayesianProposer(kb)
    retrieval = RetrievalOnlyBaseline(kb)
    result = single_pass(kb, llm, cases, args.repeats, args.no_complaint)
    bayes_ok = retrieval_ok = 0
    for case in cases:
        findings = _observe_everything(case)
        b = bayes.propose(findings, case.presenting_complaint).top.label
        r = retrieval.run(case).prediction
        bayes_ok += b == case.diagnosis
        retrieval_ok += r == case.diagnosis
        answers = result.answers[case.case_id]
        print(
            f"{case.case_id:<14} {case.diagnosis:<29} "
            f"bayes {'ok  ' if b == case.diagnosis else 'MISS'}  "
            f"model {sum(a == case.diagnosis for a in answers)}/{len(answers)}  "
            f"{dict(Counter(answers))}"
        )

    loop = run_loop(kb, BayesianProposer(kb), cases)
    n, runs = len(cases), result.runs
    print(f"\nbaselines on the {n} real cases (brief section 9; the fixture table in run_eval.py")
    print("is on invented cases and flatters every complete-record system)")
    print(f"  retrieval-only, complete record      top-1 {retrieval_ok}/{n}")
    print(f"  Bayesian single pass, complete       top-1 {bayes_ok}/{n}")
    print(
        f"  model single pass, complete          top-1 {result.correct(cases)}/{runs} runs "
        f"({result.correct(cases) / runs:.1%}); {result.fallbacks} fallbacks; "
        f"same answer every run on {result.stable}/{n} cases"
    )
    print(f"  agentic loop (Bayesian), asks        {tally(loop)}")

    print("\nstated confidence (verbalised) against accuracy earned")
    if result.stated:
        mean_stated = sum(s for _, s, _ in result.stated) / len(result.stated)
        mean_ok = sum(ok for _, _, ok in result.stated) / len(result.stated)
        wrong = [s for _, s, ok in result.stated if not ok]
        print(
            f"  {len(result.stated)} answers with a stated confidence: mean stated "
            f"{mean_stated:.0%}, accuracy {mean_ok:.0%}, gap {mean_stated - mean_ok:+.0%}"
        )
        if wrong:
            print(f"  on the wrong answers the model stated {min(wrong):.0%}-{max(wrong):.0%}")
        # Repeats of one case are not independent evidence about calibration,
        # so the calibrator gets one answer per case, and declines below its
        # minimum rather than fitting noise.
        first = {}
        for case_id, stated, ok in result.stated:
            first.setdefault(case_id, (stated, ok))
        calibrator = VerbalisedCalibrator().fit(list(first.values()))
        print(
            f"  calibrator fitted: {calibrator.fitted} "
            f"({len(first)} independent cases; it needs {calibrator.min_fit_samples})"
        )
    else:
        print("  the model stated no confidence on any answer")
    print(
        "\nTop-1 on a complete record: no abstention, no cost. A system that must\n"
        "commit on every case is not comparable to one that may escalate, and\n"
        f"n={n} settles nothing."
    )


# Share of a case's model turns that may fall back before the case is treated
# as not answered (a stray parse failure is tolerated; a silent model is not).
MAX_CASE_FALLBACK_SHARE = 0.2


def report_partial(cases, bayes_only, saved) -> None:
    """What the finished cases say so far, clearly labelled as partial."""
    if not saved:
        return
    changed = [cid for cid, v in saved.items() if v["verdict"] != bayes_only[cid][0]]
    print(
        f"\npartial ({len(saved)} of {len(cases)} cases, model answering): "
        f"{len(changed)} changed outcome"
        + (f" ({', '.join(changed)})" if changed else "")
        + f"; turns {sum(v['turns'] for v in saved.values())}, "
        f"fallbacks {sum(v['fallbacks'] for v in saved.values())}"
    )


def checkpoint_path(llm) -> Path:
    name = str(getattr(llm, "model", "model")).replace(":", "_").replace("/", "_")
    return Path(tempfile.gettempdir()) / f"dxagent_llm_in_loop_{name}.json"


def report_in_loop(kb, llm, fresh: bool = False) -> None:
    """About 500 model calls; one progress line per case, resumable.

    The first version printed nothing until every case was done, so a run
    stalled on one unanswered request looked exactly like a run in
    progress for two hours. Each finished case is now printed and saved,
    and a rerun continues from the checkpoint instead of starting over.
    """
    cases = REAL_CASES
    bayes_only = run_loop(kb, BayesianProposer(kb), cases)
    path = checkpoint_path(llm)
    saved = {} if fresh or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    # A case the model did not fully answer is not done. The first long run
    # lost its daily quota at case 13 and saved the rest as finished with
    # every turn a fallback, which a resume would then have skipped.
    redo = [cid for cid, v in saved.items() if v["fallbacks"] > MAX_CASE_FALLBACK_SHARE * v["turns"]]
    for cid in redo:
        del saved[cid]
    if saved or redo:
        print(f"resuming: {len(saved)} of {len(cases)} cases already done"
              + (f", {len(redo)} redone because the model did not answer" if redo else "")
              + f" ({path})")

    consensus = ConsensusProposer(primary=BayesianProposer(kb), secondary=LLMProposer(kb, llm))
    agent = DiagnosticAgent(
        kb=kb, proposer=consensus, gate=AbstentionGate(kb=kb), limits=LoopLimits(**REAL_LIMITS)
    )
    started = time.monotonic()
    for n, case in enumerate(cases, 1):
        if case.case_id in saved:
            continue
        turns, fallbacks = consensus.turns, consensus.secondary_fallbacks
        outcome = agent.run(case)
        if outcome.verdict is Verdict.COMMITTED:
            verdict = "correct" if outcome.prediction == case.diagnosis else "WRONG"
            detail = outcome.prediction
        else:
            verdict = "escalated"
            detail = (outcome.escalation.reason if outcome.escalation else "")[:90]
        record = {
            "verdict": verdict, "detail": detail,
            "turns": consensus.turns - turns,
            "fallbacks": consensus.secondary_fallbacks - fallbacks,
        }
        print(
            f"[{n:>2}/{len(cases)}] {case.case_id:<14} {verdict:<9} "
            f"model turns {record['turns']:>2}, fallbacks {record['fallbacks']}  "
            f"({time.monotonic() - started:.0f}s)",
            flush=True,
        )
        if record["fallbacks"] > MAX_CASE_FALLBACK_SHARE * record["turns"]:
            # The model has gone quiet (typically the daily quota). Going on
            # would only produce Bayes-against-Bayes cases; stop, keep what
            # is finished, and let a later run continue from here.
            print(
                f"\nstopped: the model gave no answer on {record['fallbacks']} of "
                f"{record['turns']} turns for {case.case_id} (quota or outage).\n"
                f"{len(saved)} of {len(cases)} cases are saved; run the same command "
                "later to continue from this case."
            )
            report_partial(cases, bayes_only, saved)
            return
        saved[case.case_id] = record
        path.write_text(json.dumps(saved, indent=1), encoding="utf-8")

    with_model = {cid: (v["verdict"], v["detail"]) for cid, v in saved.items()}
    total_turns = sum(v["turns"] for v in saved.values())
    total_fallbacks = sum(v["fallbacks"] for v in saved.values())
    print()
    changed = 0
    for case in cases:
        a, b = bayes_only[case.case_id], with_model[case.case_id]
        if a[0] != b[0]:
            changed += 1
            print(f"{case.case_id:<14} {case.diagnosis:<29} {a[0]:>9} -> {b[0]:<9} {b[1]}")
    print(f"\n{changed} of {len(cases)} cases changed outcome")
    print(f"  Bayesian loop alone        {tally(bayes_only)}")
    print(f"  with the model alongside   {tally(with_model)}")
    print(
        f"  model turns {total_turns}, of which it had no opinion (fallback) "
        f"on {total_fallbacks}"
    )
    if total_turns and total_fallbacks / total_turns > 0.05:
        print(
            "  NOT A RESULT: on more than 5% of turns the model gave no opinion\n"
            "  (quota, network or parsing), so those turns compared Bayes with\n"
            "  itself. Rerun when the model is reachable."
        )
    print(
        "\nThe consensus proposer never replaces the Bayesian ranking; the model's\n"
        "only lever is disagreement, which makes the gate escalate. So it can turn\n"
        "commits into escalations and never the reverse."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-complaint", action="store_true")
    parser.add_argument("--in-loop", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore the --in-loop checkpoint")
    args = parser.parse_args()

    from dxagent.llm import from_provider

    kb = build_knowledge_base(correlated=True)
    llm = from_provider(args.provider, args.model)
    mode = "in loop (consensus)" if args.in_loop else (
        f"single pass, {args.repeats} runs per case, complaint "
        f"{'withheld' if args.no_complaint else 'included'}"
    )
    print(f"{args.provider}:{getattr(llm, 'model', '?')}, {mode}\n")
    # One probe before hundreds of calls: an unreachable or exhausted model
    # would otherwise run every case on fallbacks and look like a result.
    try:
        llm.complete("Reply with the word ready.", max_tokens=5)
    except Exception as exc:  # noqa: BLE001 -- any failure means do not start
        print(f"model not usable right now: {str(exc)[:160]}")
        return 2
    if args.in_loop:
        report_in_loop(kb, llm, fresh=args.fresh)
    else:
        report_single_pass(kb, llm, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
