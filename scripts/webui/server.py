#!/usr/bin/env python3
"""A live-consultation web UI over the real agent -- no reimplemented logic.

    python scripts/webui/server.py
    uvicorn scripts.webui.server:app --reload   # equivalent, for development

Same pattern as ``consult.py``: the only new code is an ``Environment`` that
answers ``respond()`` from a live source instead of a lookup table. There
``input()`` blocks a terminal; here a ``queue.Queue`` blocks a background
thread running the real, unmodified ``DiagnosticAgent.run()`` loop until the
browser posts an answer. Nothing about the reasoning path is reimplemented
for the web -- the same ``kb``, ``proposer``, ``gate`` and ``LoopLimits`` as
every other script, so this cannot silently diverge from what the tests
cover.

Not for clinical use. The knowledge base is invented; see fixtures.py.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict  # noqa: E402
from dxagent.belief import BayesianProposer  # noqa: E402
from dxagent.datasets import build_knowledge_base  # noqa: E402
from dxagent.environment import Case  # noqa: E402
from dxagent.schemas import Action, Citation, Finding, Polarity  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


def humanise(concept: str) -> str:
    """`exam:raised_jvp` -> `raised jvp (examination)`. Copied from demo.py
    rather than imported, since that module has a CLI ``main()`` this server
    should not pull in."""
    kind, _, rest = concept.partition(":")
    label = (rest or kind).replace("_", " ")
    prefix = {"exam": "examination", "lab": "blood test", "imaging": "imaging"}
    return f"{label} ({prefix[kind]})" if rest and kind in prefix else label


@dataclass
class WebOracle:
    """Answers ``Environment.respond()`` from a queue instead of a prompt.

    Keeps its own running list of findings so the API can show the *current*
    differential while a question is still pending -- ``agent.run()`` builds
    its own ``CaseState`` internally and does not hand it out mid-loop, so
    this is a second, display-only call to the same ``proposer`` on the same
    findings, not a second source of truth for any decision.
    """

    kb: object
    proposer: BayesianProposer
    presenting_complaint: str
    answers: "queue.Queue[Polarity]" = field(default_factory=queue.Queue)
    findings: list[Finding] = field(default_factory=list)
    pending: Action | None = None
    history: list[dict] = field(default_factory=list)
    cost_spent: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def respond(self, action: Action) -> list[Finding]:
        with self.lock:
            self.pending = action
        polarity = self.answers.get()
        finding = Finding(
            concept=action.target,
            polarity=polarity,
            provenance=f"{action.kind.value} (user-reported)",
        )
        with self.lock:
            self.findings.append(finding)
            self.cost_spent += action.cost
            self.history.append(
                {
                    "concept": action.target,
                    "label": humanise(action.target),
                    "answer": polarity.value,
                }
            )
            self.pending = None
        return [finding]

    def live_differential(self) -> list[dict]:
        with self.lock:
            findings = list(self.findings)
        differential = self.proposer.propose(findings, self.presenting_complaint)
        return [
            _hypothesis(h, self.kb, differential)
            for h in differential.hypotheses[:8]
        ]


@dataclass
class Session:
    oracle: WebOracle
    thread: threading.Thread
    done: threading.Event
    kb: object
    limits: LoopLimits
    result: dict = field(default_factory=dict)


SESSIONS: dict[str, Session] = {}

app = FastAPI(title="dxagent live consultation")


class StartRequest(BaseModel):
    complaint: str


class AnswerRequest(BaseModel):
    polarity: Literal["present", "absent", "unknown"]


def _citation(c: Citation) -> dict:
    return {"source_id": c.source_id, "snippet": c.snippet}


def _hypothesis(h, kb, differential=None) -> dict:
    """One hypothesis, with its grounding, for the browser.

    The citations travel on every turn rather than only with the final
    result: the point of the grounding is that it is inspectable *while* the
    differential moves, not a justification assembled afterwards.

    ``grounding`` is sent separately from ``support`` and rendered separately.
    It says where the knowledge base entry came from, which for this project
    is "synthetic entry, not sourced"; it used to arrive at the front of the
    supporting list, so a disclosure that a hypothesis has no source was
    displayed as a reason to believe it.

    ``eliminated`` is what the panel was missing. Probabilities normalise
    across the eight causes, so a diagnosis can climb because its rivals were
    ruled out rather than because anything argued for it. A hypothesis with
    one weak supporting finding and a high probability is not a contradiction;
    it is a diagnosis reached by elimination, and until this field existed
    the interface showed the thin support and hid the elimination.
    """
    entry = kb.get(h.label)
    eliminated: list[dict] = []
    if differential is not None:
        seen: set[str] = set()
        for other in differential.hypotheses:
            if other.label == h.label:
                continue
            for c in other.against:
                if c.snippet and c.snippet not in seen:
                    seen.add(c.snippet)
                    eliminated.append(
                        {**_citation(c), "rival": humanise(other.label)}
                    )
    return {
        "label": h.label,
        "probability": h.probability,
        "red_flag": bool(entry is not None and entry.red_flag),
        "support": [_citation(c) for c in h.support if c.snippet],
        "against": [_citation(c) for c in h.against if c.snippet],
        "grounding": [_citation(c) for c in h.grounding if c.snippet],
        "eliminated": eliminated,
    }


def _wait_for_next(
    session: Session, timeout: float = 10.0, previous: Action | None = None
) -> None:
    """Block until the background loop reaches a *new* question or finishes.

    ``previous`` guards a real race: right after an answer is queued, the
    background thread has not yet cleared ``oracle.pending`` (it does that
    after ``.get()`` unblocks, before computing the next action), so a poll
    that only checks "is something pending" can return immediately with the
    question that was just answered rather than the next one. Comparing
    identity against the action that was pending *before* this answer was
    submitted is what actually detects progress.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session.done.is_set():
            return
        pending = session.oracle.pending
        if pending is not None and pending is not previous:
            return
        time.sleep(0.01)


def _status(session_id: str) -> dict:
    session = SESSIONS[session_id]
    oracle = session.oracle
    if session.done.is_set():
        outcome = session.result["outcome"]
        payload = {
            "session_id": session_id,
            "finished": True,
            "verdict": outcome.verdict.value,
            "confidence": outcome.confidence,
            "budget_spent": outcome.budget_spent,
            "max_cost": session.limits.max_cost,
            "max_turns": session.limits.max_turns,
            "turn": len(outcome.steps),
            "differential": [
                _hypothesis(h, session.kb, outcome.differential)
                for h in outcome.differential.hypotheses[:8]
            ],
            "prediction": outcome.prediction,
            "history": oracle.history,
        }
        if outcome.verdict is Verdict.ESCALATED:
            payload["escalation"] = {
                "reason": outcome.escalation.reason,
                "unresolved_question": (
                    humanise(outcome.escalation.unresolved_question)
                    if outcome.escalation.unresolved_question
                    else None
                ),
            }
        return payload

    action = oracle.pending
    return {
        "session_id": session_id,
        "finished": False,
        "turn": len(oracle.history) + 1,
        "cost_spent": oracle.cost_spent,
        "max_cost": session.limits.max_cost,
        "max_turns": session.limits.max_turns,
        "question": (
            {
                "concept": action.target,
                "label": humanise(action.target),
                "kind": action.kind.value,
                "cost": action.cost,
                # The selector's own stated reason for this question -- a
                # rule-out, a mandated workup item, or expected information
                # gain. Shown rather than hidden: it is the difference
                # between a questionnaire and a reasoning trace.
                "rationale": action.rationale,
            }
            if action
            else None
        ),
        "differential": oracle.live_differential(),
        "history": oracle.history,
    }


@app.post("/api/session")
def create_session(req: StartRequest) -> dict:
    complaint = req.complaint.strip() or "unspecified presentation"
    kb = build_knowledge_base(correlated=True)
    proposer = BayesianProposer(kb)
    oracle = WebOracle(kb=kb, proposer=proposer, presenting_complaint=complaint)
    case = Case(
        case_id=f"web-{uuid.uuid4().hex[:8]}",
        presenting_complaint=complaint,
        features={},
        diagnosis="unknown",
        initial_findings=(),
    )
    limits = LoopLimits()
    agent = DiagnosticAgent(
        kb=kb,
        proposer=proposer,
        gate=AbstentionGate(kb=kb),
        limits=limits,
    )
    session = Session(
        oracle=oracle,
        thread=None,  # type: ignore[arg-type]
        done=threading.Event(),
        kb=kb,
        limits=limits,
    )
    session_id = uuid.uuid4().hex

    def _run() -> None:
        outcome = agent.run(case, environment=oracle)
        session.result["outcome"] = outcome
        session.done.set()

    thread = threading.Thread(target=_run, daemon=True)
    session.thread = thread
    SESSIONS[session_id] = session
    thread.start()
    _wait_for_next(session)
    return _status(session_id)


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(404, "no such session")
    return _status(session_id)


@app.post("/api/session/{session_id}/answer")
def answer_session(session_id: str, req: AnswerRequest) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(404, "no such session")
    session = SESSIONS[session_id]
    if session.done.is_set():
        raise HTTPException(409, "session already finished")
    previous = session.oracle.pending
    if previous is None:
        raise HTTPException(409, "no question is currently pending")
    session.oracle.answers.put(Polarity(req.polarity))
    _wait_for_next(session, previous=previous)
    return _status(session_id)


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8420)
