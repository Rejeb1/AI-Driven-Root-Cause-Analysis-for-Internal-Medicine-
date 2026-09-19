# -*- coding: utf-8 -*-
"""dxagent deep dive: a long-form explainer of what the project actually is.

A companion to build_explainer_pdf.py rather than a replacement. That one is
seven pages and answers "what is this"; this one is twelve and answers "how
does it work, what was measured, and what is wrong with it".

Every number is read from the live code through deck_figures and the checks in
plausibility_check, for the reason the whole project keeps relearning: a figure
written into a document goes stale silently, and a PDF goes stale more quietly
than anything else because it looks finished.
"""

import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from deck_figures import collect  # noqa: E402

F = collect(reason_limit=150)

from dxagent import provenance  # noqa: E402
from dxagent.datasets.fixtures import build_knowledge_base  # noqa: E402
from dxagent.vocabulary import OPERATIONAL_DEFINITIONS  # noqa: E402

KB = build_knowledge_base(correlated=True)
PARAMS = provenance.parameter_report()

_spec = importlib.util.spec_from_file_location("pc", HERE / "plausibility_check.py")
PC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PC)

CITATIONS = Counter()
CONCEPTS = set()
for _e in KB.diseases():
    CONCEPTS.update(_e.features)
    for _c in _e.features:
        _s = _e.sources.get(_c)
        if _s and _s.citation:
            CITATIONS[_s.citation.source_id] += 1


def _concepts_with(prefix):
    if prefix is None:
        return sorted(c for c in CONCEPTS if ":" not in c)
    return sorted(c.split(":", 1)[1] for c in CONCEPTS if c.startswith(prefix))


HISTORY = _concepts_with(None)
EXAM = _concepts_with("exam:")
LABS = _concepts_with("lab:")
IMAGING = _concepts_with("imaging:")

ORDER_CLAIMS = len(PC.ORDERING_CLAIMS)
ORDER_PAIRS = sum(len(x[2]) for x in PC.ORDERING_CLAIMS)
ORDER_RED = len(PC.KNOWN_ORDERING_VIOLATIONS)
ORDER_FAILS = PC.ordering_failures(KB)


def _baselines():
    """The section-9 table, recomputed rather than transcribed.

    Same split and same gate settings run_eval.py uses by default, so the
    numbers here and the ones the script prints cannot drift apart.
    """
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits
    from dxagent.baselines import RetrievalOnlyBaseline, SinglePassBaseline
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import build_cases
    from dxagent.evaluation import evaluate, run_agent, split_cases
    from dxagent.evaluation.metrics import ranking_metrics

    calibration, test = split_cases(build_cases(), calibration_fraction=0.35)
    agent = DiagnosticAgent(
        kb=KB, proposer=BayesianProposer(KB), gate=AbstentionGate(kb=KB),
        limits=LoopLimits(max_turns=12, max_cost=40.0),
    )
    result = evaluate(agent, test, calibration_cases=calibration)

    complete = sum(len(c.features) for c in test) / max(len(test), 1)
    by_id = {c.case_id: c for c in test}
    gathered = [
        len(by_id[o.case_id].initial_findings) + len(o.steps)
        for o in result.outcomes if o.case_id in by_id
    ]
    loop_evidence = sum(gathered) / max(len(gathered), 1)

    out = []
    for name, outcomes, ev in (
        ("retrieval-only (no inference)",
         run_agent(RetrievalOnlyBaseline(KB), test), complete),
        ("single-pass (KB, no loop)",
         run_agent(SinglePassBaseline(KB), test), complete),
        ("agentic loop", result.outcomes, loop_evidence),
    ):
        r = ranking_metrics(outcomes, result.truths)
        out.append((name, f"{r.top1:.1%}", f"{r.top5:.1%}", f"{r.mrr:.3f}",
                    f"{ev:.1f}"))
    return out, complete, loop_evidence, result


BASE_ROWS, COMPLETE_EV, LOOP_EV, EVAL = _baselines()


def _worked_case(diagnosis):
    """One real case run turn by turn, so the loop can be read rather than described."""
    from dxagent import AbstentionGate, DiagnosticAgent, LoopLimits, Verdict
    from dxagent.belief import BayesianProposer
    from dxagent.datasets import REAL_CASES

    case = next(c for c in REAL_CASES if c.diagnosis == diagnosis)
    agent = DiagnosticAgent(
        kb=KB, proposer=BayesianProposer(KB), gate=AbstentionGate(kb=KB),
        limits=LoopLimits(uninformative_turns_still_count=False,
                          unanswered_actions_still_cost=False),
    )
    outcome = agent.run(case)
    rows = []
    for step in outcome.steps:
        finding = step.findings[0] if step.findings else None
        answer = "not recorded"
        if finding is not None:
            answer = {"present": "present", "absent": "absent"}.get(
                finding.polarity.value, "not recorded")
        top = step.differential_after.top
        rows.append((
            str(step.index + 1),
            step.action.target.split(":", 1)[-1].replace("_", " "),
            step.action.rationale,
            answer,
            f"{top.label.replace('_', ' ')} {top.probability:.0%}",
        ))
    verdict = ("committed" if outcome.verdict is Verdict.COMMITTED else "escalated")
    return case, rows, verdict, outcome


WORKED_CASE, WORKED_ROWS, WORKED_VERDICT, WORKED_OUT = _worked_case(
    "community_acquired_pneumonia")
ESC_CASE, _, _, ESC_OUT = _worked_case("pericarditis")

# ---------------------------------------------------------------- palette --
INK = colors.HexColor("#16202b")
MUTED = colors.HexColor("#5d6b7a")
LINE = colors.HexColor("#d8dee5")
PANEL = colors.HexColor("#ffffff")
BG_SOFT = colors.HexColor("#f6f8fa")
NAVY = colors.HexColor("#12263c")

TEAL, TEAL_BG = colors.HexColor("#0e7c7b"), colors.HexColor("#d9f0ef")
BLUE, BLUE_BG = colors.HexColor("#1d4e8f"), colors.HexColor("#dbe6f5")
AMBER, AMBER_BG = colors.HexColor("#b06b00"), colors.HexColor("#fbeacd")
PLUM, PLUM_BG = colors.HexColor("#6b3fa0"), colors.HexColor("#e8dff5")
GREEN, GREEN_BG = colors.HexColor("#1f7a3d"), colors.HexColor("#dcf0e2")
RED, RED_BG = colors.HexColor("#a4243b"), colors.HexColor("#f7dfe3")

PAGE_W, PAGE_H = LETTER
MARGIN = 1.7 * cm
FULLW = PAGE_W - 2 * MARGIN

S = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16.5, leading=20,
                         textColor=INK, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                         textColor=INK, spaceBefore=9, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=14,
                           textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.6, leading=12.2,
                            textColor=MUTED, spaceAfter=4),
    "card": ParagraphStyle("card", fontName="Helvetica", fontSize=8.9, leading=12.6,
                           textColor=INK, spaceAfter=2),
    "cardt": ParagraphStyle("cardt", fontName="Helvetica-Bold", fontSize=10, leading=13,
                            spaceAfter=3),
    "big": ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=19, leading=22,
                          alignment=TA_CENTER),
    "biglb": ParagraphStyle("biglb", fontName="Helvetica", fontSize=7.4, leading=9.6,
                            textColor=MUTED, alignment=TA_CENTER),
    "cover": ParagraphStyle("cover", fontName="Helvetica-Bold", fontSize=28, leading=33,
                            textColor=colors.white, alignment=TA_CENTER),
    "csub": ParagraphStyle("csub", fontName="Helvetica", fontSize=12, leading=17,
                           textColor=colors.HexColor("#a9c4e4"), alignment=TA_CENTER),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.2, leading=11.4,
                           textColor=INK, spaceAfter=4),
}


class Rule(Flowable):
    def __init__(self, w, color, t=3.5):
        Flowable.__init__(self)
        self.width, self.color, self.t, self.height = w, color, t, t

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.t, self.t / 2, stroke=0, fill=1)


def card(title, lines, fill, edge):
    body = [Paragraph(title, ParagraphStyle("t", parent=S["cardt"], textColor=edge))]
    body += [Paragraph(l, S["card"]) for l in lines]
    t = Table([[body]])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def row(cards, gap=0.28 * cm):
    n = len(cards)
    w = (FULLW - gap * (n - 1)) / n
    t = Table([cards], colWidths=[w] * n)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), gap),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def stat(value, label, color):
    inner = [Paragraph(value, ParagraphStyle("v", parent=S["big"], textColor=color)),
             Paragraph(label, S["biglb"])]
    t = Table([[inner]])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_SOFT),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def table(data, widths, header_bg=INK, small=False):
    size = 8.0 if small else 8.6
    head = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=size,
                          leading=size + 2.8, textColor=colors.white)
    body = ParagraphStyle("tb", fontName="Helvetica", fontSize=size,
                          leading=size + 2.8, textColor=INK)
    wrapped = [[c if not isinstance(c, str) else Paragraph(c, head if r == 0 else body)
                for c in rr] for r, rr in enumerate(data)]
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PANEL, BG_SOFT]),
    ]))
    return t


def quote(text, edge=PLUM):
    t = Table([[Paragraph(text, ParagraphStyle("q", parent=S["body"],
                                               textColor=INK, fontName="Helvetica-Oblique"))]])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BG_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "&hellip;"


story = []


def cover_bg(c, d):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    for i, col in enumerate((TEAL, BLUE, PLUM, AMBER, GREEN)):
        c.setFillColor(col)
        c.rect(MARGIN + i * (FULLW / 5), PAGE_H - 4.6 * cm, FULLW / 5, 0.34 * cm,
               stroke=0, fill=1)
        c.rect(MARGIN + i * (FULLW / 5), 2.2 * cm, FULLW / 5, 0.12 * cm,
               stroke=0, fill=1)
    c.restoreState()


story += [Spacer(1, 5.4 * cm)]
story += [Paragraph("dxagent", S["cover"])]
story += [Paragraph("An agentic diagnostic reasoner for acute dyspnoea and chest pain"
                    "<br/>how it works, what was measured, and what is wrong with it",
                    S["csub"])]
story += [Spacer(1, 1.0 * cm)]
story += [Paragraph(
    f"{F.total} likelihoods &middot; {F.coverage}% carrying a citation &middot; "
    f"{F.real_total} real patients &middot; {F.tests} tests",
    ParagraphStyle("cf", parent=S["csub"], fontSize=10.5,
                   textColor=colors.HexColor("#7f9dc4")))]
story += [Spacer(1, 0.5 * cm)]
story += [Paragraph(
    "Mohamed Aziz Ben Rejeb &nbsp;&middot;&nbsp; DeepShift AI Summer Internship 2026, "
    "Project 1 (Healthcare)",
    ParagraphStyle("cauth", parent=S["csub"], fontSize=9.5,
                   textColor=colors.HexColor("#6f8db4")))]

story += [Spacer(1, 2.6 * cm)]
_toc = ParagraphStyle("toc", fontName="Helvetica", fontSize=9.4, leading=15.5,
                      textColor=colors.HexColor("#9fb8d6"), alignment=TA_CENTER)
story += [Paragraph(
    "1&nbsp; The problem, and the scope decision behind it<br/>"
    "2&nbsp; How it works<br/>"
    "3&nbsp; The knowledge base, and the number that matters<br/>"
    "4&nbsp; What sourcing found, including what it took back<br/>"
    "5&nbsp; What checks the numbers, and what checks the documents<br/>"
    "6&nbsp; How it is evaluated, and which numbers to distrust<br/>"
    f"7&nbsp; The {F.real_total} real patients<br/>"
    "8&nbsp; What is wrong with it<br/>"
    "9&nbsp; What this is, in one page", _toc)]

story += [Spacer(1, 2.2 * cm)]
story += [Paragraph(
    "Not for clinical use. No part of this has been reviewed by a clinician.",
    ParagraphStyle("cwarn", parent=S["csub"], fontSize=9,
                   textColor=colors.HexColor("#c98a95")))]
story += [PageBreak()]

# =========================================================== 1. THE PROBLEM
story += [Paragraph("1 &nbsp; The problem, and the scope decision behind it", S["h1"]),
          Rule(FULLW, TEAL), Spacer(1, 0.3 * cm)]
story += [Paragraph(
    "A patient arrives short of breath, with chest pain, and nobody yet knows why. "
    "Eight conditions could explain it and they compete for the same evidence. Three "
    "of them kill people within hours. The task is not to name a disease from a "
    "complete record &mdash; it is to decide, one question at a time, what to ask "
    "next, and when the answer is certain enough to act on.", S["body"])]

story += [Paragraph("The eight candidates", S["h2"])]
story += [table([
    ["Cause", "Why it is in scope", "Red flag"],
    ["Pulmonary embolism", "Time-critical, commonly missed, published rules (Wells, PERC)", "yes"],
    ["Community-acquired pneumonia", "The most common competitor", ""],
    ["Acute coronary syndrome", "Time-critical, published rule (HEART)", "yes"],
    ["Acute pulmonary oedema", "The acute decompensation of heart failure", "yes"],
    ["COPD exacerbation", "Overlaps heavily on dyspnoea and wheeze", ""],
    ["Asthma exacerbation", "Separating it from COPD is a real clinical task", ""],
    ["Pericarditis", "Shares pleuritic pain with pulmonary embolism", ""],
    ["Panic attack", "The benign cause that must not be reached by exclusion alone", ""],
], [5.0 * cm, 9.6 * cm, FULLW - 14.6 * cm])]

story += [Spacer(1, 0.25 * cm)]
story += [Paragraph("One presentation, not three", S["h2"])]
story += [Paragraph(
    "The brief suggested three to five presentations. This covers one, deliberately. "
    "Hyponatraemia, anaemia and unexplained dyspnoea are separate diagnostic problems "
    "with disjoint evidence sets; three of them covered thinly produces a knowledge "
    "base too shallow to discriminate <i>within</i> any one of them. One presentation "
    "with eight genuinely competing causes exercises the same machinery against a "
    "differential where the candidates actually fight each other.", S["body"])]

story += [Spacer(1, 0.2 * cm)]
story += [row([
    card("Asymmetric by design", [
        "Three causes are time-critical. The gate will not commit to a "
        "<i>more likely</i> benign diagnosis while one of them still holds "
        "meaningful probability.",
        "<font color='#5d6b7a'>Missing a panic attack and missing a pulmonary "
        "embolism are not the same error.</font>"], BLUE_BG, BLUE),
    card(f"{len(CONCEPTS)} findings, {F.total} cells", [
        "Eight diseases against the findings each one describes gives the "
        "likelihood grid the rest of this document keeps returning to.",
        "<font color='#5d6b7a'>It is simultaneously the engine and the "
        "principal weakness.</font>"], AMBER_BG, AMBER),
])]

story += [Paragraph("The findings it can ask about", S["h2"])]
story += [table([
    ["Where it comes from", "n", "Findings"],
    ["History &mdash; free, instant", str(len(HISTORY)),
     ", ".join(h.replace("_", " ") for h in HISTORY)],
    ["Examination &mdash; cheap, one turn", str(len(EXAM)),
     ", ".join(e.replace("_", " ") for e in EXAM)],
    ["Laboratory &mdash; costed, delayed", str(len(LABS)),
     ", ".join(l.replace("_", " ") for l in LABS)],
    ["Imaging &mdash; most expensive", str(len(IMAGING)),
     ", ".join(i.replace("_", " ") for i in IMAGING)],
], [4.4 * cm, 0.9 * cm, FULLW - 5.3 * cm], header_bg=TEAL, small=True)]
story += [Paragraph(
    "The costs are one of the invented families counted in section 5: an ordinal "
    "ranking a non-clinician can defend, not a tariff from any health system.",
    S["small"])]
story += [PageBreak()]

# ========================================================= 2. ARCHITECTURE
story += [Paragraph("2 &nbsp; How it works", S["h1"]), Rule(FULLW, BLUE),
          Spacer(1, 0.3 * cm)]
story += [Paragraph(
    "Four layers and a loop. The loop runs until the gate commits, escalates, or a "
    "budget runs out.", S["body"])]

story += [table([
    ["Layer", "What lives there"],
    ["Knowledge", "The likelihood tables, Wells / PERC / CURB-65 / HEART with citations, "
                  "Merck Manual sentences, the findings-and-causes graph, and the "
                  "provenance record of where every number came from"],
    ["Reasoning", "Naive Bayes in log space with correlation weighting, the "
                  "information-gain action selector, temperature scaling, conformal "
                  "prediction and the abstention gate"],
    ["Orchestration", "A hand-written reference loop and the same loop as a LangGraph "
                      "state machine, held equivalent by a test that runs both over "
                      "every case and asserts identical verdicts"],
    ["Evaluation", "Ranking, calibration, selective prediction, DDx recall, and the two "
                   "baselines the brief requires"],
], [3.4 * cm, FULLW - 3.4 * cm])]

story += [Paragraph("The five steps of one turn", S["h2"])]
for n, name, colr, bg, text in [
    ("1", "Propose", TEAL, TEAL_BG,
     "Rank the eight causes on the findings so far. Correlation weighting stops four "
     "facets of one clinical picture counting as four independent pieces of evidence."),
    ("2", "Calibrate", BLUE, BLUE_BG,
     "Temperature-scale the posterior. The scaler refuses to fit below 30 labelled "
     "cases, and the report now says so rather than printing a default as a finding."),
    ("3", "Check the gate", PLUM, PLUM_BG,
     "Six conditions, all of which must hold before committing."),
    ("4", "Choose the next step", AMBER, AMBER_BG,
     "Highest expected information gain per unit cost, with a red-flag weight and a "
     "two-tier rule preferring a decisive test over a marginally cheaper question."),
    ("5", "Ask", GREEN, GREEN_BG,
     "Put it to the patient or the record. A finding the source never recorded costs "
     "neither a turn nor money &mdash; no test was performed."),
]:
    story += [row([card(f"{n} &nbsp; {name}", [text], bg, colr)])]
    story += [Spacer(1, 0.12 * cm)]

story += [Paragraph("Two engines, held equivalent by a test", S["h2"])]
story += [Paragraph(
    "The same loop exists twice: a hand-written reference implementation, and a "
    "LangGraph state machine of four nodes &mdash; propose, calibrate, decide, observe "
    "&mdash; with the decide node routing back to propose or out to a verdict. Having "
    "two is only defensible if they cannot disagree, so a test runs both over every "
    "case and asserts identical verdicts, identical predictions and identical evidence "
    "sequences. The graph is the version a reviewer can read as a diagram; the loop is "
    "the version that runs when LangGraph is not installed.", S["body"])]

story += [row([
    card("Selector: gain per unit cost", [
        "Expected reduction in entropy over the eight causes, divided by what the "
        "action costs, with an extra weight when it bears on a time-critical cause."],
        BLUE_BG, BLUE),
    card("The tie-break that had to be added", [
        "Pure gain-per-cost prefers a nearly free question with slight gain over a "
        "decisive scan. A two-tier rule takes the decisive action when the cheap one "
        "would not move the gate."], AMBER_BG, AMBER),
])]

story += [PageBreak()]
story += [Paragraph("The six conditions that must all hold before it answers",
                    S["h2"])]
story += [table([
    ["#", "Condition", "Why"],
    ["1", "Calibrated top-1 probability &ge; 0.65", "A floor on certainty"],
    ["2", "Top-two margin &ge; 0.15", "Catches contested posteriors a probability "
                                      "threshold alone waves through"],
    ["3", "Top hypothesis grounded in the KB, with a citation", "No answer without a source"],
    ["4", "<b>No red-flag diagnosis above 0.10</b>", "The asymmetry: will not commit to a "
                                                     "likelier benign cause while a killer is live"],
    ["5", "Proposer disagreement below threshold", "When two proposers run, disagreement is a signal"],
    ["6", "The evidence is explained by something", "Refuses to answer on findings the KB cannot account for"],
], [0.9 * cm, 7.4 * cm, FULLW - 8.3 * cm], header_bg=PLUM)]
story += [Spacer(1, 0.2 * cm)]
story += [Paragraph(
    "Failing any one produces an escalation carrying a stated reason and a named "
    "unresolved question &mdash; not a silent low-confidence answer. Several now read "
    "<i>&ldquo;Pulmonary embolism exclusion could not be completed &mdash; sought but "
    "not available: lab:raised_d_dimer&rdquo;</i>.", S["small"])]

story += [Paragraph("One real case, turn by turn", S["h2"])]
story += [Paragraph(
    f"<i>{WORKED_CASE.presenting_complaint}</i> &mdash; case "
    f"<font face='Courier'>{WORKED_CASE.case_id}</font>, a published case report, "
    "run through the shipped configuration. Nothing here is scripted: the "
    "&ldquo;why that one&rdquo; column is the selector's own reason.", S["small"])]
_rows = [["#", "Asked for", "Why that one", "Answer", "Top-1 after"]]
_rows += [list(r) for r in WORKED_ROWS]
story += [table(_rows, [0.9 * cm, 3.2 * cm, 6.0 * cm, 2.1 * cm,
                        FULLW - 12.2 * cm], header_bg=GREEN, small=True)]
story += [Paragraph(
    f"It <b>{WORKED_VERDICT}</b> on "
    f"{WORKED_OUT.prediction.replace('_', ' ')}, which is the right answer. Note turn "
    "one: it asks for a D-dimer <i>to rule pulmonary embolism out</i>, not because "
    "embolism is likely &mdash; the red-flag weight makes exclusion worth paying for. "
    "Note also how many answers come back <i>not recorded</i>: a case report is not a "
    "questionnaire, and the loop has to reach a verdict on what the record happens to "
    "contain.", S["small"])]
story += [PageBreak()]

# ==================================================== 3. THE KNOWLEDGE BASE
story += [Paragraph("3 &nbsp; The knowledge base, and the number that matters", S["h1"]),
          Rule(FULLW, AMBER), Spacer(1, 0.3 * cm)]

story += [row([
    stat(str(F.invented), "likelihoods invented", AMBER),
    stat(f"{F.coverage}%", "carrying a citation", GREEN),
    stat(f"{F.priors_sourced}/{F.priors_total}", "disease priors sourced", TEAL),
    stat(str(PARAMS.total), "other invented parameters", RED),
])]
story += [Spacer(1, 0.3 * cm)]

story += [Paragraph(
    f"<b>{F.invented} of {F.total} likelihoods are invented.</b> That is the most "
    "important fact about this project, and it is stated on its own line because "
    "burying it would be the error the whole design is organised against. Every "
    "number carries a tier, and the count is reported on every run.", S["body"])]

story += [row([
    card("MEASURED", ["A frequency counted in a named cohort or dataset. The only tier "
                      "that supports a claim about the world rather than about a text."],
         GREEN_BG, GREEN),
    card("NARRATIVE", ["A phrase from a reference text through a <i>fixed</i> rubric: "
                       "&ldquo;always&rdquo;&rarr;0.95, &ldquo;often&rdquo;&rarr;0.55. "
                       "Never re-tuned to produce a preferred answer."], BLUE_BG, BLUE),
    card("INVENTED", ["A plausible guess by a non-clinician. Useful for exercising code "
                      "paths and for nothing else."], AMBER_BG, AMBER),
])]

story += [Paragraph("Where the sourced numbers came from", S["h2"])]
_src_names = {
    "PIOPED-II-2007": "PIOPED II clinical tables &mdash; the arm investigated for embolism who did not have one",
    "MSD-19E": "Merck Manual 19e, narrative phrases through the fixed rubric",
    "DDXPLUS": "Frequencies counted in a simulator of ~1.3M patients",
    "WANG-2005": "JAMA review of dyspnoea in the ED, recovered by inverting likelihood ratios",
    "PIOPED-II-CTA-2006": "PIOPED II accuracy paper &mdash; CT angiography sensitivity and specificity",
    "MINIATI-2012": "360 confirmed pulmonary embolism patients, Firenze",
    "STATPEARLS-PERICARDITIS": "StatPearls, acute pericarditis",
    "CAP-DIAGNOSTIC-MODEL-2024": "954 acutely admitted patients, 265 with confirmed pneumonia",
    "DDIMER-TURBIDIMETRIC-META": "D-dimer meta-analysis, 9 studies, n=1901",
    "TROPONIN-APE-2019": "220 consecutive acute pulmonary embolism admissions",
    "NTPROBNP-APE-2012": "63 consecutive emergency embolism patients",
    "ZEGRE-HEMSEY-2018": "Acute coronary syndrome cohort",
    "DDIMER-AECOPD-2013": "148 COPD exacerbations investigated for embolism",
    "NOORAIN-2016": "COPD exacerbation troponin cohort",
    "AECOPD-IMAGING-SR-2020": "Systematic review of imaging at COPD exacerbation",
}
_rows = [["Source", "Cells", "What it is"]]
for _sid, _n in CITATIONS.most_common():
    _rows.append([_sid, str(_n), _src_names.get(_sid, _sid)])
story += [table(_rows, [5.4 * cm, 1.3 * cm, FULLW - 6.7 * cm], header_bg=GREEN, small=True)]

story += [Spacer(1, 0.2 * cm)]
story += [Paragraph(
    "Both bulk routes are <b>proven</b> exhausted rather than assumed so. DDXPlus was "
    "closed by computing all 80 pairs its mapping can reach: 15 already sourced, and "
    "13 unfilled pairs reading <i>exactly</i> 0.000 &mdash; the simulator omitting a "
    "finding from its rule base, not a frequency. Writing them would have asserted "
    "fever in 0% of pericarditis patients. Merck was closed by two complete passes "
    "over all eight relevant chapters.", S["small"])]
story += [PageBreak()]

# ====================================================== 4. THE SOURCING ARC
story += [Paragraph("4 &nbsp; What sourcing found, including what it took back", S["h1"]),
          Rule(FULLW, GREEN), Spacer(1, 0.3 * cm)]
story += [Paragraph(
    "Sourcing a number is not the same as improving a model, and this project has the "
    "record to prove it. Every correction below replaced a guess with a measurement. "
    "Several made the results worse, and those are kept.", S["body"])]

story += [table([
    ["Cell", "Was", "Now", "What it means"],
    ["P(rest dyspnoea | pneumonia)", "0.05", "0.67",
     "A faithful transcription of a Merck sentence about pneumonia at <i>all</i> "
     "severities, applied to an ED population who came in <i>because</i> they were "
     "breathless. It was arguing against the correct diagnosis in real patients."],
    ["P(raised BNP | embolism)", "0.20", "0.62",
     "Asserted the opposite of the mechanism: embolism strains the right ventricle, "
     "which is what releases the peptide."],
    ["P(troponin | embolism)", "0.25", "0.53",
     "Sat <i>below</i> the sourced 0.32 for COPD &mdash; claiming a COPD flare raises "
     "troponin more often than a pulmonary embolism does."],
    ["P(CTPA defect | embolism)", "0.95", "0.83",
     "A safety correction. At 0.95 a negative scan read as near-conclusive; the "
     "measured sensitivity means roughly <b>one embolism in six is missed</b>."],
    ["P(immobility | embolism)", "0.61", "0.25",
     "A simulator value against 192 real patients. The same gap as pleuritic pain, "
     "71% simulated against 33% observed."],
    ["P(leg swelling | oedema)", "0.999", "0.50",
     "DDXPlus recorded it in 7202 of 7205 simulated patients. A likelihood of 0.999 "
     "should have been suspicious on sight; it survived because it carried a citation."],
], [4.6 * cm, 1.3 * cm, 1.3 * cm, FULLW - 7.2 * cm], header_bg=GREEN, small=True)]

story += [Paragraph("A claim withdrawn, not just a number changed", S["h2"])]
story += [Paragraph(
    "The scope document once recorded, as a finding, that <i>&ldquo;exertional chest "
    "pain no longer raises acute coronary syndrome&rdquo;</i> &mdash; DDXPlus scored it "
    "at 0.77 in pulmonary oedema against 0.36 in ACS, and that was written up as "
    "something the sourced numbers had revealed.", S["body"])]
story += [quote(
    "It was a mapping error. The DDXPlus question behind it asks &ldquo;Do you have "
    "<b>symptoms</b> that are increased with physical exertion but alleviated with "
    "rest?&rdquo; &mdash; symptoms, not chest pain. Heart-failure patients answer yes "
    "because their breathlessness does exactly that. The mapping was withdrawn and "
    "coverage fell.", RED)]
story += [Paragraph(
    "The mapping file's own docstring, written before the mistake was made, predicts it "
    "precisely: a mis-mapped number is <i>worse</i> than the invented one it replaced, "
    "because it arrives with 1.3M patients behind it. Three of the ten original mappings "
    "have now failed the same way: the comment beside each paraphrased the question "
    "instead of quoting it.", S["small"])]

story += [Paragraph("The rubric that turns a sentence into a number", S["h2"])]
story += [Paragraph(
    "A reference text says a finding is &ldquo;usually&rdquo; present. Turning that into "
    "0.75 is a judgement, and the only way to stop it being made afresh each time "
    "&mdash; conveniently, in whichever direction helps &mdash; is to fix the map once "
    "and apply it blindly.", S["body"])]
_seen = {}
for _phrase, _band in provenance.NARRATIVE_RUBRIC.items():
    _seen.setdefault(_band, []).append(_phrase)
_rub = [(" / ".join(_ps), _band) for _band, _ps in _seen.items()]
_half = (len(_rub) + 1) // 2
_rows = [["Phrase in the source text", "Value", "Range",
          "Phrase in the source text", "Value", "Range"]]
_left, _right = _rub[:_half], _rub[_half:] + [None] * (_half - len(_rub[_half:]))
for _a, _b in zip(_left, _right):
    _cells = [_a[0], f"{_a[1][0]:.2f}", f"{_a[1][1]:.2f}&ndash;{_a[1][2]:.2f}"]
    _cells += ([_b[0], f"{_b[1][0]:.2f}", f"{_b[1][1]:.2f}&ndash;{_b[1][2]:.2f}"]
               if _b else ["", "", ""])
    _rows.append(_cells)
_w = FULLW / 6
story += [table(_rows, [_w * 1.45, _w * 0.7, _w * 0.85] * 2, header_bg=BLUE,
                small=True)]
story += [Paragraph(
    f"{F.merck} of the {F.sourced} sourced cells came through this map, which has never "
    "been re-tuned to change an outcome.", S["small"])]
story += [PageBreak()]

# ============================================================= 5. THE AUDITS
story += [Paragraph("5 &nbsp; What checks the numbers, and what checks the documents",
                    S["h1"]), Rule(FULLW, PLUM), Spacer(1, 0.3 * cm)]
story += [Paragraph(
    "Every test in a normal project checks outputs. The knowledge base is inputs, and "
    "for most of this project's life nothing looked at it at all. Three audits now do.",
    S["body"])]

story += [row([
    card(f"Ordering audit &mdash; {ORDER_CLAIMS} claims, {ORDER_PAIRS} comparisons", [
        "For each finding: which disease must lead the column, over which rivals, "
        "and why. Reasons are textbook-level so a clinician can accept or reject "
        "each in one sentence.",
        "They constrain <i>ordering</i>, not magnitude &mdash; ordering is what a "
        "non-clinician can assert honestly.",
        f"<font color='#a4243b'><b>{ORDER_RED} comparisons currently fail</b>, "
        "listed with the reason each is tolerated.</font>"], PLUM_BG, PLUM),
    card("Documentation drift guard", [
        "Every coverage figure quoted in SCOPE, DESIGN, RESPONSIBLE_AI and the "
        "write-up is checked against a live run. Prose drifts silently because "
        "nothing executes it.",
        "<font color='#5d6b7a'>The guard once drifted itself &mdash; it hardcoded a "
        "denominator that stopped matching any document, and passed. It now reads "
        "the denominator from the knowledge base.</font>"], TEAL_BG, TEAL),
])]
story += [Spacer(1, 0.2 * cm)]
story += [row([
    card(f"{PARAMS.total} parameters nobody was counting", [
        "The provenance report covered likelihoods and priors. It did not count "
        + ", ".join(f"{g.count} {g.name}" for g in PARAMS.groups) + ".",
        "The scope document already said why that is dangerous, about the priors: "
        "<i>an invented number the audit cannot name reads as an absence of a "
        "problem.</i> The hole was closed there and nobody asked whether it existed "
        "elsewhere."], AMBER_BG, AMBER),
    card(f"{len(OPERATIONAL_DEFINITIONS)} concepts given a threshold", [
        "Every concept had an HPO label naming a qualitative state &mdash; "
        "&ldquo;Hypoxemia&rdquo;, &ldquo;Elevated circulating D-dimer&rdquo; "
        "&mdash; and no cutoff, while every study reports one.",
        "That is <i>why</i> three columns were unsourceable: matching a source to a "
        "concept needed a judgement call every time."], BLUE_BG, BLUE),
])]

story += [Paragraph("What an ordering claim looks like", S["h2"])]
_rows = [["Finding", "Must lead", "Over", "Because"]]
for _concept, _leader, _rivals, _why in PC.ORDERING_CLAIMS[:7]:
    _rows.append([_concept, _leader.replace("_", " "), str(len(_rivals)), _why])
story += [table(_rows, [4.8 * cm, 3.7 * cm, 1.1 * cm, FULLW - 9.6 * cm],
                header_bg=PLUM, small=True)]
story += [Paragraph(
    f"Seven of {ORDER_CLAIMS}. Each reason is short on purpose: a clinician reading it "
    "can accept or reject the claim without reading any code, and rejecting one "
    "invalidates a specific set of cells rather than the whole table.", S["small"])]

story += [Paragraph("What the audit caught that the tests did not", S["h2"])]
story += [Paragraph(
    "Five errors, none found by a failing test, and four of them in pulmonary embolism "
    "&mdash; in a system built around not missing it. The technique costs nothing: "
    "sort each column, ask whether the ordering matches what the diseases do, and look "
    "hardest where a sourced value sits below an invented one. Two of the five are in "
    "the table in section 4; the natriuretic-peptide and troponin inversions are the "
    "others, and both were <i>sourced</i> cells sitting below invented rivals.",
    S["body"])]
story += [PageBreak()]

# ============================================================ 6. EVALUATION
story += [Paragraph("6 &nbsp; How it is evaluated, and which numbers to distrust",
                    S["h1"]), Rule(FULLW, BLUE), Spacer(1, 0.3 * cm)]

story += [Paragraph("Three case sets, doing three different jobs", S["h2"])]
story += [table([
    ["Set", "n", "What it is for", "Result"],
    ["Fixtures", str(F.fixtures_total),
     "Hand-written during development. <b>Saturated</b> &mdash; the system aces them "
     "however the mechanisms are set, so they can no longer separate a good change "
     "from a bad one.",
     f"{F.fixtures_correct}/{F.fixtures_total} top-1"],
    ["Hard cases", "5",
     "Five named diagnostic traps written to replace them: cardiac asthma, "
     "myopericarditis, silent ischaemia, pneumonia in a COPD patient, asthma in a "
     "never-smoker. Written from the clinical picture <i>before</i> the model saw them.",
     "5/5 top-1, 0 wrong"],
    ["Real patients", str(F.real_total),
     "Hand-extracted from open-access PMC case reports. Not invented, not generated, "
     "never seen by the knowledge base.",
     f"{F.real_committed_correct} correct, "
     f"<b>{F.real_committed_wrong} wrong</b>"],
], [2.5 * cm, 0.9 * cm, FULLW - 7.2 * cm, 3.8 * cm], header_bg=BLUE, small=True)]

story += [Paragraph("The required baselines, and the column that makes them fair",
                    S["h2"])]
_rows = [["System", "top-1", "top-5", "MRR", "evidence seen"]]
_rows += [list(r) for r in BASE_ROWS]
story += [table(_rows, [7.4 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm,
                        FULLW - 14.3 * cm], header_bg=BLUE, small=True)]
story += [Paragraph(
    f"Both baselines are handed the <b>complete</b> record &mdash; all {COMPLETE_EV:.1f} "
    f"findings. The loop must ask for what it wants and pay per question, and here "
    f"reaches its answer on {LOOP_EV:.1f} of them, "
    f"{LOOP_EV / COMPLETE_EV:.0%} of the evidence. <b>Read the accuracy column and the "
    "loop loses.</b> That is reported rather than smoothed away, and so is the second "
    "uncomfortable reading: a retrieval-only baseline with no inference at all lands "
    "close to the loop, which is a finding about how easy this benchmark is, not about "
    "how good the system is.", S["body"])]

story += [Paragraph("Calibration and selective prediction", S["h2"])]
story += [row([
    stat(f"{EVAL.calibration.ece:.3f}", "expected calibration error", BLUE),
    stat(f"{EVAL.calibration.brier:.3f}", "Brier score", BLUE),
    stat(f"{EVAL.selective.coverage:.0%}", "coverage (cases answered)", TEAL),
    stat(f"{EVAL.selective.aurc:.3f}", "AURC (lower is better)", TEAL),
])]
story += [Spacer(1, 0.2 * cm)]
story += [Paragraph(
    f"Mean confidence is {EVAL.calibration.mean_confidence:.1%} against "
    f"{EVAL.calibration.accuracy:.1%} accuracy &mdash; on this split the system is "
    f"<i>under</i>confident by {abs(EVAL.calibration.overconfidence):.1%}, not over. "
    f"The gate answers {EVAL.selective.coverage:.0%} of cases and is right on "
    f"{EVAL.selective.selective_accuracy:.0%} of those, against "
    f"{EVAL.selective.full_coverage_accuracy:.0%} if it were forced to answer "
    "everything. That gap is the entire claim: the value is in what it declines. It is "
    f"also {EVAL.selective.n} cases, so the calibration figures above have almost no "
    "resolution &mdash; three bins would each hold two cases.", S["small"])]

story += [Paragraph("The numbers that are not results", S["h2"])]
story += [row([
    card("20/20 on synthetic cases", [
        "Worse than uninformative. Those cases were generated <i>from the knowledge "
        "base the agent reasons over</i>, so the score measures self-consistency. "
        "Every generated id carries a <font face='Courier'>synthetic-</font> prefix so "
        "one reaching a metrics table is visible."], RED_BG, RED),
    card("Agreement with DDXPlus", [
        "Bounded by what DDXPlus is: a rule-based simulator, which a naive-Bayes model "
        "recovers partly by construction. Where a real cohort disagreed &mdash; "
        "pleuritic pain, 71% simulated against 33% observed &mdash; the disagreement "
        "is reported rather than reconciled."], RED_BG, RED),
])]
story += [PageBreak()]

# ========================================================== 7. REAL PATIENTS
story += [Paragraph(f"7 &nbsp; The {F.real_total} real patients", S["h1"]),
          Rule(FULLW, TEAL), Spacer(1, 0.3 * cm)]
story += [Paragraph(
    "The only evidence here that comes from actual people: ten in a first pass, eight "
    "more in a second under a rule fixed before any full text was read. Still far too "
    "few to support an accuracy claim, and reported this small on purpose rather than "
    "not reported at all.", S["small"])]

_rows = [["Case", "True diagnosis", "Verdict", "Why it stopped"]]
for _c, _t, _r in F.real_rows:
    _verdict, _, _why = _r.partition(";")
    _rows.append([_clip(_c, 200), _t, _clip(_verdict, 60), _clip(_why, 200)])
story += [table(_rows, [5.6 * cm, 3.2 * cm, 3.1 * cm, FULLW - 11.9 * cm],
                header_bg=TEAL, small=True)]

story += [Spacer(1, 0.25 * cm)]
story += [Paragraph(
    f"<b>{F.real_committed_wrong} wrong commits &mdash; that is the number to look "
    "at.</b> Calibrated abstention is the claim, so the count that matters is not how "
    f"many it answered ({F.real_committed_correct} of {F.real_total}) but how many it "
    "answered wrongly. For the first ten patients that number was zero, and the "
    "documents said so. The second pass took it away, and getting it back cost three "
    "separate fixes and a caveat.", S["body"])]
story += [quote(
    "A 23-year-old woman, a week of pleuritic chest pain, tachycardic, white count "
    "13.0, a &ldquo;pulmonary infiltrate&rdquo; on chest imaging, CT negative for "
    "embolism. The model committed to pneumonia at 78%. She had pericarditis, "
    "confirmed on tissue after a large pericardial effusion was drained through a "
    "surgical window. The knowledge base has no concept for a pericardial effusion, "
    "cardiomegaly or PR depression &mdash; the three findings that decide the case "
    "&mdash; so pneumonia explained everything it could see, and the gate's sixth "
    "condition, which asks whether the findings are explained by <i>something</i>, "
    "was satisfied. Section 8 describes this failure in the abstract. This was it on a "
    "real patient, and the rule was to keep whatever came.", RED)]
story += [Paragraph(
    "Nothing about the extraction is strained: &ldquo;infiltrate&rdquo; was mapped to "
    "consolidation the same way the first pass mapped &ldquo;basal infiltrate&rdquo;, "
    "and the report says what it says. The first-pass claim of zero wrong commits was "
    "a statement about ten patients whose deciding findings happened to be in the "
    "vocabulary. One patient whose deciding finding is not was enough to end it.",
    S["small"])]
story += [KeepTogether([Paragraph(
    "What it took to get that case right, and why to distrust it", S["h2"]), row([
    card("1 &nbsp; Vocabulary", [
        "Effusion and PR depression added as concepts, sourced for pericarditis, "
        "with 21 invented rival cells because a concept only one disease lists is "
        "inert. Coverage <i>fell</i> to 38%. With both findings visible, pericarditis "
        "rose from 2% to 15%. Not enough, and the loop never asked for either."],
        AMBER_BG, AMBER),
    card("2 &nbsp; A weighting defect", [
        "Findings asked for and never recorded were counting as correlated partners "
        "and discounting the ones that were observed. Fixed: real-case mean rank "
        "3.22 to 2.50, one more correct commit &mdash; and this case went to "
        "<b>93%</b> pneumonia, because the wrong answer's evidence stopped being "
        "discounted too."], BLUE_BG, BLUE),
    card("3 &nbsp; A workup the mechanism lacked", [
        "ESC 2015: ECG and echocardiography are Class I in suspected pericarditis. "
        "A third presentation-triggered workup, armed by pleuritic pain, asks for "
        "both. The case now commits correctly at 92%.",
        "<font color='#a4243b'>Written after seeing the case it fixes. It transcribes "
        "a guideline, reads the presentation and never the posterior, and moves no "
        "other set. Tested on the next pericarditis chosen by rule: it fired as "
        "designed and changed nothing &mdash; escalated at 60%, truth first, either "
        "way.</font>"],
        RED_BG, RED),
])])]

story += [KeepTogether([Paragraph(f"How the {F.real_total} were chosen", S["h2"]), row([
    card("The extraction protocol", [
        "Each case comes from an open-access PMC case report with a confirmed final "
        "diagnosis. Findings are transcribed as present, absent, or "
        "<i>not recorded</i> &mdash; the third value matters most, because it is the "
        "one a generated corpus never produces.",
        "<font color='#5d6b7a'>Only what the source states explicitly becomes a "
        "True or False. Silence stays silence, because reading it as a negative "
        "would manufacture data the report never gave.</font>"], TEAL_BG, TEAL),
    card("First pass: ten", [
        "Four picked to be genuinely hard, four for the opposite reason &mdash; "
        "clear-cut, explicit vitals &mdash; and two more for the commonest causes on "
        "epidemiology, under a rule fixed <i>before</i> any was run.",
        "<font color='#5d6b7a'>Two candidates rejected, both written down.</font>"],
        BLUE_BG, BLUE),
    card("Second pass: eight, one per diagnosis", [
        "One title-restricted PMC query per condition, candidates taken <b>in "
        "order</b>, each accepted or rejected on four stated criteria, every rejection "
        "logged with its letter. All eight were added before any touched the model.",
        "<font color='#5d6b7a'>27 rejections, mostly mimics and in-patient events "
        "that were never a presentation. Result, under the vocabulary of the time: "
        "0 correct, 1 wrong, 7 escalations with the truth ranked first in four.</font>"],
        PLUM_BG, PLUM),
])])]
story += [Spacer(1, 0.2 * cm)]
_esc = [r for _, _, r in F.real_rows if r.startswith("Escalated")]
_red = sum("not excluded" in r for r in _esc)
_budget = sum("budget" in r for r in _esc)
_stuck = len(_esc) - _red - _budget
story += [Paragraph(
    f"Of {len(_esc)} escalations, {_red} stop because a time-critical diagnosis is "
    f"still above the red-flag tolerance and {_stuck + _budget} because the top "
    "hypothesis is stuck between about 27% and 65% against a 65% floor with nothing "
    "informative left to ask. That last column used to read differently. The packet appends a "
    "note when a workup item was sought and not recorded &mdash; <i>D-dimer, not "
    "available</i> on most of these &mdash; and the engines once put the blocker "
    "last, so the note was what a reader saw first and took for the cause. It was "
    "not: pulmonary embolism sat at 2&ndash;5% on every one of them, and an earlier "
    "draft of this page, and the author, proposed a fix for a problem that was not "
    "there. The order is now fixed where the reason is made. The real blocker is a "
    "thin record meeting a knowledge base that cannot separate pneumonia, pulmonary "
    "oedema and COPD on the findings a case report happens to give.", S["body"])]

story += [Paragraph("What an escalation actually hands back", S["h2"])]
story += [Paragraph(
    f"Case <font face='Courier'>{ESC_CASE.case_id}</font>, the pericarditis. Not a "
    "shrug &mdash; a ranked differential, the reason it stopped, and the cited evidence "
    "behind each line.", S["small"])]
_rows = [["Rank", "Hypothesis", "P", "Evidence the ranking rests on"]]
for _i, _h in enumerate(ESC_OUT.differential.hypotheses[:4], 1):
    _ev = "; ".join(_clip(c.snippet, 150) for c in _h.support[:1])
    _rows.append([str(_i), _h.label.replace("_", " "), f"{_h.probability:.0%}",
                  _ev or "no positive finding in the record"])
story += [table(_rows, [1.2 * cm, 3.7 * cm, 1.0 * cm, FULLW - 5.9 * cm],
                header_bg=PLUM, small=True)]
story += [Paragraph(
    f"<b>Why it stopped:</b> <i>{ESC_OUT.escalation.reason}</i>. The true diagnosis "
    "does not lead here, which is a real miss and is counted as one &mdash; but the "
    "system hands back a ranked list and a reason instead of committing to the "
    "hypothesis at the top.", S["small"])]
story += [PageBreak()]

# ======================================================= 8. WHAT IS WRONG
story += [Paragraph("8 &nbsp; What is wrong with it", S["h1"]), Rule(FULLW, RED),
          Spacer(1, 0.3 * cm)]

story += [row([
    card("One standing red mark", [
        "P(hypoxia | acute pulmonary oedema) is an invented 0.40, below COPD and "
        "pulmonary embolism, when alveolar flooding is the mechanism that "
        "<i>defines</i> the disease.",
        "Unsourceable for a structural reason: saturation is part of how the severe "
        "presentation is identified. Trials enrol on it; the AHEAD registry uses it "
        "in the classification criterion itself. Taking that would be circular."],
        RED_BG, RED),
    card("Five D-dimer rivals understated", [
        "Three independent lines of evidence say they are about twice too low. "
        "The sources either contradict each other or need a distributional "
        "assumption no other value here depends on.",
        "<font color='#5d6b7a'>Recorded as a measured defect rather than papered "
        "over with a number nobody chose.</font>"], AMBER_BG, AMBER),
])]
story += [Spacer(1, 0.2 * cm)]
story += [row([
    card("Confident errors the gate cannot catch", [
        "Condition 6 asks whether the findings are explained by <i>something</i>. In a "
        "masquerade they are. It happened on a real patient: a pericarditis "
        "committed as pneumonia at 78%, then 93%. What finally caught it was not a "
        "threshold but a presentation-triggered workup that asks for the echo a "
        "myopic selector never would, because pericarditis sat at 4% when the loop "
        "decided. <b>The gate still cannot see a masquerade; it can only be made to "
        "look.</b>"], PLUM_BG, PLUM),
    card("Calibration is not fitted at all", [
        "The scaler refuses below 30 labelled cases, correctly. The calibration split "
        "is three, so the temperature is the untouched default.",
        "<font color='#5d6b7a'>The report used to print &ldquo;temperature "
        "(fitted) 1.00&rdquo;, which reads as a fit that found the posterior already "
        "calibrated. It now says no fit happened.</font>"], BLUE_BG, BLUE),
])]

story += [Paragraph("The failing comparisons, printed rather than described", S["h2"])]
_rows = [["Finding", "Should lead", "P", "But this rival has", "P"]]
for _concept, _leader, _rival, _top, _other in ORDER_FAILS:
    _rows.append([_concept, _leader.replace("_", " "), f"{_top:.2f}",
                  _rival.replace("_", " "), f"{_other:.2f}"])
story += [table(_rows, [4.2 * cm, 4.0 * cm, 1.1 * cm, 4.0 * cm,
                        FULLW - 13.3 * cm], header_bg=RED, small=True)]
story += [Paragraph(
    f"That is the whole list &mdash; {len(ORDER_FAILS)} of {ORDER_PAIRS} comparisons. "
    "The audit script exits non-zero on any failure that is not on the tolerated list, "
    "so a new inversion cannot enter quietly.", S["small"])]

story += [Paragraph("Not delivered, and it will not be from here", S["h2"])]
story += [table([
    ["Deliverable", "Why not"],
    ["Clinician-graded evidence chains", "<b>No collaborating clinician was available.</b> "
     "Published decision rules substitute for which findings support which cause, and "
     "likelihood ratios for strength. Grading the chain has <b>no substitute and is "
     "not delivered</b>."],
    ["The mandated model, run live", "No API credit for the project's duration. The LLM "
     "layer has been a swappable Protocol since week 1 for exactly this reason, and "
     "that default has never been called."],
    ["AgentClinic as the harness", "Needs its own API key plus a fork to accept this "
     "agent in place of its own. Not a configuration change, and not attempted."],
    ["UMLS relations", "Probed and found unusable: overwhelmingly translations, ICD "
     "crosswalks and MedDRA groupings, with one clinical label that mixes symptoms, "
     "risk factors and treatment complications without marking which is which."],
], [5.4 * cm, FULLW - 5.4 * cm], header_bg=RED, small=True)]

story += [Paragraph("What would have to happen before any of this is usable", S["h2"])]
story += [Paragraph(
    "In order, and none of it is engineering: a clinician reads the "
    f"{ORDER_CLAIMS} ordering claims and rejects the ones that are wrong; the remaining "
    f"{F.invented} invented likelihoods are sourced or removed; thirty or more labelled "
    "cases are assembled so the temperature can actually be fitted; and the real-case "
    "set grows past the point where a two-case swing is noise. The code is ready for "
    "all four. None of them is a code change.", S["body"])]
story += [PageBreak()]

# ============================================================= 9. CLOSING
story += [Paragraph("9 &nbsp; What this is, in one page", S["h1"]), Rule(FULLW, NAVY),
          Spacer(1, 0.3 * cm)]

story += [Paragraph(
    "A working research prototype whose engineering is complete and whose clinical "
    "content is not validated. That is the honest one-line description and everything "
    "above is an elaboration of it.", S["body"])]

story += [Paragraph("What would survive a hostile read", S["h2"])]
story += [row([
    card("It knows what it does not know", [
        f"{F.invented} of {F.total} likelihoods invented, {PARAMS.total} further "
        "parameters invented, reported on every run, with a regression test that "
        "fails when the documents drift from the live figure."], GREEN_BG, GREEN),
    card("It refuses rather than guesses", [
        f"{F.real_committed_wrong} wrong commits across {F.real_total} real patients "
        f"and 5 deliberately hard ones, against "
        f"{F.real_total - F.real_committed_correct - F.real_committed_wrong} escalations "
        "that each name what was sought and not obtained. The one wrong commit there "
        "was is written up, with the three fixes it took and why to distrust the "
        "last of them."],
        GREEN_BG, GREEN),
])]
story += [Spacer(1, 0.2 * cm)]
story += [row([
    card("It records the conclusions it took back", [
        "A correlation measurement that reversed eight times before anyone recognised "
        "the instrument was spent &mdash; and the one that never had, three wrong "
        "commits against none on real patients, which a pneumonia sourcing pass took "
        "to none against none. A published &ldquo;finding&rdquo; that was a mapping "
        "bug. A claim of zero wrong commits that lasted until the eighteenth patient."],
        PLUM_BG, PLUM),
    card("It checks its own inputs", [
        f"{ORDER_CLAIMS} ordering claims over {ORDER_PAIRS} comparisons, "
        f"{len(PC.CLAIMS)} direction claims from the scope document, "
        f"{len(PC.QUARTILE_BOUNDS)} distribution-free bounds from cohort quartiles, and "
        f"{len(OPERATIONAL_DEFINITIONS)} concepts given the operational threshold "
        "that makes sourcing possible at all."], TEAL_BG, TEAL),
])]

story += [Spacer(1, 0.3 * cm)]
story += [quote(
    "Three separate times, sourcing a number overturned something the author believed; "
    "twice it made a result worse, and those results were kept and written down. A "
    "system that cannot tell you which of its beliefs are invented cannot be "
    "corrected, and a project that only records the corrections that flattered it has "
    "not really recorded anything.", NAVY)]

story += [Spacer(1, 0.35 * cm)]
story += [Paragraph("Running it", S["h2"])]
story += [Paragraph(
    "All of it runs from a clean checkout; the scripts as "
    "<font face='Courier'>python3 scripts/&lt;name&gt;</font>.", S["small"])]
story += [table([
    ["Command", "What it gives you"],
    ["pytest tests/ -q", f"{F.tests} tests, including the two-engine equivalence check "
                         "and the documentation drift guard"],
    ["run_eval.py", "The full metrics report and the two required baselines, with the "
                    "evidence column"],
    ["eval_real_cases.py", f"The {F.real_total} real patients, one verdict at a time, "
                           "with the reason for each escalation"],
    ["plausibility_check.py", f"The knowledge-base audits: {ORDER_CLAIMS} ordering "
                              f"claims over {ORDER_PAIRS} comparisons, and "
                              f"{len(PC.CLAIMS)} direction claims from the scope document"],
    ["consult.py", "An interactive consultation in the terminal, one question at a time"],
    ["webui/server.py", "The same loop on a web page, with a live differential and the "
                        "provenance of every cited entry"],
], [5.0 * cm, FULLW - 5.0 * cm], header_bg=NAVY, small=True)]
story += [Paragraph(
    "No install, no API key, fully offline and deterministic.", S["small"])]

story += [Spacer(1, 0.5 * cm)]
story += [row([card("A closing caution", [
    "Nothing here has been reviewed by a clinician, most of the knowledge base is "
    f"invented, and the strongest evidence in the project is {F.real_total} patients. "
    "<b>This is not a diagnostic tool and must not be used as one.</b> What it is, is "
    "a system that will tell you exactly which parts of itself you should not trust "
    "&mdash; and that is the part worth keeping."], RED_BG, RED)])]


# An explicit output path lets the test suite render into a temporary
# directory without overwriting the committed PDF on every run.
OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "dxagent_deep_dive.pdf")


def on_page(c, d):
    c.saveState()
    c.setStrokeColor(LINE)
    c.setLineWidth(0.7)
    c.line(MARGIN, 1.15 * cm, PAGE_W - MARGIN, 1.15 * cm)
    c.setFont("Helvetica", 7.6)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 0.8 * cm,
                 f"dxagent — not for clinical use; {F.footer_claim}")
    c.drawRightString(PAGE_W - MARGIN, 0.8 * cm, f"page {d.page}")
    c.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=LETTER,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=1.5 * cm, bottomMargin=1.6 * cm,
    title="dxagent deep dive", author="Mohamed Aziz Ben Rejeb",
)
doc.build(story, onFirstPage=cover_bg, onLaterPages=on_page)
print("wrote", OUT)
