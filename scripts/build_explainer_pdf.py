# -*- coding: utf-8 -*-
"""dxagent explainer PDF: what was built, current as of tonight's fixes."""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import cm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_figures import collect  # noqa: E402

F = collect()
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.platypus.flowables import Flowable

INK      = colors.HexColor("#16202b")
MUTED    = colors.HexColor("#5d6b7a")
LINE     = colors.HexColor("#d8dee5")
PANEL    = colors.white
BG_SOFT  = colors.HexColor("#f2f5f8")

TEAL     = colors.HexColor("#0e7c7b")
TEAL_BG  = colors.HexColor("#d9f0ef")
BLUE     = colors.HexColor("#1d4e8f")
BLUE_BG  = colors.HexColor("#dbe6f5")
AMBER    = colors.HexColor("#b06b00")
AMBER_BG = colors.HexColor("#fbeacd")
RED      = colors.HexColor("#a92929")
RED_BG   = colors.HexColor("#f8dcdb")
PLUM     = colors.HexColor("#6b3fa0")
PLUM_BG  = colors.HexColor("#e8dff5")
GREEN    = colors.HexColor("#1f7a3d")
GREEN_BG = colors.HexColor("#dcf0e2")

PAGE_W, PAGE_H = LETTER
FULLW = PAGE_W - 3.4 * cm

S = {
 "h1":   ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16.5, leading=20,
                        textColor=INK, spaceAfter=7),
 "h2":   ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                        textColor=INK, spaceBefore=9, spaceAfter=5),
 "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=14,
                        textColor=INK, spaceAfter=6),
 "small":ParagraphStyle("small", fontName="Helvetica", fontSize=8.6, leading=12.2,
                        textColor=MUTED, spaceAfter=4),
 "card": ParagraphStyle("card", fontName="Helvetica", fontSize=8.9, leading=12.6,
                        textColor=INK, spaceAfter=2),
 "cardt":ParagraphStyle("cardt", fontName="Helvetica-Bold", fontSize=10, leading=13,
                        spaceAfter=3),
 "big":  ParagraphStyle("big", fontName="Helvetica-Bold", fontSize=19, leading=22,
                        alignment=TA_CENTER),
 "biglb":ParagraphStyle("biglb", fontName="Helvetica", fontSize=7.4, leading=9.6,
                        textColor=MUTED, alignment=TA_CENTER),
 "cover":ParagraphStyle("cover", fontName="Helvetica-Bold", fontSize=29, leading=34,
                        textColor=colors.white, alignment=TA_CENTER),
 "csub": ParagraphStyle("csub", fontName="Helvetica", fontSize=12.5, leading=17,
                        textColor=colors.HexColor("#a9c4e4"), alignment=TA_CENTER),
}


class Rule(Flowable):
    def __init__(self, w, color, t=3.5):
        Flowable.__init__(self); self.width=w; self.color=color; self.t=t; self.height=t
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.t, self.t/2, stroke=0, fill=1)


def card(title, lines, fill, edge):
    body = [Paragraph(title, ParagraphStyle("t", parent=S["cardt"], textColor=edge))]
    body += [Paragraph(l, S["card"]) for l in lines]
    t = Table([[body]])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), fill),
        ("LINEBEFORE", (0,0), (0,-1), 3, edge),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
        ("LEFTPADDING", (0,0), (-1,-1), 9), ("RIGHTPADDING", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    return t


def row(cards, gap=0.28*cm):
    n = len(cards); w = (FULLW - gap*(n-1)) / n
    t = Table([cards], colWidths=[w]*n)
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                           ("LEFTPADDING",(0,0),(-1,-1),0),
                           ("RIGHTPADDING",(0,0),(-1,-1),gap),
                           ("TOPPADDING",(0,0),(-1,-1),0),
                           ("BOTTOMPADDING",(0,0),(-1,-1),0)]))
    return t


def stat(value, label, color):
    inner = [Paragraph(value, ParagraphStyle("v", parent=S["big"], textColor=color)),
             Paragraph(label, S["biglb"])]
    t = Table([[inner]])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), BG_SOFT),
        ("BOX",(0,0),(-1,-1),0.6,LINE),
        ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
    ]))
    return t


def table(data, widths, header_bg=INK, small=False):
    size = 8.0 if small else 8.6
    head_style = ParagraphStyle("thead", fontName="Helvetica-Bold", fontSize=size,
                                leading=size + 2.8, textColor=colors.white)
    body_style = ParagraphStyle("tbody", fontName="Helvetica", fontSize=size,
                                leading=size + 2.8, textColor=INK)
    wrapped = []
    for r, row_ in enumerate(data):
        style = head_style if r == 0 else body_style
        wrapped.append([c if not isinstance(c, str) else Paragraph(c, style) for c in row_])
    t = Table(wrapped, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND",(0,0),(-1,0), header_bg),
        ("GRID",(0,0),(-1,-1),0.5, LINE),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),4.5),("BOTTOMPADDING",(0,0),(-1,-1),4.5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[PANEL, BG_SOFT]),
    ]
    t.setStyle(TableStyle(cmds))
    return t


story = []

# =============================================================== COVER =====
def cover_bg(c, d):
    c.saveState()
    c.setFillColor(colors.HexColor("#12263c")); c.rect(0,0,PAGE_W,PAGE_H,stroke=0,fill=1)
    for i, col in enumerate([TEAL, BLUE, PLUM, AMBER, GREEN, RED]):
        c.setFillColor(col); c.rect(i*PAGE_W/6, 0, PAGE_W/6, 0.42*cm, stroke=0, fill=1)
    c.restoreState()

story += [Spacer(1, 5.4*cm),
          Paragraph("dxagent", S["cover"]),
          Spacer(1, 0.3*cm),
          Paragraph("What I built", S["csub"]),
          Spacer(1, 0.15*cm),
          Paragraph("Agentic diagnostic reasoning for dyspnoea and chest pain",
                    ParagraphStyle("c3", parent=S["csub"], fontSize=10,
                                   textColor=colors.HexColor("#7f9dc4"))),
          Spacer(1, 1.1*cm),
          Paragraph("DeepShift AI Summer Internship 2026 &middot; Project 1 (Healthcare) &middot; Mohamed Aziz Ben Rejeb",
                    ParagraphStyle("c4", parent=S["csub"], fontSize=8.6,
                                   textColor=colors.HexColor("#5c7799")))]
story.append(PageBreak())

# ========================================================== 1. WHAT IT IS ==
story += [Paragraph("What it does", S["h1"]),
          Paragraph(
    "A patient presents with breathlessness or chest pain. Eight conditions could explain it, "
    "three of them time-critical: pulmonary embolism, acute coronary syndrome, acute pulmonary "
    "oedema. The agent asks one useful question at a time, updates its belief about all eight "
    "diagnoses after every answer, and either commits to a diagnosis with cited evidence or "
    "abstains and names exactly what's still unresolved. Not a chatbot &mdash; there's no "
    "free-text interface. It's a decision loop that redraws itself after every answer, closer "
    "to how a doctor actually works a case than to a single LLM prompt.", S["body"]),
          Spacer(1, 0.2*cm)]

story += [row([stat(str(F.tests), "tests in the suite", TEAL),
               stat(f"{F.fixtures_correct}/{F.fixtures_total}", "fixture cases correct", GREEN),
               stat(f"{F.coverage}%", "likelihoods sourced", AMBER),
               stat(str(F.real_total), "real PMC cases evaluated", BLUE)]),
          Spacer(1, 0.35*cm)]

story += [Paragraph("The reasoning loop", S["h2"])]
steps = [
    ("1", "Propose", TEAL, TEAL_BG, "Naive-Bayes over 8 diseases, correlation-weighted so one clinical picture doesn't get counted as several pieces of evidence."),
    ("2", "Calibrate", BLUE, BLUE_BG, "Temperature scaling / conformal prediction turn a raw score into a trustworthy confidence."),
    ("3", "Check the gate", PLUM, PLUM_BG, "Six conditions must all hold before it's allowed to commit &mdash; confidence, margin, grounding, a red-flag rule, proposer agreement, evidence fit."),
    ("4", "Choose the next step", AMBER, AMBER_BG, "Highest information gain per unit cost, with mandatory workups (D-dimer, troponin+ECG) that fire on the presentation regardless of current belief."),
    ("5", "Decide", GREEN, GREEN_BG, "Commit with citations, or escalate naming the unresolved question."),
]
for num, title, edge, fill, text in steps:
    t = Table([[Paragraph(num, ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=13,
                                              textColor=colors.white, alignment=TA_CENTER)),
                [Paragraph(title, ParagraphStyle("st", parent=S["cardt"], textColor=edge)),
                 Paragraph(text, S["card"])]]],
               colWidths=[0.9*cm, FULLW-0.9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0), edge), ("BACKGROUND",(1,0),(1,0), fill),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("ALIGN",(0,0),(0,0),"CENTER"),
        ("BOX",(0,0),(-1,-1),0.6,LINE),
        ("LEFTPADDING",(1,0),(1,0),9),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story += [t, Spacer(1, 0.12*cm)]

story.append(PageBreak())

# ====================================================== 2. WHAT'S BUILT ====
story += [Paragraph("What's built and working", S["h1"]),
          Paragraph("Every row is tested code, not a plan.", S["small"]),
          Spacer(1, 0.2*cm)]

built = [
    ["Component", "What it does"],
    ["Bayesian reasoner", "Naive-Bayes in log-space over 8 diseases x 27 findings, correlation-weighted"],
    ["Information-gain selector", "Picks the next question/test by expected entropy reduction per cost"],
    ["Abstention gate", "Six-condition commit/escalate decision; calibration + conformal prediction"],
    ["LangGraph engine", "Same reasoning loop as a state machine, proven equivalent to the reference loop by test"],
    ["RAG retrieval", "BGE-M3 + Qdrant over 46 passages: 32 from 4 decision rules, 14 from the Merck Manual"],
    ["Clinical ontology", "8 causes, 27 findings, 84 labelled edges, derived from the KB and provably consistent with it"],
    ["UMLS/SNOMED concept layer", "Findings and diagnoses resolved to standard clinical vocabularies"],
    ["Provenance tracking", "Every likelihood and prior tagged measured / narrative / invented, reported every run"],
    ["Synthetic case pipeline", "Generate, self-critique, near-miss targeting, PHI screen, duplicate detection"],
    ["PHI screening, two layers", "Regex over structured identifiers, plus a spaCy NER pass that reads the narrative itself"],
    ["Evaluation harness", "Both required baselines, ECE/Brier, risk-coverage, DDx recall, abstention proxy"],
    ["Live interactive consult", "scripts/consult.py &mdash; a real person plays the patient, y/n/u, turn by turn"],
    ["Guideline workups", "Wells/PERC/CURB-65/HEART, plus mandatory + confirmatory tests that bypass a wrong posterior"],
]
story += [table(built, [4.6*cm, 12.9*cm], small=True)]

story.append(PageBreak())

# ==================================================== 3. THE HONEST NUMBER =
story += [Paragraph("The knowledge base, honestly", S["h1"]),
          Paragraph(
    "The single most important fact about this project. Every number is tagged by where it "
    "came from, and reported on every run &mdash; nothing is hidden or smoothed over.", S["body"]),
          Spacer(1, 0.2*cm)]

story += [row([
    card(F.invented_of_total, [
        "Both bulk routes are <b>proven</b> exhausted, not just untried: DDXPlus by running its "
        "own extractor over the full release, Merck by two complete passes over all 8 chapters.",
        "<font color='#5d6b7a'>Deleting them drops fixture accuracy 10/10 &rarr; 1/10. "
        "Individually insensitive, collectively load-bearing.</font>"],
        AMBER_BG, AMBER),
    card(f"{F.priors_sourced}/{F.priors_total} disease priors sourced", [
        "Derived from StatPearls presentation-conditional aetiology (of patients presenting this "
        "way, what fraction have each cause) &mdash; the right quantity, found only after an "
        "earlier search targeted the wrong one.",
        "<font color='#5d6b7a'>Bands span a real 20-fold disagreement between sources on ACS.</font>"],
        GREEN_BG, GREEN),
])]

story += [Spacer(1, 0.25*cm), Paragraph("Sources, by strength", S["h2"])]
srcs = [
    ["Source", "Entries", "What it is"],
    ["DDXPlus", str(F.ddxplus), "Frequencies counted directly in ~1M simulated patients &mdash; a generating model, not real people"],
    ["Merck Manual 19e", str(F.merck), "Narrative phrases converted through a fixed, pre-committed rubric"],
    ["Published cohorts / StatPearls", str(F.cohorts), "Miniati 2012 (PE), Z\u00e8gre-Hemsey 2018 (ACS), StatPearls friction rub and ECG changes (pericarditis)"],
]
story += [table(srcs, [4.4*cm, 1.7*cm, 11.4*cm], header_bg=TEAL, small=True)]

story += [Spacer(1, 0.2*cm),
          Paragraph(
    "The most recent addition: pericarditis' ECG-change likelihood was invented at 0.60. Traced "
    "to a specific real-case failure, then sourced from StatPearls (\"more than half of patients "
    "... exhibit characteristic ECG changes\") at 0.50 &mdash; lower than the invented value, "
    "kept anyway, because the question was whether the number is real, not whether it's convenient.",
    S["small"])]

story.append(PageBreak())

# ================================================ 4. REAL PATIENTS =========
story += [Paragraph("Tested against real patients", S["h1"]),
          Paragraph(
    f"{F.real_total} cases hand-extracted from open-access PMC case reports &mdash; not invented, "
    "not generated, never seen by the knowledge base. A further candidate (aortic dissection) was "
    "found and rejected rather than mislabelled, because its true diagnosis falls outside "
    "the 8 modelled causes.", S["body"]),
          Spacer(1, 0.2*cm)]


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "&hellip;"


realcases = [["Case", "True diagnosis", "Result"]]
realcases += [[_clip(c, 58), t, _clip(r, 62)] for c, t, r in F.real_rows]
story += [table(realcases, [6.9*cm, 3.9*cm, 6.7*cm], header_bg=BLUE, small=True)]

story += [Spacer(1, 0.25*cm),
          Paragraph(
    f"{F.real_committed_wrong} wrong commit{'' if F.real_committed_wrong == 1 else 's'} "
    "&mdash; that is the number to look at", S["h2"]),
          Paragraph(
    "Calibrated abstention is this project's claim, so the count that matters is not how many "
    f"cases it answered ({F.real_committed_correct} of {F.real_total}) but how many it answered "
    "wrongly. For the first ten patients that number was zero. The second pass of eight, chosen "
    "under a rule fixed before any full text was read, produced one: a pericarditis committed as "
    "pneumonia at 78%, because the findings that decide the case &mdash; a large pericardial "
    "effusion, cardiomegaly, PR depression &mdash; have no concept in this vocabulary, and "
    "pneumonia explained everything the model could see. It is the confident error the gate "
    "was always documented as unable to catch, now on a real patient, and it is kept.",
    S["body"])]

story += [Spacer(1, 0.15*cm),
          Paragraph(
    f"No threshold was tuned to make these {F.real_total} cases pass. That would fit the knowledge "
    f"base to its own n={F.real_total} benchmark rather than to the literature &mdash; the opposite "
    "of what provenance tracking exists to prevent.", S["small"])]

story += [Spacer(1, 0.22*cm), Paragraph("One structural gap, found and closed &mdash; and one found and left alone", S["h2"]),
          Paragraph(
    "Traced why the PE case never reaches its decisive CTPA: the guideline layer mandates "
    "ordering a D-dimer once PE is a live concern, but never escalated a <i>positive</i> result "
    "to the confirmatory scan. Fixed generally &mdash; any future patient with a positive "
    "D-dimer and PE unexcluded now has CTPA marked mandatory, not case-specific tuning.", S["card"]),
          Paragraph(
    "It didn't flip this case anyway, and the reason why is itself a finding: the rule-out "
    "mechanism scores candidate tests as discrimination &divide; (cost + 1.5), and at the turn "
    "in question that formula ranked a weak, cheap exam finding (JVP: 0.27 &divide; 2.5 = 0.11) "
    "above the far more decisive but expensive CTPA (&asymp;0.9 &divide; 21.5 = 0.04). That "
    "scoring formula is shared with the ordinary selector and has a documented prior regression "
    "&mdash; not touched tonight, reported as an open structural finding instead.", S["card"])]

story.append(PageBreak())

# ==================================================== 5. SAFETY & HONESTY ==
story += [Paragraph("Safety and honesty, by construction", S["h1"]),
          Paragraph("Not a bolt-on policy &mdash; these are architectural properties, checked by tests.", S["body"]),
          Spacer(1, 0.2*cm)]

story += [row([
    card("No model authors a citation", [
        "Every citation is built by string-formatting on a passage that was actually retrieved "
        "&mdash; the retrieval index and the likelihood table share the same source object, not "
        "an independently maintained copy.",
        "<font color='#5d6b7a'>Confirmed by code inspection, not assumed.</font>"],
        TEAL_BG, TEAL),
    card("PHI screening, two independent layers", [
        "Regex catches structured identifiers (MRNs, phone numbers). A spaCy NER pass reads free "
        "narrative text and catches a name the regex is documented to miss.",
        "<font color='#5d6b7a'>Verified live against the exact case the old screen failed on.</font>"],
        GREEN_BG, GREEN),
])]
story += [Spacer(1, 0.22*cm)]
story += [row([
    card("Two independent implementations of the loop", [
        "A hand-written reference loop and a LangGraph state machine implement the same "
        "reasoning &mdash; proven equivalent by test, not merely intended to be."],
        BLUE_BG, BLUE),
    card("Six-condition abstention gate", [
        "Confidence, margin, grounding, a stricter red-flag tolerance for PE / ACS / pulmonary "
        "oedema, proposer disagreement, and evidence-fit all have to hold before the system "
        "commits to anything."],
        PLUM_BG, PLUM),
])]

story += [Spacer(1, 0.3*cm), Rule(FULLW, TEAL), Spacer(1, 0.25*cm)]
story += [Paragraph(f"{F.tests} tests on a clean checkout", S["h2"]),
          Paragraph(
    "No install, no API key, fully offline and deterministic, under 20 seconds. A separate "
    "regression test checks that every sourcing figure quoted in the documentation still "
    "matches a live run, and it reads the denominator from the knowledge base rather than "
    "hardcoding it &mdash; an earlier version pinned the number 135, and when sourcing moved "
    "the total past it the guard silently stopped matching anything. The numbers on these "
    "pages are now computed at build time for the same reason.",
    S["body"])]

story.append(PageBreak())

# ================================================ 6. WHAT'S STILL OPEN =====
story += [Paragraph("What's still open", S["h1"]),
          Paragraph(
    "Checked against the actual specification document, not just against scope notes. "
    "Named as open, not hidden.", S["body"]),
          Spacer(1, 0.2*cm)]

story += [row([
    card("The mandated model hasn't run live", [
        "No API credit available all project. The LLM layer has been swappable since week 1 "
        "&mdash; the default now points at the current Opus-tier model as the honest equivalent "
        "of what the brief asked for, but it is unverified live."],
        AMBER_BG, AMBER),
    card("AgentClinic", [
        "Named as the evaluation tool in the mandated stack. Genuinely blocked, not untried: it "
        "needs an OpenAI/Replicate key plus forking its code to accept this agent instead."],
        AMBER_BG, AMBER),
])]

story += [Spacer(1, 0.3*cm), Paragraph("The constraint behind almost everything else", S["h2"]),
          Paragraph(
    "The brief's own clinical-partnership table calls for weekly meetings with a collaborating "
    "physician, with their feedback \"part of your evaluation.\" No clinician was available. "
    "That single fact is why every real patient in this evaluation came from a published case "
    "report rather than a clinician-reviewed one, and why the knowledge base still leans on "
    "DDXPlus and Merck rather than expert-elicited frequencies.", S["body"])]

story += [Spacer(1, 0.25*cm),
          Paragraph(
    "<b>4 real patients is a genuinely small sample</b>, picked to be reasonably clear-cut where "
    "possible, not representative of anything. It adds a few real data points to an evaluation "
    "that would otherwise be entirely invented &mdash; not evidence of real-world accuracy in "
    "either direction.", S["small"])]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "dxagent_what_i_built.pdf")
doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=1.7*cm, rightMargin=1.7*cm,
                        topMargin=1.9*cm, bottomMargin=1.6*cm,
                        title="dxagent — What I Built", author="Mohamed Aziz Ben Rejeb")

HEAD = {
 2: ("What it does", "the reasoning loop, in five steps", INK),
 3: ("What's built", "every row is tested code", TEAL),
 4: ("The knowledge base", "honestly, with the numbers", AMBER),
 5: ("Real patients",
     f"{F.real_total} PMC cases, {F.real_committed_wrong} wrong commits, named reasons",
     BLUE),
 6: ("Safety & honesty", "by construction, checked by tests", GREEN),
 7: ("What's still open", "named, not hidden", PLUM),
}

def on_page(c, d):
    if d.page == 1:
        cover_bg(c, d); return
    info = HEAD.get(d.page)
    if info:
        c.saveState()
        c.setFillColor(info[2]); c.rect(0, PAGE_H-1.55*cm, PAGE_W, 1.55*cm, stroke=0, fill=1)
        c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 12)
        c.drawString(1.7*cm, PAGE_H-0.95*cm, info[0])
        c.setFont("Helvetica", 8.4); c.setFillColor(colors.HexColor("#cddae8"))
        c.drawRightString(PAGE_W-1.7*cm, PAGE_H-0.95*cm, info[1])
        c.restoreState()
    c.saveState()
    c.setStrokeColor(LINE); c.setLineWidth(0.7)
    c.line(1.7*cm, 1.15*cm, PAGE_W-1.7*cm, 1.15*cm)
    c.setFont("Helvetica", 7.6); c.setFillColor(MUTED)
    c.drawString(1.7*cm, 0.8*cm, f"dxagent — not for clinical use; {F.footer_claim}")
    c.drawRightString(PAGE_W-1.7*cm, 0.8*cm, f"page {d.page}")
    c.restoreState()

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
