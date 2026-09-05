# -*- coding: utf-8 -*-
"""Slide-shaped PDF version of the dxagent presentation (16:9, canvas-drawn).

No LibreOffice is available on this machine to convert the .pptx directly, so
this mirrors the same 11 slides' content and layout using reportlab canvas
calls at the same 13.333in x 7.5in aspect ratio -- a parallel build, not a
conversion, kept content-identical to the .pptx by construction.
"""
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
import textwrap
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_figures import collect  # noqa: E402

F = collect()

PW, PH = 13.333 * inch, 7.5 * inch
OUT = r"C:\Users\azizb\Desktop\internal_medicine\dxagent_presentation.pdf"

NAVY   = HexColor("#12263C")
TEAL   = HexColor("#0E7C7B"); TEALBG = HexColor("#E4F3F2")
BLUE   = HexColor("#1D4E8F"); BLUEBG = HexColor("#DBE6F5")
AMBER  = HexColor("#B06B00"); AMBERBG= HexColor("#FBEACD")
GREEN  = HexColor("#1F7A3D"); GREENBG= HexColor("#DCF0E2")
PLUM   = HexColor("#6B3FA0"); PLUMBG = HexColor("#E8DFF5")
RED    = HexColor("#A92929"); REDBG  = HexColor("#F8DCDB")
INK    = HexColor("#16202B")
MUTED  = HexColor("#5D6B7A")
LINE   = HexColor("#D8DEE5")
WHITE  = HexColor("#FFFFFF")
SOFT   = HexColor("#F2F5F8")
SKY    = HexColor("#A9C4E4")
SKY2   = HexColor("#7F9DC4")
SKY3   = HexColor("#5C7799")

MX = 0.7 * inch
CW = PW - 2 * MX

c = canvas.Canvas(OUT, pagesize=(PW, PH))
c.setTitle("dxagent — Validation Presentation")


def wrap_text(text, font, size, max_w):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_para(x, y_top, w, text, font="Helvetica", size=11, color=INK, leading=None, align="left"):
    leading = leading or size * 1.25
    lines = wrap_text(text, font, size, w)
    c.setFont(font, size)
    c.setFillColor(color)
    y = y_top
    for ln in lines:
        if align == "center":
            c.drawCentredString(x + w / 2, y, ln)
        else:
            c.drawString(x, y, ln)
        y -= leading
    return y  # y after last line (top of next content)


def rrect(x, y, w, h, r, fill):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, r, stroke=0, fill=1)


def card(x, y_top, w, h, title, paras, fill, edge):
    y = y_top - h
    rrect(x, y, w, h, 5, fill)
    c.setFillColor(edge)
    c.rect(x, y, 4, h, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(edge)
    c.drawString(x + 16, y_top - 24, title)
    ty = y_top - 46
    for p in paras:
        ty = draw_para(x + 16, ty, w - 32, p, "Helvetica", 10.5, INK, 14) - 5
    return y


def stat_card(x, y_top, w, h, value, label, color):
    y = y_top - h
    rrect(x, y, w, h, 5, SOFT)
    c.setStrokeColor(LINE); c.setLineWidth(0.75)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(color)
    c.drawCentredString(x + w / 2, y_top - h + 34, value)
    c.setFont("Helvetica", 9.5)
    c.setFillColor(MUTED)
    c.drawCentredString(x + w / 2, y_top - h + 14, label)


def circle_num(cx, cy, d, num, color, fs=15):
    c.setFillColor(color)
    c.circle(cx, cy, d / 2, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", fs)
    c.setFillColor(WHITE)
    c.drawCentredString(cx, cy - fs * 0.35, str(num))


def kicker_title(kicker, title):
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(TEAL)
    c.drawString(MX, PH - 0.7 * inch, kicker.upper())
    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(INK)
    c.drawString(MX, PH - 1.05 * inch, title)


def light_bg():
    c.setFillColor(WHITE)
    c.rect(0, 0, PW, PH, stroke=0, fill=1)


def dark_bg():
    c.setFillColor(NAVY)
    c.rect(0, 0, PW, PH, stroke=0, fill=1)


def simple_table(x, y_top, col_w, rows, header_bg=TEAL, row_h=0.42 * inch, fs=9.6):
    y = y_top
    for ri, row in enumerate(rows):
        rh = row_h
        bg = header_bg if ri == 0 else (SOFT if ri % 2 else WHITE)
        xx = x
        for ci, cell in enumerate(row):
            w = col_w[ci]
            c.setFillColor(bg)
            c.rect(xx, y - rh, w, rh, stroke=0, fill=1)
            c.setStrokeColor(LINE); c.setLineWidth(0.5)
            c.rect(xx, y - rh, w, rh, stroke=1, fill=0)
            fg = WHITE if ri == 0 else INK
            fnt = "Helvetica-Bold" if ri == 0 else "Helvetica"
            lines = wrap_text(cell, fnt, fs, w - 14)[:3]
            c.setFont(fnt, fs)
            c.setFillColor(fg)
            ty = y - rh / 2 + (len(lines) - 1) * (fs + 1.5) / 2 + (fs * 0.32)
            for ln in lines:
                c.drawString(xx + 7, ty, ln)
                ty -= fs + 1.5
            xx += w
        y -= rh
    return y


# ============================================================ 1. COVER =====
dark_bg()
c.setFont("Helvetica-Bold", 54)
c.setFillColor(WHITE)
c.drawCentredString(PW / 2, PH - 3.0 * inch, "dxagent")
c.setFont("Helvetica", 17)
c.setFillColor(SKY)
c.drawCentredString(PW / 2, PH - 3.55 * inch, "Agentic diagnostic reasoning for dyspnoea and chest pain")
c.setFont("Helvetica-Oblique", 12.5)
c.setFillColor(SKY2)
c.drawCentredString(PW / 2, PH - 3.95 * inch, "Validation meeting \u2014 what's built, what's honest, what's still open")
c.setFont("Helvetica", 10.5)
c.setFillColor(SKY3)
c.drawCentredString(PW / 2, 0.75 * inch,
    "DeepShift AI Summer Internship 2026  \u00b7  Project 1 (Healthcare)  \u00b7  Mohamed Aziz Ben Rejeb")
cols = [TEAL, BLUE, PLUM, AMBER, GREEN, RED]
for i, col in enumerate(cols):
    c.setFillColor(col)
    c.circle(1.0 * inch + i * 0.5 * inch, PH - 1.15 * inch, 0.14 * inch, stroke=0, fill=1)
c.showPage()

# ======================================================= 2. THE PROBLEM ====
light_bg()
kicker_title("The problem", "Why a differential needs more than an LLM prompt")
items = [
    ("No provenance", "A single-prompt diagnosis carries no verifiable source for any claim it makes.", REDBG, RED),
    ("No calibration", "A rank is not a confidence \u2014 the model can be as sure about a guess as about a certainty.", AMBERBG, AMBER),
    ("Always answers", "It responds even when the evidence genuinely cannot decide \u2014 the failure mode that matters most in medicine.", BLUEBG, BLUE),
]
w3 = (CW - 0.5 * inch) / 3
for i, (t, body, fill, edge) in enumerate(items):
    card(MX + i * (w3 + 0.25 * inch), PH - 1.7 * inch, w3, 2.3 * inch, t, [body], fill, edge)
rrect(MX, PH - 5.9 * inch, CW, 1.35 * inch, 6, NAVY)
c.setFont("Helvetica-BoldOblique", 17)
c.setFillColor(WHITE)
lines = wrap_text("A system that doesn't know when to stay silent is more dangerous than one that's sometimes wrong.",
                   "Helvetica-BoldOblique", 17, CW - 1.0 * inch)
ty = PH - 5.9 * inch + 1.35 * inch / 2 + (len(lines) - 1) * 11
for ln in lines:
    c.drawCentredString(PW / 2, ty, ln)
    ty -= 22
c.showPage()

# ================================================== 3. WHAT IT DOES ========
light_bg()
kicker_title("What it does", "From patient findings to a cited, calibrated differential")
steps3 = [
    "Eight modelled diagnoses across dyspnoea and chest pain \u2014 three of them time-critical",
    "One useful question or test at a time, chosen by expected information gain per unit cost",
    "The belief over all eight diagnoses updates after every single answer",
    "Every hypothesis carries citations back to a real, retrieved source \u2014 never authored by a model",
    "Either commits with cited evidence, or escalates naming exactly what's unresolved",
]
cols5 = [TEAL, BLUE, PLUM, AMBER, GREEN]
y = PH - 1.75 * inch
for i, t in enumerate(steps3):
    circle_num(MX + 0.2 * inch, y - 0.2 * inch, 0.4 * inch, i + 1, cols5[i])
    c.setFont("Helvetica", 14)
    c.setFillColor(INK)
    c.drawString(MX + 0.62 * inch, y - 0.27 * inch, t)
    y -= 0.85 * inch
c.setFont("Helvetica-Oblique", 11)
c.setFillColor(MUTED)
c.drawString(MX, 0.85 * inch, "Not a chatbot \u2014 there is no free-text interface. A decision loop that redraws itself after every answer.")
c.showPage()

# ======================================================== 4. THE LOOP ======
light_bg()
kicker_title("Architecture", "The reasoning loop \u2014 five steps, run every turn")
steps = [
    ("Propose", "Naive-Bayes over 8 diseases, correlation-weighted so one clinical picture isn't double-counted as several findings.", TEAL, TEALBG),
    ("Calibrate", "Temperature scaling + conformal prediction turn a raw score into a trustworthy confidence.", BLUE, BLUEBG),
    ("Gate", "Six conditions must all hold before commit: confidence, margin, grounding, red-flag tolerance, proposer agreement, evidence fit.", PLUM, PLUMBG),
    ("Select", "Highest information gain per cost \u2014 plus mandatory guideline workups that fire on the presentation itself.", AMBER, AMBERBG),
    ("Decide", "Commit with citations, or escalate naming the unresolved question. Loop back to Propose on every new finding.", GREEN, GREENBG),
]
w5 = (CW - 0.4 * inch * 4) / 5
top = PH - 1.65 * inch
h5 = 4.1 * inch
for i, (title, body, edge, fill) in enumerate(steps):
    x = MX + i * (w5 + 0.4 * inch)
    rrect(x, top - h5, w5, h5, 6, fill)
    circle_num(x + w5 / 2, top - 0.5 * inch, 0.56 * inch, i + 1, edge, fs=16)
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(edge)
    c.drawCentredString(x + w5 / 2, top - 1.05 * inch, title)
    draw_para(x + 0.14 * inch, top - 1.35 * inch, w5 - 0.28 * inch, body, "Helvetica", 9.8, INK, 12.5)
    if i < len(steps) - 1:
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(MUTED)
        c.drawCentredString(x + w5 + 0.2 * inch, top - 0.75 * inch, "\u2192")
c.showPage()

# ==================================================== 5. WHAT'S BUILT ======
light_bg()
kicker_title("What's built", "Every number below is tested code, not a plan")
stats = [(str(F.tests), "tests in the suite", TEAL),
         (f"{F.fixtures_correct}/{F.fixtures_total}", "fixture cases correct", GREEN),
         (f"{F.coverage}%", "likelihoods sourced", AMBER),
         (str(F.real_total), "real PMC cases evaluated", BLUE)]
sw = (CW - 0.3 * inch * 3) / 4
for i, (v, l, col) in enumerate(stats):
    stat_card(MX + i * (sw + 0.3 * inch), PH - 1.65 * inch, sw, 1.35 * inch, v, l, col)

rows5 = [
    ("Bayesian reasoner + LangGraph engine", "Two independent implementations of the same loop, proven equivalent by test"),
    ("RAG retrieval", "BGE-M3 + Qdrant over 46 passages from 4 decision rules and the Merck Manual"),
    ("Clinical ontology", "8 causes, 27 findings, 84 labelled edges \u2014 derived from the KB, consistent by construction"),
    ("Guideline workups", "Wells / PERC / CURB-65 / HEART, plus mandatory + confirmatory tests that bypass a wrong posterior"),
    ("PHI screening, two layers", "Regex for structured identifiers, spaCy NER for names inside free narrative text"),
    ("Live interactive consult", "scripts/consult.py \u2014 a real person plays the patient, turn by turn"),
]
colW = (CW - 0.3 * inch) / 2
top5 = PH - 3.25 * inch
for i, (t, body) in enumerate(rows5):
    col, row = i % 2, i // 2
    x = MX + col * (colW + 0.3 * inch)
    yy = top5 - row * 1.15 * inch
    rrect(x, yy - 1.0 * inch, colW, 1.0 * inch, 5, SOFT)
    c.setStrokeColor(LINE); c.setLineWidth(0.75)
    c.roundRect(x, yy - 1.0 * inch, colW, 1.0 * inch, 5, stroke=1, fill=0)
    c.setFont("Helvetica-Bold", 11.5)
    c.setFillColor(INK)
    c.drawString(x + 0.15 * inch, yy - 0.28 * inch, t)
    draw_para(x + 0.15 * inch, yy - 0.55 * inch, colW - 0.3 * inch, body, "Helvetica", 9.4, MUTED, 12)
c.showPage()

# ============================================== 6. KNOWLEDGE BASE HONEST ===
light_bg()
kicker_title("The knowledge base", "Honestly \u2014 every number tagged by where it came from")
half = (CW - 0.3 * inch) / 2
card(MX, PH - 1.65 * inch, half, 1.95 * inch, F.invented_of_total, [
    "Both bulk routes proven exhausted, not untried \u2014 DDXPlus by running its own extractor, Merck by two full passes over all 8 chapters.",
    "Deleting them drops fixture accuracy 10/10 to 1/10. Individually insensitive, collectively load-bearing.",
], AMBERBG, AMBER)
card(MX + half + 0.3 * inch, PH - 1.65 * inch, half, 1.95 * inch, f"{F.priors_sourced} of {F.priors_total} disease priors sourced", [
    "Derived from StatPearls presentation-conditional aetiology \u2014 the right quantity, found only after an earlier search targeted the wrong one.",
    "Bands span a real 20-fold disagreement between sources on ACS.",
], GREENBG, GREEN)

srcs = [
    ("Source", "Entries", "What it is"),
    ("DDXPlus", str(F.ddxplus), "Frequencies counted in ~1M simulated patients \u2014 a generating model, not real people"),
    ("Merck Manual 19e", str(F.merck), "Narrative phrases converted through a fixed, pre-committed rubric"),
    ("Published cohorts / StatPearls", str(F.cohorts), "Miniati 2012 (PE), Z\u00e8gre-Hemsey 2018 (ACS), StatPearls (pericarditis)"),
]
simple_table(MX, PH - 3.85 * inch, [3.0 * inch, 1.1 * inch, CW - 4.1 * inch], srcs, header_bg=TEAL, row_h=0.42 * inch)
c.setFont("Helvetica-Oblique", 10.5)
c.setFillColor(MUTED)
draw_para(MX, PH - 5.9 * inch, CW,
    "Most recent addition: pericarditis' ECG-change likelihood, invented at 0.60, sourced from StatPearls at 0.50 "
    "\u2014 lower than the invented value, kept anyway.", "Helvetica-Oblique", 10.5, MUTED, 14)
c.showPage()

# ================================================== 7. REAL PATIENTS =======
light_bg()
kicker_title("Real patients",
             f"{F.real_total} cases: {F.real_committed_correct} correct, "
             f"{F.real_committed_wrong} wrong, the rest escalated")

def _short(text, limit=52):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

rows7 = [("Case", "True diagnosis", "Result")]
rows7 += [(_short(c), t, _short(r, 62)) for c, t, r in F.real_rows]
simple_table(MX, PH - 1.45 * inch, [3.9 * inch, 2.5 * inch, CW - 6.4 * inch], rows7,
             header_bg=BLUE, row_h=0.395 * inch, fs=9.0)

box_h = 1.35 * inch
box_y = PH - 1.45 * inch - len(rows7) * 0.395 * inch - 0.28 * inch - box_h
rrect(MX, box_y, CW, box_h, 6, SOFT)
c.setStrokeColor(LINE); c.setLineWidth(0.75)
c.roundRect(MX, box_y, CW, box_h, 6, stroke=1, fill=0)
c.setFont("Helvetica-Bold", 11.5)
c.setFillColor(INK)
c.drawString(MX + 0.25 * inch, box_y + box_h - 0.3 * inch,
             f"{F.real_committed_wrong} wrong commits — that is the number to look at.")
draw_para(MX + 0.25 * inch, box_y + box_h - 0.55 * inch, CW - 0.5 * inch,
    "Calibrated abstention is the claim, so the count that matters is not how many it answered but how many it "
    "answered wrongly. Every escalation names what it sought and could not get — several say the source "
    "report never recorded a D-dimer. n is small and the cases were hand-picked for clarity, so this is a few "
    "real data points rather than an accuracy result, and no threshold was tuned to make any of them pass.",
    "Helvetica", 10.0, INK, 13.5)
c.showPage()

# ============================================== 8. STRUCTURAL FIX ==========
light_bg()
kicker_title("A structural fix", "Found, closed generally, and traced honestly")
card(MX, PH - 1.65 * inch, CW, 2.0 * inch, "The gap", [
    "MandatoryWorkup enforced ordering a D-dimer once PE was a live concern, but never escalated a positive "
    "result to the confirmatory CTPA \u2014 a real gap for any future patient, not just this case.",
    "Fixed with a new ConfirmatoryWorkup rule, cited from the same Wells literature already in the codebase. "
    f"All {F.tests} tests still pass; fixture accuracy unaffected.",
], BLUEBG, BLUE)
card(MX, PH - 3.85 * inch, CW, 2.35 * inch, "Why it still didn't flip the case", [
    "The rule-out mechanism scores every candidate as discrimination \u00f7 (cost + 1.5). At the decisive turn "
    "it ranked a weak, cheap exam finding (JVP: 0.27 \u00f7 2.5 = 0.11) above the far more decisive but expensive "
    "CTPA (\u22480.9 \u00f7 21.5 = 0.04).",
    "That scoring formula is shared with the ordinary selector and has a documented prior regression \u2014 not "
    "touched tonight, reported as an open structural finding for the roadmap instead of patched blind.",
], AMBERBG, AMBER)
c.showPage()

# ============================================ 9. SAFETY & HONESTY ==========
light_bg()
kicker_title("Safety & honesty", "Architectural properties, checked by tests \u2014 not a bolt-on policy")
items9 = [
    ("No model authors a citation", "Every citation is built by string-formatting on a passage that was actually retrieved \u2014 confirmed by code inspection, not assumed.", TEALBG, TEAL),
    ("Two independent loop implementations", "A hand-written reference loop and a LangGraph state machine implement the same reasoning, proven equivalent by test.", BLUEBG, BLUE),
    ("PHI screening, two layers", "Regex catches structured identifiers; a spaCy NER pass reads the narrative and catches a name the regex is documented to miss.", GREENBG, GREEN),
    ("Six-condition abstention gate", "Confidence, margin, grounding, red-flag tolerance, proposer disagreement, evidence-fit all have to hold before commit.", PLUMBG, PLUM),
]
w9 = (CW - 0.3 * inch) / 2
for i, (t, body, fill, edge) in enumerate(items9):
    col, row = i % 2, i // 2
    x = MX + col * (w9 + 0.3 * inch)
    y_top = PH - 1.65 * inch - row * 2.25 * inch
    card(x, y_top, w9, 2.0 * inch, t, [body], fill, edge)
c.showPage()

# ============================================== 10. WHAT'S STILL OPEN ======
light_bg()
kicker_title("What's still open", "Named, not hidden")
half10 = (CW - 0.3 * inch) / 2
card(MX, PH - 1.65 * inch, half10, 1.95 * inch, "The mandated model hasn't run live", [
    "No API credit all project. The LLM layer's been swappable since week 1 \u2014 the default now points at the "
    "current Opus-tier model as the honest equivalent, unverified live.",
], AMBERBG, AMBER)
card(MX + half10 + 0.3 * inch, PH - 1.65 * inch, half10, 1.95 * inch, "AgentClinic", [
    "Named in the mandated stack. Genuinely blocked, not untried \u2014 needs an OpenAI/Replicate key plus forking "
    "its code to accept this agent.",
], AMBERBG, AMBER)

navy_top = PH - 3.9 * inch
navy_h = 2.35 * inch
rrect(MX, navy_top - navy_h, CW, navy_h, 6, NAVY)
c.setFont("Helvetica-Bold", 15)
c.setFillColor(WHITE)
c.drawString(MX + 0.4 * inch, navy_top - 0.35 * inch, "The constraint behind almost everything else")
draw_para(MX + 0.4 * inch, navy_top - 0.7 * inch, CW - 0.8 * inch,
    "The brief's own clinical-partnership table calls for weekly meetings with a collaborating physician, "
    "feedback \u201cpart of your evaluation.\u201d No clinician was available. That single fact is why every real "
    "patient came from a published case report rather than a clinician-reviewed one, and why the knowledge base "
    "still leans on DDXPlus and Merck rather than expert-elicited frequencies.", "Helvetica", 12, SKY, 17)
c.showPage()

# ==================================================== 11. THANK YOU ========
dark_bg()
c.setFont("Helvetica-Bold", 46)
c.setFillColor(WHITE)
c.drawCentredString(PW / 2, PH - 2.9 * inch, "Thank you.")
c.setFont("Helvetica", 18)
c.setFillColor(SKY)
c.drawCentredString(PW / 2, PH - 3.55 * inch, "Questions?")
c.setFont("Helvetica", 10)
c.setFillColor(SKY3)
c.drawCentredString(PW / 2, 0.75 * inch,
    f"dxagent  \u00b7  not for clinical use \u2014 {F.footer_claim}, and every run says so")
c.showPage()

c.save()
print("wrote", OUT)
