# dxagent

Agentic diagnostic reasoning for internal medicine: iterative evidence
gathering, KB-grounded differential ranking, calibrated confidence, and an
abstain/escalate gate for a collaborating physician.

Project 1, DeepShift AI summer internship 2026. Week 3 implementation scaffold.

**Not for clinical use.** The bundled knowledge base is synthetic and invented
(see `datasets/fixtures.py`); no output from it means anything clinically.
`RESPONSIBLE_AI.md` states what that rules out, what privacy and safety
properties the system does have, and what would have to be true before any of
this went near a patient.

## Running it

Python 3.10+. The core runs with no install, no API key and no credentials.

```bash
python3 -m pytest tests/ -q     # 148 pass
python3 scripts/run_eval.py     # evaluation report, with the required baselines
python3 scripts/demo.py         # one consultation, as a readable transcript
python3 scripts/consult.py      # a live consultation -- you play the patient
```

Everything above is offline and deterministic. Tests that need the optional
stack skip themselves when it is absent, so a clean checkout is always green --
that is deliberate, because it keeps the dependency-free loop usable as the
reference the LangGraph engine is checked against.

More options:

```bash
python3 scripts/run_eval.py --trace     # per-turn reasoning trace
python3 scripts/run_eval.py --sweep     # gate threshold sweep
python3 scripts/run_eval.py --noisy     # lossy history taking
python3 scripts/run_eval.py --engine loop   # force the reference loop
python3 scripts/run_eval.py --ddxplus /path/to/release --limit 200
python3 scripts/ddxplus_verify.py --root /path/to/release
python3 scripts/build_vocabulary.py     # downloads hp.obo (11 MB)
python3 scripts/demo.py --case fx-009   # a case it gets wrong, and why
python3 scripts/demo.py --list          # available cases
python3 scripts/sensitivity.py          # which invented numbers actually matter
python3 scripts/sensitivity.py --ablate # what deleting them costs, vs moving them
python3 scripts/sensitivity.py --bands  # do the sourced ranges change anything
python3 scripts/plausibility_check.py   # does SCOPE.md still match the numbers
python3 scripts/build_ontology.py       # the finding/cause graph, as text or DOT
python3 scripts/synthesize.py --dry-run # synthetic case plan (section 5.2)
```

### Synthetic cases (section 5.2)

`--dry-run` needs nothing. Generating for real needs a model:

```bash
python3 scripts/synthesize.py --provider gemini --per-disease 2 --out cases.json
```

`--provider anthropic` (the default) uses the brief's mandated model and needs
`ANTHROPIC_API_KEY`; `--provider gemini` needs `GEMINI_API_KEY` and a free key
from [aistudio.google.com](https://aistudio.google.com). Gemini is a documented
deviation from the mandated stack, not an equivalent — a run says so, and each
case records which model wrote it.

**The corpus is never a headline metric.** The agent scores 20/20 on it, and
that number means nothing: cases generated from the knowledge base the agent
reasons over measure agreement with itself. Every generated id carries a
`synthetic-` prefix so one reaching a metrics table is visible in the per-case
output rather than resting on someone having remembered.

`--ddxplus` reads the published `.zip` splits directly; nothing needs
extracting. Keep `--limit` small at first -- the loop runs about a second per
case and the baselines add two more passes over the same cases.

**PHI screening.** `synthesis.phi_scan` (regex, structured identifiers) runs
with nothing installed. The NER layer that also catches a name in prose
(`synthesis.phi_scan_ner`) needs:

```bash
pip install -e ".[phi_ner]"
python3 -m spacy download en_core_web_sm
```

Without it, `screen()` still runs and still works -- it reports
`ner_screening_active: false` in its output so the weaker screen is visible
rather than assumed away.

### The mandated stack

```bash
pip install -e ".[stack]"                # LangGraph, BGE-M3, Qdrant
python3 scripts/run_eval.py --grounded   # cite retrieved guideline passages
python3 scripts/run_eval.py --llm        # add an LLM proposer (needs ANTHROPIC_API_KEY)
```

LangGraph is used automatically once installed. Retrieval is opt-in because the
first `--grounded` run downloads BGE-M3 (~2.3 GB) and indexing takes about a
minute; `--llm` runs the LLM *alongside* the Bayesian proposer rather than
instead of it, so their disagreement stays visible to the gate.

## Layout

| Module | Role |
|---|---|
| `schemas.py` | Findings, hypotheses, differentials, actions, escalation packet |
| `knowledge.py` | Vetted KB interface + in-memory likelihood-table backend |
| `belief.py` | Bayesian / LLM / consensus differential proposers |
| `actions.py` | Information-gain selection and red-flag rule-out policy |
| `gate.py` | Temperature scaling, conformal prediction, abstain/escalate gate |
| `agent.py` | The reference loop |
| `graph.py` | The same loop as a LangGraph state machine (asserted equivalent) |
| `guidelines.py` | Wells, PERC, CURB-65, HEART; presentation-triggered workup |
| `retrieval.py` | BGE-M3 + Qdrant over the decision rules and MSD Manual sentences; grounded proposer |
| `merck.py` | The Merck Manual quotes: one source, cited both as likelihoods and as retrievable passages |
| `baselines.py` | The two comparisons the brief requires |
| `provenance.py` | Where each likelihood came from: measured, narrative, invented |
| `ontology.py` | Findings and causes as a traversable graph, edges labelled supports/contradicts |
| `synthesis.py` | Synthetic case generation, critique and screening |
| `environment.py` | Case oracle, noisy oracle |
| `datasets/` | Synthetic fixtures; DDXPlus adapter |
| `evaluation/` | Ranking, calibration, selective prediction, DDx recall |

## The documents

| File | What it is for |
|---|---|
| `SCOPE.md` | Step 1 deliverable: presentation, causes, reasoning schema, and every constraint the project ran into |
| `DESIGN.md` | The running record of what was measured, including the conclusions later measurements took back |
| `RESPONSIBLE_AI.md` | Privacy, safety and risk, with the limitations stated as limitations |
| `VOCABULARY.md` | The concept layer: HPO, UMLS CUIs, SNOMED CT codes |

## Design traceability

Each choice maps to a specific system from the Weeks 1–2 note, keeping the
structural principle of the survey intact:

| Choice | Traces to |
|---|---|
| Sequential, cost-aware evidence gathering | MAI-DxO |
| Structured history-taking state | AMIE |
| Two independent proposers, disagreement as signal | MedAgents, MedAgent-Pro |
| Mandatory KB citation per hypothesis | grounding requirement |
| Calibrated abstain/escalate gate, risk–coverage evaluation | Wen et al. |
| Simulated patient/environment boundary | AgentClinic |
| Public benchmark adapter | DDXPlus |

## Swap points for blocked dependencies

Written so that MIMIC and UMLS clearing changes configuration, not code:

- `KnowledgeBase` is a Protocol — replace `InMemoryKnowledgeBase` with a
  UMLS-normalised backend.
- `Finding.concept` vs `Finding.raw` — concept normalisation drops in without
  touching call sites.
- `Environment` is a Protocol — MIMIC-derived cases replace the oracle.
- `LLMClient` is a Protocol — `NullLLM` is the default so nothing silently
  depends on a model being reachable.
