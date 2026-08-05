# dxagent

Agentic diagnostic reasoning for internal medicine: iterative evidence
gathering, KB-grounded differential ranking, calibrated confidence, and an
abstain/escalate gate for a collaborating physician.

Project 1, DeepShift AI summer internship 2026. Week 3 implementation scaffold.

**Not for clinical use.** The bundled knowledge base is synthetic and invented
(see `datasets/fixtures.py`); no output from it means anything clinically.

## Running it

Python 3.10+. No install, no API key, no credentials. Run from the repo root.

```bash
python3 -m pytest tests/ -q     # 43 pass, 10 skip until step 2
python3 scripts/run_eval.py     # evaluation report
```

The 10 skips are the vocabulary tests; they need the HPO file:

```bash
python3 scripts/build_vocabulary.py   # downloads hp.obo (11 MB), then 53 pass
```

That download is the only step needing internet. Everything else is offline and
deterministic.

More options:

```bash
python3 scripts/run_eval.py --trace   # per-turn reasoning trace
python3 scripts/run_eval.py --sweep   # gate threshold sweep
python3 scripts/run_eval.py --noisy   # lossy history taking
python3 scripts/run_eval.py --ddxplus /path/to/release --limit 5000
python3 scripts/build_vocabulary.py --refresh   # re-download HPO
```

If you prefer an editable install: `pip install -e ".[dev]"`.

## Layout

| Module | Role |
|---|---|
| `schemas.py` | Findings, hypotheses, differentials, actions, escalation packet |
| `knowledge.py` | Vetted KB interface + in-memory likelihood-table backend |
| `belief.py` | Bayesian / LLM / consensus differential proposers |
| `actions.py` | Information-gain selection and red-flag rule-out policy |
| `gate.py` | Temperature calibration and the abstain/escalate gate |
| `agent.py` | The loop |
| `environment.py` | Case oracle, noisy oracle |
| `datasets/` | Synthetic fixtures; DDXPlus adapter |
| `evaluation/` | Ranking, calibration, and selective-prediction metrics |

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
