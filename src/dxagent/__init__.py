"""dxagent -- agentic diagnostic reasoning with grounded evidence and abstention.

Project 1, DeepShift AI summer internship 2026.
"""

from .actions import InformationGainSelector
from .baselines import RetrievalOnlyBaseline, SinglePassBaseline
from .agent import DiagnosticAgent, LoopLimits
from .belief import BayesianProposer, ConsensusProposer, LLMProposer
from .environment import Case, CaseOracle, NoisyOracle
from .gate import (
    AbstentionGate,
    ConformalPredictor,
    GateDecision,
    TemperatureScaler,
)
from .knowledge import DiseaseEntry, InMemoryKnowledgeBase
from .llm import AnthropicLLM, NullLLM, ScriptedLLM
from .schemas import (
    Action,
    ActionKind,
    CaseOutcome,
    Citation,
    Differential,
    Escalation,
    Finding,
    Hypothesis,
    Polarity,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "AbstentionGate",
    "RetrievalOnlyBaseline",
    "SinglePassBaseline",
    "Action",
    "ActionKind",
    "AnthropicLLM",
    "BayesianProposer",
    "Case",
    "CaseOracle",
    "CaseOutcome",
    "ConformalPredictor",
    "Citation",
    "ConsensusProposer",
    "DiagnosticAgent",
    "Differential",
    "DiseaseEntry",
    "Escalation",
    "Finding",
    "GateDecision",
    "Hypothesis",
    "InMemoryKnowledgeBase",
    "InformationGainSelector",
    "LLMProposer",
    "LoopLimits",
    "NoisyOracle",
    "NullLLM",
    "Polarity",
    "ScriptedLLM",
    "TemperatureScaler",
    "Verdict",
]
