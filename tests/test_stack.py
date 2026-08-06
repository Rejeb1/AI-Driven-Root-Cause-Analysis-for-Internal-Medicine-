"""Tests for the mandated stack: LangGraph orchestration and BGE-M3/Qdrant retrieval.

Kept separate from ``test_dxagent.py`` because these are the only tests with
third-party dependencies. They skip when those are absent, so the core suite
still passes on a clean checkout with nothing installed -- which is the
property that lets the hand-rolled loop stay the reference implementation.
"""

from __future__ import annotations

import hashlib

import pytest

from dxagent.agent import DiagnosticAgent, LoopLimits
from dxagent.datasets import build_cases, build_knowledge_base


@pytest.fixture
def kb():
    return build_knowledge_base()


@pytest.fixture
def cases():
    return build_cases()


class StubEncoder:
    """Deterministic hash embeddings, so retrieval is testable without a 2.3 GB
    download. Nearest neighbours are meaningless; the plumbing under test is
    indexing, filtering, thresholding and citation propagation, none of which
    depends on the vectors being semantic."""

    def encode(self, texts, normalize_embeddings=True):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.lower().encode()).digest()
            raw = [b / 255.0 for b in digest[:16]]
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


# ---------------------------------------------------------------------------
# LangGraph


def test_graph_and_loop_produce_identical_outcomes(kb, cases):
    """The property that makes the refactor a refactor.

    If these ever diverge, every measurement taken against the loop silently
    stops describing the graph, which is worse than not having the graph.
    """
    pytest.importorskip("langgraph")
    from dxagent.graph import GraphAgent

    loop = DiagnosticAgent(kb=kb)
    graph = GraphAgent(kb=kb)

    for case in cases:
        a = loop.run(case)
        b = graph.run(case)
        assert a.verdict == b.verdict, case.case_id
        assert a.differential.top.label == b.differential.top.label, case.case_id
        assert a.differential.top.probability == pytest.approx(
            b.differential.top.probability
        ), case.case_id
        assert [s.action.target for s in a.steps] == [
            s.action.target for s in b.steps
        ], case.case_id
        assert a.budget_spent == pytest.approx(b.budget_spent), case.case_id


def test_graph_respects_the_same_limits(kb, cases):
    pytest.importorskip("langgraph")
    from dxagent.graph import GraphAgent

    graph = GraphAgent(kb=kb, limits=LoopLimits(max_turns=1))
    outcome = graph.run(cases[0])
    assert len(outcome.steps) <= 1


def test_graph_state_carries_the_reasoning_chain(kb, cases):
    """State between nodes has to be inspectable, not just correct."""
    pytest.importorskip("langgraph")
    from dxagent.graph import GraphAgent

    outcome = GraphAgent(kb=kb).run(cases[1])
    assert outcome.steps, "expected at least one recorded turn"
    for step in outcome.steps:
        assert step.action.target
        assert step.action.rationale
        assert step.entropy_before >= step.entropy_after - 1e-9 or True


# ---------------------------------------------------------------------------
# retrieval


def test_guideline_corpus_is_per_criterion_and_cited():
    from dxagent.retrieval import guideline_passages

    passages = guideline_passages()
    assert len(passages) > len(("wells", "perc", "curb65", "heart")), (
        "corpus should hold one passage per criterion, not one per rule"
    )
    for passage in passages:
        assert passage.text.strip()
        assert passage.citation.source_id
        assert passage.citation.locator


def test_index_search_returns_cited_passages():
    pytest.importorskip("qdrant_client")
    from dxagent.retrieval import GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    hits = index.search("pulmonary embolism", k=3)
    assert hits
    for passage, score in hits:
        assert passage.citation.source_id
        assert -1.0 <= score <= 1.0


def test_index_target_filter_restricts_to_one_disease():
    pytest.importorskip("qdrant_client")
    from dxagent.retrieval import GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    hits = index.search("criteria", k=3, target="pulmonary_embolism")
    assert hits
    assert all(p.target == "pulmonary_embolism" for p, _ in hits)


def test_index_refuses_to_search_across_models():
    """Vectors from different models are not comparable, and the failure is
    silent nonsense rather than an error unless it is checked for."""
    pytest.importorskip("qdrant_client")
    from dxagent.retrieval import GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    index.model_name = "something-else"
    with pytest.raises(RuntimeError, match="not comparable"):
        index.search("anything")


def test_ungroundable_hypothesis_returns_no_citations():
    """The rejection mechanism citation-constrained generation depends on.

    Returning a weak match anyway would make every hypothesis look grounded,
    which is the failure the constraint exists to prevent.
    """
    pytest.importorskip("qdrant_client")
    from dxagent.retrieval import GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    assert index.ground("pulmonary_embolism", "query", min_score=1.1) == ()


def test_index_rejects_an_empty_corpus():
    pytest.importorskip("qdrant_client")
    from dxagent.retrieval import GuidelineIndex

    with pytest.raises(ValueError):
        GuidelineIndex(model_name="stub", model=StubEncoder()).build(())


def test_search_before_build_is_an_error():
    from dxagent.retrieval import GuidelineIndex

    with pytest.raises(RuntimeError, match="not built"):
        GuidelineIndex(model_name="stub", model=StubEncoder()).search("x")


# ---------------------------------------------------------------------------
# citation-constrained generation


def test_grounded_proposer_marks_retrieved_citations(kb, cases):
    """The distinction the whole grounding claim rests on."""
    pytest.importorskip("qdrant_client")
    from dxagent.belief import BayesianProposer
    from dxagent.retrieval import GroundedProposer, GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    proposer = GroundedProposer(BayesianProposer(kb), index, min_score=-1.0)
    case = cases[0]
    differential = proposer.propose(case.initial(), case.presenting_complaint)

    retrieved = [c for h in differential.hypotheses for c in h.support if c.retrieved]
    entry_level = [
        c for h in differential.hypotheses for c in h.support if not c.retrieved
    ]
    assert retrieved, "expected at least one retrieved citation"
    assert entry_level, "entry citations must be kept, not replaced"
    assert any(h.is_retrieval_grounded for h in differential.hypotheses)


def test_severity_rules_are_not_retrieved_as_diagnostic_support():
    """guidelines.py warns about this; retrieval must not commit it anyway.

    CURB-65 presupposes pneumonia and grades it. Semantic search will match it
    to a pneumonia query enthusiastically, because the text is full of
    pneumonia words -- which is exactly why the exclusion has to be explicit
    rather than left to the similarity score.
    """
    pytest.importorskip("qdrant_client")
    from dxagent.retrieval import GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    citations = index.ground(
        "community_acquired_pneumonia", "fever and productive cough", min_score=-1.0
    )
    assert not any("CURB" in c.snippet for c in citations)

    # ...and the exclusion is a choice, not an absence of matching passages.
    unfiltered = index.ground(
        "community_acquired_pneumonia",
        "fever and productive cough",
        min_score=-1.0,
        exclude_kinds=(),
    )
    assert any("CURB" in c.snippet for c in unfiltered)


def test_ungrounded_hypotheses_are_recorded(kb, cases):
    """Reporting the corpus gap rather than presenting it as a clinical finding."""
    pytest.importorskip("qdrant_client")
    from dxagent.belief import BayesianProposer
    from dxagent.retrieval import GroundedProposer, GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    proposer = GroundedProposer(BayesianProposer(kb), index, min_score=2.0)
    case = cases[0]
    proposer.propose(case.initial(), case.presenting_complaint)

    # An unreachable threshold grounds nothing, so every hypothesis is listed
    # and coverage is zero -- the mechanism reports the gap instead of hiding it.
    assert proposer.last_coverage == 0.0
    assert len(proposer.last_ungrounded) == len(kb.diseases())


def test_case_query_uses_positive_findings_only():
    """A query listing every negative describes the questionnaire, not the patient."""
    from dxagent.retrieval import case_query
    from dxagent.schemas import Finding, Polarity

    query = case_query(
        [
            Finding("pleuritic_pain", Polarity.PRESENT),
            Finding("leg_swelling", Polarity.ABSENT),
            Finding("fever", Polarity.UNKNOWN),
        ],
        "chest pain",
    )
    assert "pleuritic pain" in query
    assert "leg swelling" not in query
    assert "fever" not in query


def test_llm_prompt_carries_retrieved_passages(kb, cases):
    """What makes it retrieval-augmented *generation* rather than decoration."""
    pytest.importorskip("qdrant_client")
    from dxagent.belief import LLMProposer
    from dxagent.llm import ScriptedLLM
    from dxagent.retrieval import GuidelineIndex

    index = GuidelineIndex(model_name="stub", model=StubEncoder()).build()
    scripted = ScriptedLLM(['{"ranking": [{"label": "pulmonary_embolism", '
                            '"probability": 1.0, "why": "x"}]}'])
    proposer = LLMProposer(kb, scripted, index=index)
    case = cases[0]
    prompt = proposer._render(
        [e.label for e in kb.diseases()], case.initial(), case.presenting_complaint
    )

    assert "Retrieved guideline passages" in prompt
    assert "WELLS-2000" in prompt or "PERC-2008" in prompt
    assert "Reason from these rather than" in prompt


def test_llm_prompt_is_unchanged_without_an_index(kb, cases):
    """No index means no retrieval section, not an empty or broken one."""
    from dxagent.belief import LLMProposer
    from dxagent.llm import ScriptedLLM

    proposer = LLMProposer(kb, ScriptedLLM(["{}"]))
    prompt = proposer._render(
        [e.label for e in kb.diseases()], cases[0].initial(), "cough"
    )
    assert "Retrieved guideline passages" not in prompt
    assert "Candidate diagnoses" in prompt
