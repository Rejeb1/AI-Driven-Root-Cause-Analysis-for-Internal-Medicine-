"""Citation-grounded retrieval: BGE-M3 embeddings indexed in Qdrant.

The brief requires that every claim about a candidate cause cite a retrieved
passage rather than the model's parametric memory, and that hypotheses which
cannot be grounded are rejected or flagged. This module supplies the retrieval
half of that: a corpus built from the encoded clinical rules, embedded and
searchable, where every hit carries the citation of the source it came from.

The distinction from what the knowledge base already does is worth being
precise about, because it is easy to claim more grounding than is delivered.
``InMemoryKnowledgeBase.evidence_split`` cites the KB *entry* a likelihood came
from -- provenance for a number. This module cites the *text* a clinical claim
was read from. The first says where the arithmetic came from; only the second
lets a clinician check the reasoning against a source they can read.

Qdrant runs in memory. A server would add an operational dependency for no gain
at this corpus size -- the guideline corpus is tens of passages, not millions --
and the same client API points at a real instance by changing one argument, so
nothing about the code changes when the corpus does.

On the embedding model
----------------------
BGE-M3 is mandated and is the default. It is a ~2.3 GB download on first use
and is loaded lazily, so importing this module costs nothing and the test suite
can inject a small model instead. Which model produced an index matters: the
vectors are not comparable across models, so ``GuidelineIndex`` records the
model name and refuses to search an index built by a different one rather than
silently returning nonsense neighbours.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .guidelines import RULES, WORKUPS
from .schemas import Citation


@dataclass(frozen=True)
class Passage:
    """One retrievable unit of guideline text, with the citation it came from."""

    text: str
    citation: Citation
    concepts: tuple[str, ...] = ()  # vocabulary terms this passage speaks about
    target: str = ""  # disease label, when the passage is about one

    @property
    def identifier(self) -> str:
        return f"{self.citation.source_id}:{self.citation.locator}:{hash(self.text) & 0xFFFF}"


def guideline_passages() -> tuple[Passage, ...]:
    """Build the corpus from the encoded rules.

    One passage per criterion rather than one per rule. A rule embedded whole
    retrieves as a single blurry vector: a query about haemoptysis and a query
    about tachycardia would return the same passage with the same score, and
    the citation would point at the rule rather than at the line that justifies
    the claim. Per-criterion passages keep the citation as specific as the
    claim it supports.
    """
    passages: list[Passage] = []

    for rule in RULES:
        passages.append(
            Passage(
                text=(
                    f"{rule.name} ({rule.kind.value}) applies to "
                    f"{rule.target.replace('_', ' ')}. {rule.note}"
                ),
                citation=rule.citation,
                target=rule.target,
            )
        )
        for criterion in rule.criteria:
            passages.append(
                Passage(
                    text=(
                        f"{rule.name}: {criterion.description} "
                        f"scores {criterion.weight:g}."
                    ),
                    citation=rule.citation,
                    concepts=(criterion.concept,) if criterion.concept else (),
                    target=rule.target,
                )
            )

    for workup in WORKUPS:
        passages.append(
            Passage(
                text=f"{workup.name}. {workup.rationale}",
                citation=workup.citation,
                concepts=workup.required,
            )
        )

    return tuple(passages)


@dataclass
class GuidelineIndex:
    """Embedded, searchable guideline corpus.

    ``model`` may be injected to avoid the BGE-M3 download in tests; anything
    exposing sentence-transformers' ``encode`` works.
    """

    model_name: str = "BAAI/bge-m3"
    collection: str = "guidelines"
    model: Any = None
    _client: Any = field(default=None, init=False, repr=False)
    _passages: dict[str, Passage] = field(default_factory=dict, init=False, repr=False)
    _built_with: str = field(default="", init=False, repr=False)

    def _load_model(self) -> Any:
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    def build(self, passages: tuple[Passage, ...] | None = None) -> "GuidelineIndex":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        passages = passages if passages is not None else guideline_passages()
        if not passages:
            raise ValueError("cannot build an index from an empty corpus")

        vectors = self._encode([p.text for p in passages])
        dimension = len(vectors[0])

        self._client = QdrantClient(":memory:")
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        points = []
        for passage, vector in zip(passages, vectors):
            point_id = str(uuid.uuid4())
            self._passages[point_id] = passage
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": passage.text,
                        "source_id": passage.citation.source_id,
                        "locator": passage.citation.locator,
                        "target": passage.target,
                    },
                )
            )
        self._client.upsert(collection_name=self.collection, points=points)
        self._built_with = self.model_name
        return self

    def search(
        self, query: str, k: int = 5, target: str | None = None
    ) -> tuple[tuple[Passage, float], ...]:
        """Nearest passages to ``query``, most similar first.

        ``target`` filters to passages about one disease, which is what makes
        this usable for grounding a specific hypothesis rather than the case
        as a whole.
        """
        if self._client is None:
            raise RuntimeError("index not built; call build() first")
        if self._built_with != self.model_name:
            raise RuntimeError(
                f"index was built with {self._built_with!r} but the model is now "
                f"{self.model_name!r}; vectors from different models are not "
                "comparable, so rebuild rather than search across them"
            )

        vector = self._encode([query])[0]
        # query_points, not the removed search(): qdrant-client dropped the
        # latter, and it returns a response object rather than a bare list.
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            # Over-fetch when filtering in Python so the filter cannot starve
            # the result set below k.
            limit=k * 4 if target else k,
        )
        results: list[tuple[Passage, float]] = []
        for hit in response.points:
            passage = self._passages.get(str(hit.id))
            if passage is None:
                continue
            if target and passage.target != target:
                continue
            results.append((passage, float(hit.score)))
            if len(results) >= k:
                break
        return tuple(results)

    def ground(
        self, label: str, query: str, k: int = 3, min_score: float = 0.3
    ) -> tuple[Citation, ...]:
        """Citations for a hypothesis, drawn from retrieved text.

        Returns empty when nothing clears ``min_score``. That is the case the
        brief calls for rejecting or flagging: a hypothesis with no retrievable
        support is exactly what citation-constrained generation is supposed to
        catch, and returning a weak match anyway would defeat the mechanism.
        """
        grounded: list[Citation] = []
        for passage, score in self.search(query, k=k, target=label):
            if score < min_score:
                continue
            grounded.append(
                Citation(
                    source_id=passage.citation.source_id,
                    locator=passage.citation.locator,
                    snippet=f"{passage.text} [retrieved, similarity {score:.2f}]",
                )
            )
        return tuple(grounded)


__all__ = ["GuidelineIndex", "Passage", "guideline_passages"]
