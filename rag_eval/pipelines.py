"""Retrieval + generation conditions for the evaluation grid.

The pilot compares two of the four baselines from the paper plan:

  (b) vanilla  — dense-only retrieval, no lexical branch, no tier prior,
                 no deduplication. The "just embed everything" baseline.
  (c) hybrid   — dense + lexical fused with RRF, tier prior, canonical chunks
                 only. What `corpus_index` actually implements.

Both use the same generator model, so any difference is the retrieval pipeline
rather than the LLM — the whole point of the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corpus_index.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from corpus_index.embedder import Embedder
from corpus_index.graph import VECTOR_INDEX, GraphStore

ANSWER_PROMPT = """You are assisting a perioperative gynecologic nurse.

Answer the question using ONLY the numbered context passages below. If the
context does not contain the answer, say so plainly rather than guessing.
Be specific about doses, timings and thresholds when the context gives them.
Cite the passages you used as [1], [2], etc.

Question: {question}

Context:
{context}

Answer:"""

DENSE_ONLY = """
CALL db.index.vector.queryNodes($vector_index, $k, $embedding)
YIELD node, score
RETURN node.chunk_id AS chunk_id, node.text AS text, node.section AS section,
       node.page AS page, node.tier AS tier, node.source AS source,
       node.title AS title, node.url AS url, node.publication_year AS year,
       score
"""


@dataclass
class Retrieved:
    question: str
    contexts: list[str] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)


class Pipeline:
    """A retrieval condition plus a generator."""

    def __init__(self, name: str, store: GraphStore, embedder: Embedder,
                 model: str = CHAT_MODEL, top_k: int = 5):
        self.name = name
        self.store = store
        self.embedder = embedder
        self.model = model
        self.top_k = top_k
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
        return self._client

    def retrieve(self, question: str) -> Retrieved:
        raise NotImplementedError

    def generate(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "The provided context does not contain information to answer this question."
        joined = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": ANSWER_PROMPT.format(question=question, context=joined)}],
            temperature=0,
            max_tokens=4000,
        )
        return (response.choices[0].message.content or "").strip()

    def answer(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        return {
            "question": question,
            "answer": self.generate(question, retrieved.contexts),
            "contexts": retrieved.contexts,
            "meta": retrieved.meta,
            "pipeline": self.name,
        }


class VanillaDense(Pipeline):
    """(b) Dense-only, no tiering, no dedup filter — the naive baseline."""

    def retrieve(self, question: str) -> Retrieved:
        embedding = self.embedder.embed_one(question)
        with self.store.session() as session:
            rows = [dict(r) for r in session.run(DENSE_ONLY, {
                "vector_index": VECTOR_INDEX,
                "embedding": embedding,
                "k": self.top_k,
            })]
        return Retrieved(question, [r["text"] for r in rows], rows)


class HybridTiered(Pipeline):
    """(c) Hybrid dense + lexical, RRF, tier prior, canonical chunks only."""

    def retrieve(self, question: str) -> Retrieved:
        embedding = self.embedder.embed_one(question)
        rows = self.store.search(question, embedding, limit=self.top_k)
        return Retrieved(question, [r["text"] for r in rows], rows)


def build_pipelines(top_k: int = 5) -> dict[str, Pipeline]:
    store = GraphStore()
    embedder = Embedder()
    return {
        "vanilla_dense": VanillaDense("vanilla_dense", store, embedder, top_k=top_k),
        "hybrid_tiered": HybridTiered("hybrid_tiered", store, embedder, top_k=top_k),
    }


# --- relational conditions ---------------------------------------------------

class NextExpansion(Pipeline):
    """Baseline for the relational pilot: hybrid retrieval + :NEXT-chain context.

    This is what the agent did before the knowledge graph existed. The graph has
    to beat it to justify its cost.
    """

    def retrieve(self, question: str) -> Retrieved:
        embedding = self.embedder.embed_one(question)
        seeds = self.store.search(question, embedding, limit=self.top_k)
        seen = {r["chunk_id"] for r in seeds}
        expanded: list[dict] = []

        with self.store.session() as session:
            for row in seeds[:4]:
                for record in session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})
                    OPTIONAL MATCH (c)-[:NEXT]->(nxt:Chunk)
                    OPTIONAL MATCH (prv:Chunk)-[:NEXT]->(c)
                    WITH [x IN [nxt, prv] WHERE x IS NOT NULL] AS ns
                    UNWIND ns AS n
                    RETURN n.chunk_id AS chunk_id, n.text AS text,
                           n.section AS section, n.page AS page, n.tier AS tier,
                           n.source AS source, n.title AS title, n.url AS url
                    """, {"cid": row["chunk_id"]}):
                    item = dict(record)
                    if item["chunk_id"] in seen:
                        continue
                    seen.add(item["chunk_id"])
                    item["score"] = row["score"] * 0.5
                    expanded.append(item)

        rows = (seeds + expanded)[:self.top_k * 2]
        return Retrieved(question, [r["text"] for r in rows], rows)


class KGTraversal(Pipeline):
    """Hybrid retrieval plus clinical-KG traversal, grounded in cited chunks."""

    def retrieve(self, question: str) -> Retrieved:
        from clinical_kg.retrieve import format_paths, format_relations, traverse

        embedding = self.embedder.embed_one(question)
        seeds = self.store.search(question, embedding, limit=self.top_k)
        result = traverse(self.store, question)

        rows = list(seeds)
        seen = {r["chunk_id"] for r in rows}
        for row in result["contexts"]:
            if row["chunk_id"] not in seen:
                seen.add(row["chunk_id"])
                rows.append(row)
        rows = rows[:self.top_k * 2]

        contexts = [r["text"] for r in rows]
        # The relation block is prepended as extra evidence, with qualifiers —
        # it is not a chunk, so it is excluded from chunk-recall scoring.
        relation_block = format_relations(result["relations"])
        path_block = format_paths(result["paths"])
        if relation_block:
            contexts.insert(0, "Extracted clinical relations (with the population "
                               "and condition each guideline attaches):\n"
                            + relation_block
                            + (("\n\nTwo-step connections:\n" + path_block)
                               if path_block else ""))
        return Retrieved(question, contexts, rows)


def build_relational_pipelines(top_k: int = 5) -> dict[str, Pipeline]:
    store = GraphStore()
    embedder = Embedder()
    return {
        "rel_hybrid_only":   HybridTiered("rel_hybrid_only", store, embedder, top_k=top_k),
        "rel_next_expand":   NextExpansion("rel_next_expand", store, embedder, top_k=top_k),
        "rel_kg_traversal":  KGTraversal("rel_kg_traversal", store, embedder, top_k=top_k),
    }


class GraphOnly(Pipeline):
    """Pure graph path: question -> entity/relation extraction -> lookup -> chunks.

    No dense or lexical retrieval. This is the condition that actually tests
    whether the knowledge graph is a competent retriever, rather than whether it
    is harmless when bolted onto one.
    """

    def retrieve(self, question: str) -> Retrieved:
        from clinical_kg.query import graph_only_retrieve

        result = graph_only_retrieve(self.store, question,
                                     chunk_limit=self.top_k * 2)
        rows = result["contexts"]
        contexts = [r["text"] for r in rows]

        if result["relations"]:
            from clinical_kg.retrieve import format_relations

            contexts.insert(0, "Clinical relations from the knowledge graph:\n"
                            + format_relations(result["relations"]))
        return Retrieved(question, contexts, rows)


def build_relational_pipelines_v2(top_k: int = 15) -> dict[str, Pipeline]:
    store = GraphStore()
    embedder = Embedder()
    return {
        "hybrid_only":  HybridTiered("hybrid_only", store, embedder, top_k=top_k),
        "next_expand":  NextExpansion("next_expand", store, embedder, top_k=top_k),
        "kg_plus_hybrid": KGTraversal("kg_plus_hybrid", store, embedder, top_k=top_k),
        "kg_only":      GraphOnly("kg_only", store, embedder, top_k=top_k),
    }
