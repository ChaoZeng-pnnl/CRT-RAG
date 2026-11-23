"""Query-side entity and relation extraction, and pure graph lookup.

The previous relational condition ran hybrid retrieval *and* graph traversal and
merged them, so the graph's contribution could never be isolated: it returned 178
chunks across 30 questions but only 2 gold ones, and only 2 that hybrid had not
already found. That design cannot answer "does the graph retrieve well?" — only
"does adding it hurt?".

This module implements the clean path:

    question -> LLM extracts entities + intended relation
             -> entity linking against the graph (exact, alias, then embedding)
             -> traversal
             -> the source chunks those edges cite

No dense or lexical retrieval anywhere. If the graph is a competent retriever,
this path finds the evidence on its own; if it is not, no amount of merging with
hybrid will show that.

Entity linking here is also stronger than the substring CONTAINS matching used
before, which was both too loose on short names and blind to morphology.
"""

from __future__ import annotations

import json
import re

from corpus_index.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from corpus_index.graph import GraphStore

from .schema import EDGE_TYPES, NODE_TYPES

QUERY_PROMPT = """Extract the clinical entities and the intended relation from a
perioperative nursing question, for lookup in a knowledge graph.

ENTITY TYPES: {node_types}
RELATION TYPES: {edge_types}

Question: {question}

Give the entities the question is *about*, using full clinical terms (expand
abbreviations: "VTE" -> "venous thromboembolism"). Include both the thing being
asked about and any patient factor, drug, device or procedure mentioned.
Give the relation(s) the question is asking for, or [] if none fits.

Return JSON only:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": ["..."]}}"""

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def extract_query_terms(question: str, model: str = CHAT_MODEL) -> dict:
    """LLM-extracted entities and relation intent from the question."""
    prompt = QUERY_PROMPT.format(
        node_types=", ".join(NODE_TYPES), edge_types=", ".join(EDGE_TYPES),
        question=question)
    try:
        response = client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=600)
        content = response.choices[0].message.content or ""
    except Exception as exc:
        return {"entities": [], "relations": [], "error": str(exc)[:120]}

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {"entities": [], "relations": []}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}

    entities = [{"name": (e.get("name") or "").strip().lower(),
                 "type": (e.get("type") or "").strip()}
                for e in (data.get("entities") or [])
                if isinstance(e, dict) and (e.get("name") or "").strip()]
    relations = [r.strip().upper() for r in (data.get("relations") or [])
                 if isinstance(r, str) and r.strip().upper() in EDGE_TYPES]
    return {"entities": entities, "relations": relations}


LINK_EXACT = """
UNWIND $names AS n
MATCH (e:Entity)
WHERE e.name = n
RETURN DISTINCT e.name AS name, e.type AS type, 1.0 AS score
"""

ALL_ENTITIES = "MATCH (e:Entity) RETURN e.name AS name, e.type AS type"

TRAVERSE_TYPED = """
UNWIND $names AS anchor
MATCH (a:Entity {name: anchor})-[r:RELATES]-(b:Entity)
WHERE size($relations) = 0 OR r.relation IN $relations
RETURN a.name AS subject, r.relation AS relation, b.name AS object,
       r.population AS population, r.condition AS condition,
       r.evidence AS evidence, r.chunk_id AS chunk_id, r.tier AS tier,
       r.source AS source, r.page AS page, r.confidence AS confidence,
       startNode(r) = a AS outgoing
ORDER BY r.tier, r.confidence DESC
LIMIT $limit
"""

FETCH_CHUNKS = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
RETURN c.chunk_id AS chunk_id, c.text AS text, c.section AS section,
       c.page AS page, c.tier AS tier, c.source AS source, c.title AS title,
       c.url AS url, c.publication_year AS year
"""

_ENTITY_CACHE: list[dict] | None = None
_ENTITY_VECTORS: list[list[float]] | None = None


def _entity_index(store: GraphStore):
    """All entity names plus their embeddings, cached per process."""
    global _ENTITY_CACHE, _ENTITY_VECTORS
    if _ENTITY_CACHE is None:
        from corpus_index.embedder import Embedder

        with store.session() as session:
            _ENTITY_CACHE = [dict(r) for r in session.run(ALL_ENTITIES)]
        _ENTITY_VECTORS = Embedder().embed(
            [e["name"] for e in _ENTITY_CACHE], progress=False)
    return _ENTITY_CACHE, _ENTITY_VECTORS


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def link(store: GraphStore, terms: list[dict], threshold: float = 0.80,
         per_term: int = 2) -> list[dict]:
    """Link extracted terms to Entity nodes: exact first, then embedding."""
    names = [t["name"] for t in terms]
    if not names:
        return []

    linked: dict[str, dict] = {}
    with store.session() as session:
        for row in session.run(LINK_EXACT, {"names": names}):
            linked[row["name"]] = {**dict(row), "how": "exact"}

    unmatched = [t for t in terms if t["name"] not in linked]
    if unmatched:
        from corpus_index.embedder import Embedder

        entities, vectors = _entity_index(store)
        query_vectors = Embedder().embed([t["name"] for t in unmatched],
                                         progress=False)
        for term, qv in zip(unmatched, query_vectors):
            if not qv:
                continue
            scored = []
            for entity, ev in zip(entities, vectors):
                if not ev:
                    continue
                # Only link within the same entity type when the LLM gave one.
                if term.get("type") and entity["type"] != term["type"]:
                    continue
                score = _cosine(qv, ev)
                if score >= threshold:
                    scored.append((score, entity))
            scored.sort(key=lambda s: -s[0])
            for score, entity in scored[:per_term]:
                if entity["name"] not in linked:
                    linked[entity["name"]] = {**entity, "score": score,
                                              "how": f"embedding({score:.2f})"}
    return list(linked.values())


def graph_only_retrieve(store: GraphStore, question: str, limit: int = 40,
                        chunk_limit: int = 15, rerank: bool = False,
                        max_anchor_degree: int | None = None) -> dict:
    """question -> extraction -> linking -> traversal -> source chunks. No hybrid.

    `rerank` orders the traversed chunks by embedding similarity to the question
    instead of by edge tier/confidence. This isolates the failure mode: entity
    anchoring produces a large candidate pool with no question-specific ordering,
    because an entity appears in every passage that mentions it. If reranking
    recovers the gap, the graph found the right neighbourhood and simply could
    not discriminate within it.

    `max_anchor_degree` drops hub anchors ("venous thromboembolism" carries 275
    edges), testing whether candidate explosion is what dilutes the pool.
    """
    terms = extract_query_terms(question)
    anchors = link(store, terms["entities"])

    if max_anchor_degree is not None and anchors:
        with store.session() as session:
            degrees = {r["name"]: r["d"] for r in session.run(
                "UNWIND $names AS n MATCH (e:Entity {name: n}) "
                "RETURN e.name AS name, COUNT { (e)-[:RELATES]-() } AS d",
                {"names": [a["name"] for a in anchors]})}
        kept = [a for a in anchors if degrees.get(a["name"], 0) <= max_anchor_degree]
        anchors = kept or anchors      # never leave the query with nothing
        for a in anchors:
            a["degree"] = degrees.get(a["name"], 0)

    if not anchors:
        return {"terms": terms, "anchors": [], "relations": [], "contexts": []}

    with store.session() as session:
        relations = [dict(r) for r in session.run(TRAVERSE_TYPED, {
            "names": [a["name"] for a in anchors],
            "relations": terms["relations"],
            "limit": limit,
        })]

        chunk_ids: list[str] = [r["chunk_id"] for r in relations if r.get("chunk_id")]
        seen: set[str] = set()
        # When reranking, keep a wider pool so the reranker has something to
        # choose from — the point is to test ordering, not truncation.
        pool = chunk_limit * 3 if rerank else chunk_limit
        ordered = [c for c in chunk_ids if not (c in seen or seen.add(c))][:pool]
        contexts = [dict(r) for r in session.run(
            FETCH_CHUNKS, {"chunk_ids": ordered})] if ordered else []

    weight = {cid: chunk_ids.count(cid) for cid in ordered}
    for row in contexts:
        row["score"] = 0.5 + 0.1 * weight.get(row["chunk_id"], 1)
        row["branches"] = ["graph_only"]

    if rerank and contexts:
        from corpus_index.embedder import Embedder

        embedder = Embedder()
        qv = embedder.embed_one(question)
        cvs = embedder.embed([c["text"] for c in contexts], progress=False)
        for row, cv in zip(contexts, cvs):
            row["score"] = _cosine(qv, cv) if cv else 0.0
            row["branches"] = ["graph_only_reranked"]

    contexts.sort(key=lambda r: r["score"], reverse=True)

    return {"terms": terms, "anchors": anchors, "relations": relations,
            "contexts": contexts[:chunk_limit]}
