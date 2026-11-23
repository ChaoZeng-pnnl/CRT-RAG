"""Graph traversal for the agent's relational branch.

Three stages:

  1. link    — find the Entity nodes a question mentions
  2. traverse— walk 1-2 hops from those entities along typed relations
  3. ground  — return the evidence chunks the traversed edges cite, so the
               generator quotes real passages rather than the triple's own words

Every returned row carries `population` and `condition`. A bare relation is not
safe to act on — v1 of this graph scored 0.591 precisely because those
qualifications were dropped — so the retriever surfaces them and the answer
prompt is expected to preserve them.

Paths are deduplicated by evidence chunk, not by path: "reduced mobility raises
VTE risk" reaches the same NICE passage through a dozen prevention edges, and
handing the generator twelve copies of one recommendation wastes context and
biases it toward whatever is repeated.
"""

from __future__ import annotations

import re

from corpus_index.graph import GraphStore

# Entities whose names are too generic to be useful anchors: matching them
# retrieves half the graph.
_STOPWORD_ENTITIES = {"patient", "patients", "women", "surgery", "treatment",
                      "care", "risk", "management", "assessment"}

LINK_EXACT = """
UNWIND $terms AS term
MATCH (e:Entity)
WHERE e.name = term OR e.name CONTAINS term OR term CONTAINS e.name
RETURN DISTINCT e.name AS name, e.type AS type,
       size([(e)-[r:RELATES]-() | r]) AS degree
ORDER BY degree DESC
LIMIT $limit
"""

# One hop out from an anchor, in either direction, with the qualifiers.
ONE_HOP = """
UNWIND $names AS anchor
MATCH (a:Entity {name: anchor})-[r:RELATES]-(b:Entity)
RETURN a.name AS subject, a.type AS subject_type,
       r.relation AS relation, b.name AS object, b.type AS object_type,
       r.population AS population, r.condition AS condition,
       r.evidence AS evidence, r.chunk_id AS chunk_id, r.tier AS tier,
       r.source AS source, r.page AS page, r.title AS title,
       r.confidence AS confidence,
       startNode(r) = a AS outgoing
ORDER BY r.tier, r.confidence DESC
LIMIT $limit
"""

# Two hops: the query shape chunk retrieval genuinely cannot serve, e.g.
# "risk factor -> complication <- prevention" joined across two documents.
#
# Traversal is undirected (a path is a path), but direction is returned so the
# rendering can respect it — printing "VTE -[PREVENTS]- fondaparinux" inverts
# the clinical meaning and would mislead the generator.
#
# `$max_hub_degree` excludes hub nodes: "venous thromboembolism" carries 275
# edges, so any two VTE-related facts are 2 hops apart through it and the path
# asserts nothing. Paths are only informative through a mid node of modest degree.
TWO_HOP = """
UNWIND $names AS anchor
MATCH (a:Entity {name: anchor})-[r1:RELATES]-(mid:Entity)-[r2:RELATES]-(c:Entity)
WHERE c <> a AND mid <> a
  AND mid.type IN ['Complication', 'Procedure', 'RiskFactor']
  AND COUNT { (mid)-[:RELATES]-() } <= $max_hub_degree
RETURN a.name AS start, r1.relation AS rel1, mid.name AS middle,
       r2.relation AS rel2, c.name AS end,
       startNode(r1).name = a.name AS r1_forward,
       startNode(r2).name = mid.name AS r2_forward,
       r1.chunk_id AS chunk1, r2.chunk_id AS chunk2,
       r1.evidence AS evidence1, r2.evidence AS evidence2,
       r1.population AS population1, r1.condition AS condition1,
       r2.population AS population2, r2.condition AS condition2,
       r1.tier AS tier1, r2.tier AS tier2,
       r1.source AS source1, r2.source AS source2
ORDER BY r1.tier + r2.tier
LIMIT $limit
"""

FETCH_CHUNKS = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
RETURN c.chunk_id AS chunk_id, c.text AS text, c.section AS section,
       c.page AS page, c.tier AS tier, c.source AS source, c.title AS title,
       c.url AS url, c.publication_year AS year
"""


def _terms(question: str) -> list[str]:
    """Candidate entity mentions: noun-ish spans, longest first."""
    cleaned = re.sub(r"[^a-z0-9 \-]", " ", question.lower())
    words = [w for w in cleaned.split() if len(w) > 2]
    spans: list[str] = []
    # Try 3-, 2- and 1-word spans so "low molecular weight heparin" is found
    # before "heparin".
    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            span = " ".join(words[i:i + size])
            if span not in _STOPWORD_ENTITIES:
                spans.append(span)
    return spans[:60]


def link_entities(store: GraphStore, question: str, limit: int = 6) -> list[dict]:
    """Entity nodes the question plausibly mentions, most connected first."""
    with store.session() as session:
        rows = [dict(r) for r in session.run(
            LINK_EXACT, {"terms": _terms(question), "limit": limit})]
    return [r for r in rows if r["name"] not in _STOPWORD_ENTITIES]


def traverse(store: GraphStore, question: str, hops: int = 2,
             limit_per_hop: int = 30, max_hub_degree: int = 40) -> dict:
    """Link, walk, and return relations plus the chunks that evidence them."""
    anchors = link_entities(store, question)
    if not anchors:
        return {"anchors": [], "relations": [], "paths": [], "contexts": []}

    names = [a["name"] for a in anchors]
    with store.session() as session:
        relations = [dict(r) for r in session.run(
            ONE_HOP, {"names": names, "limit": limit_per_hop})]

        paths = []
        if hops >= 2:
            paths = [dict(r) for r in session.run(
                TWO_HOP, {"names": names, "limit": limit_per_hop,
                          "max_hub_degree": max_hub_degree})]

        # Ground everything in the passages the edges cite.
        chunk_ids: list[str] = []
        for row in relations:
            if row.get("chunk_id"):
                chunk_ids.append(row["chunk_id"])
        for row in paths:
            chunk_ids.extend(c for c in (row.get("chunk1"), row.get("chunk2")) if c)

        seen: set[str] = set()
        ordered_ids = [c for c in chunk_ids if not (c in seen or seen.add(c))][:10]
        contexts = [dict(r) for r in session.run(
            FETCH_CHUNKS, {"chunk_ids": ordered_ids})] if ordered_ids else []

    # Rank the grounding chunks by how many traversed edges cite them: a passage
    # supporting several relations is the one the answer should lead with.
    weight = {cid: chunk_ids.count(cid) for cid in ordered_ids}
    for row in contexts:
        row["score"] = 0.5 + 0.1 * weight.get(row["chunk_id"], 1)
        row["branches"] = ["graph"]

    contexts.sort(key=lambda r: r["score"], reverse=True)
    return {"anchors": anchors, "relations": relations, "paths": paths,
            "contexts": contexts}


def format_paths(paths: list[dict], max_rows: int = 6) -> str:
    """Render 2-hop paths with each edge pointing the way it actually points."""
    lines = []
    for row in paths[:max_rows]:
        left = (f"({row['start']}) -[{row['rel1']}]-> ({row['middle']})"
                if row.get("r1_forward")
                else f"({row['middle']}) -[{row['rel1']}]-> ({row['start']})")
        right = (f"({row['middle']}) -[{row['rel2']}]-> ({row['end']})"
                 if row.get("r2_forward")
                 else f"({row['end']}) -[{row['rel2']}]-> ({row['middle']})")
        lines.append(f"{left}  AND  {right}"
                     f"  (T{row.get('tier1','?')} {row.get('source1','')}"
                     f" + T{row.get('tier2','?')} {row.get('source2','')})")
    return "\n".join(lines)


def format_relations(relations: list[dict], max_rows: int = 12) -> str:
    """Render relations for the prompt, qualifications included."""
    lines = []
    for row in relations[:max_rows]:
        arrow = "->" if row.get("outgoing") else "<-"
        line = f"({row['subject']}) {arrow}[{row['relation']}]{arrow} ({row['object']})"
        qualifiers = []
        if row.get("population"):
            qualifiers.append(f"for: {row['population']}")
        if row.get("condition"):
            qualifiers.append(f"when: {row['condition']}")
        if qualifiers:
            line += "  [" + "; ".join(qualifiers) + "]"
        line += f"  (T{row.get('tier','?')} {row.get('source','')}"
        line += f" p{row['page']})" if row.get("page") else ")"
        lines.append(line)
    return "\n".join(lines)
