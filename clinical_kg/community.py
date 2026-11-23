"""Community detection and summarisation — the global GraphRAG layer.

Local GraphRAG (entity -> traversal -> chunks) is what `query.py` implements, and
it lost to hybrid retrieval on local lookup. That is consistent with the original
GraphRAG paper, whose claimed advantage is *global* sensemaking: questions whose
answer exists in no single chunk ("what themes recur across these guidelines?",
"where do NICE and RCOG disagree?").

Answering those needs a different architecture:

    entity graph -> communities -> LLM summary per community -> map-reduce answer

Communities are found with label propagation rather than Leiden: Neo4j GDS is not
installed here, and label propagation needs no plugin, is deterministic under a
fixed seed and iteration count, and on a graph this size gives comparable
partitions. The method is a reportable implementation choice, not a hidden one.

Hub handling matters here as much as it did for local retrieval: `venous
thromboembolism` (degree 484) will otherwise absorb the whole graph into one
community. High-degree nodes are excluded from propagation and attached to the
community of their most frequent neighbour afterwards, so they inform summaries
without collapsing the partition.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from corpus_index.graph import GraphStore

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"

HUB_DEGREE = 60          # excluded from propagation, re-attached after
MAX_ITERATIONS = 20
MIN_COMMUNITY = 3        # smaller groups are noise, not themes

SUMMARY_PROMPT = """You are summarising one thematic community from a clinical
knowledge graph built over perioperative gynecologic nursing guidelines.

The community contains these entities and the relations between them, each with
the population and condition its guideline attaches, and its source:

{relations}

Write a summary that a nurse educator or quality-improvement lead could use to
understand this theme across the guideline corpus. Cover:
- what clinical theme this community represents
- the main recommendations, WITH their populations and conditions
- which guideline bodies contribute, and any points where they differ or where
  one is more specific than another
- what the community does NOT establish

Be concrete: name drugs, thresholds, timings and populations. 200-350 words.
Do not invent anything absent from the relations above.

Return JSON only:
{{"title": "<6 words>", "summary": "...", "sources": ["..."],
  "key_entities": ["..."]}}"""

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def load_edges(store: GraphStore) -> list[dict]:
    with store.session() as session:
        return [dict(r) for r in session.run("""
            MATCH (a:Entity)-[r:RELATES]->(b:Entity)
            RETURN a.name AS subject, a.type AS subject_type,
                   b.name AS object, b.type AS object_type,
                   r.relation AS relation, r.population AS population,
                   r.condition AS condition, r.evidence AS evidence,
                   r.source AS source, r.tier AS tier, r.page AS page,
                   r.chunk_id AS chunk_id
        """)]


def detect_communities(edges: list[dict], hub_degree: int = HUB_DEGREE,
                       iterations: int = MAX_ITERATIONS) -> dict[str, int]:
    """Label propagation over non-hub nodes; hubs attached afterwards."""
    adjacency: dict[str, Counter] = defaultdict(Counter)
    degree: Counter = Counter()
    for edge in edges:
        a, b = edge["subject"], edge["object"]
        adjacency[a][b] += 1
        adjacency[b][a] += 1
        degree[a] += 1
        degree[b] += 1

    hubs = {n for n, d in degree.items() if d >= hub_degree}
    nodes = sorted(n for n in adjacency if n not in hubs)
    labels = {n: i for i, n in enumerate(nodes)}

    for _ in range(iterations):
        changed = False
        for node in nodes:                     # deterministic order
            neighbours = Counter()
            for other, weight in adjacency[node].items():
                if other in labels:
                    neighbours[labels[other]] += weight
            if not neighbours:
                continue
            best = min(l for l, c in neighbours.items()
                       if c == max(neighbours.values()))   # tie-break by id
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break

    # Attach hubs to the community they touch most; they inform summaries but
    # never drive the partition.
    for hub in hubs:
        neighbours = Counter()
        for other, weight in adjacency[hub].items():
            if other in labels:
                neighbours[labels[other]] += weight
        labels[hub] = (min(l for l, c in neighbours.items()
                           if c == max(neighbours.values()))
                       if neighbours else -1)
    return labels


def group_edges(edges: list[dict], labels: dict[str, int],
                min_size: int = MIN_COMMUNITY) -> dict[int, list[dict]]:
    """Edges whose endpoints share a community, keyed by community id."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for edge in edges:
        a = labels.get(edge["subject"])
        b = labels.get(edge["object"])
        if a is None or b is None:
            continue
        # An edge belongs to a community if either endpoint is in it; hub edges
        # therefore appear in several, which is what makes summaries coherent.
        for community in {a, b}:
            groups[community].append(edge)

    sizes = {c: len({e["subject"] for e in es} | {e["object"] for e in es})
             for c, es in groups.items()}
    return {c: es for c, es in groups.items() if sizes[c] >= min_size}


def _format(edges: list[dict], limit: int = 60) -> str:
    lines = []
    for edge in edges[:limit]:
        line = f"({edge['subject']}) -[{edge['relation']}]-> ({edge['object']})"
        qualifiers = [q for q in (
            f"for: {edge['population']}" if edge.get("population") else "",
            f"when: {edge['condition']}" if edge.get("condition") else "") if q]
        if qualifiers:
            line += "  [" + "; ".join(qualifiers) + "]"
        line += f"  ({edge.get('source','')} T{edge.get('tier','?')}"
        line += f" p{edge['page']})" if edge.get("page") else ")"
        lines.append(line)
    return "\n".join(lines)


def summarise_community(community_id: int, edges: list[dict],
                        model: str = CHAT_MODEL) -> dict | None:
    try:
        response = client().chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": SUMMARY_PROMPT.format(relations=_format(edges))}],
            temperature=0, max_tokens=4000)
        content = response.choices[0].message.content or ""
    except Exception as exc:
        print(f"    ! community {community_id}: {type(exc).__name__}: {str(exc)[:80]}")
        return None

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    entities = {e["subject"] for e in edges} | {e["object"] for e in edges}
    return {
        "community_id": community_id,
        "title": data.get("title", ""),
        "summary": data.get("summary", ""),
        "key_entities": data.get("key_entities", [])[:15],
        "n_entities": len(entities),
        "n_relations": len(edges),
        "sources": sorted({e.get("source", "") for e in edges if e.get("source")}),
        "tiers": sorted({e.get("tier") for e in edges if e.get("tier")}),
        "chunk_ids": sorted({e["chunk_id"] for e in edges if e.get("chunk_id")})[:40],
    }


def build(concurrency: int = 6, min_size: int = 3) -> dict:
    """Summarise communities of at least `min_size` entities.

    The threshold is a reportable parameter, not a tuning knob. Measured on the
    full Tier 1-2 graph (3,806 labelled entities, 5,066 relations):

        >=3  ->  487 communities, 3,246 entities (85%), 6,163 relations
        >=5  ->  269 communities, 2,776 entities (73%), 5,581 relations
        >=8  ->  157 communities, 2,426 entities (64%), 4,953 relations
        >=20 ->   51 communities, 1,828 entities (48%), 3,526 relations

    Set to 3: a global question should be answerable from the whole corpus, and
    excluding a third of the entities would silently put part of the corpus out
    of reach — which is exactly the coverage error that invalidated the first
    relational pilot. Small communities give thin summaries, but a thin summary
    is honest; a missing one is a blind spot the evaluation cannot see.
    """
    store = GraphStore()
    try:
        edges = load_edges(store)
        print(f"  Relations: {len(edges)}")
        labels = detect_communities(edges)
        groups = group_edges(edges, labels, min_size=min_size)
        print(f"  Communities (>= {min_size} entities): {len(groups)}")
        sizes = sorted((len({e['subject'] for e in es} | {e['object'] for e in es})
                        for es in groups.values()), reverse=True)
        print(f"  Sizes: {sizes[:12]}")

        summaries = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result in pool.map(lambda kv: summarise_community(*kv),
                                   sorted(groups.items())):
                if result:
                    summaries.append(result)
                    print(f"    summarised {len(summaries)}/{len(groups)}",
                          end="\r", flush=True)
        print()

        out = OUT_DIR / "communities.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in summaries:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote {out} ({len(summaries)} community summaries)")
        return {"communities": len(groups), "summarised": len(summaries),
                "sizes": sizes}
    finally:
        store.close()
