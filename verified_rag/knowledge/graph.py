"""Write recommendations and links into Neo4j.

    (:Recommendation {rec_id, start, end, text, grade, grade_start, grade_end,
                      population, pop_start, pop_end}) -[:STATED_IN]-> (:Chunk)
    (:Recommendation) -[:SUPERSEDES {grade_from, grade_to, grade_changed}]-> (:Recommendation)
    (:Document) -[:REFINES {population}]-> (:Document)

Recommendations inherit visibility from their chunk's document, so the release
rules (corpus_index.graph) apply to them unchanged.
"""

from __future__ import annotations

from typing import Dict, List

WRITE_RECS = """
UNWIND $recs AS r
MATCH (c:Chunk {chunk_id: r.chunk_id})
MERGE (x:Recommendation {rec_id: r.rec_id})
SET x += r
MERGE (x)-[:STATED_IN]->(c)
RETURN count(x) AS n
"""


def write_recommendations(store, recs: List[dict]) -> int:
    clean = [{k: v for k, v in r.items() if v is not None} for r in recs]
    with store.session() as s:
        s.run("CREATE CONSTRAINT rec_id IF NOT EXISTS FOR (x:Recommendation) REQUIRE x.rec_id IS UNIQUE")
        return s.run(WRITE_RECS, {"recs": clean}).single()["n"]


def link_refines(store, specific_doc: str, general_doc: str, population: str) -> None:
    with store.session() as s:
        s.run("""MATCH (a:Document {doc_id: $a}), (b:Document {doc_id: $b})
                 MERGE (a)-[r:REFINES]->(b) SET r.population = $p""",
              {"a": specific_doc, "b": general_doc, "p": population})


def link_editions(store, embedder, new_doc: str, old_doc: str, min_cos: float = 0.90) -> List[dict]:
    """Pair each recommendation of the new edition with the most similar one of
    the old edition (cosine of embeddings of the stated text) and record whether
    the grade changed: exact string comparison, no judge."""
    import math
    with store.session() as s:
        # Graded recommendations only: each edition states a recommendation in
        # the summary (no grade) and the body (graded), worded slightly
        # differently, and pairing a graded copy with an ungraded one reports a
        # grade "change" from None that never happened.
        q = """MATCH (x:Recommendation)-[:STATED_IN]->(:Chunk {doc_id: $d})
               WHERE x.grade IS NOT NULL
               RETURN x.rec_id AS id, x.text AS text, x.grade AS grade"""
        recs = s.run(q, {"d": new_doc}).data(), s.run(q, {"d": old_doc}).data()
    new, old = recs
    if not new or not old:
        return []
    vec = embedder.embed([" ".join(r["text"].split()) for r in new + old], progress=False)
    nv, ov = vec[:len(new)], vec[len(new):]
    cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    pairs = []
    for r, v in zip(new, nv):
        best = max(range(len(old)), key=lambda j: cos(v, ov[j]))
        c = cos(v, ov[best])
        if c >= min_cos:
            o = old[best]
            pairs.append({"new": r["id"], "old": o["id"], "cos": round(c, 4),
                          "grade_from": o["grade"], "grade_to": r["grade"],
                          "grade_changed": (o["grade"] or "") != (r["grade"] or "")})
    with store.session() as s:
        s.run("""UNWIND $pairs AS p
                 MATCH (n:Recommendation {rec_id: p.new}), (o:Recommendation {rec_id: p.old})
                 MERGE (n)-[e:SUPERSEDES]->(o)
                 SET e.cos = p.cos, e.grade_from = p.grade_from, e.grade_to = p.grade_to,
                     e.grade_changed = p.grade_changed""", {"pairs": pairs})
    return pairs


SAME_PRACTICE = ("Do `older` and `newer` give guidance on the same clinical practice "
                 "(the same intervention for the same purpose), so that `newer` replaces `older`?")


def link_partial(store, embedder, jev, new_doc: str, old_doc: str,
                 top: int = 3, min_cos: float = 0.70, min_p: float = 0.5) -> List[dict]:
    """Pair recommendations across a PARTIAL update, where the newer document
    rewrites some recommendations and leaves others in force.

    Candidates: the most similar graded older recommendations by embedding
    cosine; the verifier then confirms they address the same practice. What
    changed is recorded mechanically: the grade strings, and the quantities
    stated in each (exact comparison)."""
    import math
    from ..answer.details import quantities
    q = """MATCH (x:Recommendation)-[:STATED_IN]->(c:Chunk {doc_id: $d})
           WHERE x.grade IS NOT NULL
           RETURN x.rec_id AS id, x.text AS text, x.grade AS grade, c.chunk_id AS chunk"""
    with store.session() as s:
        new, old = s.run(q, {"d": new_doc}).data(), s.run(q, {"d": old_doc}).data()
    if not new or not old:
        return []
    vec = embedder.embed([" ".join(r["text"].split()) for r in new + old], progress=False)
    nv, ov = vec[:len(new)], vec[len(new):]
    cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    # Score every plausible (newer, older) pair, then match one-to-one on the
    # best overall score. Greedy matching in document order let a general 2017
    # antibiotic recommendation claim the 1999 caesarean one, leaving its real
    # successor ("before skin incision in all cesarean section procedures")
    # unpaired: the most important change in the pair went missing.
    scored = []
    for i, v in enumerate(nv):
        for c, j in sorted(((cos(v, ov[j]), j) for j in range(len(old))), reverse=True)[:top]:
            if c >= min_cos:
                scored.append((i, j, c))
    if not scored:
        return []
    probs = {}
    by_new = {}
    for i, j, c in scored:
        by_new.setdefault(i, []).append((j, c))
    for i, cands in by_new.items():
        state = {"newer": " ".join(new[i]["text"].split())}
        state.update({f"older_{k}": " ".join(old[j]["text"].split()) for k, (j, c) in enumerate(cands)})
        ans = jev.ask(state, {f"c{k}": jev.noul_q(SAME_PRACTICE.replace("`older`", f"`older_{k}`"))
                              for k in range(len(cands))})
        for k, (j, c) in enumerate(cands):
            probs[(i, j)] = (ans.noul(f"c{k}") or 0.0, c)
    pairs, used_new, used_old = [], set(), set()
    for (i, j), (p, c) in sorted(probs.items(), key=lambda kv: (kv[1][0] * kv[1][1]), reverse=True):
        if p < min_p or i in used_new or j in used_old:
            continue
        used_new.add(i); used_old.add(j)
        r, o = new[i], old[j]
        qn, qo = quantities(r["text"]), quantities(o["text"])
        pairs.append({"new": r["id"], "old": o["id"], "old_chunk": o["chunk"], "new_chunk": r["chunk"],
                      "cos": round(c, 4), "p_same": round(p, 3),
                      "grade_from": o["grade"], "grade_to": r["grade"],
                      "grade_changed": o["grade"].split("–")[0].split("-")[0].strip()
                      != r["grade"].split("–")[0].split("-")[0].strip(),
                      "quantities_changed": qn != qo})
    with store.session() as s:
        s.run("""UNWIND $pairs AS p
                 MATCH (n:Recommendation {rec_id: p.new}), (o:Recommendation {rec_id: p.old})
                 MERGE (n)-[e:SUPERSEDES]->(o)
                 SET e.cos = p.cos, e.p_same = p.p_same, e.grade_from = p.grade_from,
                     e.grade_to = p.grade_to, e.grade_changed = p.grade_changed,
                     e.quantities_changed = p.quantities_changed, e.partial = true""",
              {"pairs": pairs})
    return pairs
