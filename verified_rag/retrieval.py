"""Retrieval: snapshot-pinned hybrid search, then the graph's view of each hit.

Search finds passages (corpus_index.GraphStore.search_text, pinned to one
release). The graph then says, for each passage's document: is it current,
what did it replace, and is it specific to one population (REFINES). A
population-specific document is demoted below general guidance unless the
question is about that population; that is what stops a caesarean-only
guideline from crowding out the general SSI guideline for a hysterectomy
question (design doc §20, the failing replay check).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .types import Source

ENRICH = """
UNWIND $docs AS did
MATCH (d:Document {doc_id: did})
OPTIONAL MATCH (d)-[:SUPERSEDES]->(old:Document)
OPTIONAL MATCH (d)-[rf:REFINES]->(gen:Document)
RETURN did AS doc_id,
       collect(DISTINCT old.title + ' (' + old.publication_year + ')') AS replaces,
       collect(DISTINCT rf.population) AS populations,
       collect(DISTINCT gen.doc_id) AS refines_docs
"""

# Recommendations stated in the retrieved chunks that a newer document, live at
# this release, has replaced (partial updates such as CDC 1999 -> 2017, where
# the old passage stays searchable because it also states current guidance).
REPLACED = """
UNWIND $chunks AS cid
MATCH (o:Recommendation)-[:STATED_IN]->(c:Chunk {chunk_id: cid})
MATCH (n:Recommendation)-[:SUPERSEDES]->(o)
MATCH (n)-[:STATED_IN]->(nc:Chunk)
MATCH (nd:Document {doc_id: nc.doc_id})
WHERE nd.live_from IS NOT NULL AND nd.live_from <= $snapshot
  AND (nd.superseded_in IS NULL OR nd.superseded_in > $snapshot)
  AND nd.doc_id <> c.doc_id
RETURN cid AS chunk_id, o.text AS old_text, n.text AS new_text, nc.chunk_id AS new_chunk,
       nd.publication_year AS new_year, n.start AS new_start, n.end AS new_end
"""

# Grades of recommendations stated in the retrieved chunks, and whether the
# previous edition graded the same recommendation differently. Both strings are
# positions in the source text, compared exactly (knowledge.graph.link_editions).
GRADES = """
UNWIND $chunks AS cid
MATCH (x:Recommendation)-[:STATED_IN]->(c:Chunk {chunk_id: cid})
WHERE x.grade IS NOT NULL
OPTIONAL MATCH (x)-[e:SUPERSEDES]->(o:Recommendation)-[:STATED_IN]->(oc:Chunk)
WHERE e.grade_changed
RETURN cid AS chunk_id, x.grade AS grade, e.grade_from AS grade_from,
       oc.publication_year AS from_year
"""


class Retriever:
    def __init__(self, store, embedder, jev=None, limit: int = 6,
                 population_threshold: float = 0.5, max_per_doc: int = 3,
                 versioned: bool = True):
        self.store, self.embedder, self.jev = store, embedder, jev
        self.limit, self.population_threshold = limit, population_threshold
        self.max_per_doc = max_per_doc
        # versioned=False is the conventional baseline (system A): one static
        # index, no release filtering and no graph knowledge of change.
        self.versioned = versioned

    def retrieve(self, question: str, snapshot: Optional[int] = None
                 ) -> Tuple[List[Source], int, Dict[str, float]]:
        t0 = time.perf_counter()
        rows = self.store.search_text(question, self.embedder, limit=self.limit * 6,
                                      snapshot=snapshot, versioned=self.versioned)
        timings = {"search_s": time.perf_counter() - t0}
        if not rows:
            return [], snapshot or self.store.current_release(), timings
        seq = rows[0]["snapshot"]
        if not self.versioned:
            sources = [Source(label=f"S{i}", chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"],
                              title=r.get("title", ""), tier=r.get("tier", 4), year=str(r.get("year", "")),
                              page=r.get("page"), snapshot=seq)
                       for i, r in enumerate(rows[:self.limit], start=1)]
            return sources, seq, timings

        t1 = time.perf_counter()
        with self.store.session() as s:
            info = {r["doc_id"]: r for r in s.run(
                ENRICH, {"docs": sorted({r["doc_id"] for r in rows})}).data()}
            grades = {}
            for g in s.run(GRADES, {"chunks": [r["chunk_id"] for r in rows]}).data():
                grades.setdefault(g["chunk_id"], g)
            replaced = {}
            for x in s.run(REPLACED, {"chunks": [r["chunk_id"] for r in rows], "snapshot": seq}).data():
                replaced.setdefault(x["chunk_id"], []).append(x)
        timings["graph_s"] = time.perf_counter() - t1

        # population-aware ranking: ask once per distinct population present
        t2 = time.perf_counter()
        pops = sorted({p for r in info.values() for p in r["populations"] if p})
        applies: Dict[str, float] = {}
        if pops and self.jev is not None:
            qs = {f"p{i}": self.jev.noul_q(
                f"Is `question` specifically about patients in this population: {p}? "
                "Answer yes only if the question concerns that population.")
                for i, p in enumerate(pops)}
            ans = self.jev.ask({"question": question}, qs)
            applies = {p: (ans.noul(f"p{i}") or 0.0) for i, p in enumerate(pops)}
        timings["population_s"] = time.perf_counter() - t2

        def demoted(row) -> bool:
            p = [x for x in info.get(row["doc_id"], {}).get("populations", []) if x]
            return bool(p) and all(applies.get(x, 0.0) < self.population_threshold for x in p)

        ranked = [r for r in rows if not demoted(r)] + [r for r in rows if demoted(r)]

        # At most max_per_doc passages from one document, so a single source
        # cannot fill the answer (a caesarean question otherwise got 8 of 8
        # passages from the one caesarean guideline).
        per_doc: Dict[str, int] = {}
        chosen = []
        for r in ranked:
            if per_doc.get(r["doc_id"], 0) < self.max_per_doc:
                chosen.append(r)
                per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
        chosen = chosen[:self.limit]

        # A refining document is shown WITH the general guidance it refines:
        # if that guidance is in the candidate pool but not chosen, its best
        # passage replaces the last slot.
        for r in list(chosen):
            for gen_doc in info.get(r["doc_id"], {}).get("refines_docs", []) or []:
                if not gen_doc or any(c["doc_id"] == gen_doc for c in chosen):
                    continue
                pool = [x for x in ranked if x["doc_id"] == gen_doc]
                if not pool:   # not in the general pool: search that document alone
                    pool = self.store.search_text(question, self.embedder, limit=1,
                                                  snapshot=seq, allowed_docs={gen_doc},
                                                  k_dense=400, k_lexical=400)
                if pool:
                    if len(chosen) < self.limit:
                        chosen.append(pool[0])
                    else:
                        chosen[-1] = pool[0]
                    info.setdefault(gen_doc, {"doc_id": gen_doc, "replaces": [],
                                              "populations": [], "refines_docs": []})
        # A passage stating a replaced recommendation brings its replacement
        # with it, so the writer and the checks have the current text.
        have = {c["chunk_id"] for c in chosen}
        wanted = [x["new_chunk"] for c in chosen for x in replaced.get(c["chunk_id"], [])
                  if x["new_chunk"] not in have]
        if wanted:
            from corpus_index.graph import FETCH_CHUNKS
            with self.store.session() as s:
                extra = {r["chunk_id"]: dict(r, snapshot=seq) for r in s.run(
                    FETCH_CHUNKS, {"chunk_ids": sorted(set(wanted))})}
            for cid in dict.fromkeys(wanted):
                if cid in extra:
                    chosen.append(extra[cid])
        sources = []
        for i, r in enumerate(chosen, start=1):
            meta = info.get(r["doc_id"], {})
            pops_r = [x for x in meta.get("populations", []) if x]
            sources.append(Source(
                label=f"S{i}", chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"],
                title=r.get("title", ""), tier=r.get("tier", 4), year=str(r.get("year", "")),
                page=r.get("page"), snapshot=seq,
                superseded_note=("replaces " + "; ".join(meta["replaces"])) if meta.get("replaces") else "",
                refines=" ".join(pops_r[0].split()) if pops_r else "",
                grade=" ".join((grades.get(r["chunk_id"], {}).get("grade") or "").split()),
                grade_change=(f"was '{' '.join(grades[r['chunk_id']]['grade_from'].split())}' in the "
                              f"{grades[r['chunk_id']]['from_year']} edition")
                if grades.get(r["chunk_id"], {}).get("grade_from") else "",
                replaced=[dict(x) for x in replaced.get(r["chunk_id"], [])]))
        by_chunk = {s.chunk_id: s.label for s in sources}
        for s in sources:
            for x in s.replaced:
                x["new_label"] = by_chunk.get(x["new_chunk"], "")
        return sources, seq, timings
