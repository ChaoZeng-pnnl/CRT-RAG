"""Build the knowledge layer for a set of documents.

    python -m verified_rag.knowledge pilot     # the three WHO documents
    python -m verified_rag.knowledge relink    # recompute links only (no Jev calls)
    python -m verified_rag.knowledge cdc       # CDC 1999 -> 2017 partial update, no publish
    python -m verified_rag.knowledge cdc --publish   # retire the replaced 1999 passages

Writes Recommendation nodes, REFINES and edition SUPERSEDES links, and a JSON
report (runs/knowledge_pilot.json) that the figures read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from corpus_index.config import CHUNKS_FILE
from corpus_index.embedder import Embedder
from corpus_index.graph import GraphStore

from ..clients.jev import JevClient
from ..config import ROOT, load
from .extract import extract_document
from .graph import link_editions, link_partial, link_refines, write_recommendations

WHO16, WHO18, CS21 = "e7ad0c398aa2", "7c3781feaf22", "f3e61e207cc2"
CDC99, CDC17 = "99fccebb0aaa", "1a12d3183519"


def chunks_for(doc_ids):
    out = {d: [] for d in doc_ids}
    with Path(CHUNKS_FILE).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["doc_id"] in out:
                out[r["doc_id"]].append(r)
    return out


def _norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def refinement_population(emb, specific: list, general: list):
    """The population of the specific document's recommendation that refines a
    general one: most similar stated action, AND a population the general
    guideline does not state in any of its recommendations. The second
    condition is exact (substring on normalized text) and is what separates a
    refinement from the specific document merely quoting the general one."""
    import math
    # A population the general guideline already states anywhere is not a
    # narrowing of it; the specific document is quoting.
    general_text = " | ".join(_norm(g["text"]) for g in general)
    cands = [r for r in specific if r.get("population")
             and _norm(r["population"]) not in general_text]
    if not cands or not general:
        return None, None
    vec = emb.embed([_norm(r["text"]) for r in cands + general], progress=False)
    cv, gv = vec[:len(cands)], vec[len(cands):]
    cos = lambda a, b: sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    best = None
    for i, a in enumerate(cv):
        for j, b in enumerate(gv):
            c = cos(a, b)
            if best is None or c > best[0]:
                best = (c, i, j)
    if best is None:
        return None, None
    c, i, j = best
    return cands[i]["population"], {"specific": cands[i]["rec_id"], "general": general[j]["rec_id"],
                                    "cos": round(c, 4)}


def load_recs(store, doc):
    with store.session() as s:
        return s.run("""MATCH (x:Recommendation)-[:STATED_IN]->(:Chunk {doc_id: $d})
                        RETURN x.rec_id AS rec_id, x.text AS text, x.grade AS grade,
                               x.population AS population""", {"d": doc}).data()


def relink() -> int:
    """Recompute links from recommendations already in the graph (no Jev calls)."""
    store, emb = GraphStore(), Embedder()
    try:
        report = json.loads((ROOT / "runs" / "knowledge_pilot.json").read_text())
        recs = {d: load_recs(store, d) for d in (WHO16, WHO18, CS21)}
        with store.session() as s:
            s.run("MATCH ()-[r:REFINES]->() DELETE r")
            s.run("MATCH (:Recommendation)-[r:SUPERSEDES]->(:Recommendation) DELETE r")
        population, paired = refinement_population(emb, recs[CS21], recs[WHO18])
        link_refines(store, CS21, WHO18, population)
        report["refines"] = {"specific": CS21, "general": WHO18, "population": population}
        report["refines_pair"] = paired
        print(f"  REFINES {CS21} -> {WHO18}: population = {' '.join((population or '-').split())!r}  {paired}")
        pairs = link_editions(store, emb, WHO18, WHO16)
        changed = [p for p in pairs if p["grade_changed"]]
        report["editions"] = {"pairs": len(pairs), "grade_changed": changed,
                              "graded_2018": sum(1 for r in recs[WHO18] if r["grade"]),
                              "graded_2016": sum(1 for r in recs[WHO16] if r["grade"])}
        print(f"  editions: {len(pairs)} graded recommendations paired 2018 -> 2016, {len(changed)} changed grade")
        for p in changed:
            print(f"     '{p['grade_from']}'  ->  '{p['grade_to']}'  (cos {p['cos']})")
        (ROOT / "runs" / "knowledge_pilot.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        store.close()


def cdc(publish: bool = False, relink: bool = False) -> int:
    """CDC SSI 1999 -> 2017, a PARTIAL update: pair recommendations, record what
    changed, and (with --publish) retire only the 1999 passages whose graded
    recommendations were all replaced."""
    s = load()
    jev, store, emb = JevClient(s.jev_api_key, s.jev_model), GraphStore(), Embedder()
    out = ROOT / "runs" / "knowledge_cdc.json"
    try:
        report = {"documents": {}}
        if relink:
            with store.session() as se:
                se.run("""MATCH (:Recommendation)-[e:SUPERSEDES {partial: true}]->(:Recommendation) DELETE e""")
            pairs = link_partial(store, emb, jev, CDC17, CDC99)
            prev = json.loads(out.read_text()) if out.exists() else {}
            report = {"documents": prev.get("documents", {}), "pairs": pairs}
        elif not publish:
            by_doc = chunks_for([CDC99, CDC17])
            with store.session() as se:
                se.run("""MATCH (x:Recommendation)-[:STATED_IN]->(c:Chunk) WHERE c.doc_id IN $d
                          DETACH DELETE x""", {"d": [CDC99, CDC17]})
            for d, chunks in by_doc.items():
                recs, candidates = extract_document(jev, chunks)
                write_recommendations(store, recs)
                report["documents"][d] = {"chunks": len(chunks), "candidates": candidates,
                                          "recommendations": len(recs),
                                          "graded": sum(1 for r in recs if r.get("grade"))}
                print(f"  {d}: {len(chunks)} chunks -> {candidates} candidates -> {len(recs)} "
                      f"recommendations ({report['documents'][d]['graded']} graded)")
            pairs = link_partial(store, emb, jev, CDC17, CDC99)
            report["pairs"] = pairs
        else:
            report = json.loads(out.read_text())
            pairs = report["pairs"]

        with store.session() as se:
            graded_by_chunk = {}
            for r in se.run("""MATCH (x:Recommendation)-[:STATED_IN]->(c:Chunk {doc_id: $d})
                               WHERE x.grade IS NOT NULL
                               RETURN c.chunk_id AS chunk, collect(x.rec_id) AS recs""", {"d": CDC99}):
                graded_by_chunk[r["chunk"]] = set(r["recs"])
        replaced = {p["old"] for p in pairs}
        retire = sorted(c for c, recs in graded_by_chunk.items() if recs and recs <= replaced)
        mixed = sorted(c for c, recs in graded_by_chunk.items() if recs & replaced and not recs <= replaced)
        report["retire_chunks"], report["mixed_chunks_kept"] = retire, mixed
        print(f"  pairs 2017 -> 1999: {len(pairs)}; grade changed {sum(p['grade_changed'] for p in pairs)}, "
              f"quantities changed {sum(p['quantities_changed'] for p in pairs)}")
        print(f"  1999 graded recommendations: {sum(len(v) for v in graded_by_chunk.values())}; "
              f"replaced: {len(replaced)}; still in force: {sum(len(v) for v in graded_by_chunk.values()) - len(replaced)}")
        print(f"  1999 passages to retire: {len(retire)}; kept because they also state a "
              f"still-current recommendation: {len(mixed)}")
        if publish:
            mapping = {p["old_chunk"]: p["new_chunk"] for p in pairs if p["old_chunk"] in retire}
            seq = store.retire_passages(retire, mapping,
                                        note="CDC SSI 2017 partially updates 1999: replaced recommendations retired")
            report["release"] = seq
            print(f"  published release {seq}: {len(retire)} passages retired")
        report["jev_calls"] = jev.calls
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  report -> {out}")
        return 0
    finally:
        store.close()


def pilot() -> int:
    s = load()
    jev, store, emb = JevClient(s.jev_api_key, s.jev_model), GraphStore(), Embedder()
    report = {"documents": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        by_doc = chunks_for([WHO16, WHO18, CS21])
        all_recs = {}
        for d, chunks in by_doc.items():
            t = time.perf_counter()
            recs, candidates = extract_document(jev, chunks)
            n = write_recommendations(store, recs)
            all_recs[d] = recs
            report["documents"][d] = {
                "chunks": len(chunks), "candidates": candidates, "recommendations": len(recs),
                "with_grade": sum(1 for r in recs if r.get("grade")),
                "with_population": sum(1 for r in recs if r.get("population")),
                "seconds": round(time.perf_counter() - t, 1), "written": n}
            print(f"  {d}: {len(chunks)} chunks -> {candidates} candidates -> {len(recs)} recommendations "
                  f"({report['documents'][d]['with_grade']} graded, "
                  f"{report['documents'][d]['with_population']} with population)")

        # refinement: population comes from the 2021 recommendation itself
        # The population must be a position in the 2021 document. No fallback
        # string: if extraction finds none, the link is written without one and
        # population-aware ranking cannot use it (reported, not papered over).
        population, paired = refinement_population(emb, all_recs[CS21], all_recs[WHO18])
        link_refines(store, CS21, WHO18, population)
        report["refines_pair"] = paired
        report["refines"] = {"specific": CS21, "general": WHO18, "population": population}
        print(f"  REFINES {CS21} -> {WHO18}: population = {population!r}")

        pairs = link_editions(store, emb, WHO18, WHO16)
        changed = [p for p in pairs if p["grade_changed"]]
        report["editions"] = {"pairs": len(pairs), "grade_changed": changed,
                              "recs_2018": len(all_recs[WHO18]), "recs_2016": len(all_recs[WHO16])}
        print(f"  editions: {len(pairs)} recommendations paired 2018 -> 2016, "
              f"{len(changed)} with a changed grade")
        for p in changed:
            print(f"     {p['old']} '{p['grade_from']}'  ->  {p['new']} '{p['grade_to']}'  (cos {p['cos']})")
        report["jev_calls"] = jev.calls
        report["recommendations"] = {d: [{k: r.get(k) for k in ("rec_id", "text", "grade", "population")}
                                         for r in recs] for d, recs in all_recs.items()}
        out = ROOT / "runs" / "knowledge_pilot.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  report -> {out}   ({jev.calls} Jev calls)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd == ["pilot"]:
        sys.exit(pilot())
    if cmd == ["relink"]:
        sys.exit(relink())
    if cmd == ["cdc"]:
        sys.exit(cdc())
    if cmd == ["cdc", "--relink"]:
        sys.exit(cdc(relink=True))
    if cmd == ["cdc", "--publish"]:
        sys.exit(cdc(publish=True))
    print(__doc__)
    sys.exit(2)
