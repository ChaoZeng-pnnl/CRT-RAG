"""Ingestion timing: the work each system does when a new edition arrives.

    python -m verified_rag.timing ingest     # -> runs/timing_ingest.json

Event timed: WHO SSI second edition (2018), 508 passages, arriving while the
2016 edition is in the corpus. Every step is run for real on that document and
labelled "measured", except the two expensive graph steps, which are timed on
a sample and scaled up ("extrapolated", with the sample size and the
parallelism used). Nothing persists: the temporary Neo4j load is deleted and
the publish transaction is rolled back.

Which steps each system needs:
  hybrid RAG        extract+chunk, embed every passage, duplicates, load
  Neo4j GraphRAG    hybrid + LLM triple extraction per passage + load triples
  GraphRAG          Neo4j GraphRAG + community detection (whole graph)
                    + regenerate community summaries (whole graph)
  ours              extract+chunk, embed only changed passages, duplicates, load,
                    recommendation extraction (Jev), link editions, publish
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT, load

DOC = "7c3781feaf22"          # WHO SSI 2018
OLD = "e7ad0c398aa2"          # WHO SSI 2016
PDF = ROOT / "corpus/tier1/WHO/WHO_Global_SSI_Prevention_Guidelines_2nd_ed_2018.pdf"
CHEAP = "gemini-3.7-flash-project"


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def ingest() -> int:
    from corpus_dedup.__main__ import load_chunks as load_all
    from corpus_dedup.detect import cluster, elect
    from corpus_index.embedder import Embedder
    from corpus_index.graph import GraphStore
    from corpus_index.load import _embedding_input
    from corpus_ingest.chunker import chunk_sections
    from corpus_ingest.extractors import extract

    s = load()
    steps = {}
    store, emb = GraphStore(), Embedder()
    rows = [json.loads(l) for l in open(ROOT / "chunks/chunks.jsonl")]
    doc_rows = [r for r in rows if r["doc_id"] == DOC]
    old_texts = {" ".join(r["text"].split()) for r in rows if r["doc_id"] == OLD}
    changed = [r for r in doc_rows if " ".join(r["text"].split()) not in old_texts]

    # 1. extract + chunk
    chunks, t = timed(lambda: chunk_sections(extract(PDF, "PDF")))
    steps["extract_chunk"] = {"seconds": t, "kind": "measured", "passages": len(chunks)}

    # 2. embeddings, bypassing the cache: every passage vs changed passages only
    def embed_fresh(texts, batch=32):
        for i in range(0, len(texts), batch):
            emb._embed_batch(texts[i:i + batch])
    _, t = timed(lambda: embed_fresh([_embedding_input(r) for r in doc_rows]))
    steps["embed_all"] = {"seconds": t, "kind": "measured", "passages": len(doc_rows)}
    _, t = timed(lambda: embed_fresh([_embedding_input(r) for r in changed]))
    steps["embed_changed"] = {"seconds": t, "kind": "measured", "passages": len(changed)}

    # 3. duplicate detection over the whole corpus
    all_rows = load_all()
    _, t = timed(lambda: elect(all_rows, cluster(all_rows)))
    steps["duplicates"] = {"seconds": t, "kind": "measured", "passages": len(all_rows)}

    # 4. load into Neo4j (temporary copy, deleted afterwards)
    vecs = [emb._read_cache(_embedding_input(r)) for r in doc_rows]
    payload = [{"chunk_id": "TIMING_TMP:%04d" % i, "doc_id": "TIMING_TMP", "text": r["text"],
                "embedding": v, "seq": i} for i, (r, v) in enumerate(zip(doc_rows, vecs)) if v]

    def write():
        with store.session() as se:
            for i in range(0, len(payload), 200):
                se.run("""UNWIND $rows AS row
                          MERGE (c:Chunk {chunk_id: row.chunk_id})
                          SET c.text = row.text, c.doc_id = row.doc_id, c.seq = row.seq
                          WITH c, row CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
                          RETURN count(*)""", {"rows": payload[i:i + 200]}).consume()
    _, t = timed(write)
    with store.session() as se:
        se.run("MATCH (c:Chunk {doc_id: 'TIMING_TMP'}) DETACH DELETE c").consume()
    steps["load_neo4j"] = {"seconds": t, "kind": "measured", "passages": len(payload)}

    # 5. ours: recommendation extraction (Jev), no write
    from .clients.jev import JevClient
    from .knowledge.extract import extract_document
    jev = JevClient(s.jev_api_key, s.jev_model)
    (recs, cands), t = timed(lambda: extract_document(jev, doc_rows))
    steps["ours_extract_recommendations"] = {"seconds": t, "kind": "measured",
                                             "candidates": cands, "recommendations": len(recs),
                                             "jev_calls": jev.calls}

    # 6. ours: link editions (idempotent MERGE of the same pairs)
    from .knowledge.graph import link_editions
    pairs, t = timed(lambda: link_editions(store, emb, DOC, OLD))
    steps["ours_link_editions"] = {"seconds": t, "kind": "measured", "pairs": len(pairs)}

    # 7. ours: publish, inside a transaction that is rolled back
    def publish_rollback():
        with store.session() as se:
            tx = se.begin_transaction()
            seq = tx.run("MATCH (r:Release) RETURN coalesce(max(r.seq),0)+1 AS s").single()["s"]
            # the same pre-publication checks GraphStore.publish runs
            tx.run("""MATCH (d:Document {doc_id: $doc})
                      OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                      RETURN count(c) AS n, count(c.embedding) AS e, count(c.is_canonical) AS f""",
                   {"doc": DOC}).consume()
            tx.run("CREATE (:Release {seq: $s, at: datetime(), note: 'timing, rolled back'})", {"s": seq}).consume()
            tx.run("MATCH (d:Document {doc_id: $d}) SET d.live_from = $s", {"d": DOC, "s": seq}).consume()
            tx.run("MATCH (d:Document {doc_id: $d}) SET d.superseded_in = $s", {"d": OLD, "s": seq}).consume()
            tx.rollback()
    _, t = timed(publish_rollback)
    steps["ours_publish"] = {"seconds": t, "kind": "measured (rolled back)"}

    # 8. Neo4j GraphRAG / GraphRAG: LLM triple extraction, sample -> 508 passages
    import clinical_kg.extract as kx
    sample = doc_rows[:: max(1, len(doc_rows) // 24)][:24]
    stats = {"errors": 0, "proposed": 0, "kept": 0, "dropped": 0, "dropped_reasons": {}}
    stats = __import__("collections").defaultdict(int, stats)
    per = []

    def one(r):
        t0 = time.perf_counter()
        out = kx.extract_one(r, stats, model=CHEAP)
        per.append(time.perf_counter() - t0)
        return out
    out, t = timed(lambda: list(ThreadPoolExecutor(max_workers=8).map(one, sample)))
    triples = sum(len(x) for x in out)
    steps["kg_extract_triples"] = {
        "seconds": t / len(sample) * len(doc_rows), "kind": "extrapolated",
        "sample": len(sample), "sample_seconds": t, "mean_call_s": statistics.mean(per),
        "concurrency": 8, "calls": len(doc_rows), "triples_per_passage": triples / len(sample),
        "model": CHEAP}

    # 9. GraphRAG: community detection over the whole graph (measured)
    import clinical_kg.community as kc
    edges, _ = timed(lambda: kc.load_edges(store))
    labels, t = timed(lambda: kc.detect_communities(edges))
    groups = kc.group_edges(edges, labels)
    big = list(groups.items())          # group_edges already applies the minimum size
    steps["graphrag_detect_communities"] = {"seconds": t, "kind": "measured",
                                            "edges": len(edges), "communities": len(big)}

    # 10. GraphRAG: regenerate community summaries, sample -> all communities
    sample_c = big[:8]
    per = []

    def summ(item):
        t0 = time.perf_counter()
        kc.summarise_community(item[0], item[1], model=CHEAP)
        per.append(time.perf_counter() - t0)
    _, t = timed(lambda: list(ThreadPoolExecutor(max_workers=6).map(summ, sample_c)))
    steps["graphrag_summaries"] = {
        "seconds": t / len(sample_c) * len(big), "kind": "extrapolated",
        "sample": len(sample_c), "sample_seconds": t, "mean_call_s": statistics.mean(per),
        "concurrency": 6, "calls": len(big), "model": CHEAP}
    store.close()

    S = lambda *k: sum(steps[x]["seconds"] for x in k)
    systems = {
        "hybrid_rag": {"steps": ["extract_chunk", "embed_all", "duplicates", "load_neo4j"]},
        "neo4j_graphrag": {"steps": ["extract_chunk", "embed_all", "duplicates", "load_neo4j",
                                     "kg_extract_triples"]},
        "graphrag": {"steps": ["extract_chunk", "embed_all", "duplicates", "load_neo4j",
                               "kg_extract_triples", "graphrag_detect_communities", "graphrag_summaries"]},
        "ours": {"steps": ["extract_chunk", "embed_changed", "duplicates", "load_neo4j",
                           "ours_extract_recommendations", "ours_link_editions", "ours_publish"]},
    }
    for v in systems.values():
        v["seconds"] = round(S(*v["steps"]), 1)
        v["llm_calls"] = sum(steps[x].get("calls", 0) for x in v["steps"] if x.startswith(("kg_", "graphrag_s")))
        v["jev_calls"] = sum(steps[x].get("jev_calls", 0) for x in v["steps"])
    out = {"event": "WHO SSI 2018 second edition arrives (508 passages; 2016 edition in corpus)",
           "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": steps, "systems": systems,
           "notes": ["Embedding times bypass the cache.",
                     "Extrapolated steps scale the sample's wall time at the stated concurrency.",
                     "Hybrid RAG and the graph systems have no versioning: after ingestion the 2016 "
                     "edition stays searchable beside 2018."]}
    path = ROOT / "runs/timing_ingest.json"
    path.write_text(json.dumps(out, indent=2))
    for k, v in systems.items():
        print(f"  {k:<15} {v['seconds']:>8.1f} s   LLM calls {v['llm_calls']:>4}   Jev calls {v['jev_calls']}")
    print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(ingest() if sys.argv[1:] == ["ingest"] else (print(__doc__) or 2))
