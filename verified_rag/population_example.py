"""Supplementary Figure S9 data: guidance for the right patient.

For a caesarean and a hysterectomy question, record which documents fill the
passage slots (a) with search alone and (b) with CVT-RAG's retrieval
(population-aware ranking, at most 3 passages per document, and the general
guidance a refinement narrows). Read-only against the live release.

    PYTHONPATH=. python -m verified_rag.population_example   ->  runs/population_example.json
"""
import json

from corpus_index.embedder import Embedder
from corpus_index.graph import GraphStore

from .clients.jev import JevClient
from .config import ROOT, load
from .retrieval import Retriever

QUESTIONS = {
    "caesarean": "What skin antiseptic should I use before an emergency caesarean section?",
    "hysterectomy": "What skin antiseptic should I use before an abdominal hysterectomy?",
}


def main():
    s = load()
    store, emb = GraphStore(), Embedder()
    jev = JevClient(s.jev_api_key, s.jev_model)
    ret = Retriever(store, emb, jev, limit=s.retrieve_limit)
    titles = {}
    with store.session() as se:
        for r in se.run("MATCH (d:Document) RETURN d.doc_id AS d, d.title AS t, d.year AS y"):
            titles[r["d"]] = (r["t"], r["y"])
    out = {"release": store.current_release(), "limit": s.retrieve_limit, "questions": {}}
    for key, q in QUESTIONS.items():
        rows = store.search_text(q, emb, limit=s.retrieve_limit, use_cache=False)
        sources, seq, timings = ret.retrieve(q)
        out["questions"][key] = {
            "question": q,
            "search_only": [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"]} for r in rows[:s.retrieve_limit]],
            "cvt_rag": [{"doc_id": x.doc_id, "chunk_id": x.chunk_id, "refines": x.refines} for x in sources],
        }
    out["titles"] = {d: {"title": t, "year": y} for d, (t, y) in titles.items()
                     if any(d in json.dumps(v) for v in out["questions"].values())}
    (ROOT / "runs/population_example.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: [x["doc_id"] for x in v["search_only"]] + ["|"] + [x["doc_id"] for x in v["cvt_rag"]]
                      for k, v in out["questions"].items()}, indent=1))
    store.close()


if __name__ == "__main__":
    main()
