"""Retrieval accuracy: does each system's retrieval find the passage a question
was written from? Recall@k and MRR over the 290 source-linked test questions.

    PYTHONPATH=. python -m verified_rag.retrieval_eval   ->  runs/retrieval_eval.json

Read-only against the live release. A retrieved passage counts as the gold one
if it IS the source passage or a duplicate of it (either direction), so a
system is not penalised for returning the canonical copy of the same text.
Questions whose source passage is not live at the current release (retired
WHO 2016 text with no 2018 duplicate) are reported separately, since CVT-RAG
is meant not to return them. Microsoft GraphRAG retrieves community summaries,
not passages, so it has no passage recall.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from corpus_index.embedder import Embedder
from corpus_index.graph import GraphStore

from .baselines import CHEAP, StaticStore
from .clients.jev import JevClient
from .config import ROOT, load
from .retrieval import Retriever

K = 6
RETIRED_DOC = "e7ad0c398aa2"


def main():
    from rag_eval.pipelines import HybridTiered, KGTraversal
    s = load()
    store, emb = GraphStore(), Embedder()
    static = StaticStore(store)
    jev = JevClient(s.jev_api_key, s.jev_model)
    systems = {"hybrid_rag": HybridTiered("hybrid_rag", static, emb, model=CHEAP, top_k=K),
               "neo4j_graphrag": KGTraversal("neo4j_graphrag", static, emb, model=CHEAP, top_k=K)}
    cvt = Retriever(store, emb, jev, limit=K)
    dup = {}
    for l in open(ROOT / "chunks/chunks.jsonl"):
        r = json.loads(l)
        if r.get("duplicate_of"):
            dup[r["chunk_id"]] = r["duplicate_of"]
    rows = [json.loads(l) for l in open(ROOT / "eval/full_testset.jsonl")]
    qs = [r for r in rows if not r.get("excluded") and r.get("source_chunk_id")]

    def same(a, b):
        return a == b or dup.get(a) == b or dup.get(b) == a or (dup.get(a) and dup.get(a) == dup.get(b))

    def rank(ids, gold):
        for i, c in enumerate(ids, 1):
            if same(c, gold):
                return i
        return None

    def one(q):
        gold = q["source_chunk_id"]
        out = {"question": q["question"], "gold": gold,
               "gold_live": not (gold.startswith(RETIRED_DOC) and not dup.get(gold))}
        for name, p in systems.items():
            r = p.retrieve(q["question"])
            ids = [m.get("chunk_id") for m in r.meta if m.get("chunk_id")]
            out[name] = rank(ids[:K], gold)
            if name == "neo4j_graphrag":           # its writer receives up to 2K passages
                out["neo4j_graphrag_k12"] = rank(ids[:2 * K], gold)
        src, _, _ = cvt.retrieve(q["question"])
        out["ours"] = rank([x.chunk_id for x in src][:K], gold)
        return out
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(one, qs))

    def metrics(R, name):
        n = len(R)
        ranks = [r[name] for r in R]
        return {"n": n, **{f"recall@{k}": round(sum(1 for x in ranks if x and x <= k) / n, 3) for k in (1, 3, K)},
                "mrr@6": round(sum(1 / x for x in ranks if x) / n, 3)}
    live = [r for r in res if r["gold_live"]]
    import math
    # paired comparison CVT-RAG vs hybrid at k=6: exact McNemar on discordant questions
    b = sum(1 for r in res if r["hybrid_rag"] and r["hybrid_rag"] <= K and not (r["ours"] and r["ours"] <= K))
    c = sum(1 for r in res if r["ours"] and r["ours"] <= K and not (r["hybrid_rag"] and r["hybrid_rag"] <= K))
    n_d = b + c
    p_val = min(1.0, 2 * sum(math.comb(n_d, i) for i in range(0, min(b, c) + 1)) / 2 ** n_d) if n_d else 1.0
    def wilson(k, n, z=1.96):
        ph = k / n; d = 1 + z * z / n; ce = (ph + z * z / (2 * n)) / d
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return [round(ce - h, 3), round(ce + h, 3)]
    summary = {"k": K, "questions": len(res), "gold_live": len(live),
               "all": {n: metrics(res, n) for n in ("hybrid_rag", "neo4j_graphrag", "ours")},
               "live_gold_only": {n: metrics(live, n) for n in ("hybrid_rag", "neo4j_graphrag", "ours")},
               "neo4j_graphrag_recall@12": round(sum(1 for r in res if r["neo4j_graphrag_k12"]) / len(res), 3),
               "recall@6_ci": {n: wilson(sum(1 for r in res if r[n] and r[n] <= K), len(res))
                               for n in ("hybrid_rag", "neo4j_graphrag", "ours")},
               "ours_vs_hybrid_k6": {"hybrid_only": b, "ours_only": c, "mcnemar_exact_p": round(p_val, 3)},
               "seconds": round(time.time() - t0)}
    (ROOT / "runs/retrieval_eval.json").write_text(json.dumps({"summary": summary, "rows": res}, indent=1))
    print(json.dumps(summary, indent=1))
    store.close()


if __name__ == "__main__":
    main()
