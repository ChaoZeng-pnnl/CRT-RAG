"""Supplementary Figure S3 data: consistency while guidance is updated.

Replays the real release history on a FRESH, throwaway database (all
documents staged, nothing live): release 1 (base corpus incl. WHO SSI 2016),
release 2 (WHO SSI 2018 published, 2016 retired) and release 3 (WHO 2021
caesarean antisepsis), each published while four workers search
continuously. Every search records when it ran, which release it read, and
whether its results mix releases.

    NEO4J_URI=bolt://localhost:7688 PYTHONPATH=. python -m verified_rag.consistency_replay
        ->  runs/consistency_replay.json

Refuses to run unless the database is at release 0 (never touches a live one).
"""
import json
import threading
import time

from corpus_index.embedder import Embedder
from corpus_index.graph import GraphStore

from .config import ROOT

OLD, NEW, CS = "e7ad0c398aa2", "7c3781feaf22", "f3e61e207cc2"
QUERIES = ["should intubated adults receive 80% inspired oxygen FiO2 during surgery to reduce surgical site infection",
           "skin preparation antiseptic before emergency caesarean section"]
WORKERS, BEFORE, BETWEEN, AFTER = 4, 4.0, 4.0, 4.0


def main():
    store, emb = GraphStore(), Embedder()
    if store.current_release() != 0:
        raise SystemExit("refusing: database already has releases (use a fresh, staged-only database)")
    staged = store.staged_documents()
    for q in QUERIES:                       # warm the query-embedding cache
        emb.embed_one(q)
    log, events, stop = [], [], threading.Event()
    T0 = time.perf_counter()

    def publish(docs, sup, note):
        t = time.perf_counter()
        seq = store.publish(docs, supersedes=sup, note=note)
        events.append({"release": seq, "start": t - T0, "end": time.perf_counter() - T0, "note": note,
                       "documents": len(docs)})

    publish([d for d in staged if d not in (NEW, CS)], [], "base corpus incl. WHO SSI 2016")

    def worker(wid):
        i = 0
        while not stop.is_set():
            q = QUERIES[i % 2]; i += 1
            t0 = time.perf_counter()
            rows = store.search_text(q, emb, limit=10, use_cache=False)
            t1 = time.perf_counter()
            if not rows:
                continue
            snap = rows[0]["snapshot"]
            docs = {r["doc_id"] for r in rows}
            mixed = (snap < 2 and NEW in docs) or (snap >= 2 and OLD in docs) or (snap < 3 and CS in docs)
            log.append({"worker": wid, "query": i % 2, "start": t0 - T0, "end": t1 - T0, "snapshot": snap,
                        "old": OLD in docs, "new": NEW in docs, "cs": CS in docs, "mixed": mixed,
                        "same_snapshot_all_rows": len({r["snapshot"] for r in rows}) == 1})

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    time.sleep(BEFORE)
    publish([NEW], [(NEW, OLD)], "WHO SSI 2nd edition supersedes 2016")
    time.sleep(BETWEEN)
    publish([CS], [], "WHO caesarean antisepsis 2021 (refines SSI 2018)")
    time.sleep(AFTER)
    stop.set()
    for t in threads:
        t.join()
    out = {"workers": WORKERS, "queries": QUERIES, "events": events, "searches": len(log),
           "by_release": {s: sum(x["snapshot"] == s for x in log) for s in sorted({x["snapshot"] for x in log})},
           "mixed": sum(x["mixed"] for x in log),
           "inconsistent_rows": sum(not x["same_snapshot_all_rows"] for x in log),
           "log": log}
    (ROOT / "runs/consistency_replay.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=1))
    store.close()


if __name__ == "__main__":
    main()
