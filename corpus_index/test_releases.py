"""Replay test for releases and snapshot isolation.

Requires a graph in which every document is STAGED and nothing is published:
load into an empty database with `python -m corpus_index --stage-only`.
Run: PYTHONPATH=. python corpus_index/test_releases.py

Replays three releases: (1) base corpus incl. WHO SSI 2016 (e7ad0c398aa2);
(2) WHO SSI 2018 (7c3781feaf22) superseding 2016, published WHILE searches run;
(3) WHO caesarean antisepsis 2021 (f3e61e207cc2), a refinement.

First run 2026-11-20: 21 of 22 checks passed. The failure is a real retrieval
gap, not a release bug: after the caesarean-only 2021 guideline went live, a
hysterectomy question lost WHO SSI 2018 from its top 8. That check tests the
search layer alone, which has no notion of population, so it still fails here
by design; population-aware ranking in verified_rag.retrieval fixes it at the
answer layer (design doc §21).
"""
import random
import statistics
import string
import threading
import time

from corpus_index.embedder import Embedder
from corpus_index.graph import VISIBLE_AT_SNAPSHOT, GraphStore, PublishRefused

OLD, NEW, CS = "e7ad0c398aa2", "7c3781feaf22", "f3e61e207cc2"
Q_OXY = "should intubated adults receive 80% inspired oxygen FiO2 during surgery to reduce surgical site infection"
FAIL = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


store, emb = GraphStore(), Embedder()


def visible(doc, seq):
    with store.session() as s:
        return s.run("MATCH (node:Chunk {doc_id: $doc}) WITH node, 0 AS score "
                     + VISIBLE_AT_SNAPSHOT + " RETURN count(node) AS n",
                     {"doc": doc, "snapshot": seq, "include_duplicates": False}).single()["n"]


print("== before any release")
staged = store.staged_documents()
check("all documents staged, release 0", store.current_release() == 0 and len(staged) == 239,
      f"release {store.current_release()}, staged {len(staged)}")
check("search returns nothing while staged", store.search_text(Q_OXY, emb, use_cache=False) == [])

print("== refusals change nothing")
for args, label in ((dict(doc_ids=["nope000000"]), "unknown document"),
                    (dict(doc_ids=[NEW], supersedes=[(NEW, OLD)]), "retiring a document that is not live")):
    try:
        store.publish(**args)
        check(f"refuses {label}", False, "it published")
    except PublishRefused:
        check(f"refuses {label}", store.current_release() == 0 and len(store.staged_documents()) == 239)

print("== release 1: base corpus")
r1 = store.publish([d for d in staged if d not in (NEW, CS)], note="base corpus incl. WHO SSI 2016")
check("release 1 created", r1 == 1)
check("2016 fully visible at r1 (incl. 420 chunks flagged duplicates of the unpublished 2018)",
      visible(OLD, 1) == 495, visible(OLD, 1))
check("2018 invisible at r1 (staged)", visible(NEW, 1) == 0)
res = store.search_text(Q_OXY, emb, limit=5, use_cache=False)
top = res[0]
check("r1 oxygen answer comes from the 2016 edition", top["doc_id"] == OLD, top["doc_id"])
check("r1 passage says Strong", any("Strong recommendation" in r["text"] and "80%" in r["text"]
                                    for r in res if r["doc_id"] == OLD))

print("== release 2 published while searches run")
seen, stop = [], threading.Event()


def worker():
    while not stop.is_set():
        rows = store.search_text(Q_OXY, emb, limit=10, use_cache=False)
        if rows:
            seen.append((rows[0]["snapshot"], {r["doc_id"] for r in rows}))


threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads:
    t.start()
time.sleep(3)
r2 = store.publish([NEW], supersedes=[(NEW, OLD)], note="WHO SSI 2nd edition supersedes 2016")
time.sleep(3)
stop.set()
for t in threads:
    t.join()
at1 = [d for s, d in seen if s == 1]
at2 = [d for s, d in seen if s == 2]
mixed = [d for s, d in seen if (s == 1 and NEW in d) or (s == 2 and OLD in d)]
check("searches ran on both sides of the publish", len(at1) > 0 and len(at2) > 0,
      f"{len(at1)} at r1, {len(at2)} at r2")
check("no search mixed releases", not mixed, f"{len(mixed)} of {len(seen)} inconsistent")

print("== release 2 contents")
check("2016 invisible at r2 (superseded)", visible(OLD, 2) == 0, visible(OLD, 2))
check("2018 visible at r2", visible(NEW, 2) > 500, visible(NEW, 2))
res2 = store.search_text(Q_OXY, emb, limit=5, use_cache=False)
check("r2 oxygen answer comes from the 2018 edition", res2[0]["doc_id"] == NEW, res2[0]["doc_id"])
check("r2 passage says Conditional", any("Conditional recommendation" in r["text"] and "80%" in r["text"]
                                         for r in res2 if r["doc_id"] == NEW))
check("2016 no longer returned at all", all(r["doc_id"] != OLD for r in res2))
past = store.search_text(Q_OXY, emb, limit=5, snapshot=1, use_cache=False)
check("audit replay: searching as of r1 still returns the 2016 'Strong' passage",
      past[0]["doc_id"] == OLD and past[0]["snapshot"] == 1)

print("== cache keyed by release")
store.cache_hits = store.cache_misses = 0
store.search_text(Q_OXY, emb, limit=5)
store.search_text(Q_OXY, emb, limit=5)
check("second identical query is a cache hit", store.cache_hits == 1 and store.cache_misses == 1,
      f"hits {store.cache_hits} misses {store.cache_misses}")
r3 = store.publish([CS], note="WHO caesarean antisepsis 2021 (refines SSI 2018)")
store.search_text(Q_OXY, emb, limit=5)
check("after release 3 the same query misses (stale entry not served)", store.cache_misses == 2,
      f"misses {store.cache_misses}")

print("== release 3: refinement does not retire anything")
check("2018 still visible at r3", visible(NEW, 3) == visible(NEW, 2))
q_cs = "skin preparation antiseptic before emergency caesarean section"
rows = store.search_text(q_cs, emb, limit=8, use_cache=False)
check("caesarean question reaches the 2021 guideline", any(r["doc_id"] == CS for r in rows))
q_gen = "skin preparation antiseptic before hysterectomy surgical site"
rows = store.search_text(q_gen, emb, limit=8, use_cache=False)
check("general question still reaches the SSI 2018 guideline", any(r["doc_id"] == NEW for r in rows))
check("staged documents left: none", store.staged_documents() == [])

print("== latency: overlapped vs sequential")
def fresh(q):
    return q + " " + "".join(random.choice(string.ascii_lowercase) for _ in range(6))
seq_t, ovl_t, embed_t = [], [], []
for base in ("catheter removal after laparoscopic hysterectomy", "postoperative nausea prophylaxis",
             "early mobilisation after gynaecological surgery", "chlorhexidine vaginal preparation",
             "venous thromboembolism prophylaxis duration", "postpartum haemorrhage blood loss threshold"):
    q = fresh(base)
    t = time.perf_counter(); v = emb.embed_one(q); te = time.perf_counter() - t
    store.search(q, v, limit=8); seq_t.append(time.perf_counter() - t); embed_t.append(te)
    q = fresh(base)
    t = time.perf_counter(); store.search_text(q, emb, limit=8, use_cache=False); ovl_t.append(time.perf_counter() - t)
print(f"  embed alone        median {statistics.median(embed_t):.3f}s")
print(f"  sequential total   median {statistics.median(seq_t):.3f}s  (search share {statistics.median(seq_t)-statistics.median(embed_t):.3f}s)")
print(f"  overlapped total   median {statistics.median(ovl_t):.3f}s")
t = time.perf_counter(); store.search_text(Q_OXY, emb, limit=5); hit = time.perf_counter() - t
print(f"  cache hit          {hit*1000:.1f} ms")

print("\nreleases:")
for r in store.releases():
    print(f"  {r['seq']}  +{r['published']} -{r['retired']}  {r['note']}")
store.close()
print("\nFAILED: %d" % len(FAIL) if FAIL else "\nALL PASSED")
