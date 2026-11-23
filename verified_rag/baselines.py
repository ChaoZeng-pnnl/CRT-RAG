"""Four systems, same writer, same questions, same sentence-level scoring.

    python -m verified_rag.baselines run      # -> runs/baselines.json

  hybrid_rag      vector + keyword search, tier weighting, static index
  neo4j_graphrag  hybrid seeds + entity-relation graph traversal (static)
  graphrag        map-reduce over community summaries (top-20 by similarity)
  ours            versioned releases, currency, four checks, repair

Every sentence a system shows is scored the same way: current (rests on a
replaced recommendation, or cites a superseded document version), exact details
against what it cites, verifier support, and traceability (does it cite a
specific guideline passage). Seconds and model calls are recorded per answer.
For ours, the writer's sentences BEFORE the checks are scored too.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer.details import check_details, rests_on_replaced
from .config import ROOT, load

CHEAP = "gemini-3.7-flash-project"
# Two phrasings of the same support question. Calibrated on controls
# (runs/support_calibration.json): both reject unrelated / number-changed /
# negated claims equally, but the `cited` phrasing also rejects 15% of TRUE
# word-for-word sentences (vs 7% for `cited_passage`). The `cited_passage`
# phrasing is the verifier's own, so its numbers are reported as primary and
# the stricter `cited` phrasing alongside as a cross-check.
from .answer.verify import SUPPORT as SUPPORT_PRIMARY
SUPPORT = ("Does `cited` fully support `claim`, including every number, unit, population "
           "and recommendation strength it states?")
CTX_CHARS = 12000


def _norm(t):
    return " ".join(re.sub(r"[^a-z0-9%]+", " ", t.lower()).split())
_CIT = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_LOCK = threading.Lock()


class StaticStore:
    """A GraphStore view that ignores releases: how a conventional index behaves."""
    def __init__(self, store):
        self._s = store

    def search(self, *a, **kw):
        kw.setdefault("versioned", False)
        return self._s.search(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._s, name)


_META = re.compile(r"\b(?:provided (?:context|information|passages?|material)|the context|"
                   r"context (?:does|do) not|(?:does|do) not (?:contain|provide|mention|specify)|"
                   r"based on the (?:provided|given|available))", re.I)


def is_claim(s: str) -> bool:
    """A clinical statement, not a heading or a remark about the context. The
    same filter applies to every system; excluded sentences are counted."""
    words = s.split()
    if _META.search(s) or len(words) < 6:
        return False
    if not re.search(r"[.)%]\s*$", s) and len(words) < 10:
        return False          # a heading or label
    return True


def sentences(answer: str):
    text = re.sub(r"[*_#>`]+", " ", answer)
    out = []
    for line in text.splitlines():
        line = line.strip(" -•\t")
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", line):
            s = s.strip()
            if len(s) >= 25 and not s.endswith(":"):
                out.append(s)
    return out


def cited_indices(sentence: str):
    idx = set()
    for m in _CIT.finditer(sentence):
        for part in re.split(r"\s*,\s*", m.group(1)):
            if re.search(r"[–-]", part):
                a, b = re.split(r"\s*[–-]\s*", part)[:2]
                idx.update(range(int(a), int(b) + 1))
            elif part.strip().isdigit():
                idx.add(int(part))
    return idx


def score_sentences(jev, sents, event, superseded):
    """sents: [{text, cited_text, cited_docs, traceable}] -> measures."""
    excluded = [s for s in sents if not is_claim(s["text"])]
    sents = [s for s in sents if is_claim(s["text"])]
    m = {"n": len(sents), "excluded_non_claims": len(excluded), "traceable": 0, "detail_mismatch": 0,
         "unsupported": 0, "unsupported_strict": 0, "copies": 0, "copies_rejected": 0, "uncheckable": 0, "uses_superseded": False, "states_current": False,
         "per_sentence": []}
    jobs = []
    for s in sents:
        m["traceable"] += bool(s["traceable"])
        det = check_details(s["text"], s["cited_text"]) if s["cited_text"] else None
        if det is not None and not det.passed:
            m["detail_mismatch"] += 1
        s["_detail"] = None if det is None else det.detail
        if any(d in superseded for d in s["cited_docs"]):
            m["uses_superseded"] = True
        if event is not None:
            if rests_on_replaced(s["text"], [event]):
                m["uses_superseded"] = True
            if rests_on_replaced(s["text"], [{"old_text": event["new_text"], "new_text": event["old_text"]}]):
                m["states_current"] = True
        jobs.append(s)

    def sup(s):
        # No guideline text to check against (e.g. a sentence built from
        # model-written community summaries): counted as uncheckable, not as
        # unsupported, so the two failures are not conflated.
        if not s["cited_text"]:
            return None
        ctx = s["cited_text"][:CTX_CHARS]
        p = jev.ask({"claim": s["text"], "cited_passage": ctx},
                    {"s": jev.noul_q(SUPPORT_PRIMARY)}).noul("s") or 0.0
        q = jev.ask({"claim": s["text"], "cited": ctx},
                    {"s": jev.noul_q(SUPPORT)}).noul("s") or 0.0
        return p, q
    with ThreadPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(sup, jobs))
    m["unsupported"] = sum(1 for r in res if r is not None and r[0] < 0.5)
    m["unsupported_strict"] = sum(1 for r in res if r is not None and r[1] < 0.5)
    m["uncheckable"] = sum(r is None for r in res)
    per = []
    for s, r in zip(jobs, res):
        # positive control inside the real answers: the sentence is a
        # word-for-word copy of its cited text, so it IS supported.
        copy = bool(s["cited_text"]) and len(_norm(s["text"]).split()) >= 6 and _norm(s["text"]) in _norm(s["cited_text"])
        if copy and r is not None:
            m["copies"] += 1
            m["copies_rejected"] += r[0] < 0.5
        per.append({"text": s["text"], "origin": s.get("origin"), "traceable": s["traceable"],
                    "cited_chunks": s.get("cited_chunks", []), "detail": s.get("_detail"),
                    "copy_of_source": copy,
                    "p_support": None if r is None else round(r[0], 3),
                    "p_support_strict": None if r is None else round(r[1], 3)})
    m["per_sentence"] = per
    return m


def prose_sentences(answer, contexts, meta, offset):
    """Map a prose answer's [n] citations to passages. contexts[0] may be a
    non-passage block (relations, summaries): `offset` says how many."""
    out = []
    for s in sentences(answer):
        idx = cited_indices(s)
        texts, docs, passage = [], [], False
        for i in idx:
            k = i - 1
            if 0 <= k < len(contexts):
                texts.append(contexts[k])
                mk = k - offset
                if 0 <= mk < len(meta) and meta[mk].get("chunk_id"):
                    docs.append(meta[mk]["chunk_id"].split(":")[0])
                    passage = True
        if not texts:     # uncited sentence: lenient, check against everything retrieved
            texts = list(contexts)
        chunks = [meta[i - 1 - offset]["chunk_id"] for i in idx
                  if 0 <= i - 1 - offset < len(meta) and meta[i - 1 - offset].get("chunk_id")]
        out.append({"text": _CIT.sub("", s).strip(), "cited_text": "\n".join(texts),
                    "cited_docs": docs, "traceable": passage, "cited_chunks": chunks})
    return out


def main(argv=None, limit=None) -> int:
    from corpus_index.embedder import Embedder
    from corpus_index.graph import GraphStore
    from rag_eval.global_controlled import graphrag, load_communities
    from rag_eval.pipelines import HybridTiered, KGTraversal
    from .clients.jev import JevClient
    from .compare import general_questions, QFILE
    from .pipeline import Pipeline

    s = load()
    store, emb = GraphStore(), Embedder()
    static = StaticStore(store)
    jev = JevClient(s.jev_api_key, s.jev_model)
    hybrid = HybridTiered("hybrid_rag", static, emb, model=CHEAP, top_k=6)
    neo = KGTraversal("neo4j_graphrag", static, emb, model=CHEAP, top_k=6)
    comms = load_communities(ROOT / "kg/communities.jsonl")
    ours = Pipeline()
    with store.session() as se:
        superseded = frozenset(r["d"] for r in se.run(
            "MATCH (d:Document) WHERE d.superseded_in IS NOT NULL RETURN d.doc_id AS d"))

    change = [json.loads(l) for l in QFILE.open()][:limit]
    general = general_questions()[:limit]
    qs = [("change", q) for q in change] + [("general", q) for q in general]
    graphrag_qs = {id(q) for k, q in qs if k == "change"} | {id(q) for q in general[:20]}

    def run_one(job):
        kind, q, system = job
        ev = q if kind == "change" else None
        t0 = time.perf_counter()
        try:
            if system in ("hybrid_rag", "neo4j_graphrag"):
                p = hybrid if system == "hybrid_rag" else neo
                r = p.answer(q["question"])
                offset = len(r["contexts"]) - len(r["meta"])
                sents = prose_sentences(r["answer"], r["contexts"], r["meta"], max(0, offset))
                calls, status = 1, ("not_stated" if not sents else "answered")
                raw = None
            elif system == "graphrag":
                ans, calls, st = graphrag(q["question"], comms, emb, top_communities=20, concurrency=6,
                                          map_model=CHEAP, reduce_model=CHEAP, map_max_tokens=3000)
                sents = [{"text": x, "cited_text": "", "cited_docs": [], "traceable": False}
                         for x in sentences(ans)]
                status = "invalid" if ans.startswith("__INVALID__") else ("not_stated" if not sents else "answered")
                raw = None
            else:
                before_jev, before_chat = jev.calls, 0
                resp = ours.answer(q["question"])
                by = {x.label: x for x in resp.sources}
                shown = [c for c in resp.claims if c.origin != "withdrawn" and c.text]
                sents = [{"text": c.text, "cited_text": "\n".join(by[l].text for l in c.sources if l in by),
                          "cited_docs": [by[l].doc_id for l in c.sources if l in by], "traceable": True,
                          "cited_chunks": [by[l].chunk_id for l in c.sources if l in by], "origin": c.origin}
                         for c in shown]
                raw = [{"text": (c.history[0]["text"] if c.history else c.text),
                        "cited_text": "\n".join(by[l].text for l in (c.history[0]["sources"] if c.history else c.sources) if l in by),
                        "cited_docs": [by[l].doc_id for l in (c.history[0]["sources"] if c.history else c.sources) if l in by],
                        "cited_chunks": [by[l].chunk_id for l in (c.history[0]["sources"] if c.history else c.sources) if l in by],
                        "traceable": True} for c in resp.claims]
                n_repair = sum(1 for c in resp.claims if c.origin == "repaired")
                calls = 1 + n_repair
                status = resp.status
            seconds = time.perf_counter() - t0
            m = score_sentences(jev, sents, ev, superseded)
            out = {"kind": kind, "system": system, "question": q["question"], "status": status,
                   "seconds": round(seconds, 2), "llm_calls": calls, **m,
                   "sentences": [x["text"] for x in sents]}
            if raw is not None:
                out["before_checks"] = score_sentences(jev, raw, ev, superseded)
                out["origins"] = [c.origin for c in resp.claims]
            return out
        except Exception as exc:                          # noqa: BLE001
            return {"kind": kind, "system": system, "question": q["question"], "error": str(exc)[:200]}

    jobs = [(k, q, sysname) for k, q in qs for sysname in ("hybrid_rag", "neo4j_graphrag", "ours")]
    jobs += [(k, q, "graphrag") for k, q in qs if id(q) in graphrag_qs]
    started = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(run_one, jobs))
    store.close(); ours.store.close()

    summary = {}
    for kind in ("change", "general"):
        for system in ("hybrid_rag", "neo4j_graphrag", "graphrag", "ours"):
            R = [r for r in rows if r["kind"] == kind and r["system"] == system and "error" not in r]
            if not R:
                continue
            sec = sorted(r["seconds"] for r in R)
            n = sum(r["n"] for r in R)
            d = {"answers": len(R),
                 "errors": sum(1 for r in rows if r["kind"] == kind and r["system"] == system and "error" in r),
                 "not_stated_or_invalid": sum(r["status"] in ("not_stated", "invalid") for r in R),
                 "uses_superseded": sum(r["uses_superseded"] for r in R),
                 "states_current": sum(r["states_current"] for r in R),
                 "sentences": n, "traceable": sum(r["traceable"] for r in R),
                 "detail_mismatch": sum(r["detail_mismatch"] for r in R),
                 "unsupported": sum(r["unsupported"] for r in R),
                 "unsupported_strict": sum(r["unsupported_strict"] for r in R),
                 "copies_of_source": sum(r["copies"] for r in R),
                 "copies_rejected": sum(r["copies_rejected"] for r in R),
                 "uncheckable": sum(r["uncheckable"] for r in R),
                 "excluded_non_claims": sum(r["excluded_non_claims"] for r in R),
                 "median_s": sec[len(sec) // 2], "p90_s": sec[int(0.9 * (len(sec) - 1))],
                 "mean_llm_calls": round(sum(r["llm_calls"] for r in R) / len(R), 1)}
            if system == "ours":
                verb = [x for r in R for x in r["per_sentence"] if x["origin"] == "verbatim" and x["p_support"] is not None]
                d["verifier_false_negative_floor"] = {
                    "verbatim_sentences": len(verb),
                    "rejected": sum(1 for x in verb if x["p_support"] < 0.5)}
                B = [r["before_checks"] for r in R]
                d["before_checks"] = {k: sum(b[k] for b in B) for k in
                                      ("n", "detail_mismatch", "unsupported", "unsupported_strict")}
                d["before_checks"]["uses_superseded"] = sum(b["uses_superseded"] for b in B)
            summary[f"{kind}:{system}"] = d
    out = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - started),
           "writer": CHEAP, "graphrag_questions": "15 change + first 20 general (cost)",
           "superseded_docs": sorted(superseded), "summary": summary, "rows": rows}
    name = "baselines.json" if limit is None else f"baselines_smoke{limit}.json"
    (ROOT / "runs" / name).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=1))
    return 0



LLM_ONLY_PROMPT = """You are assisting a perioperative gynecologic nurse.

Answer the question from your own knowledge. Be specific about doses, timings
and thresholds where they apply. Answer in short sentences.

Question: {question}

Answer:"""


def llm_only():
    """Ablation arm: the same writer with NO retrieval, on the same 15 change and
    58 general questions as run 3. Appends system 'llm_only' to runs/baselines.json
    without touching the other systems' rows."""
    from openai import OpenAI
    from rag_eval.pipelines import OPENAI_API_KEY, OPENAI_BASE_URL
    from .clients.jev import JevClient
    from .compare import general_questions, QFILE
    s = load()
    jev = JevClient(s.jev_api_key, s.jev_model)
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    path = ROOT / "runs/baselines.json"
    D = json.loads(path.read_text())
    superseded = frozenset(D["superseded_docs"])
    change = [json.loads(l) for l in QFILE.open()]
    qs = [("change", q) for q in change] + [("general", q) for q in general_questions()]

    def one(job):
        kind, q = job
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(model=CHEAP, temperature=0, max_tokens=4000,
                messages=[{"role": "user", "content": LLM_ONLY_PROMPT.format(question=q["question"])}])
            ans = (r.choices[0].message.content or "").strip()
            sents = [{"text": x, "cited_text": "", "cited_docs": [], "traceable": False} for x in sentences(ans)]
            m = score_sentences(jev, sents, q if kind == "change" else None, superseded)
            return {"kind": kind, "system": "llm_only", "question": q["question"],
                    "status": "answered" if sents else "not_stated", "seconds": round(time.perf_counter() - t0, 2),
                    "llm_calls": 1, **m, "sentences": [x["text"] for x in sents], "answer": ans}
        except Exception as exc:                          # noqa: BLE001
            return {"kind": kind, "system": "llm_only", "question": q["question"], "error": str(exc)[:200]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one, qs))
    D["rows"] = [r for r in D["rows"] if r.get("system") != "llm_only"] + rows
    for kind in ("change", "general"):
        R = [r for r in rows if r["kind"] == kind and "error" not in r]
        sec = sorted(r["seconds"] for r in R)
        D["summary"][f"{kind}:llm_only"] = {
            "answers": len(R), "errors": sum(1 for r in rows if r["kind"] == kind and "error" in r),
            "not_stated_or_invalid": sum(r["status"] != "answered" for r in R),
            "uses_superseded": sum(r["uses_superseded"] for r in R),
            "states_current": sum(r["states_current"] for r in R),
            "sentences": sum(r["n"] for r in R), "traceable": 0, "detail_mismatch": 0, "unsupported": 0,
            "unsupported_strict": 0, "uncheckable": sum(r["uncheckable"] for r in R),
            "excluded_non_claims": sum(r["excluded_non_claims"] for r in R),
            "median_s": sec[len(sec) // 2], "p90_s": sec[int(0.9 * (len(sec) - 1))], "mean_llm_calls": 1.0}
    D["llm_only_note"] = ("system 'llm_only' added 2026-11-23: same writer, temperature 0, max_tokens 4000, "
                          "no retrieval; prompt LLM_ONLY_PROMPT")
    path.write_text(json.dumps(D, indent=2, default=str))
    print(json.dumps({k: v for k, v in D["summary"].items() if "llm_only" in k}, indent=1))
    return 0

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["llm-only"]:
        sys.exit(llm_only())
    if a[:1] == ["run"]:
        sys.exit(main(limit=int(a[2]) if a[1:2] == ["--limit"] else None))
    print(__doc__); sys.exit(2)
