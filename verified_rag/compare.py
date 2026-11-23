"""Systems A / B / C on the nurse's needs (paper plan: comparison).

    python -m verified_rag.compare questions   # write change-sensitive questions
    python -m verified_rag.compare run         # run A, B, C; score; runs/comparison.json

A: conventional RAG (one static index, no checks)
B: A + versioned releases and graph knowledge of change (no sentence checks)
C: full system (B + four checks and repair)
Same retrieval code, same writer model, same questions. Every measure is
mechanical and applied identically to all three systems.
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer.details import check_details, rests_on_replaced
from .answer.verify import verify_claim
from .config import ROOT

OUT = ROOT / "eval" / "comparison"
QFILE = OUT / "change_questions.jsonl"

QPROMPT = """A recommendation in perioperative nursing guidance was replaced.

Older recommendation: "{old}"
Newer recommendation: "{new}"

Write 3 different, realistic questions a perioperative nurse might ask at work whose correct answer depends on which of these is current. Do not mention years, editions, guideline names or that anything changed. One question per line, no numbering."""


def change_events(store):
    """Every recommendation-level replacement in the graph whose newer side is live."""
    with store.session() as s:
        rows = s.run("""
            MATCH (n:Recommendation)-[e:SUPERSEDES]->(o:Recommendation)
            MATCH (n)-[:STATED_IN]->(nc:Chunk), (o)-[:STATED_IN]->(oc:Chunk)
            WHERE e.partial = true OR e.grade_changed = true
            RETURN n.rec_id AS new_rec, o.rec_id AS old_rec, n.text AS new_text, o.text AS old_text,
                   nc.chunk_id AS new_chunk, oc.chunk_id AS old_chunk,
                   nc.doc_id AS new_doc, oc.doc_id AS old_doc,
                   nc.publication_year AS new_year, oc.publication_year AS old_year
            ORDER BY new_rec""").data()
    return rows


def write_questions() -> int:
    from corpus_index.graph import GraphStore
    from .clients.chat import ChatClient
    from .config import load
    s = load()
    chat = ChatClient(s.chat_base_url, s.chat_api_key, s.chat_model)
    store = GraphStore()
    try:
        events = change_events(store)
    finally:
        store.close()
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with QFILE.open("w") as fh:
        for ev in events:
            text = chat.complete(QPROMPT.format(old=" ".join(ev["old_text"].split()),
                                                new=" ".join(ev["new_text"].split())), max_tokens=4000).text
            qs = [q.strip(" -•\t") for q in text.splitlines() if q.strip() and "?" in q][:3]
            for q in qs:
                fh.write(json.dumps({"question": q, **ev, "writer": chat.model}) + "\n")
                n += 1
            print(f"  {ev['old_year']} -> {ev['new_year']}: {len(qs)} questions | {' '.join(ev['new_text'].split())[:70]}")
    print(f"  {n} questions -> {QFILE}")
    return 0


def general_questions():
    manifest = json.loads((ROOT / "eval/annotation/study_manifest.json").read_text())
    rows = [json.loads(l) for l in (ROOT / "eval/full_testset.jsonl").open()]
    return [{"question": rows[int(q[1:])]["question"], "qid": q} for q in manifest["qids"]
            if not rows[int(q[1:])].get("excluded")]


def score_answer(pipe, resp, event=None, superseded_docs=frozenset()):
    """Measures for one answer, identical for A, B and C."""
    by = {s.label: s for s in resp.sources}
    shown = [c for c in resp.claims if c.origin != "withdrawn" and c.text]
    out = {"n_shown": len(shown), "detail_fail": 0, "support_fail": 0,
           "uses_superseded": False, "states_current": False, "cites_superseded_doc": False}
    for c in shown:
        cited = [by[l] for l in c.sources if l in by]
        text = "\n".join(s.text for s in cited)
        if text and not check_details(c.text, text).passed:
            out["detail_fail"] += 1
        if any(s.doc_id in superseded_docs for s in cited):
            out["cites_superseded_doc"] = True
        if event is not None:
            if rests_on_replaced(c.text, [event]):
                out["uses_superseded"] = True
            if rests_on_replaced(c.text, [{"old_text": event["new_text"], "new_text": event["old_text"]}]):
                out["states_current"] = True
    if out["cites_superseded_doc"]:
        out["uses_superseded"] = True
    # verifier support on every displayed sentence (same instrument for all modes)
    for c in shown:
        if c.origin == "verbatim":
            continue
        r = verify_claim(pipe.jev, c, resp.sources, pipe.settings.support_threshold)
        if r["support"] < pipe.settings.support_threshold:
            out["support_fail"] += 1
    return out


def run() -> int:
    from .pipeline import Pipeline
    pipe = Pipeline()
    with pipe.store.session() as s:
        superseded = frozenset(r["d"] for r in s.run(
            "MATCH (d:Document) WHERE d.superseded_in IS NOT NULL RETURN d.doc_id AS d"))
    change = [json.loads(l) for l in QFILE.open()]
    general = general_questions()
    jobs = [("change", q, m) for q in change for m in "ABC"] + [("general", q, m) for q in general for m in "ABC"]
    started = time.time()

    def one(job):
        kind, q, mode = job
        for attempt in range(3):
            try:
                resp = pipe.answer(q["question"], mode=mode)
                sc = score_answer(pipe, resp, q if kind == "change" else None, superseded)
                return {"kind": kind, "mode": mode, "question": q["question"],
                        "event": q.get("new_rec"), "status": resp.status,
                        "seconds": resp.timings.get("total_s"), **sc,
                        "claims": [{"origin": c.origin, "text": c.text,
                                    "sources": [s.chunk_id for s in resp.sources if s.label in c.sources]}
                                   for c in resp.claims]}
            except Exception as exc:                  # noqa: BLE001
                err = str(exc)[:160]; time.sleep(3 * (attempt + 1))
        return {"kind": kind, "mode": mode, "question": q["question"], "error": err}

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(one, jobs))
    pipe.store.close()

    summary = {}
    for kind in ("change", "general"):
        for mode in "ABC":
            R = [r for r in rows if r["kind"] == kind and r["mode"] == mode and "error" not in r]
            shown = sum(r["n_shown"] for r in R)
            summary[f"{kind}:{mode}"] = {
                "answers": len(R), "errors": sum(1 for r in rows if r["kind"] == kind and r["mode"] == mode and "error" in r),
                "not_stated": sum(r["status"] == "not_stated" for r in R),
                "uses_superseded": sum(r["uses_superseded"] for r in R),
                "states_current": sum(r["states_current"] for r in R),
                "sentences_shown": shown,
                "detail_mismatch": sum(r["detail_fail"] for r in R),
                "verifier_unsupported": sum(r["support_fail"] for r in R),
                "median_seconds": sorted(r["seconds"] for r in R)[len(R) // 2] if R else None}
    out = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - started),
           "writer": pipe.chat.model, "superseded_docs": sorted(superseded),
           "summary": summary, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "comparison.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1:]
    sys.exit(write_questions() if cmd == ["questions"] else run() if cmd == ["run"] else (print(__doc__) or 2))
