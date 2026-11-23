"""Run the full pipeline over a fixed question set and save every measurement.

    python -m verified_rag.bench                 # the 60-question study sample
    python -m verified_rag.bench --limit 10

Writes runs/bench.json: per question, the stage timings, status, and for each
claim its origin, checks and Jev support. The figures read this file. Nothing
here judges whether an answer is clinically right; that needs the nurse
annotations. It measures what the pipeline did.
"""

from __future__ import annotations

import argparse
import json
import time

from .config import ROOT
from .pipeline import Pipeline


def questions(limit=None):
    manifest = json.loads((ROOT / "eval" / "annotation" / "study_manifest.json").read_text())
    rows = [json.loads(l) for l in (ROOT / "eval" / "full_testset.jsonl").open()]
    out = []
    for qid in manifest["qids"]:
        r = rows[int(qid[1:])]
        if not r.get("excluded"):
            out.append((qid, r["question"], r.get("question_type", "")))
    return out[:limit] if limit else out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verified_rag.bench")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    pipe = Pipeline()
    results = []
    started = time.time()
    try:
        for i, (qid, q, qtype) in enumerate(questions(args.limit), 1):
            r = pipe.answer(q)
            results.append({
                "qid": qid, "question_type": qtype, "status": r.status, "snapshot": r.snapshot,
                "timings": r.timings, "n_sources": len(r.sources),
                "claims": [{"origin": c.origin, "support": c.support,
                            "failed_checks": [h_c["name"] for h in c.history[:1]
                                              for h_c in h["checks"] if not h_c["passed"]],
                            "repair_reason": c.history[0]["reason"] if c.history else ""}
                           for c in r.claims],
                "signals": vars(r.signals)})
            print(f"  {i:>2} {qid} {r.status:<10} {r.timings.get('total_s', 0):5.2f}s  "
                  + " ".join(c.origin[0] for c in r.claims))
    finally:
        pipe.store.close()
    out = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - started, 1),
           "chat_model": pipe.chat.model, "jev_versions": pipe.jev.versions,
           "questions": results}
    path = ROOT / "runs" / "bench.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n  {len(results)} questions -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
