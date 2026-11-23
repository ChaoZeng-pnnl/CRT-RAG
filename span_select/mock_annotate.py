"""MOCK gold spans from a large chat model, in the annotation sheet's format.

    python3 -m span_select.mock_annotate --model claude-opus-5-project

Purpose: exercise the scorer, the calibration analysis and the figures end to
end before nurse annotations exist. It is NOT gold:

* Every row carries annotator "MOCK-<model>" and mock=true. score_spans prints
  a MOCK banner when it sees them; figures built from it must say MOCK.
* Numbers computed against it measure agreement with that model, not clinical
  accuracy. When the selector being scored is the same model family, the
  agreement is inflated by construction.

The model sees exactly what the nurse sees: the question and the same
candidate ballot (span_select.annotate.pools), never the reference answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .annotate import FIELDS, load, pools
from .selector import JevStyleSelector

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """You are an experienced perioperative nurse annotating evidence for a study.

Question a nurse asked: {question}

Below is a numbered list of passages from one guideline chunk. Decide which passages provide {need}.

{options}

Instructions:
- verdict "present" if at least one passage states it; "absent" if none does (do not stretch to a near-match); "unclear" if you cannot tell.
- If present: "best" is the single best passage number; "acceptable" lists EVERY passage number you would be content to show a nurse (include "best").
- Judge the question as asked, for a general patient. Passages may be line fragments; judge what they state.

Reply with ONLY this JSON, no other text:
{{"verdict": "present|absent|unclear", "best": <int or null>, "acceptable": [<ints>], "note": "<one short line>"}}"""


def ask(sel: JevStyleSelector, question: str, need: str, spans) -> dict:
    options = "\n".join("[%d] %s" % (i, " ".join(s.text.split())[:300]) for i, s in enumerate(spans))
    data = sel._post({"model": sel.model, "max_tokens": 4000, "temperature": 0,
                      "messages": [{"role": "user", "content": PROMPT.format(
                          question=question, need=need, options=options)}]})
    text = data["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group(0)) if m else {}
    n = len(spans)
    best = out.get("best") if isinstance(out.get("best"), int) and 0 <= out.get("best") < n else None
    acc = sorted({a for a in (out.get("acceptable") or []) if isinstance(a, int) and 0 <= a < n} |
                 ({best} if best is not None else set()))
    verdict = out.get("verdict") if out.get("verdict") in ("present", "absent", "unclear") else "unclear"
    if verdict == "present" and best is None:
        verdict = "unclear"
    return {"verdict": verdict, "best": best if verdict == "present" else None,
            "acceptable": acc if verdict == "present" else [], "note": str(out.get("note", ""))[:200],
            "raw_ok": bool(m)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="span_select.mock_annotate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "eval/annotation/study_manifest.json"))
    args = ap.parse_args(argv)
    cfg = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("="); cfg[k.strip()] = v.strip().strip('"').strip("'")
    sel = JevStyleSelector(cfg["OPENAI_BASE_URL"], cfg["OPENAI_API_KEY"], args.model, timeout=180)
    qids = json.loads(Path(args.manifest).read_text())["qids"]
    questions, chunks = load(qids=qids)
    jobs = []
    for q in questions:
        if q.get("excluded") or q["source_chunk_id"] not in chunks:
            continue
        pool = pools(chunks[q["source_chunk_id"]])
        for field, need in FIELDS:
            jobs.append((q, field, need, pool[field]))

    def run(job):
        q, field, need, spans = job
        row = {"qid": q["qid"], "chunk_id": q["source_chunk_id"], "field": field,
               "annotator": "MOCK-" + args.model, "mock": True, "n_candidates": len(spans),
               "reference_revealed_first": False}
        if not spans:
            row.update(verdict="", best=None, acceptable=[], note="no candidates")
            return row
        for attempt in range(3):
            try:
                row.update(ask(sel, q["question"], need, spans))
                return row
            except Exception as exc:                       # noqa: BLE001
                err = str(exc)[:120]; time.sleep(2 * (attempt + 1))
        row.update(verdict="unclear", best=None, acceptable=[], note="ERROR " + err)
        return row

    t = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool_:
        rows = list(pool_.map(run, jobs))
    out = ROOT / "eval/annotation/mock" / f"gold_spans_MOCK_{args.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print("decisions %d in %.0fs -> %s" % (len(rows), time.time() - t, out))
    print("verdicts", dict(Counter(r["verdict"] or "no candidates" for r in rows)))
    print("errors", sum(1 for r in rows if r["note"].startswith("ERROR")),
          "| unparsed", sum(1 for r in rows if r.get("raw_ok") is False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
