"""Calibrate the Jev support question on controls, two phrasings.

Controls are built from real corpus passages: a true sentence of the passage
(must be supported), an unrelated sentence from another passage, the true
sentence with a number changed, and with a negation added (must not be).
Every case's score is saved, so the figure can show distributions.

    PYTHONPATH=. python -m verified_rag.support_calibration  ->  runs/support_calibration.json
"""
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from . import baselines as B
from .answer import verify as V
from .clients.jev import JevClient
from .config import ROOT, load

SKIP_DOCS = {"e7ad0c398aa2"}          # retired WHO 2016 edition: not live at the current release
NUM = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def sents(t):
    return [x for x in re.split(r"(?<=[.!?])\s+", " ".join(t.split())) if 12 <= len(x.split()) <= 45]


def build(seed=7, n_passages=400, max_cases=320):
    rng = random.Random(seed)
    rows = [r["text"] for r in map(json.loads, open(ROOT / "chunks/chunks.jsonl"))
            if r.get("is_canonical") and r["doc_id"] not in SKIP_DOCS and len(r["text"]) > 700]
    rng.shuffle(rows)
    cases = []
    for i, t in enumerate(rows[:n_passages]):
        ss = sents(t)
        if not ss:
            continue
        c = rng.choice(ss)
        other = sents(rows[(i + 200) % len(rows)])
        cases.append(("verbatim", 1, c, t))
        if other:
            cases.append(("unrelated", 0, rng.choice(other), t))
        m = NUM.search(c)
        if m and m.group(1) not in ("19", "20"):
            v = m.group(1)
            nv = str(int(float(v)) * 3 + 7) if "." not in v else str(round(float(v) * 3 + 0.7, 2))
            cases.append(("number_changed", 0, c[:m.start()] + nv + c[m.end():], t))
        neg = re.sub(r"\b(is|are|should)\b", lambda k: k.group(1) + " not", c, count=1)
        if neg != c:
            cases.append(("negated", 0, neg, t))
        if len(cases) >= max_cases:
            break
    return cases


def main():
    s = load()
    jev = JevClient(s.jev_api_key, s.jev_model)
    cases = build()

    def go(x):
        kind, y, claim, t = x
        a = jev.ask({"claim": claim, "cited": t[:B.CTX_CHARS]}, {"s": jev.noul_q(B.SUPPORT)}).noul("s") or 0.0
        b = jev.ask({"claim": claim, "cited_passage": t[:B.CTX_CHARS]}, {"s": jev.noul_q(V.SUPPORT)}).noul("s") or 0.0
        return {"kind": kind, "label": y, "p_cited": a, "p_cited_passage": b}
    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(go, cases))
    summary = {}
    for form in ("p_cited", "p_cited_passage"):
        d = {}
        for k in ("verbatim", "unrelated", "number_changed", "negated"):
            v = sorted(r[form] for r in res if r["kind"] == k)
            ok = [(x >= 0.5) == (k == "verbatim") for x in v]
            d[k] = {"n": len(v), "correct_at_0.5": round(sum(ok) / len(v), 3), "median_p": round(v[len(v) // 2], 2)}
        pos = [r[form] for r in res if r["label"] == 1]
        neg = [r[form] for r in res if r["label"] == 0]
        d["auc"] = round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)
        summary[form] = d
    out = {"cases": len(res), "source": "chunks/chunks.jsonl (canonical, live documents), seed 7",
           "results": {"scorer_cited": summary["p_cited"], "verifier_cited_passage": summary["p_cited_passage"]},
           "per_case": res}
    (ROOT / "runs/support_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
