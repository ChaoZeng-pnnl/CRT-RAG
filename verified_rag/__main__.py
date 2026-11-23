"""Command line.

    python -m verified_rag ask "should my intubated patient get 80% oxygen?"
    python -m verified_rag ask "..." --as-of 1          # as the system answered at release 1
    python -m verified_rag ask "..." --json
    python -m verified_rag feedback RESPONSE_ID 0 right  # nurse verdict on claim 0
"""

from __future__ import annotations

import argparse
import json
import sys

MARK = {"synthesized": "✓", "repaired": "✓ repaired", "verbatim": "❝ source text", "withdrawn": "✗ withdrawn"}


def render(r) -> str:
    out = [f"\nQ: {r.question}", f"   release {r.snapshot} · {r.status} · {r.timings.get('total_s', 0):.2f}s\n"]
    by = {s.label: s for s in r.sources}
    for i, c in enumerate(r.claims):
        if c.origin == "withdrawn":
            reason = c.history[-1]["reason"] if c.history else ""
            out.append(f"  [{i}] ✗ withdrawn: {reason}")
            continue
        refs = ", ".join(f"{by[l].title[:38]} {by[l].year} p.{by[l].page} T{by[l].tier}"
                         for l in c.sources if l in by)
        sup = f" · P={c.support:.2f}" if c.support is not None else ""
        out.append(f"  [{i}] {c.text}\n       {MARK.get(c.origin, c.origin)}{sup} · {refs}")
    if r.status == "not_stated":
        out.append("  The retrieved guidance does not state this. Searched: " +
                   "; ".join(sorted({f'{s.title[:40]} {s.year}' for s in r.sources})))
    sg = r.signals
    out += ["", f"  integrity  {sg.integrity}", f"  evidence   {sg.evidence}",
            f"  consensus  {sg.consensus}", f"  relevance  {sg.relevance}",
            f"  response   {r.response_id}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verified_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--as-of", type=int, default=None)
    a.add_argument("--json", action="store_true")
    f = sub.add_parser("feedback")
    f.add_argument("response_id")
    f.add_argument("claim_index", type=int)
    f.add_argument("verdict", choices=["right", "wrong", "not_stated"])
    f.add_argument("--note", default="")
    args = ap.parse_args(argv)

    if args.cmd == "feedback":
        from .config import load
        from .store import Store
        Store(load().audit_db).feedback(args.response_id, args.claim_index, args.verdict, args.note)
        print("recorded")
        return 0

    from .pipeline import Pipeline
    pipe = Pipeline()
    try:
        r = pipe.answer(args.question, snapshot=args.as_of)
        print(json.dumps(r.to_dict(), indent=2, default=str) if args.json else render(r))
    finally:
        pipe.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
