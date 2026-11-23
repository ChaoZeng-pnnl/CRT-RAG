"""Step 4: the response object and its four separate signals."""

from __future__ import annotations

from typing import Callable, List, Optional

from ..types import Claim, Response, Signals, Source


def assemble(question: str, snapshot: int, claims: List[Claim], sources: List[Source],
             calibrate: Optional[Callable] = None) -> Response:
    shown = [c for c in claims if c.origin != "withdrawn" and c.text]
    withdrawn = [c for c in claims if c.origin == "withdrawn"]
    status = ("not_stated" if not shown else "partial" if withdrawn else "answered")

    by_label = {s.label: s for s in sources}
    cited = [by_label[l] for c in shown for l in c.sources if l in by_label]

    origins = {o: sum(c.origin == o for c in shown) for o in ("synthesized", "repaired", "verbatim", "unchecked")}
    integrity = ("verified" if shown and all(c.passed for c in shown) else "none" if not shown else "partial")
    integrity += " (" + ", ".join(f"{n} {o}" for o, n in origins.items() if n) + \
                 (f", {len(withdrawn)} withdrawn" if withdrawn else "") + ")"

    if cited:
        tiers = sorted({s.tier for s in cited})
        years = sorted({s.year for s in cited if s.year})
        notes = sorted({s.superseded_note for s in cited if s.superseded_note})
        evidence = f"Tier {'/'.join(map(str, tiers))}; {', '.join(years)}; current at release {snapshot}"
        graded = sorted({f"{s.grade} ({s.title[:30]} {s.year} p.{s.page})" +
                         (f", {s.grade_change}" if s.grade_change else "")
                         for s in cited if s.grade})
        if graded:
            evidence += "; grade: " + "; ".join(graded)
        if notes:
            evidence += "; " + "; ".join(notes)
    else:
        evidence = "no source cited"

    refined = sorted({f"{s.title} ({s.year}) applies to {s.refines}" for s in cited if s.refines})
    docs = sorted({s.doc_id for s in cited})
    consensus = ("; ".join(refined) + "; " if refined else "") + \
                (f"{len(docs)} source document{'s' if len(docs) != 1 else ''} cited" if docs else "no sources")
    consensus += "; agreement between sources not yet assessed"   # hubs not built

    scored = [c for c in shown if c.support is not None]
    if scored:
        weakest = min(scored, key=lambda c: c.support)
        cal = calibrate(weakest.support, weakest.origin) if calibrate else None
        relevance = (f"{cal[0]:.0%} of similar claims accepted by nurses (n={cal[1]})" if cal
                     else f"not yet calibrated (weakest claim raw P={weakest.support:.2f})")
        for c in scored:
            got = calibrate(c.support, c.origin) if calibrate else None
            c.calibrated = got[0] if got else None
    else:
        relevance = "no claims to score"

    return Response(question=question, snapshot=snapshot, status=status, claims=claims,
                    sources=sources, signals=Signals(integrity, evidence, consensus, relevance))
