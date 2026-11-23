"""Step 2: three checks per claim.

  1. citation  (code)  every cited label exists, and at least one is cited
  2. details   (code)  quantities, strength words and negations match the source
  3. support   (Jev)   P(the cited passage supports the claim)

One Jev request per claim carries three questions: support by the cited
passage, support by ANY retrieved passage (so "nothing supports this" is
distinguishable from "wrong citation"), and which passage supports it best (so
a wrong citation can be repaired in code). Claims are checked in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from ..types import Check, Claim, Source
from .details import check_details, rests_on_replaced

SUPPORT = ("Does `cited_passage` fully support `claim`, including every number, unit, "
           "population and recommendation strength it states?")
ANY = "Does at least one passage in `passages` fully support `claim`?"
BEST = "Which passage best supports `claim`?"


def cited_text(claim: Claim, by_label: Dict[str, Source]) -> str:
    return "\n".join(by_label[l].text for l in claim.sources if l in by_label)


def verify_claim(jev, claim: Claim, sources: List[Source], threshold: float) -> dict:
    by_label = {s.label: s for s in sources}
    checks = []
    valid = [l for l in claim.sources if l in by_label]
    checks.append(Check("citation", bool(valid) and len(valid) == len(claim.sources),
                        "" if valid and len(valid) == len(claim.sources)
                        else f"cites {claim.sources}; available {sorted(by_label)}"))
    text = cited_text(claim, by_label)
    checks.append(check_details(claim.text, text) if text
                  else Check("details", False, "no valid cited passage"))
    # check 4 (code): does the sentence rest on a recommendation that a newer,
    # live document has replaced?
    replaced = [x for l in valid for x in by_label[l].replaced]
    old = rests_on_replaced(claim.text, replaced)
    checks.append(Check("current", old is None,
                        "" if old is None else f"rests on a recommendation replaced in {old['new_year']}"))
    state = {"claim": claim.text, "cited_passage": text or "(none)",
             "passages": {s.label: " ".join(s.text.split())[:1500] for s in sources}}
    ans = jev.ask(state, {
        "support": jev.noul_q(SUPPORT),
        "any": jev.noul_q(ANY),
        "best": jev.choice_q(BEST, {s.label: " ".join(s.text.split())[:600] for s in sources}),
    })
    p = ans.noul("support") or 0.0
    checks.append(Check("support", p >= threshold, f"P(supported)={p:.2f}", score=p))
    best, probs = ans.choice("best")
    return {"checks": checks, "support": p, "any": ans.noul("any") or 0.0,
            "best": best, "best_p": probs.get(best) if best else None,
            "jev_s": ans.seconds, "replaced": old}


def verify_all(jev, claims: List[Claim], sources: List[Source], threshold: float) -> List[dict]:
    if not claims:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(claims))) as pool:
        results = list(pool.map(lambda c: verify_claim(jev, c, sources, threshold), claims))
    for claim, r in zip(claims, results):
        claim.checks, claim.support = r["checks"], r["support"]
    return results
