"""Step 3: repair, by failure type, then the fallback ladder.

    synthesized -> repaired (at most ONE rewrite) -> verbatim source -> withdrawn

Rules (design doc, repair loop):
* Wrong citation -> re-cite in code, using Jev's choice of supporting passage.
* A quantity / strength / negation mismatch -> never rewritten: replaced by
  the source's own sentence (verbatim), because these are the details a
  rewriter is most likely to get wrong again.
* Partly supported -> ONE rewrite. The rewriter is told why it failed and
  shown the passage, never Jev's score, so it cannot tune wording to pass.
* Supported by no retrieved passage -> withdrawn as "not stated". Never
  rewritten: asking a model to fix an unsupported claim invites it to make the
  claim sound supported.
Every failed version is kept in claim.history for the audit log.
"""

from __future__ import annotations

from typing import List

from span_select.candidates import enumerate_spans, resolve

from ..types import Check, Claim, Source
from .verify import verify_claim

REWRITE = """This sentence failed verification against its source: "{text}"
Reason: {reason}

Source passage:
{passage}

Rewrite the sentence so that it states only what the passage says, for the population the passage names. Copy numbers, units and recommendation-strength words exactly. Return only the sentence."""

PICK = "Which sentence of the passage states the information in `claim`?"
ANY_SENTENCE = "Does any option state the information in `claim`?"


def _snapshot_history(claim: Claim, reason: str) -> None:
    claim.history.append({"text": claim.text, "origin": claim.origin, "sources": claim.sources,
                          "checks": [vars(c) for c in claim.checks], "reason": reason})


def verbatim(jev, claim: Claim, source: Source) -> bool:
    """Replace the claim with the source sentence that states it. True if done."""
    spans = [s for s in enumerate_spans(source.chunk_id, source.text, max_options=0,
                                        soft_wraps=True) if s.kind == "segment"]
    if not spans:
        return False
    options = {f"o{i}": " ".join(s.text.split())[:500] for i, s in enumerate(spans)}
    ans = jev.ask({"claim": claim.text, "options": options},
                  {"pick": jev.choice_q(PICK, options), "fits": jev.noul_q(ANY_SENTENCE)})
    pick, probs = ans.choice("pick")
    fits = ans.noul("fits") or 0.0
    if pick not in options or fits < 0.5:
        return False
    i = int(pick[1:])
    span = spans[i]
    resolve(span, source.text)                        # the read-back invariant
    start, end = span.start, span.end
    # Keep the grade with its recommendation. The segmenter splits "...risk of
    # SSI." from "(Conditional recommendation, ...)", and quoting the sentence
    # alone would drop exactly the strength a repaired claim most often got
    # wrong. Extending over the next segment keeps one contiguous range of the
    # source, so the read-back guarantee still holds.
    if i + 1 < len(spans):
        nxt = spans[i + 1]
        if nxt.text.lstrip().startswith("(") and "recommendation" in nxt.text.lower():
            end = nxt.end
    text = source.text[start:end]
    claim.text, claim.sources, claim.origin = " ".join(text.split()), [source.label], "verbatim"
    claim.span = {"chunk_id": span.chunk_id, "start": start, "end": end}
    claim.support = probs.get(pick)
    claim.checks = [Check("citation", True), Check("details", True, "verbatim source text"),
                    Check("support", True, f"verbatim; P(sentence)={fits:.2f}", score=fits)]
    return True


def repair(chat, jev, claim: Claim, result: dict, sources: List[Source],
           threshold: float) -> None:
    if claim.passed:
        return
    by_label = {s.label: s for s in sources}
    failed = {c.name: c for c in claim.checks if not c.passed}

    if "current" in failed and result.get("replaced"):
        # Never rewrite towards old guidance: quote the recommendation that
        # replaced it, read back from its passage by position.
        x = result["replaced"]
        _snapshot_history(claim, f"rests on a recommendation replaced in {x['new_year']}")
        src = by_label.get(x.get("new_label", ""))
        if src is not None and x.get("new_start") is not None:
            text = src.text[x["new_start"]:x["new_end"]]
            claim.text, claim.sources, claim.origin = " ".join(text.split()), [src.label], "verbatim"
            claim.span = {"chunk_id": src.chunk_id, "start": x["new_start"], "end": x["new_end"]}
            claim.checks = [Check("citation", True), Check("details", True, "verbatim source text"),
                            Check("current", True, f"replacement ({x['new_year']}) quoted"),
                            Check("support", True, "verbatim")]
            claim.support = None
        else:
            claim.origin, claim.text = "withdrawn", ""
        return

    if result["any"] < threshold:
        _snapshot_history(claim, "no retrieved passage supports it")
        claim.origin, claim.text = "withdrawn", ""
        return

    best = by_label.get(result["best"]) or (by_label.get(claim.sources[0]) if claim.sources else None)

    if "citation" in failed and best is not None and set(failed) <= {"citation", "support"}:
        _snapshot_history(claim, "re-cited to the passage Jev found supporting")
        claim.sources = [best.label]
        r = verify_claim(jev, claim, sources, threshold)
        claim.checks, claim.support = r["checks"], r["support"]
        if claim.passed:
            claim.origin = "repaired"
            return
        failed = {c.name: c for c in claim.checks if not c.passed}

    if "details" in failed or "citation" in failed:
        _snapshot_history(claim, failed.get("details", failed.get("citation")).detail)
        if best is None or not verbatim(jev, claim, best):
            claim.origin, claim.text = "withdrawn", ""
        return

    # support only: one rewrite, then re-check from scratch
    _snapshot_history(claim, "not fully supported by the cited passage")
    source = by_label.get(claim.sources[0]) if claim.sources else best
    if source is None:
        claim.origin, claim.text = "withdrawn", ""
        return
    reason = ("states more than the passage does (population, scope or certainty)")
    rewritten = chat.complete(REWRITE.format(text=claim.text, reason=reason,
                                             passage=" ".join(source.text.split())[:1200]),
                              max_tokens=3000).text.strip().strip('"')
    if rewritten:
        claim.text, claim.sources = " ".join(rewritten.split()), [source.label]
        r = verify_claim(jev, claim, sources, threshold)
        claim.checks, claim.support = r["checks"], r["support"]
        if claim.passed:
            claim.origin = "repaired"
            return
        _snapshot_history(claim, "rewrite still failed verification")
    if not verbatim(jev, claim, source):
        claim.origin, claim.text = "withdrawn", ""
