"""Find recommendation statements in a document and fill their slots.

1. candidates   segments mentioning recommend/suggest (fixed rule, no model)
2. is it one?   Jev noul: a formal recommendation, not rationale or evidence?
3. grade        the "(Strong|Conditional recommendation, ... evidence)" that
                follows it, by exact pattern; else none
4. population   Jev choice over the sentence's clauses + noul "is any of these
                the population?"; else not stated
Output: dicts with chunk-relative offsets for every slot.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from span_select.candidates import enumerate_spans, resolve

from .clauses import clauses

_CANDIDATE = re.compile(r"\b(recommend(s|ed)?|suggest(s|ed)?)\b", re.I)
_GRADE = re.compile(r"\(\s*((?:strong|conditional)\s+recommendation[^)]{0,120})\)", re.I)
# CDC formats, read only from immediately AFTER the sentence (never inside it,
# where CDC defines its categories: "Category IA and IB recommendations are ..."):
#   1999  "... before the operation. Category IB"
#   2017  "... (Category IA–strong recommendation; high-quality evidence.)"
_CDC_GRADE = re.compile(r"\A[\s.;]{0,6}\(?\s*(Category\s+(?:I[ABC]|II)\b(?:[–-][^)\n]{0,90})?)\)?")

IS_REC = ("Is `sentence` a formal guideline recommendation statement telling clinicians "
          "what to do (not background, rationale, a summary of evidence, or a description "
          "of the guideline process)?")
POP_ANY = "Does any of `clauses` state which patients this recommendation applies to?"
POP_PICK = "Which clause states which patients `sentence` applies to?"


def _graded_after(text: str, end: int) -> bool:
    after = text[end:min(len(text), end + 140)]
    return bool(_CDC_GRADE.search(after)) or bool(re.match(
        r"\A[\s.;]{0,6}\(\s*(?:strong|conditional)\s+recommendation", after, re.I))


def _one(jev, chunk: dict, span, spans) -> dict | None:
    text = chunk["text"]
    sentence = resolve(span, text)
    ans = jev.ask({"sentence": " ".join(sentence.split())}, {"rec": jev.noul_q(IS_REC)})
    p_rec = ans.noul("rec") or 0.0
    if p_rec < 0.5:
        return None
    rec = {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "start": span.start,
           "end": span.end, "text": sentence, "p_rec": p_rec}
    # grade: the parenthetical right after the sentence (same or next segment)
    tail = text[span.start:min(len(text), span.end + 200)]
    g = _GRADE.search(tail)
    if g:
        rec.update(grade=g.group(1).strip(), grade_start=span.start + g.start(1),
                   grade_end=span.start + g.start(1) + len(g.group(1).strip()))
    else:
        after = text[span.end:min(len(text), span.end + 140)]
        c = _CDC_GRADE.search(after)
        if c:
            gtxt = c.group(1).strip().rstrip(".; ")
            rec.update(grade=gtxt, grade_start=span.end + c.start(1),
                       grade_end=span.end + c.start(1) + len(gtxt))
    cl = clauses(text, span.start, span.end)
    if cl:
        options = {f"c{i}": " ".join(c[2].split()) for i, c in enumerate(cl)}
        a2 = jev.ask({"sentence": " ".join(sentence.split()), "clauses": options},
                     {"pick": jev.choice_q(POP_PICK, options), "any": jev.noul_q(POP_ANY)})
        pick, probs = a2.choice("pick")
        if pick in options and (a2.noul("any") or 0.0) >= 0.5:
            a, b, t = cl[int(pick[1:])]
            assert text[a:b] == t                          # read-back
            rec.update(population=t, pop_start=a, pop_end=b, p_pop=probs.get(pick))
    return rec


def extract_document(jev, chunks: List[dict]) -> tuple:
    jobs = []
    for ch in chunks:
        spans = [s for s in enumerate_spans(ch["chunk_id"], ch["text"], max_options=0,
                                            soft_wraps=True) if s.kind == "segment"]
        for s in spans:
            # Two ways to be a candidate: the sentence says "recommend"/"suggest"
            # (WHO style: "The panel recommends ..."), or a grade marker follows
            # it immediately (CDC style: imperative "Administer ... Category IB").
            if len(s.text) > 40 and (_CANDIDATE.search(s.text) or _graded_after(ch["text"], s.end)):
                jobs.append((ch, s, spans))
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(lambda j: _one(jev, *j), jobs))
    recs = [r for r in found if r]
    # A recommendation stated twice (executive summary and body) is one
    # recommendation. Keep the occurrence that carries the grade: the summary
    # copy usually has none, and keeping it would lose exactly the strength
    # that changes between editions.
    seen: Dict[str, dict] = {}
    for r in recs:
        key = " ".join(r["text"].split()).lower()
        if key not in seen or (r.get("grade") and not seen[key].get("grade")):
            seen[key] = r
    out = list(seen.values())
    for r in out:
        r["formal"] = bool(r.get("grade")) or r["text"].lstrip().lower().startswith("recommendation")
    for i, r in enumerate(out):
        r["rec_id"] = f"{r['doc_id']}:R{i:03d}"
    return out, len(jobs)
