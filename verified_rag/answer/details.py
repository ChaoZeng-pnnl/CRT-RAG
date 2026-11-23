"""Check 2: the exact-detail check. Pure code, no model.

Every quantity, percentage, recommendation-strength word and negation in a
generated claim must also appear in the passage it cites. These are the errors
that matter most clinically and that a semantic verifier is least reliable on:
"strong" vs "conditional", 24 h vs 72 h, "are" vs "are not" recommended.

Comparison is on normalized text (span_select.candidates.normalize: NFKC,
whitespace, soft hyphens) plus a small, explicit unit-synonym table. Nothing
here is fuzzy: a detail is present or it is not.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from span_select.candidates import enumerate_spans, normalize

from ..types import Check

_UNIT = {
    "hours": "h", "hour": "h", "hrs": "h", "hr": "h", "h": "h",
    "minutes": "min", "minute": "min", "mins": "min", "min": "min",
    "days": "d", "day": "d", "weeks": "wk", "week": "wk", "months": "mo", "month": "mo",
    "mg": "mg", "mcg": "mcg", "µg": "mcg", "μg": "mcg", "micrograms": "mcg", "microgram": "mcg",
    "g": "g", "kg": "kg", "ml": "ml", "l": "l", "units": "units", "unit": "units",
    "iu": "iu", "mmhg": "mmhg", "%": "%", "°c": "c", "degrees": "c",
}
_NUM = r"\d+(?:[.,]\d+)?"
_THOUSANDS = re.compile(r"(?<=\d)[,\u00a0\u2009](?=\d{3}(?!\d))")
_QTY = re.compile(r"(" + _NUM + r")(?:\s*[-–to]+\s*(" + _NUM + r"))?\s*(%|°c|[a-zµμ]+)?", re.I)
_STRONG = re.compile(r"\bstrong(?:ly)?\b|\bmust\b", re.I)
_CONDITIONAL = re.compile(r"\bconditional(?:ly)?\b|\bsuggest(?:s|ed)?\b|\bconsider(?:ed)?\b", re.I)
_NEGATION = re.compile(r"\bnot\b|\bno\b|\bnever\b|\bavoid(?:ed)?\b|\bcontraindicated\b|\bagainst\b", re.I)
# A source that says "not recommended" / "should not" while the claim
# recommends the same thing: the dropped negation, the most dangerous error of all.
_NEGATED_ADVICE = re.compile(r"\b(?:not|no longer)\s+(?:be\s+)?(?:recommended|advised|indicated|used)\b"
                             r"|\bshould\s+not\b|\bdo\s+not\b|\bavoid\b", re.I)
_ADVICE = re.compile(r"\brecommend|\bshould\b|\badvise|\bindicated\b|\buse\b|\bgive\b", re.I)


def _canon(text: str) -> str:
    return normalize(text).lower().replace("–", "-").replace("—", "-")


def quantities(text: str) -> Set[Tuple[str, str]]:
    """{(value-or-range, unit)} found in text; unit '' when bare."""
    out = set()
    # Thousands separators: "1,000" / "1 000" is one thousand, not the decimal
    # 1.000 (a comma followed by exactly three digits). Fixed 2026-11-23.
    canon = _THOUSANDS.sub("", _canon(text))
    for m in _QTY.finditer(canon):
        lo, hi, unit = m.group(1), m.group(2), (m.group(3) or "")
        unit = _UNIT.get(unit.lower(), "") if unit else ""
        value = lo.replace(",", ".") + ("-" + hi.replace(",", ".") if hi else "")
        out.add((value, unit))
    return out


_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")
_STOP = {"the", "and", "for", "with", "that", "this", "are", "was", "were", "from", "should",
         "can", "may", "not", "use", "used", "patients", "patient", "women", "recommended"}


def _stem(w: str) -> str:
    """Crude, fixed suffix stripping so diabetic/diabetes and level/levels
    compare equal. Deliberately simple and deterministic."""
    for suf in ("ically", "ations", "ation", "ities", "ing", "ies", "ics", "ic", "es", "ed", "ly", "s", "e"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def claim_sentences(claim: str, passage: str, min_overlap: float = 0.34) -> List[str]:
    """The passage sentences a claim draws on: those sharing at least a third of
    the claim's content words (or the single best one). Word overlap only
    LOCATES the sentence to check; it never decides whether a claim is
    supported. Without this, an "avoid" anywhere in a long passage flagged
    unrelated claims (15 false alarms in the first benchmark)."""
    words = _content(claim)
    if not words:
        return [passage]
    sents = [s.text for s in enumerate_spans("p", passage, max_options=0, soft_wraps=True)
             if s.kind == "segment"] or [passage]
    scored = sorted(((len(words & _content(s)) / len(words), s) for s in sents), reverse=True)
    best = scored[0][0]
    # Near-best only: a neighbouring sentence about a different population
    # shares some words ("preterm", "membranes") without being the claim's source.
    picked = [s for sc, s in scored if sc >= max(min_overlap, 0.8 * best)]
    return picked or [scored[0][1]]


def rests_on_replaced(claim: str, replaced: list, min_share: float = 0.4):
    """The replaced recommendation a claim draws on, or None. Pure code: the
    claim must share at least `min_share` of its content words with the OLD
    recommendation, and more with it than with the newer one that replaced it.
    Used both as check 4 ("current") and, independently, as the currency
    measure in the A/B/C comparison."""
    words = {_stem(w) for w in _content(claim)}
    if not words:
        return None
    c_qty = quantities(claim)
    best = None
    for x in replaced or []:
        # A quantity that only the newer recommendation states decides it:
        # "<200 mg/dL" is the 2017 rule, whatever words surround it.
        new_only = quantities(x["new_text"]) - quantities(x["old_text"])
        if c_qty & new_only:
            continue
        o = len(words & {_stem(w) for w in _content(x["old_text"])}) / len(words)
        n = len(words & {_stem(w) for w in _content(x["new_text"])}) / len(words)
        if o >= min_share and o > n and (best is None or o > best[0]):
            best = (o, x)
    return best[1] if best else None


_CITE_TAG = re.compile(r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def check_details(claim: str, cited_text: str) -> Check:
    problems: List[str] = []
    # Not clinical details: citation tags ("[S1, S6]") and bare publication
    # years ("the 2018 guideline"). Both produced false alarms in the A/B/C run.
    claim = _CITE_TAG.sub(" ", claim)
    c_text, p_text = _canon(claim), _canon(cited_text)

    p_qty = quantities(p_text)
    p_values = {v for v, _ in p_qty} | {part for v, _ in p_qty for part in v.split("-")}
    for value, unit in quantities(c_text):
        if (value, unit) in p_qty:
            continue
        if not unit and _YEAR.fullmatch(value):
            continue          # a publication year, not a clinical quantity
        if not unit and value in p_values:
            continue          # bare number present in the passage
        if unit and any(v == value and (u == unit or not u) for v, u in p_qty):
            continue          # passage states the number; unit implied by context
        problems.append(f"quantity '{value}{(' ' + unit) if unit else ''}' not in the cited passage")

    if _STRONG.search(c_text) and not _STRONG.search(p_text):
        problems.append("claim says strong/must; the cited passage does not")
    if _CONDITIONAL.search(c_text) and not _CONDITIONAL.search(p_text):
        problems.append("claim says conditional/suggest; the cited passage does not")
    if _NEGATION.search(c_text) and not _NEGATION.search(p_text):
        problems.append("claim contains a negation the cited passage lacks")
    if _ADVICE.search(c_text) and not _NEGATION.search(c_text):
        local = " ".join(_canon(s) for s in claim_sentences(claim, cited_text))
        if _NEGATED_ADVICE.search(local):
            problems.append("cited passage advises against this; the claim omits the negation")

    return Check("details", not problems, "; ".join(problems))
