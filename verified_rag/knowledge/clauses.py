"""Clause-level candidate spans inside one recommendation sentence.

Sentence-level passages are too coarse for slots: the population ("adult
patients undergoing general anaesthesia with tracheal intubation") sits inside
the sentence. Clauses are cut at fixed connectives, deterministically, with
chunk-relative offsets so every slot reads back exactly.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_CUT = re.compile(r",\s+|\s+(?=that\s|who\s|undergoing\s|prior to\s|before\s|after\s|"
                  r"for\s|with\s|if\s|unless\s|in patients\s|in women\s|during\s|"
                  r"(?:is|are)\s+(?:recommended|suggested|not recommended)\b|should\s)"
                  r"|\s*(?=\()", re.I)     # a parenthetical (e.g. the grade) is never part of a slot
_MIN = 12


def clauses(text: str, start: int, end: int) -> List[Tuple[int, int, str]]:
    """(abs_start, abs_end, text) clauses of text[start:end], plus cumulative
    joins of neighbours so a slot spanning two clauses is still selectable."""
    seg = text[start:end]
    cuts = [0] + [m.end() for m in _CUT.finditer(seg)] + [len(seg)]
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        while a < b and seg[a].isspace():
            a += 1
        while b > a and seg[b - 1] in " ,.;":
            b -= 1
        if b - a >= _MIN:
            pieces.append((start + a, start + b))
    out = list(pieces)
    for i in range(len(pieces) - 1):                 # two-clause joins
        out.append((pieces[i][0], pieces[i + 1][1]))
    seen, result = set(), []
    for a, b in out:
        if (a, b) not in seen:
            seen.add((a, b))
            result.append((a, b, text[a:b]))
    return result
