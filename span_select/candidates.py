"""Deterministic candidate enumeration and the resolver.

No model touches anything in this file. Candidates are produced from chunk text
by segmentation and regex, carry absolute offsets into that text, and are
resolved by reading the chunk back. That is the whole trick: the model's output
vocabulary is a set of integers this file minted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

# Jev's Choice type accepts at most 255 options, so that is the budget a
# selection question has to fit into. Chunks that overflow it must be split
# across chained questions rather than silently truncated.
MAX_CHOICE_OPTIONS = 255

# A clinical specific: a number carrying a unit. \s matches U+00A0 in Python 3
# str patterns, but the non-breaking space is named explicitly because it is the
# exact character that breaks naive matching in this corpus (see normalize()).
QUANTITY = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?"
    r"[\s ]?"
    r"(?:mg|mcg|µg|g|kg|mL|ml|L|l|units?|IU|hours?|hrs?|hr|h|minutes?|mins?|min|"
    r"days?|weeks?|months?|%|mmHg|cmH2O|/min|bpm|beats?|°C|degrees?)\b",
    re.IGNORECASE,
)

# Segment boundaries. Guideline text is line-structured (numbered
# recommendations, bullets, table rows), so newlines matter at least as much as
# sentence punctuation.
_BOUNDARY = re.compile(r"\n+|(?<=[.;:])[\s ]+(?=[A-Z(•\-\d])")

# With soft_wraps=True, a single newline is a boundary only where it plausibly
# ends a unit: after sentence punctuation, before a list marker, or as part of
# a blank line. PDF extraction hard-wraps prose every ~60 characters; in 51% of
# chunks a newline falls mid-sentence, so splitting on every newline turns
# sentences into line fragments.
_BOUNDARY_SOFT = re.compile(
    r"\n\s*\n+"                                    # blank line
    r"|(?<=[.;:!?)])[ \t]*\n+"                       # newline after sentence end
    r"|\n+(?=[ \t]*(?:[•\-–*]|\d+[.)]\s))"          # newline before a list item
    r"|(?<=[.;:])[\s ]+(?=[A-Z(•\-\d])")              # sentence end within a line

_SOFT_HYPHEN = "­"
_MIN_SEGMENT_CHARS = 25


@dataclass(frozen=True)
class Span:
    """A verbatim region of one chunk. `text` MUST equal chunk[start:end]."""

    id: int
    chunk_id: str
    start: int
    end: int
    text: str
    kind: str  # "segment" | "quantity"

    def __len__(self) -> int:
        return self.end - self.start


def normalize(text: str) -> str:
    """Lexical normalization, for comparison only — never for offsets.

    NFKC, collapse every flavour of unicode whitespace to a single space, drop
    soft hyphens, and rejoin words broken across a line break.

    This exists to quantify what generate-then-verify would have to survive. In
    the selection architecture it is not on the critical path: nothing is ever
    matched by string equality, because nothing is ever regenerated.

    Deliberately lexical. Reaching for embedding similarity here would convert a
    mechanical guarantee back into a statistical judgement while still looking
    like a guarantee.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_SOFT_HYPHEN, "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[\s ]+", " ", text)
    return text.strip()


def naive_match_would_fail(span: Span) -> bool:
    """True if re-emitting this span as plain text breaks byte equality.

    A model asked to quote '2\\xa0hours' will almost always write '2 hours'.
    Under generate-then-verify that is a false alarm; under selection it cannot
    arise. Counting these measures the trap the selection design sidesteps.
    """
    return span.text != normalize(span.text)


def _segments(text: str, soft_wraps: bool = False) -> List[tuple]:
    """(start, end) for each segment, whitespace trimmed, in document order."""
    cuts = [0]
    for match in (_BOUNDARY_SOFT if soft_wraps else _BOUNDARY).finditer(text):
        cuts.append(match.end())
    cuts.append(len(text))

    out = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            out.append((start, end))
    return out


def enumerate_spans(chunk_id: str, text: str,
                    max_options: int = MAX_CHOICE_OPTIONS,
                    dedupe: bool = True, soft_wraps: bool = False) -> List[Span]:
    """Every candidate span in one chunk, with offsets into `text`.

    Two kinds, because a nurse's answer needs both: the segment that states the
    recommendation, and the quantity that is the actionable specific inside it.
    Overflow past `max_options` is reported by the caller, never silently cut.

    `dedupe` drops a candidate whose normalized text already appears earlier in
    its own kind, keeping the first offset. This is not tidying. A chunk that
    prints "6\nweeks" in a table and "6 weeks" in a sentence yields two
    candidates that render identically in the prompt, so the selector is asked
    to choose between two options it cannot tell apart, and the annotator is
    asked the same impossible question. Whichever it picks, the nurse sees the
    same string from the same page, so the two are interchangeable and only one
    belongs on the ballot. Offsets are unaffected: the survivor still resolves
    exactly, and `resolve()` is the only way text is ever produced.

    `soft_wraps` joins lines a PDF extractor hard-wrapped mid-sentence. It is
    off by default because the annotation study (eval/annotation) was built on
    the line-level ballot and its candidate indices must not move.
    """
    spans: List[Span] = []
    next_id = 0
    seen = {"segment": set(), "quantity": set()}

    def add(start: int, end: int, body: str, kind: str) -> None:
        nonlocal next_id
        if dedupe:
            key = normalize(body)
            if key in seen[kind]:
                return
            seen[kind].add(key)
        spans.append(Span(next_id, chunk_id, start, end, body, kind))
        next_id += 1

    for start, end in _segments(text, soft_wraps):
        body = text[start:end]
        if len(body) < _MIN_SEGMENT_CHARS and not QUANTITY.search(body):
            continue
        add(start, end, body, "segment")

    for match in QUANTITY.finditer(text):
        start, end = match.span()
        add(start, end, text[start:end], "quantity")

    return spans[:max_options] if max_options else spans


def resolve(span: Span, chunk_text: str) -> str:
    """Read the span back out of the chunk. The ONLY way answer text is produced.

    Raises if the offsets do not reproduce the span, which would mean the chunk
    changed under us — a corpus-integrity failure, not a model failure.
    """
    actual = chunk_text[span.start:span.end]
    if actual != span.text:
        raise ValueError(
            "span %d does not resolve in %s: offsets (%d,%d) give %r, expected %r"
            % (span.id, span.chunk_id, span.start, span.end, actual[:60], span.text[:60])
        )
    return actual


def select_by_id(spans: List[Span], chosen: Optional[int]) -> Optional[Span]:
    """Map a model's integer back to a span. Out-of-range is rejected, not clamped."""
    if chosen is None:
        return None
    for span in spans:
        if span.id == chosen:
            return span
    return None
