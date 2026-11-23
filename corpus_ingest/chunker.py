"""Section-aware chunking.

Fixed-size chunking is wrong for this corpus. Clinical guidelines are written as
numbered recommendation statements, and a recommendation split across two chunks
is the exact unit a nurse asks about — cut it and neither half answers the
question while both look plausible to a reranker.

So: sections are the primary boundary, recommendation statements are respected
inside oversized sections, and the token cap only applies as a last resort.
Short sections are packed together rather than emitted as fragments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extractors import Section

# ~4 chars/token is close enough for budgeting without a tokenizer dependency.
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 512
OVERLAP_TOKENS = 64
MIN_TOKENS = 40          # below this, pack with a neighbour instead of emitting
MAX_TOKENS = 900         # hard ceiling before forced split

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN
MIN_CHARS = MIN_TOKENS * CHARS_PER_TOKEN
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

# Numbered/lettered recommendation statements as guidelines write them:
# "1.2.3 ...", "Recommendation 4 ...", "R5.", "4.1.2." at a line start.
_RECOMMENDATION = re.compile(
    r"(?m)^(?:\s*(?:recommendation\s+)?"
    r"(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-Z]\.|R\d{1,2}\.?)\s+)"
    r"(?=[A-Z(‘“])",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    text: str
    section: str
    page: int | None
    kind: str
    index: int


def _split_preserving(text: str) -> list[str]:
    """Split oversized text on the strongest boundary that fits.

    Recommendation starts first, then paragraphs, then sentences. Each level is
    only used if the level above leaves pieces that are still too large.
    """
    for splitter in (_RECOMMENDATION, _PARAGRAPH, _SENTENCE):
        pieces = [p.strip() for p in splitter.split(text) if p.strip()]
        if len(pieces) > 1:
            return pieces
    # No boundary at all — hard-wrap, accepting the loss.
    return [text[i:i + TARGET_CHARS] for i in range(0, len(text), TARGET_CHARS)]


def _pack(pieces: list[str]) -> list[str]:
    """Greedily fill chunks to TARGET_CHARS, with overlap between them."""
    out: list[str] = []
    buf = ""

    for piece in pieces:
        if len(piece) > MAX_CHARS:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_pack(_split_preserving(piece)))
            continue

        candidate = f"{buf}\n\n{piece}".strip() if buf else piece
        if len(candidate) <= TARGET_CHARS:
            buf = candidate
        else:
            if buf:
                out.append(buf)
                tail = buf[-OVERLAP_CHARS:] if len(buf) > OVERLAP_CHARS else ""
                # Start the next chunk with trailing context, cut at a sentence.
                if tail:
                    parts = _SENTENCE.split(tail)
                    tail = parts[-1] if len(parts) > 1 else ""
                buf = f"{tail}\n\n{piece}".strip() if tail else piece
            else:
                buf = piece

    if buf:
        out.append(buf)
    return out


# Lines that are page furniture rather than content: copyright stamps, bare
# URLs, "Subject to Notice of rights". Harmless in a long chunk, but a chunk
# made only of these is pure retrieval noise.
_BOILERPLATE = re.compile(
    r"©|\(c\)\s*\d{4}|all rights reserved|subject to notice of rights|"
    r"^\s*(?:https?://|www\.)\S+\s*$|licensed under|doi:\s*\S+",
    re.IGNORECASE | re.MULTILINE,
)


def _is_boilerplate_only(text: str) -> bool:
    """True when stripping page furniture leaves almost nothing behind."""
    residue = _BOILERPLATE.sub(" ", text)
    residue = re.sub(r"[\s\W_]+", " ", residue).strip()
    return len(residue) < MIN_CHARS // 2


def _split_table(body: str) -> list[str]:
    """Split an oversized table on rows, repeating the header on each piece."""
    if len(body) <= MAX_CHARS:
        return [body]

    lines = [ln for ln in body.split("\n") if ln.strip()]
    header = "\n".join(lines[:2])
    pieces, buf = [], []
    for line in lines[2:]:
        buf.append(line)
        if sum(len(x) for x in buf) >= TARGET_CHARS:
            pieces.append(header + "\n" + "\n".join(buf))
            buf = []
    if buf:
        pieces.append(header + "\n" + "\n".join(buf))
    if not pieces:
        pieces = [body]

    # Some extractors emit a whole table as one line; nothing to split on.
    out = []
    for piece in pieces:
        if len(piece) <= MAX_CHARS:
            out.append(piece)
        else:
            out.extend(piece[i:i + TARGET_CHARS]
                       for i in range(0, len(piece), TARGET_CHARS))
    return out


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Sections -> chunks, packing short sections and splitting long ones."""
    chunks: list[Chunk] = []
    pending: list[Section] = []

    def flush() -> None:
        if not pending:
            return
        merged = "\n\n".join(
            f"{s.title}\n{s.text}".strip() if s.title else s.text for s in pending
        )
        head = pending[0]
        for body in _pack([merged] if len(merged) <= TARGET_CHARS
                          else _split_preserving(merged)):
            chunks.append(Chunk(text=body, section=head.breadcrumb,
                                page=head.page, kind=head.kind, index=len(chunks)))
        pending.clear()

    for section in sections:
        body = f"{section.title}\n{section.text}".strip() if section.title else section.text

        # Tables are atomic: a dosing table split in half is worse than useless.
        # But a 7,000-token table exceeds every embedding model's context, so
        # oversized ones are split on row boundaries with the header repeated —
        # each piece still reads as a table rather than a fragment.
        if section.kind == "table":
            flush()
            for text in _split_table(body):
                chunks.append(Chunk(text=text, section=section.breadcrumb,
                                    page=section.page, kind="table",
                                    index=len(chunks)))
            continue

        if len(body) < MIN_CHARS:
            pending.append(section)
            if sum(len(s.text) for s in pending) >= TARGET_CHARS:
                flush()
            continue

        flush()
        pieces = [body] if len(body) <= TARGET_CHARS else _split_preserving(body)
        for text in _pack(pieces):
            chunks.append(Chunk(text=text, section=section.breadcrumb,
                                page=section.page, kind=section.kind,
                                index=len(chunks)))

    flush()
    # Drop chunks that are nothing but page furniture. Short *content* chunks
    # are kept deliberately: a standalone "1.3.5 Consider an alternate-day oral
    # iron regimen..." is the exact atom a nurse asks about.
    kept = [c for c in chunks if not _is_boilerplate_only(c.text)]
    for i, chunk in enumerate(kept):
        chunk.index = i
    return kept
