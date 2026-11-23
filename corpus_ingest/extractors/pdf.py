"""PDF — 79 documents, 28 of them Tier 1 guidelines.

Section structure is recovered from the PDF outline (bookmarks) where the
publisher provided one, and from font-size heading detection where they did not.
Clinical guidelines are written as numbered recommendation statements, and
splitting one across chunks destroys the unit a nurse actually asks about, so
getting these boundaries right matters more here than in Tier 4.

Note: `corpus_curation.validate.extract_text` is NOT usable for this — it is a
deliberately crude heuristic for spotting error pages and returns mojibake on
compressed streams. Ingestion needs a real parser.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .base import Section, clean_text

# Running heads/feet repeat on most pages; detected by frequency, not by regex.
_MIN_REPEAT_FRACTION = 0.5
_HEADING_MAX_WORDS = 18


def _load(path: Path):
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        try:
            import fitz as pymupdf  # older installs
        except ImportError as exc:
            raise ImportError(
                "PDF extraction needs PyMuPDF: pip install pymupdf"
            ) from exc
    return pymupdf.open(path)


def _repeated_lines(pages: list[str], page_count: int) -> set[str]:
    """Lines appearing on most pages — running heads, footers, DOI stamps."""
    counts: Counter[str] = Counter()
    for text in pages:
        lines = {line.strip() for line in text.split("\n")[:3]}
        lines |= {line.strip() for line in text.split("\n")[-3:]}
        counts.update(line for line in lines if 3 < len(line) < 120)
    threshold = max(2, int(page_count * _MIN_REPEAT_FRACTION))
    return {line for line, n in counts.items() if n >= threshold}


def _outline_is_trustworthy(toc, pages: list[str]) -> bool:
    """Journal PDFs often carry issue-level or foreign bookmarks.

    The ERAS 2019 guideline, for instance, ships an outline naming an unrelated
    article about anaphylaxis in Lebanese day care. Trust the outline only if
    its entries actually point into distinct pages and its titles are found in
    the text they claim to head.
    """
    if len(toc) < 2:
        return False

    distinct_pages = {page for _, _, page in toc}
    if len(distinct_pages) < max(2, len(toc) // 2):
        return False

    hits = 0
    checked = 0
    for _, title, page_no in toc[:8]:
        probe = re.sub(r"\W+", " ", title).strip().lower()[:40]
        if len(probe) < 8:
            continue
        checked += 1
        window = " ".join(pages[max(0, page_no - 2): page_no + 1])
        if probe[:24] in re.sub(r"\W+", " ", window).lower():
            hits += 1
    return checked == 0 or hits >= max(1, checked // 2)


def _outline_sections(doc, pages: list[str]) -> list[Section]:
    """Use the PDF's own bookmarks when they can be trusted."""
    toc = doc.get_toc()
    if not toc or not _outline_is_trustworthy(toc, pages):
        return []

    out: list[Section] = []
    for i, (level, title, page_no) in enumerate(toc):
        start = max(0, page_no - 1)
        end = len(pages)
        for next_level, _, next_page in toc[i + 1:]:
            if next_page - 1 > start:
                end = max(start + 1, next_page - 1)
                break
        text = clean_text("\n".join(pages[start:end]))
        if not text:
            continue
        trail = [t for lvl, t, _ in toc[:i] if lvl < level][-3:]
        out.append(Section(title=clean_text(title), text=text,
                           path=trail, page=page_no))
    return out


def _font_sections(doc, pages: list[str], drop: set[str]) -> list[Section]:
    """Fallback: treat lines set in a larger-than-body font as headings."""
    sizes: Counter[int] = Counter()
    spans: list[tuple[int, str, int]] = []   # (size, text, page)

    for page_index, page in enumerate(doc):
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if not text or text in drop:
                    continue
                size = round(max((s.get("size", 0) for s in line.get("spans", [])), default=0))
                sizes[size] += len(text)
                spans.append((size, text, page_index + 1))

    if not spans:
        return []

    body_size = sizes.most_common(1)[0][0]
    sections: list[Section] = []
    title, buf, page_of = "", [], 1

    for size, text, page_no in spans:
        is_heading = (
            size > body_size
            and len(text.split()) <= _HEADING_MAX_WORDS
            and not text.endswith((".", ";", ","))
        )
        if is_heading:
            if buf:
                sections.append(Section(title=title, text=clean_text("\n".join(buf)),
                                        page=page_of))
                buf = []
            title, page_of = text, page_no
        else:
            buf.append(text)

    if buf:
        sections.append(Section(title=title, text=clean_text("\n".join(buf)), page=page_of))

    return [s for s in sections if s.text]


def extract_pdf(path: Path) -> list[Section]:
    doc = _load(path)
    try:
        pages = [page.get_text() for page in doc]
        drop = _repeated_lines(pages, len(pages))
        pages = ["\n".join(line for line in text.split("\n")
                           if line.strip() not in drop) for text in pages]

        sections = _outline_sections(doc, pages)

        if not sections:
            sections = _font_sections(doc, pages, drop)
            # Two-column journal PDFs often set headings at body size, so font
            # detection returns one enormous section. That loses page numbers,
            # which are the only citable locator such a document has — fall back
            # to per-page sections instead.
            if len(pages) > 3 and len(sections) < max(3, len(pages) // 4):
                sections = []

        if not sections:
            sections = [Section(title="", text=clean_text(text), page=i + 1)
                        for i, text in enumerate(pages) if text.strip()]

        return [s for s in sections if s.text]
    finally:
        doc.close()
