"""Shared section model and format dispatch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


@dataclass
class Section:
    """A titled span of a document — the unit chunking works on.

    `path` is the section hierarchy ("Postoperative care > Analgesia"), which is
    what lets a chunk say where in the guideline it came from.
    """

    title: str
    text: str
    path: list[str] = field(default_factory=list)
    page: int | None = None
    kind: str = "body"          # body | table | abstract | reference

    @property
    def breadcrumb(self) -> str:
        return " > ".join([*self.path, self.title]) if self.title else " > ".join(self.path)


def clean_text(text: str) -> str:
    """Normalise whitespace without destroying paragraph boundaries."""
    text = text.replace("­", "")            # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def extract(path: Path, file_format: str) -> list[Section]:
    """Dispatch on the manifest's declared format."""
    from .html import extract_html
    from .jats import extract_jats
    from .pdf import extract_pdf

    fmt = (file_format or path.suffix.lstrip(".")).upper()
    if fmt == "XML":
        return extract_jats(path)
    if fmt == "PDF":
        return extract_pdf(path)
    if fmt == "HTML":
        return extract_html(path)
    raise ValueError(f"no extractor for format {fmt!r} ({path.name})")
