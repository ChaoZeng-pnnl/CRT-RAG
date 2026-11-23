"""JATS XML — 140 of 237 documents (all of Tier 4, from PMC and Elsevier).

These arrive already marked up with <sec>, <title> and <table-wrap>, so the
section tree is read directly rather than inferred. This is the highest-quality
structure in the corpus and needs no heuristics.

Elsevier uses its own namespace (ce:section, ce:section-title) alongside the
JATS-style tags, so both vocabularies are handled.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .base import Section, clean_text

# Tags whose text is metadata or navigation, not document content.
_SKIP = {
    "ref-list", "back", "journal-meta", "author-notes", "fn-group",
    "funding-group", "permissions", "history", "contrib-group",
}

_TITLE_TAGS = {"title", "section-title", "label"}
_SECTION_TAGS = {"sec", "section"}
# Wrappers that hold sections but are not sections themselves. Without
# descending through these, <body><sec> is never reached and the whole article
# collapses into one undifferentiated blob.
_CONTAINER_TAGS = {"article", "pmc-articleset", "body", "front", "article-meta",
                   "sub-article", "serialItem", "converted-article",
                   "originalText", "doc", "article-body", "bodyText"}


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _node_text(node: ET.Element) -> str:
    """All descendant text except nested section titles (those become headings)."""
    parts: list[str] = []
    for child in node.iter():
        name = _localname(child.tag)
        if name in _SKIP or name in _TITLE_TAGS:
            continue
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return clean_text(" ".join(parts))


def _title_of(node: ET.Element) -> str:
    for child in node:
        if _localname(child.tag) in _TITLE_TAGS:
            return clean_text("".join(child.itertext()))
    return ""


def _walk(node: ET.Element, trail: list[str], out: list[Section]) -> None:
    for child in node:
        name = _localname(child.tag)
        if name in _SKIP:
            continue

        if name in _SECTION_TAGS:
            title = _title_of(child)
            nested = [c for c in child if _localname(c.tag) in _SECTION_TAGS]
            # Text belonging to this section, excluding nested subsections.
            own = ET.Element("wrapper")
            own.extend([c for c in child if c not in nested])
            text = _node_text(own)
            if text:
                out.append(Section(title=title, text=text, path=list(trail)))
            _walk(child, [*trail, title] if title else trail, out)

        elif name == "table-wrap":
            text = _node_text(child)
            if text:
                out.append(Section(title=_title_of(child) or "Table",
                                   text=text, path=list(trail), kind="table"))

        elif name == "abstract":
            text = _node_text(child)
            if text:
                out.append(Section(title="Abstract", text=text,
                                   path=list(trail), kind="abstract"))

        elif list(child):
            # Descend through any other wrapper. Allow-listing container names
            # does not survive contact with real vocabularies: Elsevier nests
            # content under full-text-retrieval-response > originalText >
            # xocs:doc > serial-item > article > body > sections, and a missing
            # name anywhere in that chain silently loses the whole article.
            _walk(child, trail, out)


def extract_jats(path: Path) -> list[Section]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"malformed XML: {exc}") from exc

    sections: list[Section] = []

    # Article title gives the top-level breadcrumb.
    title = ""
    for node in root.iter():
        if _localname(node.tag) in {"article-title", "title"}:
            title = clean_text("".join(node.itertext()))
            if title:
                break

    _walk(root, [title] if title else [], sections)

    # Some Elsevier documents carry only <ce:para> with no sections at all.
    if not sections:
        body = _node_text(root)
        if body:
            sections.append(Section(title=title, text=body, path=[]))

    return sections
