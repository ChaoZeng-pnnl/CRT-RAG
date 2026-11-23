"""HTML — 18 documents (OpenRN chapters, AHRQ PSNet primers, ACOG article).

These are full web pages, so navigation, cookie banners and site chrome have to
go before anything else; otherwise every chunk carries the same menu text and
retrieval collapses toward boilerplate.
"""

from __future__ import annotations

from pathlib import Path

from .base import Section, clean_text

_DROP_TAGS = ["script", "style", "nav", "header", "footer", "noscript",
              "aside", "form", "iframe", "svg"]
_DROP_HINTS = ["nav", "menu", "banner", "cookie", "breadcrumb", "sidebar",
               "footer", "header", "skip", "social", "related", "search"]
_HEADINGS = ["h1", "h2", "h3", "h4"]


def _looks_like_chrome(tag) -> bool:
    attrs = getattr(tag, "attrs", None) or {}
    classes = attrs.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    ident = " ".join([
        " ".join(str(c) for c in classes),
        str(attrs.get("id") or ""),
        str(attrs.get("role") or ""),
    ]).lower()
    return any(hint in ident for hint in _DROP_HINTS)


def _parse(html: str, strip_chrome: bool):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    if strip_chrome:
        for tag in soup.find_all(True):
            if _looks_like_chrome(tag):
                tag.decompose()
    return soup


def extract_html(path: Path) -> list[Section]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    sections = _extract(_parse(html, strip_chrome=True))
    if not sections:
        # The class/id heuristics matched the content wrapper itself and took
        # the page with it. Fall back to tag-level stripping only.
        sections = _extract(_parse(html, strip_chrome=False))
    return sections


def _extract(soup) -> list[Section]:

    root = (soup.find("main") or soup.find("article")
            or soup.find(attrs={"role": "main"}) or soup.body or soup)

    page_title = clean_text(soup.title.get_text()) if soup.title else ""

    sections: list[Section] = []
    title, buf = "", []

    for node in root.find_all([*_HEADINGS, "p", "li", "td", "th", "pre"]):
        text = clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in _HEADINGS:
            if buf:
                sections.append(Section(title=title, text="\n\n".join(buf),
                                        path=[page_title] if page_title else []))
                buf = []
            title = text
        else:
            buf.append(text)

    if buf:
        sections.append(Section(title=title, text="\n\n".join(buf),
                                path=[page_title] if page_title else []))

    return [s for s in sections if s.text]
