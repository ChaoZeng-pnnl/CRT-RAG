"""Headless-browser fetching for JavaScript-rendered sources.

Several authorities no longer serve document links in their HTML: ACOG returns a
branded 404 to non-browser clients, RNAO and AHRQ render their listings client
side, and search.cdc.gov paints results after load. `requests` sees a shell in
every case — which is exactly how landing pages entered the corpus in the first
place.

This module renders the page in Chromium, then either harvests the links that
appear or downloads a document through the browser session (so cookies and
bot-checks that gate `requests` are satisfied). It is deliberately isolated:
Playwright is an optional dependency and everything here degrades to a clear
error when it is absent, so the plain HTTP phases keep working without it.

Install once:
    pip install playwright && python -m playwright install chromium
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin

# A real browser UA: the headless default is filtered by several of these hosts.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

PAGE_TIMEOUT = 45_000   # ms
SETTLE_MS = 2_500       # give client-side rendering time to paint


class BrowserUnavailable(RuntimeError):
    """Raised when Playwright or its Chromium build is not installed."""


def _launch(playwright):
    """Prefer the full Chromium build over the headless shell.

    The shell is detectable and fails Cloudflare's interstitial (RNAO's download
    endpoint sits behind one); full Chromium in new-headless mode passes it.
    Install with: python -m playwright install chromium --no-shell
    """
    args = ["--disable-blink-features=AutomationControlled", "--no-sandbox"]
    try:
        return playwright.chromium.launch(headless=True, channel="chromium", args=args)
    except Exception:
        return playwright.chromium.launch(headless=True, args=args)


def _new_context(browser, accept_downloads: bool = False):
    return browser.new_context(
        user_agent=BROWSER_UA,
        viewport={"width": 1440, "height": 900},
        accept_downloads=accept_downloads,
        ignore_https_errors=True,
    )


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise BrowserUnavailable(
            "Playwright is not installed. Run:\n"
            "    pip install playwright && python -m playwright install chromium"
        ) from exc
    return sync_playwright


def render_html(url: str, wait_for: str | None = None,
                settle_ms: int = SETTLE_MS) -> str:
    """Return the fully rendered HTML of `url`."""
    sync_playwright = _require_playwright()
    with sync_playwright() as p:
        browser = _launch(p)
        try:
            page = _new_context(browser).new_page()
            page.goto(url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
            if wait_for:
                try:
                    page.wait_for_selector(wait_for, timeout=PAGE_TIMEOUT)
                except Exception:
                    pass  # fall through to whatever did render
            page.wait_for_timeout(settle_ms)
            return page.content()
        finally:
            browser.close()


def find_pdf_links(url: str, keywords: list[str] | None = None,
                   wait_for: str | None = None) -> list[tuple[str, str]]:
    """Render `url` and return [(link text, absolute PDF url)] matching keywords."""
    from bs4 import BeautifulSoup

    html = render_html(url, wait_for=wait_for)
    soup = BeautifulSoup(html, "lxml")
    terms = [k.lower() for k in (keywords or [])]

    hits: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)
        if ".pdf" not in href.lower():
            continue
        absolute = urljoin(url, href)
        if absolute in seen:
            continue
        if terms and not any(t in f"{text} {href}".lower() for t in terms):
            continue
        seen.add(absolute)
        hits.append((text, absolute))
    return hits


def download_via_browser(url: str, dest: Path, referer: str | None = None) -> bool:
    """Fetch a document inside the browser session. Returns True on success.

    Two paths: PDFs that Chromium streams as a download, and PDFs served inline
    (Chromium renders those in its viewer, so the bytes are pulled from the
    response body instead).
    """
    sync_playwright = _require_playwright()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = _launch(p)
        try:
            context = _new_context(browser, accept_downloads=True)
            page = context.new_page()

            if referer:
                # Visiting the referring page first satisfies hosts that reject
                # a cold deep link.
                try:
                    page.goto(referer, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
                    page.wait_for_timeout(1_000)
                except Exception:
                    pass

            # Case 1: navigation triggers a download. Chromium aborts the
            # navigation with "Download is starting" — that exception must be
            # swallowed *inside* the expect_download block, or the download
            # event is never awaited.
            try:
                with page.expect_download(timeout=30_000) as info:
                    try:
                        page.goto(url, timeout=PAGE_TIMEOUT)
                    except Exception:
                        pass
                info.value.save_as(dest)
                if dest.exists() and dest.stat().st_size > 0:
                    return True
            except Exception:
                pass

            # Case 2: the document renders inline. Chromium answers the
            # navigation with its own PDF-viewer HTML, so match on the response
            # *content* — the real bytes arrive as application/pdf — rather than
            # on the URL, which would capture the viewer wrapper instead.
            captured: dict[str, bytes] = {}

            def _capture(response):
                try:
                    if not response.ok:
                        return
                    ctype = response.headers.get("content-type", "").lower()
                    body = response.body()
                    if not body:
                        return
                    if "application/pdf" in ctype or body[:5] == b"%PDF-":
                        captured["body"] = body
                except Exception:
                    pass

            page.on("response", _capture)
            try:
                page.goto(url, timeout=PAGE_TIMEOUT)
                page.wait_for_timeout(2_000)
            except Exception:
                pass
            if captured.get("body"):
                dest.write_bytes(captured["body"])
                return dest.stat().st_size > 0

            # Case 3: last resort — the context's request stack (no browser TLS
            # fingerprint, so hosts like cdc.gov may still refuse it).
            try:
                response = context.request.get(url, timeout=PAGE_TIMEOUT)
                if response.ok:
                    body = response.body()
                    if body:
                        dest.write_bytes(body)
                        return dest.stat().st_size > 0
            except Exception:
                pass

            return False
        finally:
            browser.close()


def cdc_search(query: str, max_results: int = 10) -> list[tuple[str, str]]:
    """Search search.cdc.gov and return [(title, url)] for PDF/document hits.

    www.cdc.gov refuses automated clients outright (HTTP 403 on every path
    tried), but the search front end answers normally and its result links point
    at the same files.
    """
    from bs4 import BeautifulSoup

    search_url = f"https://search.cdc.gov/search/?query={query.replace(' ', '+')}"
    html = render_html(search_url, wait_for="a[href]", settle_ms=4_000)
    soup = BeautifulSoup(html, "lxml")

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True)
        if not href.startswith("http") or "search.cdc.gov" in href:
            continue
        if not re.search(r"cdc\.gov", href):
            continue
        if href in seen or not text:
            continue
        seen.add(href)
        results.append((text, href))
        if len(results) >= max_results:
            break
    return results


def download_via_click(page_url: str, selector: str, dest: Path) -> bool:
    """Load `page_url`, click `selector`, and save the resulting download.

    Needed where the download endpoint is protected by an interstitial that a
    direct navigation trips (RNAO's Cloudflare challenge): clicking from the
    already-cleared page carries the clearance.
    """
    sync_playwright = _require_playwright()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = _launch(p)
        try:
            page = _new_context(browser, accept_downloads=True).new_page()
            page.goto(page_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
            page.wait_for_timeout(5_000)   # let the interstitial clear
            with page.expect_download(timeout=60_000) as info:
                page.locator(selector).first.click(timeout=25_000)
            info.value.save_as(dest)
            return dest.exists() and dest.stat().st_size > 0
        except Exception:
            return False
        finally:
            browser.close()
