"""Phase 1-2: curated catalogue downloads and listing-page scrapes."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import time

from bs4 import BeautifulSoup
import requests

from ..config import CORPUS_ROOT, REQUEST_DELAY, WILEY_TDM_TOKEN, WILEY_TDM_URL
from ..http_utils import _get
from ..logs import _append_log, _now, _sha256
from ..validate import validate_file
from .who_iris import resolve_pdf_url


# ---------------------------------------------------------------------------
# Download a single catalogued document
# ---------------------------------------------------------------------------

def download_document(doc: dict, force_refresh: bool = False):
    dest_dir = CORPUS_ROOT / doc["dest_subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / doc["filename"]

    base_record = {
        "timestamp": _now(),
        "source":    doc["source"],
        "tier":      doc["tier"],
        "year":      doc.get("year", ""),
        "topic":     doc["topic"],
        "url":       doc["url"],
        "filename":  doc["filename"],
        "dest_path": str(dest_path),
        "status":    "",
        "http_code": "",
        "file_size_bytes": "",
        "sha256":    "",
        "error":     "",
    }

    # Skip cached documents unless an explicit refresh was requested — but only
    # if the cached file is a valid document. A cached error page is not a hit.
    if not force_refresh and dest_path.exists():
        cached_ok, cached_reason = validate_file(dest_path)
        if cached_ok:
            record = {**base_record, "status": "skipped",
                      "file_size_bytes": dest_path.stat().st_size,
                      "sha256": _sha256(dest_path)}
            _append_log(record)
            return
        print(f"     cached file invalid ({cached_reason}) — refetching")

    try:
        # Use Wiley TDM API for paywalled Wiley/BJOG articles
        wiley_doi = doc.get("wiley_doi")
        if wiley_doi and WILEY_TDM_TOKEN:
            doi_enc = wiley_doi.replace("/", "%2F")
            tdm_url = WILEY_TDM_URL.format(doi=doi_enc)
            tdm_headers = {"Wiley-TDM-Client-Token": WILEY_TDM_TOKEN,
                           "Accept": "application/pdf"}
            resp = _get(tdm_url, stream=True, extra_headers=tdm_headers)
            base_record["url"] = tdm_url  # log the actual URL used
        else:
            url = doc["url"]
            # WHO IRIS legacy bitstream paths now serve a DSpace SPA shell;
            # resolve the handle through the REST API instead.
            if "iris" in url and "who.int" in url:
                resolved = resolve_pdf_url(url)
                if resolved:
                    url = resolved
                    base_record["url"] = url
            resp = _get(url, stream=True)
        base_record["http_code"] = resp.status_code

        if resp.status_code != 200:
            record = {**base_record, "status": "failed",
                      "error": f"HTTP {resp.status_code}"}
            _append_log(record)
            return

        # For HTML saves
        if doc["filename"].endswith(".html"):
            dest_path.write_text(resp.text, encoding="utf-8")
        else:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

        size = dest_path.stat().st_size

        # Content validation — the download is not a success until the bytes
        # are shown to be the document, not a landing page or an error shell.
        valid, reason = validate_file(dest_path)
        if not valid:
            dest_path.unlink(missing_ok=True)
            record = {**base_record, "status": "failed",
                      "file_size_bytes": size,
                      "error": f"Content validation failed: {reason}"}
            _append_log(record)
            return

        record = {**base_record, "status": "success",
                  "file_size_bytes": size, "sha256": _sha256(dest_path)}
        _append_log(record)

    except requests.exceptions.RequestException as exc:
        record = {**base_record, "status": "failed", "error": str(exc)}
        _append_log(record)


# ---------------------------------------------------------------------------
# Scrape a listing page for additional PDFs
# ---------------------------------------------------------------------------

def scrape_and_download(target: dict, force_refresh: bool = False):
    source      = target["source"]
    tier        = target["tier"]
    dest_subdir = target["dest_subdir"]
    keywords    = [k.lower() for k in target.get("keywords", [])]
    page_url    = target["url"]

    print(f"\n  Scraping {source}: {page_url}")
    try:
        resp = _get(page_url)
        time.sleep(REQUEST_DELAY)
    except requests.exceptions.RequestException as exc:
        print(f"  ✗  Failed to fetch listing page: {exc}")
        return

    if resp.status_code != 200:
        print(f"  ✗  HTTP {resp.status_code} for listing page")
        return

    soup = BeautifulSoup(resp.text, "lxml")
    base_url = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

    found = 0
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(" ", strip=True).lower()

        # Must link to a PDF
        if ".pdf" not in href.lower():
            continue

        # Must match at least one keyword in link text or href
        combined = f"{text} {href.lower()}"
        if keywords and not any(kw in combined for kw in keywords):
            continue

        # Build absolute URL
        if href.startswith("http"):
            pdf_url = href
        elif href.startswith("//"):
            pdf_url = "https:" + href
        elif href.startswith("/"):
            pdf_url = base_url + href
        else:
            pdf_url = base_url + "/" + href

        # Build filename from URL
        raw_name = Path(urlparse(pdf_url).path).name
        if not raw_name.lower().endswith(".pdf"):
            raw_name += ".pdf"
        filename = f"{source}_{raw_name}"

        doc = {
            "source":      source,
            "tier":        tier,
            "year":        "",
            "topic":       a.get_text(" ", strip=True)[:120],
            "url":         pdf_url,
            "filename":    filename,
            "dest_subdir": dest_subdir,
        }

        time.sleep(REQUEST_DELAY)
        download_document(doc, force_refresh=force_refresh)
        found += 1

    if found == 0:
        print(f"  ⚠  No matching PDF links found on listing page.")
