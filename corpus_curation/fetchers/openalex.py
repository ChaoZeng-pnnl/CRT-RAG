"""Phase 4: OpenAlex open-access PDFs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time

import requests

from ..catalog.searches import OPENALEX_SEARCHES
from ..config import (
    HEADERS,
    OPENALEX_BASE,
    OPENALEX_DIR,
    OPENALEX_MAX_PER_QUERY,
    REQUEST_DELAY,
    TIMEOUT,
)
from ..http_utils import _get, _is_pdf_response
from ..validate import validate_file
from ..logs import _append_log, _now, _sha256


# ---------------------------------------------------------------------------
# Phase 4 — OpenAlex open-access article download
# ---------------------------------------------------------------------------

def _fetch_openalex_articles(since: datetime | None = None,
                             force_refresh: bool = False):
    """Phase 4: search OpenAlex for open-access articles and download PDFs."""
    OPENALEX_DIR.mkdir(parents=True, exist_ok=True)
    total_new = total_skip = total_fail = 0

    for search in OPENALEX_SEARCHES:
        print(f"\n  [{search['topic'][:70]}]")
        try:
            filters = ["open_access.oa_status:gold", "type:article"]
            if since:
                filters.append(f"from_publication_date:{since:%Y-%m-%d}")
            r = requests.get(OPENALEX_BASE, params={
                "search":   search["query"],
                "filter":   ",".join(filters),
                "sort":     "publication_date:desc" if since else "cited_by_count:desc",
                "per-page": str(search.get("max_results", OPENALEX_MAX_PER_QUERY)),
                "select":   "id,title,doi,open_access,locations,publication_year",
            }, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            works = r.json().get("results", [])
        except Exception as exc:
            print(f"    OpenAlex search error: {exc}")
            continue
        time.sleep(REQUEST_DELAY)

        if not works:
            print("    No results.")
            continue

        for work in works:
            oa = work.get("open_access", {})
            # Try locations[].pdf_url first (direct PDF), then oa_url fallback
            pdf_url = ""
            for loc in work.get("locations", []):
                if loc.get("is_oa") and loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
                    break
            if not pdf_url:
                pdf_url = oa.get("oa_url", "")
            if not pdf_url or not pdf_url.startswith("http"):
                continue
            title   = (work.get("title") or work["id"])
            year    = str(work.get("publication_year", ""))
            oa_id   = work["id"].split("/")[-1]
            safe    = re.sub(r"[^\w\-]", "_", title[:50]).strip("_")
            fname   = f"{oa_id}_{safe}.pdf"
            dest    = OPENALEX_DIR / fname

            base_record = {
                "timestamp":       _now(),
                "source":          "OpenAlex",
                "tier":            4,
                "year":            year,
                "topic":           search["topic"],
                "subtopic":        search.get("subtopic", ""),
                "document_type":   "peer-reviewed article",
                "title":           title,
                "url":             pdf_url,
                "filename":        fname,
                "dest_path":       str(dest),
                "status":          "",
                "http_code":       "",
                "file_size_bytes": "",
                "sha256":          "",
                "error":           "",
            }

            if not force_refresh and dest.exists() and dest.stat().st_size > 1024:
                record = {**base_record, "status": "skipped",
                          "file_size_bytes": dest.stat().st_size,
                          "sha256": _sha256(dest)}
                _append_log(record)
                total_skip += 1
                continue

            try:
                resp = _get(pdf_url, stream=True)
                base_record["http_code"] = resp.status_code
                if resp.status_code != 200 or not _is_pdf_response(resp):
                    record = {**base_record, "status": "failed",
                              "error": f"HTTP {resp.status_code} or not a PDF"}
                    _append_log(record)
                    total_fail += 1
                    continue
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                size = dest.stat().st_size
                valid, reason = validate_file(dest)
                if not valid:
                    dest.unlink(missing_ok=True)
                    record = {**base_record, "status": "failed",
                              "file_size_bytes": size,
                              "error": f"Content validation failed: {reason}"}
                    _append_log(record)
                    total_fail += 1
                    continue
                record = {**base_record, "status": "success",
                          "file_size_bytes": size, "sha256": _sha256(dest)}
                _append_log(record)
                total_new += 1
            except Exception as exc:
                record = {**base_record, "status": "failed", "error": str(exc)}
                _append_log(record)
                total_fail += 1
            time.sleep(REQUEST_DELAY)

    print(f"\n  OpenAlex — new: {total_new}  skipped: {total_skip}  failed: {total_fail}")
