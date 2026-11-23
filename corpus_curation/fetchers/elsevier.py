"""Phase 5: Elsevier ScienceDirect open-access XML."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time

import requests

from ..catalog.searches import OPENALEX_SEARCHES
from ..config import (
    ELSEVIER_API_KEY,
    ELSEVIER_DIR,
    ELSEVIER_SEARCH_BASE,
    HEADERS,
    REQUEST_DELAY,
    TIMEOUT,
)
from ..validate import validate_file
from ..logs import _append_log, _now, _sha256


# ---------------------------------------------------------------------------
# Phase 5 — Elsevier ScienceDirect open-access article download
# ---------------------------------------------------------------------------

def _fetch_elsevier_articles(since: datetime | None = None,
                             force_refresh: bool = False):
    """Phase 5: search ScienceDirect for open-access articles and download full-text XML."""
    if not ELSEVIER_API_KEY:
        print("    Elsevier API key not configured — skipping.")
        return
    ELSEVIER_DIR.mkdir(parents=True, exist_ok=True)
    total_new = total_skip = total_fail = 0

    for search in OPENALEX_SEARCHES:   # same search topics reused
        print(f"\n  [{search['topic'][:70]}]")
        try:
            r = requests.get(ELSEVIER_SEARCH_BASE, params={
                "query":       search["query"],
                "apiKey":      ELSEVIER_API_KEY,
                "field":       "doi,title,openaccess,coverDate,pii,prism:url",
                "open-access": "true",
                "count":       "8",
                "sort":        "-coverDate" if since else "",
            }, headers={**HEADERS, "Accept": "application/json"}, timeout=TIMEOUT)
            r.raise_for_status()
            entries = r.json().get("search-results", {}).get("entry", [])
        except Exception as exc:
            print(f"    Elsevier search error: {exc}")
            continue
        time.sleep(REQUEST_DELAY)

        for entry in entries:
            doi     = entry.get("prism:doi", "")
            title   = entry.get("dc:title", doi)
            is_oa   = str(entry.get("openaccess", "false")).lower() == "true"
            year    = str(entry.get("prism:coverDate", ""))[:4]
            art_url = entry.get("prism:url", "")
            if not is_oa or not art_url:
                continue
            if since and str(entry.get("prism:coverDate", ""))[:10] < f"{since:%Y-%m-%d}":
                continue
            safe  = re.sub(r"[^\w\-]", "_", title[:45]).strip("_")
            doi_s = re.sub(r"[^\w]", "_", doi)[:30]
            fname = f"Elsevier_{doi_s}_{safe}.xml"
            dest  = ELSEVIER_DIR / fname

            base_record = {
                "timestamp":       _now(),
                "source":          "Elsevier",
                "tier":            4,
                "year":            year,
                "topic":           search["topic"],
                "subtopic":        search.get("subtopic", ""),
                "document_type":   "peer-reviewed article",
                "title":           title,
                "url":             f"https://doi.org/{doi}" if doi else art_url,
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
                resp = requests.get(art_url, params={
                    "apiKey": ELSEVIER_API_KEY, "httpAccept": "text/xml",
                }, headers=HEADERS, timeout=TIMEOUT)
                base_record["http_code"] = resp.status_code
                if resp.status_code != 200 or len(resp.content) < 1024:
                    record = {**base_record, "status": "failed",
                              "error": f"HTTP {resp.status_code} or empty response"}
                    _append_log(record)
                    total_fail += 1
                    continue
                dest.write_bytes(resp.content)
                valid, reason = validate_file(dest)
                if not valid:
                    dest.unlink(missing_ok=True)
                    record = {**base_record, "status": "failed",
                              "error": f"Content validation failed: {reason}"}
                    _append_log(record)
                    total_fail += 1
                    continue
                record = {**base_record, "status": "success",
                          "file_size_bytes": dest.stat().st_size,
                          "sha256": _sha256(dest)}
                _append_log(record)
                total_new += 1
            except Exception as exc:
                record = {**base_record, "status": "failed", "error": str(exc)}
                _append_log(record)
                total_fail += 1
            time.sleep(REQUEST_DELAY)

    print(f"\n  Elsevier — new: {total_new}  skipped: {total_skip}  failed: {total_fail}")
