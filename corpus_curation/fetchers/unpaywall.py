"""Phase 6: Unpaywall OA lookup for key society guidelines."""

from __future__ import annotations

import json
import time

import requests

from ..catalog.searches import UNPAYWALL_DOIS
from ..config import CORPUS_ROOT, HEADERS, REQUEST_DELAY, TIMEOUT, UNPAYWALL_EMAIL
from ..http_utils import _get, _is_pdf_response
from ..validate import validate_file
from ..logs import _append_log, _now, _sha256


# ---------------------------------------------------------------------------
# Phase 6 — Unpaywall DOI-based download
# ---------------------------------------------------------------------------

def _fetch_unpaywall_dois(force_refresh: bool = False):
    """Phase 6: download key society guidelines by DOI via Unpaywall OA lookup."""
    if not UNPAYWALL_EMAIL:
        print("    UNPAYWALL_EMAIL not configured — skipping.")
        return
    total_new = total_skip = total_fail = 0

    for entry in UNPAYWALL_DOIS:
        doi      = entry["doi"]
        dest_dir = CORPUS_ROOT / entry["dest_subdir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest     = dest_dir / entry["filename"]

        base_record = {
            "timestamp":          _now(),
            "source":             entry["source"],
            "tier":               entry.get("tier", 1),
            "year":               str(entry.get("year", "")),
            "topic":              entry["topic"],
            "subtopic":           entry.get("subtopic", ""),
            "document_type":      entry.get("document_type", "clinical guideline"),
            "title":              entry.get("title", entry["topic"]),
            "retrieval_priority": entry.get("retrieval_priority", "critical"),
            "url":                f"https://doi.org/{doi}",
            "filename":           entry["filename"],
            "dest_path":          str(dest),
            "status":             "",
            "http_code":          "",
            "file_size_bytes":    "",
            "sha256":             "",
            "error":              "",
        }

        if not force_refresh and dest.exists() and dest.stat().st_size > 1024:
            record = {**base_record, "status": "skipped",
                      "file_size_bytes": dest.stat().st_size,
                      "sha256": _sha256(dest)}
            _append_log(record)
            total_skip += 1
            continue

        # Look up best open-access URL via Unpaywall
        try:
            r_oa = requests.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": UNPAYWALL_EMAIL},
                headers=HEADERS, timeout=TIMEOUT,
            )
            time.sleep(REQUEST_DELAY)
            if r_oa.status_code != 200:
                record = {**base_record, "status": "failed",
                          "error": f"Unpaywall lookup HTTP {r_oa.status_code}"}
                _append_log(record)
                total_fail += 1
                continue
            best   = r_oa.json().get("best_oa_location") or {}
            oa_url = best.get("url_for_pdf") or best.get("url")
            if not oa_url:
                record = {**base_record, "status": "failed",
                          "error": "No OA location found via Unpaywall"}
                _append_log(record)
                total_fail += 1
                continue
        except Exception as exc:
            record = {**base_record, "status": "failed", "error": str(exc)}
            _append_log(record)
            total_fail += 1
            continue

        # Download the OA PDF
        try:
            resp = _get(oa_url, stream=True)
            base_record["http_code"] = resp.status_code
            base_record["url"]       = oa_url
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

    print(f"\n  Unpaywall — new: {total_new}  skipped: {total_skip}  failed: {total_fail}")
