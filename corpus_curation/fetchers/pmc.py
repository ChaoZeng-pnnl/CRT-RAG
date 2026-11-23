"""Phase 3: PubMed Central open-access JATS XML."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
import xml.etree.ElementTree as ET

import requests

from ..catalog.searches import PMC_SEARCHES
from ..config import (
    HEADERS,
    NCBI_API_KEY,
    NCBI_BASE,
    NCBI_DELAY,
    NCBI_OA_BASE,
    PMC_DIR,
    TIMEOUT,
)
from ..validate import validate_file
from ..logs import _append_log, _now, _sha256


# ---------------------------------------------------------------------------
# PubMed Central helpers
# ---------------------------------------------------------------------------

def _ncbi_get(endpoint: str, params: dict) -> requests.Response:
    """Rate-limited GET to NCBI E-utilities."""
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}
    time.sleep(NCBI_DELAY)
    return requests.get(endpoint, params=params, headers=HEADERS, timeout=TIMEOUT)


def _fetch_pmc_oa_articles(since: datetime | None = None,
                           force_refresh: bool = False):
    """Phase 3: search PMC for open-access articles and download PDFs."""
    PMC_DIR.mkdir(parents=True, exist_ok=True)
    total_new = total_skip = total_fail = 0

    for search in PMC_SEARCHES:
        print(f"\n  [{search['topic'][:70]}]")

        # 1. Search PMC (db=pmc ensures full-text is in PMC)
        try:
            query = search["query"]
            if since:
                query = f"({query}) AND ({since:%Y/%m/%d}[PDAT] : 3000[PDAT])"
            r = _ncbi_get(f"{NCBI_BASE}/esearch.fcgi", {
                "db": "pmc",
                "term": query,
                "retmax": search["max_results"],
                "retmode": "json",
                "sort": "pub date",
            })
            r.raise_for_status()
            ids = r.json()["esearchresult"]["idlist"]
        except Exception as exc:
            print(f"    esearch error: {exc}")
            continue

        if not ids:
            print("    No results.")
            continue

        # 2. Fetch article summaries (title, journal, year)
        try:
            r2 = _ncbi_get(f"{NCBI_BASE}/esummary.fcgi", {
                "db": "pmc", "id": ",".join(ids), "retmode": "json",
            })
            r2.raise_for_status()
            summaries = r2.json().get("result", {})
        except Exception:
            summaries = {}

        for uid in ids:
            pmcid   = f"PMC{uid}"
            meta    = summaries.get(uid, {})
            title   = meta.get("title", pmcid)
            year    = str(meta.get("pubdate", ""))[:4]

            # 3. Confirm article is in the PMC Open Access subset via OA API
            try:
                r3 = _ncbi_get(NCBI_OA_BASE, {"id": pmcid})
                root_el = ET.fromstring(r3.text)
            except Exception:
                continue

            if root_el.find(".//error") is not None:
                continue  # not in OA subset — skip

            # Confirm at least one full-text link exists (tgz or pdf)
            if not root_el.findall(".//link"):
                continue

            # 4. Build filename (.xml — efetch full-text XML is reliable over FTP PDF)
            safe  = re.sub(r"[^\w\-]", "_", title[:55]).strip("_")
            fname = f"{pmcid}_{safe}.xml"
            dest  = PMC_DIR / fname

            base_record = {
                "timestamp": _now(),
                "source":    "PubMed_PMC",
                "tier":      4,
                "year":      year,
                "topic":     search["topic"],
                "url":       f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                "filename":  fname,
                "dest_path": str(dest),
                "status":    "",
                "http_code": "",
                "file_size_bytes": "",
                "sha256":    "",
                "error":     "",
            }

            # Skip cached documents unless an explicit refresh was requested.
            if not force_refresh and dest.exists() and dest.stat().st_size > 1024:
                record = {**base_record, "status": "skipped",
                          "file_size_bytes": dest.stat().st_size,
                          "sha256": _sha256(dest)}
                _append_log(record)
                total_skip += 1
                continue

            # 5. Download full-text XML via efetch (db=pmc, rettype=full, retmode=xml)
            #    This is the officially supported E-utilities method for PMC full text.
            try:
                resp = _ncbi_get(f"{NCBI_BASE}/efetch.fcgi", {
                    "db": "pmc", "id": uid,
                    "rettype": "full", "retmode": "xml",
                })
                base_record["http_code"] = resp.status_code
                if resp.status_code != 200 or len(resp.content) < 1024:
                    record = {**base_record, "status": "failed",
                              "error": f"HTTP {resp.status_code} / empty"}
                    _append_log(record)
                    total_fail += 1
                    continue
                with open(dest, "wb") as f:
                    f.write(resp.content)
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

    print(f"\n  PMC — new: {total_new}  skipped: {total_skip}  failed: {total_fail}")
