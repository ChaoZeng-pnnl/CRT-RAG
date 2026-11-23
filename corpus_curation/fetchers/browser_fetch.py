"""Phase 7: documents that only a real browser can retrieve.

Kept as its own phase so the pipeline degrades cleanly: if Playwright is not
installed the phase reports what it would have fetched and the other six phases
run unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..browser import (
    BrowserUnavailable,
    download_via_browser,
    download_via_click,
    render_html,
)
from ..catalog.browser_targets import BROWSER_TARGETS
from ..config import CORPUS_ROOT, REQUEST_DELAY
from ..logs import _append_log, _now, _sha256
from ..validate import validate_file


def _fetch_browser_targets(force_refresh: bool = False):
    total_new = total_skip = total_fail = 0

    for entry in BROWSER_TARGETS:
        dest_dir = CORPUS_ROOT / entry["dest_subdir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / entry["filename"]

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
            "url":                entry["url"],
            "filename":           entry["filename"],
            "dest_path":          str(dest),
            "status":             "",
            "http_code":          "",
            "file_size_bytes":    "",
            "sha256":             "",
            "error":              "",
        }

        if not force_refresh and dest.exists():
            ok, _ = validate_file(dest)
            if ok:
                record = {**base_record, "status": "skipped",
                          "file_size_bytes": dest.stat().st_size,
                          "sha256": _sha256(dest)}
                _append_log(record)
                total_skip += 1
                continue

        try:
            if entry.get("save_rendered_html"):
                html = render_html(entry["url"])
                dest.write_text(html, encoding="utf-8")
                fetched = True
            elif entry.get("click_selector"):
                fetched = download_via_click(
                    entry["referer"], entry["click_selector"], dest)
            else:
                fetched = download_via_browser(
                    entry["url"], dest, referer=entry.get("referer"))
        except BrowserUnavailable as exc:
            record = {**base_record, "status": "failed", "error": str(exc)}
            _append_log(record)
            total_fail += 1
            continue
        except Exception as exc:
            record = {**base_record, "status": "failed", "error": str(exc)[:200]}
            _append_log(record)
            total_fail += 1
            continue

        if not fetched or not dest.exists():
            record = {**base_record, "status": "failed",
                      "error": "browser fetch produced no file"}
            _append_log(record)
            total_fail += 1
            continue

        valid, reason = validate_file(dest)
        if not valid:
            size = dest.stat().st_size
            dest.unlink(missing_ok=True)
            record = {**base_record, "status": "failed",
                      "file_size_bytes": size,
                      "error": f"Content validation failed: {reason}"}
            _append_log(record)
            total_fail += 1
            continue

        record = {**base_record, "status": "success",
                  "file_size_bytes": dest.stat().st_size,
                  "sha256": _sha256(dest)}
        _append_log(record)
        total_new += 1
        time.sleep(REQUEST_DELAY)

    print(f"\n  Browser — new: {total_new}  skipped: {total_skip}  failed: {total_fail}")
