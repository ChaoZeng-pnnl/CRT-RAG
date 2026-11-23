"""Append-only JSONL/CSV download log."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json

from .config import LOG_CSV, LOG_JSONL


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _init_csv(path: Path):
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "source", "tier", "year", "topic",
                "url", "filename", "dest_path", "status",
                "http_code", "file_size_bytes", "sha256", "error",
            ])
            writer.writeheader()


def _append_log(record: dict):
    """Append one record to both JSONL and CSV logs."""
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()), extrasaction="ignore")
        writer.writerow(record)

    status_icon = "✓" if record["status"] == "success" else ("⊘" if record["status"] == "skipped" else "✗")
    print(f"  {status_icon}  [{record['source']}] {record['filename']}  →  {record['status']}")
    if record.get("error"):
        print(f"     error: {record['error']}")


def _run_summary(log_start: int):
    successes = skipped = failed = 0
    if LOG_JSONL.exists():
        for line in LOG_JSONL.read_text().splitlines()[log_start:]:
            try:
                status = json.loads(line).get("status")
            except json.JSONDecodeError:
                continue
            if status == "success":
                successes += 1
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed += 1
    print(f"  Downloaded (new):  {successes}")
    print(f"  Skipped (cached):  {skipped}")
    print(f"  Failed:            {failed}")
