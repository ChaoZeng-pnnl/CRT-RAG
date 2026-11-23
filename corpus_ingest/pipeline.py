"""Manifest -> chunks.jsonl.

Driven by `corpus/metadata/corpus_metadata.json`, never by a directory walk, so
only content-validated documents are ingested and every chunk inherits the
tier/source/year that tier-weighted retrieval depends on.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .chunker import CHARS_PER_TOKEN, chunk_sections
from .extractors import extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "corpus" / "metadata" / "corpus_metadata.json"
OUT_DIR = PROJECT_ROOT / "chunks"
OUT_FILE = OUT_DIR / "chunks.jsonl"
REPORT = OUT_DIR / "ingest_report.json"

# The two OpenRN textbooks are 188 MB of a 307 MB PDF corpus. Ingested whole
# they swamp the chunk distribution with general nursing fundamentals and skew
# every retrieval metric away from the perioperative gynecologic material the
# system is actually being evaluated on. Keep only chapters on the topic axes.
_OPENRN_KEEP = re.compile(
    r"periop|preop|postop|surg|infection|asepsis|aseptic|sterile|wound|"
    r"pain|analgesi|opioid|catheter|urinary|elimination|foley|"
    r"vital sign|assessment|hygiene|dressing|drain|iv therapy|intravenous|"
    r"fluid|medication administration|safety|documentation|communication",
    re.IGNORECASE,
)
_OPENRN_MARKER = "OpenRN_Nursing_"


def _tier_number(trust_level: str) -> int:
    match = re.search(r"Tier\s*(\d)", trust_level or "")
    return int(match.group(1)) if match else 4


def _doc_id(record: dict) -> str:
    basis = record.get("sha256") or record.get("filename", "")
    return hashlib.sha1(basis.encode()).hexdigest()[:12]


def _keep_openrn_section(chunk_section: str, text: str) -> bool:
    head = f"{chunk_section} {text[:400]}"
    return bool(_OPENRN_KEEP.search(head))


def ingest(manifest: Path = MANIFEST, out_file: Path = OUT_FILE,
           limit: int | None = None, stats_only: bool = False) -> dict:
    records = json.loads(manifest.read_text())
    if limit:
        records = records[:limit]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    handle = None if stats_only else out_file.open("w", encoding="utf-8")

    stats = {
        "documents_total": len(records),
        "documents_ingested": 0,
        "documents_failed": 0,
        "chunks": 0,
        "chunks_by_tier": Counter(),
        "chunks_by_format": Counter(),
        "chunks_by_source": Counter(),
        "openrn_chunks_dropped": 0,
        "failures": [],
    }

    try:
        for record in records:
            path = Path(record["dest_path"])
            if not path.exists():
                stats["documents_failed"] += 1
                stats["failures"].append({"filename": record["filename"],
                                          "error": "file missing"})
                continue

            try:
                sections = extract(path, record.get("file_format", ""))
                chunks = chunk_sections(sections)
            except Exception as exc:
                stats["documents_failed"] += 1
                stats["failures"].append({"filename": record["filename"],
                                          "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue

            is_openrn = _OPENRN_MARKER in record["filename"]
            tier = _tier_number(record.get("trust_level", ""))
            doc_id = _doc_id(record)
            kept = 0

            for chunk in chunks:
                if is_openrn and not _keep_openrn_section(chunk.section, chunk.text):
                    stats["openrn_chunks_dropped"] += 1
                    continue

                row = {
                    "chunk_id": f"{doc_id}:{chunk.index:04d}",
                    "doc_id": doc_id,
                    "text": chunk.text,
                    "n_chars": len(chunk.text),
                    "est_tokens": len(chunk.text) // CHARS_PER_TOKEN,
                    "section": chunk.section,
                    "page": chunk.page,
                    "kind": chunk.kind,
                    # Provenance carried onto every chunk — this is what makes
                    # tier-weighted ranking and citation verification possible.
                    "tier": tier,
                    "trust_level": record.get("trust_level", ""),
                    "source": record.get("source", ""),
                    "organization": record.get("organization", ""),
                    "title": record.get("title", ""),
                    "topic": record.get("topic", ""),
                    "subtopic": record.get("subtopic", ""),
                    "document_type": record.get("document_type", ""),
                    "publication_year": record.get("publication_year", ""),
                    "url": record.get("url", ""),
                    "filename": record.get("filename", ""),
                    "file_format": record.get("file_format", ""),
                    "sha256": record.get("sha256", ""),
                }
                if handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                stats["chunks_by_tier"][tier] += 1
                stats["chunks_by_format"][record.get("file_format", "")] += 1
                stats["chunks_by_source"][record.get("source", "")] += 1

            stats["chunks"] += kept
            stats["documents_ingested"] += 1
    finally:
        if handle:
            handle.close()

    stats["chunks_by_tier"] = dict(sorted(stats["chunks_by_tier"].items()))
    stats["chunks_by_format"] = dict(stats["chunks_by_format"])
    stats["chunks_by_source"] = dict(sorted(stats["chunks_by_source"].items(),
                                            key=lambda kv: -kv[1]))
    if not stats_only:
        REPORT.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats
