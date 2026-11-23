"""Content validation: proves a saved file holds the document it claims to.

The original pipeline treated HTTP 200 as success. It is not. Nine WHO
"PDFs" were 755-byte DSpace JavaScript shells, the ACOG committee opinion was
a branded "Page Not Found", and the CDC CAUTI guideline was a "Preparing to
download..." interstitial — all recorded as successes with a SHA-256 that
looked exactly like provenance.

Every file is now checked for (a) the right container format and (b) enough
extractable text to be a real document, both at download time and after the
fact via `--audit`.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .config import (
    CORPUS_ROOT,
    ERROR_PAGE_MARKERS,
    MIN_PDF_BYTES,
    MIN_TEXT_CHARS,
)

_TAG_RE = re.compile(r"<[^>]+>")
_DROP_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_WS_RE = re.compile(r"\s+")


def extract_text(path: Path) -> str:
    """Best-effort plain text, without requiring a PDF library."""
    raw = path.read_bytes()

    if raw[:5] == b"%PDF-":
        # Concatenate the text-showing operators inside the content streams.
        # Crude, but enough to tell a real guideline from an empty shell.
        chunks = re.findall(rb"\((?:\\.|[^\\()])*\)", raw)
        text = b" ".join(c[1:-1] for c in chunks).decode("latin-1", "ignore")
        return _WS_RE.sub(" ", text).strip()

    decoded = raw.decode("utf-8", "ignore")
    stripped = _TAG_RE.sub(" ", _DROP_RE.sub(" ", decoded))
    return _WS_RE.sub(" ", html.unescape(stripped)).strip()


def validate_file(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). `reason` is empty when the file is usable."""
    if not path.exists():
        return False, "file missing"

    size = path.stat().st_size
    raw_head = path.read_bytes()[:4096]
    suffix = path.suffix.lower()

    # A file named .pdf must actually be a PDF — this is what caught WHO.
    if suffix == ".pdf":
        if raw_head[:5] != b"%PDF-":
            sniff = raw_head.decode("utf-8", "ignore").lstrip()[:40].replace("\n", " ")
            return False, f"not a PDF (starts with {sniff!r})"
        if size < MIN_PDF_BYTES:
            return False, f"PDF too small ({size} bytes < {MIN_PDF_BYTES})"

    text = extract_text(path)
    lowered = text[:4000].lower()
    for marker in ERROR_PAGE_MARKERS:
        if marker in lowered:
            return False, f"error/interstitial page (matched {marker!r})"

    if suffix != ".pdf" and len(text) < MIN_TEXT_CHARS:
        return False, f"only {len(text)} chars of text (< {MIN_TEXT_CHARS})"

    return True, ""


def audit(corpus_root: Path = CORPUS_ROOT, delete: bool = False) -> list[dict]:
    """Re-validate every file on disk. Returns one record per rejected file."""
    metadata_dir = corpus_root / "metadata"
    files = sorted(
        p for p in corpus_root.rglob("*")
        if p.is_file() and metadata_dir not in p.parents
    )

    bad: list[dict] = []
    for path in files:
        ok, reason = validate_file(path)
        if ok:
            continue
        bad.append({
            "dest_path": str(path),
            "filename": path.name,
            "tier": next((part for part in path.parts if part.startswith("tier")), ""),
            "source": path.parent.name,
            "size_bytes": path.stat().st_size,
            "reason": reason,
        })

    print(f"\n  Audited {len(files)} files — {len(files) - len(bad)} valid, {len(bad)} rejected")
    for rec in bad:
        print(f"  ✗  [{rec['tier']}/{rec['source']}] {rec['filename']}")
        print(f"       {rec['reason']}  ({rec['size_bytes']} bytes)")

    report = metadata_dir / "validation_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(bad, indent=2, ensure_ascii=False))
    print(f"\n  Report: {report}")

    if delete:
        for rec in bad:
            Path(rec["dest_path"]).unlink(missing_ok=True)
        print(f"  Deleted {len(bad)} invalid files — re-run the download phases to refetch.")
    elif bad:
        print("  Re-run with --delete-invalid to remove them, then refetch.")

    return bad
