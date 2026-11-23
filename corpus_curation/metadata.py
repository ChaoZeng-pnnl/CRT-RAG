"""Builds corpus/metadata/corpus_metadata.json from the log and catalogue."""

from __future__ import annotations

from pathlib import Path
import json

from .catalog.browser_targets import BROWSER_TARGETS
from .catalog.documents import DOCUMENTS
from .config import CORPUS_ROOT, LOG_JSONL
from .manual import load_manual_entries, review_needed
from .validate import validate_file


_ORG_MAP = {
    "NICE":        "National Institute for Health and Care Excellence",
    "WHO":         "World Health Organization",
    "CDC":         "Centers for Disease Control and Prevention",
    "RCOG":        "Royal College of Obstetricians and Gynaecologists",
    "ACOG":        "American College of Obstetricians and Gynecologists",
    "RNAO":        "Registered Nurses' Association of Ontario",
    "OpenRN":      "Open Resources for Nursing (OpenRN) / WisTech Open",
    "AUGS":        "American Urogynecologic Society",
    "AHRQ_PSNet":  "Agency for Healthcare Research and Quality — Patient Safety Network",
    "PubMed_PMC":  "National Center for Biotechnology Information / PubMed Central",
    "ERAS_Society": "ERAS® Society — Enhanced Recovery After Surgery",
    "OpenAlex":    "OpenAlex / OurResearch",
    "Elsevier":    "Elsevier / ScienceDirect",
}
_TIER_LABEL = {
    1: "Tier 1 — Highest Authority",
    2: "Tier 2 — Nursing Best Practice",
    3: "Tier 3 — Reflective Learning & Safety",
    4: "Tier 4 — Primary Research Literature",
}


def _write_corpus_metadata():
    """Generate corpus/metadata/corpus_metadata.json from all DOCUMENTS entries."""
    # Index the most recent successful log entry per filename
    log_by_file: dict = {}
    if LOG_JSONL.exists():
        for line in LOG_JSONL.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "success":
                log_by_file[r["filename"]] = r

    records = []
    _skipped: list[tuple[str, str]] = []
    # Browser-retrieved documents are catalogued separately but indexed the same
    # way; without them, Phase 7 files for a source that also appears in
    # DOCUMENTS would be dropped by the source-based filter below.
    # BROWSER_TARGETS first: where a document appears in both catalogues, the
    # browser entry is the one that actually retrieved the file.
    _by_filename = {doc["filename"]: doc for doc in DOCUMENTS}
    _by_filename.update({doc["filename"]: doc for doc in BROWSER_TARGETS})
    # Hand-added documents (institutional access, anything automation cannot
    # reach) are registered separately but belong in the corpus like any other.
    _manual = load_manual_entries()
    _by_filename.update({doc["filename"]: doc for doc in _manual})
    for doc in _by_filename.values():
        fname    = doc["filename"]
        fpath    = CORPUS_ROOT / doc["dest_subdir"] / fname
        log_entry = log_by_file.get(fname, {})

        # Only index documents that are on disk AND hold real content, so the
        # published metadata never claims a document the corpus does not have.
        if not fpath.exists():
            continue
        ok, reason = validate_file(fpath)
        if not ok:
            _skipped.append((fname, reason))
            continue

        tier        = doc.get("tier", 2)
        file_format = "PDF" if fname.lower().endswith(".pdf") else (
                      "XML" if fname.lower().endswith(".xml") else "HTML")

        records.append({
            "title":              doc.get("title", doc["topic"]),
            "source":             doc["source"],
            "organization":       _ORG_MAP.get(doc["source"], doc["source"]),
            "topic":              doc["topic"],
            "subtopic":           doc.get("subtopic", ""),
            "document_type":      doc.get("document_type", "clinical guideline"),
            "publication_year":   str(doc.get("year", "")),
            "trust_level":        doc.get("trust_level", _TIER_LABEL.get(tier, "")),
            "specialty":          "Perioperative Gynecologic Nursing",
            "url":                doc["url"],
            "file_format":        file_format,
            "retrieval_priority": doc.get("retrieval_priority",
                                          "critical" if tier == 1 else
                                          "high"     if tier == 2 else "medium"),
            "filename":           fname,
            "dest_path":          str(fpath),
            "sha256":             log_entry.get("sha256", ""),
            "file_size_bytes":    log_entry.get("file_size_bytes", ""),
        })

    # Include all log-sourced articles (phases 3–6) not already covered by DOCUMENTS
    _docs_sources = {doc["source"] for doc in DOCUMENTS}
    seen_filenames = {r["filename"] for r in records}   # avoid duplicates

    for line in (LOG_JSONL.read_text().splitlines() if LOG_JSONL.exists() else []):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("source") in _docs_sources or r.get("status") != "success":
            continue
        fname = r.get("filename", "")
        if fname in seen_filenames:
            continue
        fpath = Path(r["dest_path"])
        if not fpath.exists():
            continue
        ok, reason = validate_file(fpath)
        if not ok:
            _skipped.append((fname, reason))
            continue
        tier = int(r.get("tier", 4))
        records.append({
            "title":              r.get("title", fpath.stem),
            "source":             r["source"],
            "organization":       _ORG_MAP.get(r["source"], r["source"]),
            "topic":              r.get("topic", ""),
            "subtopic":           r.get("subtopic", ""),
            "document_type":      r.get("document_type", "peer-reviewed article"),
            "publication_year":   str(r.get("year", "")),
            "trust_level":        _TIER_LABEL.get(tier, _TIER_LABEL[4]),
            "specialty":          "Perioperative Gynecologic Nursing",
            "url":                r.get("url", ""),
            "file_format":        "XML" if fname.endswith(".xml") else "PDF",
            "retrieval_priority": r.get("retrieval_priority",
                                        "critical" if tier == 1 else
                                        "high"     if tier == 2 else "medium"),
            "filename":           fname,
            "dest_path":          r["dest_path"],
            "sha256":             r.get("sha256", ""),
            "file_size_bytes":    r.get("file_size_bytes", ""),
        })
        seen_filenames.add(fname)

    out_path = CORPUS_ROOT / "metadata" / "corpus_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n  Metadata JSON: {out_path}  ({len(records)} documents indexed)")
    if _skipped:
        print(f"  Excluded {len(_skipped)} file(s) that failed content validation:")
        for fname, reason in _skipped:
            print(f"    - {fname}: {reason}")

    pending = review_needed(_manual)
    if pending:
        print(f"\n  ⚠  {len(pending)} hand-added document(s) still carry placeholder metadata.")
        print("     They are indexed, but their provenance fields are incomplete:")
        for fname, fields in pending:
            print(f"    - {fname}: {', '.join(fields)}")
        print("     Edit corpus/metadata/manual_additions.json before freezing the corpus.")
