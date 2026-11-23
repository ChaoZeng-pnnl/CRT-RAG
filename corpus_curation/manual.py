"""Registration for hand-added documents.

Files copied into `corpus/` by hand — paywalled guidelines fetched through an
institutional login, anything the automated phases cannot reach — are invisible
to `corpus_metadata.json`, which is built from the catalogues and the download
log. An unindexed file is not in the corpus as far as every downstream step is
concerned: it will not be chunked, embedded, or retrievable, and it will not
appear in the paper's corpus table.

`--register-manual` finds those files, infers what it can from their location,
and writes `corpus/metadata/manual_additions.json` for you to complete. Fields
it cannot know (title, year, url) are left with a REVIEW marker rather than
guessed, because a fabricated citation in a provenance manifest is worse than a
missing one.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import CORPUS_ROOT
from .validate import validate_file

MANUAL_JSON = CORPUS_ROOT / "metadata" / "manual_additions.json"

REVIEW = "REVIEW — fill in"

# Directory name -> (source key, tier). Mirrors the corpus layout.
_TIER_FROM_PATH = {"tier1": 1, "tier2": 2, "tier3": 3, "tier4": 4}


def load_manual_entries() -> list[dict]:
    """Registered hand-added documents, or [] when none exist."""
    if not MANUAL_JSON.exists():
        return []
    try:
        entries = json.loads(MANUAL_JSON.read_text())
    except json.JSONDecodeError as exc:
        print(f"  ⚠  {MANUAL_JSON} is not valid JSON ({exc}); ignoring it.")
        return []
    return entries if isinstance(entries, list) else []


def find_unregistered(catalogued: set[str], logged: set[str]) -> list[Path]:
    """Valid files on disk that no catalogue or log entry accounts for."""
    metadata_dir = CORPUS_ROOT / "metadata"
    known = catalogued | logged | {e.get("filename", "") for e in load_manual_entries()}

    out: list[Path] = []
    for path in sorted(CORPUS_ROOT.rglob("*")):
        if not path.is_file() or metadata_dir in path.parents:
            continue
        if path.name in known:
            continue
        ok, _ = validate_file(path)
        if ok:
            out.append(path)
    return out


def scaffold(path: Path) -> dict:
    """Infer what the file's location tells us; mark the rest for review."""
    parts = path.relative_to(CORPUS_ROOT).parts
    tier = next((_TIER_FROM_PATH[p] for p in parts if p in _TIER_FROM_PATH), 2)
    source = parts[1] if len(parts) > 2 else "manual"

    return {
        "source": source,
        "tier": tier,
        "year": REVIEW,
        "topic": REVIEW,
        "title": REVIEW,
        "document_type": "clinical guideline" if tier <= 2 else "peer-reviewed article",
        "subtopic": "",
        "url": REVIEW,
        "filename": path.name,
        "dest_subdir": str(path.parent.relative_to(CORPUS_ROOT)),
        "added_by": "manual upload",
    }


def register(catalogued: set[str], logged: set[str]) -> list[dict]:
    """Add scaffolds for every unregistered file; keep existing entries intact."""
    found = find_unregistered(catalogued, logged)
    if not found:
        print("\n  No unregistered files — every document on disk is accounted for.")
        return []

    entries = load_manual_entries()
    entries.extend(scaffold(p) for p in found)

    MANUAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_JSON.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    print(f"\n  Registered {len(found)} hand-added file(s) in {MANUAL_JSON}:")
    for path in found:
        print(f"    + [{path.parent.name}] {path.name}")
    print(f"\n  Now edit that file and replace every \"{REVIEW}\" with the real")
    print("  title, year, and source URL, then re-run --audit to rebuild the metadata.")
    return entries


def review_needed(entries: list[dict] | None = None) -> list[tuple[str, list[str]]]:
    """[(filename, [fields still marked REVIEW])] — used to warn at build time."""
    out = []
    for entry in entries if entries is not None else load_manual_entries():
        pending = [k for k, v in entry.items() if isinstance(v, str) and v == REVIEW]
        if pending:
            out.append((entry.get("filename", "?"), pending))
    return out
