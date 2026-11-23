"""Corpus curation for the perioperative gynecologic nursing RAG system.

Layout
------
    config.py            paths, HTTP settings, credentials, validation thresholds
    validate.py          content validation and the corpus-wide audit
    http_utils.py        shared requests wrapper
    logs.py              append-only JSONL/CSV download log
    metadata.py          builds corpus_metadata.json from validated files only
    cli.py               argument parsing and phase orchestration
    catalog/documents.py curated Tier 1-3 catalogue
    catalog/searches.py  PMC / OpenAlex / scrape / Unpaywall query lists
    fetchers/            one module per acquisition phase

Usage
-----
    python3 -m corpus_curation --audit
    python3 -m corpus_curation --sources curated --force-refresh
    python3 -m corpus_curation --update --since 2026-01-01
"""

from .cli import main

__all__ = ["main"]
