#!/usr/bin/env python3
"""Perioperative Gynecologic Nursing RAG — corpus downloader (compatibility shim).

The implementation now lives in the `corpus_curation` package; this file is kept
so existing commands and scripts keep working. Prefer:

    python3 -m corpus_curation --audit

Usage:
    python3 download_corpus.py
    python3 download_corpus.py --audit
    python3 download_corpus.py --update --since 2026-01-01
"""

from __future__ import annotations

import sys

from corpus_curation.cli import main

if __name__ == "__main__":
    sys.exit(main() or 0)
