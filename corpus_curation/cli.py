"""Argument parsing and phase orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import time

from .config import LOG_CSV, LOG_JSONL, REQUEST_DELAY
from .logs import _init_csv, _now, _run_summary
from .metadata import _write_corpus_metadata
from .validate import audit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SOURCE_CHOICES = {"curated", "scrape", "pmc", "openalex", "elsevier", "unpaywall", "browser"}
_UPDATE_SOURCES = {"pmc", "openalex", "elsevier", "unpaywall"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or update the nursing RAG corpus.")
    parser.add_argument("--update", action="store_true",
                        help="Run live discovery sources only.")
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help="Only discover research published on or after this date.")
    parser.add_argument("--sources", metavar="SOURCE[,SOURCE...]",
                        help="Restrict phases to: " + ", ".join(sorted(_SOURCE_CHOICES)) + ".")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Redownload selected documents even when a cached file exists.")
    parser.add_argument("--audit", action="store_true",
                        help="Validate every file already in corpus/ and exit. "
                             "Writes corpus/metadata/validation_report.json.")
    parser.add_argument("--delete-invalid", action="store_true",
                        help="With --audit: delete files that fail validation.")
    parser.add_argument("--register-manual", action="store_true",
                        help="Scaffold metadata for hand-added files in corpus/ that no "
                             "catalogue or log accounts for, then exit. Writes "
                             "corpus/metadata/manual_additions.json for you to complete.")
    args = parser.parse_args(argv)
    if args.since:
        try:
            args.since = datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            parser.error("--since must use YYYY-MM-DD")
    if args.sources:
        args.sources = {source.strip().lower() for source in args.sources.split(",") if source.strip()}
        invalid = args.sources - _SOURCE_CHOICES
        if invalid:
            parser.error("unknown source(s): " + ", ".join(sorted(invalid)))
    else:
        args.sources = _UPDATE_SOURCES if args.update else _SOURCE_CHOICES
    if args.update:
        args.sources &= _UPDATE_SOURCES
    return args


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    print("=" * 65)
    print("  Perioperative Gynecologic RAG — Corpus Downloader")
    print(f"  Started: {_now()}")

    # Registration mode: account for hand-added files, then stop.
    if args.register_manual:
        print("  Mode: register manual additions")
        print("=" * 65)
        from .catalog.browser_targets import BROWSER_TARGETS
        from .catalog.documents import DOCUMENTS
        from .manual import register

        catalogued = {d["filename"] for d in DOCUMENTS} | {d["filename"] for d in BROWSER_TARGETS}
        logged = set()
        if LOG_JSONL.exists():
            import json as _json
            for line in LOG_JSONL.read_text().splitlines():
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if rec.get("status") == "success":
                    logged.add(rec.get("filename", ""))
        register(catalogued, logged)
        _write_corpus_metadata()
        print("=" * 65)
        return 0

    # Audit mode: validate what is on disk, then stop.
    if args.audit:
        print("  Mode: audit (no downloads)")
        print("=" * 65)
        bad = audit(delete=args.delete_invalid)
        if args.delete_invalid:
            _write_corpus_metadata()
        print("=" * 65)
        return 1 if bad else 0

    # Imported here, not at module scope, so --audit works without requests/bs4.
    from .catalog.documents import DOCUMENTS
    from .catalog.searches import (
        OPENALEX_SEARCHES, PMC_SEARCHES, SCRAPE_TARGETS, UNPAYWALL_DOIS,
    )
    from .fetchers.curated import download_document, scrape_and_download
    from .fetchers.elsevier import _fetch_elsevier_articles
    from .fetchers.openalex import _fetch_openalex_articles
    from .fetchers.pmc import _fetch_pmc_oa_articles
    from .fetchers.unpaywall import _fetch_unpaywall_dois
    from .fetchers.browser_fetch import _fetch_browser_targets
    from .catalog.browser_targets import BROWSER_TARGETS

    if args.update:
        print("  Mode: update")
    if args.since:
        print(f"  Since: {args.since:%Y-%m-%d}")
    print("=" * 65)

    _init_csv(LOG_CSV)
    log_start = len(LOG_JSONL.read_text().splitlines()) if LOG_JSONL.exists() else 0

    # Phase 1 — Direct downloads from curated catalogue
    if "curated" in args.sources:
        print(f"\n── Phase 1: Curated catalogue ({len(DOCUMENTS)} documents) ──")
        for i, doc in enumerate(DOCUMENTS, 1):
            print(f"\n[{i}/{len(DOCUMENTS)}] {doc['source']} | {doc['topic'][:60]}")
            download_document(doc, force_refresh=args.force_refresh)
            time.sleep(REQUEST_DELAY)

    # Phase 2 — Scrape listing pages
    if "scrape" in args.sources:
        print(f"\n── Phase 2: Scraping source listing pages ──")
        for target in SCRAPE_TARGETS:
            scrape_and_download(target, force_refresh=args.force_refresh)

    # Phase 3 — PubMed Central open-access articles
    if "pmc" in args.sources:
        print(f"\n── Phase 3: PubMed Central open-access search ({len(PMC_SEARCHES)} queries) ──")
        _fetch_pmc_oa_articles(args.since, args.force_refresh)

    # Phase 4 — OpenAlex open-access PDF articles
    if "openalex" in args.sources:
        print(f"\n── Phase 4: OpenAlex open-access search ({len(OPENALEX_SEARCHES)} queries) ──")
        _fetch_openalex_articles(args.since, args.force_refresh)

    # Phase 5 — Elsevier ScienceDirect open-access XML articles
    if "elsevier" in args.sources:
        print(f"\n── Phase 5: Elsevier ScienceDirect open-access search ({len(OPENALEX_SEARCHES)} queries) ──")
        _fetch_elsevier_articles(args.since, args.force_refresh)

    # Phase 6 — Unpaywall DOI-based download (key society guidelines)
    if "unpaywall" in args.sources:
        print(f"\n── Phase 6: Unpaywall DOI lookup ({len(UNPAYWALL_DOIS)} DOIs) ──")
        _fetch_unpaywall_dois(args.force_refresh)

    # Phase 7 — Browser-only documents (JS-rendered or bot-protected sources)
    if "browser" in args.sources:
        print(f"\n\u2500\u2500 Phase 7: Headless-browser retrieval ({len(BROWSER_TARGETS)} documents) \u2500\u2500")
        _fetch_browser_targets(args.force_refresh)

    # Summary
    print("\n" + "=" * 65)
    _run_summary(log_start)
    print(f"\n  Log files:")
    print(f"    {LOG_JSONL}")
    print(f"    {LOG_CSV}")

    # Write full corpus metadata JSON
    _write_corpus_metadata()

    print("=" * 65)
