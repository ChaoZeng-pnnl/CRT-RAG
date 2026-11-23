"""Entry point for `python3 -m corpus_ingest`."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import OUT_FILE, ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract and chunk the validated corpus into chunks.jsonl.")
    parser.add_argument("--stats", action="store_true",
                        help="Report what would be produced without writing chunks.")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Ingest only the first N documents (for a quick check).")
    args = parser.parse_args(argv)

    print("=" * 65)
    print("  Corpus ingestion — extract + chunk")
    print("=" * 65)

    stats = ingest(limit=args.limit, stats_only=args.stats)

    print(f"\n  Documents: {stats['documents_ingested']} ingested, "
          f"{stats['documents_failed']} failed, of {stats['documents_total']}")
    print(f"  Chunks:    {stats['chunks']}")
    print(f"  By tier:   {stats['chunks_by_tier']}")
    print(f"  By format: {stats['chunks_by_format']}")
    if stats["openrn_chunks_dropped"]:
        print(f"  OpenRN off-topic chunks dropped: {stats['openrn_chunks_dropped']}")
    if stats["failures"]:
        print(f"\n  Failures ({len(stats['failures'])}):")
        for f in stats["failures"][:10]:
            print(f"    - {f['filename']}: {f['error']}")
    if not args.stats:
        print(f"\n  Wrote {OUT_FILE}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
