"""Report or apply cross-tier duplicate marking."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from corpus_index.config import CHUNKS_FILE

from .detect import JACCARD_THRESHOLD, cluster, elect

REPORT_FILE = Path(CHUNKS_FILE).parent / "dedup_report.json"

UPDATE_FLAGS = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
SET c.is_canonical = row.is_canonical,
    c.duplicate_of = row.duplicate_of,
    c.cluster_size = row.cluster_size
RETURN count(c) AS updated
"""

LINK_DUPLICATES = """
UNWIND $rows AS row
MATCH (dup:Chunk {chunk_id: row.chunk_id})
MATCH (canon:Chunk {chunk_id: row.duplicate_of})
MERGE (dup)-[:DUPLICATE_OF]->(canon)
RETURN count(*) AS links
"""


def load_chunks() -> list[dict]:
    with Path(CHUNKS_FILE).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def summarise(rows: list[dict], verdict: dict[str, dict]) -> dict:
    by_id = {r["chunk_id"]: r for r in rows}
    dupes = [cid for cid, v in verdict.items() if not v["is_canonical"]]

    # The interesting case: a cluster spanning more than one tier, where the
    # lower-authority copy would otherwise compete with the original.
    cross_tier = {cid for cid, v in verdict.items() if len(v["cluster_tiers"]) > 1}

    demoted_by_source = Counter(by_id[cid]["source"] for cid in dupes)
    demoted_by_tier = Counter(by_id[cid]["tier"] for cid in dupes)
    cluster_sizes = Counter(v["cluster_size"] for v in verdict.values()
                            if v["cluster_size"] > 1)

    # Which tier pairs collide most — this is the number the paper needs.
    pairings = Counter()
    for cid in cross_tier:
        pairings[tuple(verdict[cid]["cluster_tiers"])] += 1

    return {
        "chunks": len(rows),
        "duplicates_marked": len(dupes),
        "canonical_remaining": len(rows) - len(dupes),
        "chunks_in_cross_tier_clusters": len(cross_tier),
        "duplicates_by_source": dict(demoted_by_source.most_common()),
        "duplicates_by_tier": {str(k): v for k, v in sorted(demoted_by_tier.items())},
        "cluster_size_histogram": {str(k): v for k, v in sorted(cluster_sizes.items())},
        "cross_tier_cluster_tiers": {str(list(k)): v for k, v in pairings.most_common(10)},
        "jaccard_threshold": JACCARD_THRESHOLD,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mark cross-tier duplicate chunks.")
    parser.add_argument("--apply", action="store_true",
                        help="Write flags to chunks.jsonl and Neo4j (default: report only).")
    parser.add_argument("--examples", type=int, default=5,
                        help="Show this many cross-tier duplicate examples.")
    args = parser.parse_args(argv)

    print("=" * 65)
    print("  Cross-tier duplicate detection")
    print("=" * 65)

    rows = load_chunks()
    print(f"  Chunks: {len(rows)}  (threshold: Jaccard >= {JACCARD_THRESHOLD})")

    assignment = cluster(rows)
    verdict = elect(rows, assignment)
    stats = summarise(rows, verdict)

    print(f"\n  Duplicates marked:   {stats['duplicates_marked']}")
    print(f"  Canonical remaining: {stats['canonical_remaining']}")
    print(f"  In cross-tier clusters: {stats['chunks_in_cross_tier_clusters']}")
    print(f"\n  Demoted by source: {stats['duplicates_by_source']}")
    print(f"  Demoted by tier:   {stats['duplicates_by_tier']}")
    print(f"  Cross-tier cluster shapes: {stats['cross_tier_cluster_tiers']}")

    if args.examples:
        by_id = {r["chunk_id"]: r for r in rows}
        shown = 0
        print(f"\n  Examples of cross-tier duplicates:")
        for cid, v in verdict.items():
            if v["is_canonical"] or len(v["cluster_tiers"]) < 2:
                continue
            dup, canon = by_id[cid], by_id[v["duplicate_of"]]
            print(f"\n   demoted: [T{dup['tier']} {dup['source']}] {dup['title'][:52]}")
            print(f"   kept:    [T{canon['tier']} {canon['source']}] {canon['title'][:52]}")
            print(f"   text:    {dup['text'][:110].replace(chr(10),' ')}...")
            shown += 1
            if shown >= args.examples:
                break

    REPORT_FILE.write_text(json.dumps(stats, indent=2))
    print(f"\n  Report: {REPORT_FILE}")

    if not args.apply:
        print("\n  Report only — re-run with --apply to write the flags.")
        print("=" * 65)
        return 0

    # 1. chunks.jsonl gains the flags, so a rebuild keeps them.
    out = Path(CHUNKS_FILE)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            row.update(verdict[row["chunk_id"]])
            row.pop("cluster_tiers", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  Updated {out}")

    # 2. Neo4j gains the same flags plus an explicit :DUPLICATE_OF edge.
    from corpus_index.graph import GraphStore

    store = GraphStore()
    try:
        payload = [{"chunk_id": cid,
                    "is_canonical": v["is_canonical"],
                    "duplicate_of": v["duplicate_of"],
                    "cluster_size": v["cluster_size"]}
                   for cid, v in verdict.items()]
        updated = 0
        with store.session() as session:
            session.run("CREATE INDEX chunk_canonical IF NOT EXISTS "
                        "FOR (c:Chunk) ON (c.is_canonical)")
            for i in range(0, len(payload), 1000):
                record = session.run(UPDATE_FLAGS, {"rows": payload[i:i + 1000]}).single()
                updated += record["updated"] if record else 0
            links = [p for p in payload if p["duplicate_of"]]
            linked = 0
            for i in range(0, len(links), 1000):
                record = session.run(LINK_DUPLICATES, {"rows": links[i:i + 1000]}).single()
                linked += record["links"] if record else 0
        print(f"  Neo4j: {updated} chunks flagged, {linked} :DUPLICATE_OF edges")
    finally:
        store.close()

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
