"""Cross-tier near-duplicate detection for the chunk corpus.

    python3 -m corpus_dedup --report        # what would be marked
    python3 -m corpus_dedup --apply         # write flags to chunks.jsonl + Neo4j
"""

from .detect import cluster, elect, jaccard, minhash

__all__ = ["cluster", "elect", "jaccard", "minhash"]
