"""Multi-scale community detection — the granularity knob for global GraphRAG.

WHY THIS EXISTS
---------------
`community.py` produces a partition that is fine-grained *by construction*:
label propagation with high-degree nodes excluded. That was the right choice for
local retrieval, where hub entities ("venous thromboembolism", degree 484) cause
the candidate explosion that sank graph-only retrieval. But hubs are also the
connectors that would merge clusters into corpus-spanning themes, so excluding
them guarantees small communities:

    486 communities | median 3 chunks, 2 documents, 5 entities | max 12 documents

Edge et al.'s global method works because a coarse partition yields a handful of
summaries each covering a large swath of the corpus. A partition of 486 summaries
covering ~2 documents each has no corpus-level representation to map over, so our
global evaluation never tested the method it named. Granularity is therefore a
factor to sweep, not a fixed implementation detail.

WHAT CHANGES
------------
  * Leiden (RBConfiguration) instead of label propagation, so `resolution` is an
    explicit knob: low resolution -> few large communities, high -> many small.
  * Hubs are INCLUDED in the partition. Their exclusion is what caps community
    size; a global summary of "VTE prophylaxis across bodies" *should* be built
    around the hub entity.
  * Deterministic under a fixed seed, and the seed is recorded.

Reported per scale so "coarse" and "fine" are measured, not adjectival:
communities, entity coverage, and the median/max documents a community spans.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from corpus_index.graph import GraphStore

from .community import (
    MIN_COMMUNITY,
    group_edges,
    load_edges,
    summarise_community,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"
EDGE_CACHE = OUT_DIR / "edges_cache.jsonl"
SEED = 20260907

# Swept on the live graph; see `sweep()`. Chosen to bracket the current partition
# (~486) down to a genuinely corpus-level one (~10).
DEFAULT_RESOLUTIONS = (2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)


# ---------------------------------------------------------------- edge loading

def edges(refresh: bool = False) -> list[dict]:
    """Relations from Neo4j, cached so a resolution sweep needs one query."""
    if EDGE_CACHE.exists() and not refresh:
        return [json.loads(line) for line in EDGE_CACHE.open()]
    store = GraphStore()
    try:
        rows = load_edges(store)
    finally:
        store.close()
    EDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with EDGE_CACHE.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


# ------------------------------------------------------------------- detection

def leiden_labels(rows: list[dict], resolution: float,
                  seed: int = SEED) -> dict[str, int]:
    """Leiden partition over the entity graph, hubs included.

    Edge weight is the number of extracted relations between the two entities:
    a pair asserted by many guideline passages is a stronger tie than one
    asserted once, which is what should hold a clinical theme together.
    """
    import igraph as ig
    import leidenalg

    weights: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        a, b = row["subject"], row["object"]
        if a == b:
            continue
        weights[(a, b) if a < b else (b, a)] += 1

    nodes = sorted({n for pair in weights for n in pair})
    index = {n: i for i, n in enumerate(nodes)}
    graph = ig.Graph(n=len(nodes),
                     edges=[(index[a], index[b]) for a, b in weights])
    graph.es["weight"] = list(weights.values())

    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    return {nodes[i]: community
            for community, members in enumerate(partition)
            for i in members}


# ------------------------------------------------------------------ statistics

def partition_stats(rows: list[dict], labels: dict[str, int],
                    min_size: int = MIN_COMMUNITY) -> dict:
    """Granularity of a partition, in the units the experiment cares about.

    Documents per community is the number that matters: a community spanning one
    document cannot support a corpus-level answer no matter how good its summary.
    """
    groups = group_edges(rows, labels, min_size=min_size)
    docs, chunks, ents = [], [], []
    for group in groups.values():
        ids = {e["chunk_id"] for e in group if e.get("chunk_id")}
        chunks.append(len(ids))
        docs.append(len({i.split(":")[0] for i in ids}))
        ents.append(len({e["subject"] for e in group} | {e["object"] for e in group}))
    covered = {e["subject"] for g in groups.values() for e in g} | \
              {e["object"] for g in groups.values() for e in g}
    total = {r["subject"] for r in rows} | {r["object"] for r in rows}

    def med(values):
        return statistics.median(values) if values else 0

    return {
        "communities": len(groups),
        "entity_coverage": round(len(covered) / len(total), 3) if total else 0.0,
        "median_docs": med(docs), "mean_docs": round(statistics.mean(docs), 2) if docs else 0,
        "max_docs": max(docs) if docs else 0,
        "communities_ge5_docs": sum(d >= 5 for d in docs),
        "median_chunks": med(chunks), "max_chunks": max(chunks) if chunks else 0,
        "median_entities": med(ents), "max_entities": max(ents) if ents else 0,
    }


def sweep(resolutions=DEFAULT_RESOLUTIONS, min_size: int = MIN_COMMUNITY) -> list[dict]:
    """Granularity vs resolution. No LLM calls — this is the sizing step."""
    rows = edges()
    print(f"  Relations: {len(rows)} | seed {SEED}")
    print(f"  {'resolution':>10} {'comms':>6} {'cover':>6} {'med.doc':>8} "
          f"{'max.doc':>8} {'>=5doc':>7} {'med.chunk':>10} {'med.ent':>8}")
    out = []
    for resolution in resolutions:
        stats = partition_stats(rows, leiden_labels(rows, resolution), min_size)
        stats["resolution"] = resolution
        out.append(stats)
        print(f"  {resolution:>10.3f} {stats['communities']:>6} "
              f"{stats['entity_coverage']:>6.3f} {stats['median_docs']:>8.1f} "
              f"{stats['max_docs']:>8} {stats['communities_ge5_docs']:>7} "
              f"{stats['median_chunks']:>10.1f} {stats['median_entities']:>8.1f}")

    path = OUT_DIR / "community_scale_sweep.json"
    path.write_text(json.dumps({"seed": SEED, "min_size": min_size,
                                "scales": out}, indent=2))
    print(f"\n  Wrote {path}")
    return out


# -------------------------------------------------------------------- building

def build_scale(resolution: float, label: str, concurrency: int = 6,
                min_size: int = MIN_COMMUNITY, model: str | None = None) -> dict:
    """Summarise one scale into `kg/communities_<label>.jsonl`."""
    rows = edges()
    groups = group_edges(rows, leiden_labels(rows, resolution), min_size=min_size)
    print(f"  scale '{label}' (resolution {resolution}): {len(groups)} communities")

    from concurrent.futures import ThreadPoolExecutor

    kwargs = {"model": model} if model else {}
    summaries = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for result in pool.map(
                lambda kv: summarise_community(kv[0], kv[1], **kwargs),
                sorted(groups.items())):
            if result:
                summaries.append(result)
                print(f"    summarised {len(summaries)}/{len(groups)}",
                      end="\r", flush=True)
    print()

    out = OUT_DIR / f"communities_{label}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {out} ({len(summaries)} summaries)")
    return {"label": label, "resolution": resolution,
            "communities": len(groups), "summarised": len(summaries)}
