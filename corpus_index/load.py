"""chunks.jsonl -> embeddings -> Neo4j."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .config import CHUNKS_FILE
from .embedder import Embedder
from .graph import GraphStore

WRITE_BATCH = 200


def read_chunks(path: Path = CHUNKS_FILE, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _embedding_input(row: dict) -> str:
    """Prepend the section breadcrumb so the vector carries document context.

    A chunk reading "Remove the catheter within 24 hours" is ambiguous alone;
    with "NICE NG180 > Postoperative care > Urinary catheters" in front of it,
    the embedding lands near the questions people actually ask.
    """
    parts = [row.get("title", ""), row.get("section", ""), row["text"]]
    return "\n".join(p for p in parts if p).strip()


def _to_graph_row(row: dict, embedding: list[float], seq: int) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "text": row["text"],
        "section": row.get("section", ""),
        "page": row.get("page"),
        "kind": row.get("kind", "body"),
        "n_chars": row.get("n_chars", len(row["text"])),
        "est_tokens": row.get("est_tokens", 0),
        "tier": row.get("tier", 4),
        "trust_level": row.get("trust_level", ""),
        "source": row.get("source", ""),
        "organization": row.get("organization", ""),
        "title": row.get("title", ""),
        "topic": row.get("topic", ""),
        "subtopic": row.get("subtopic", ""),
        "document_type": row.get("document_type", ""),
        "publication_year": str(row.get("publication_year", "")),
        "url": row.get("url", ""),
        "filename": row.get("filename", ""),
        "file_format": row.get("file_format", ""),
        "sha256": row.get("sha256", ""),
        "embedding": embedding,
        "seq": seq,
        # Dedup flags travel with the chunk. They previously reached Neo4j only
        # via a separate `corpus_dedup --apply` write, so rebuilding the database
        # produced a graph with no `is_canonical` at all — and because the
        # retrieval filter treats NULL as canonical, duplicate filtering silently
        # switched off instead of failing. A rebuild must be self-sufficient.
        "is_canonical": row.get("is_canonical", True),
        "duplicate_of": row.get("duplicate_of"),
        "cluster_size": row.get("cluster_size", 1),
    }


def _batches(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def build(limit: int | None = None, embed_only: bool = False,
          skip_embedding: bool = False, stage_only: bool = False,
          note: str = "load") -> dict:
    """Embed and load chunks. Loaded documents are STAGED (invisible to search)
    and, unless `stage_only`, published together as one release at the end,
    so a half-written load is never searchable."""
    rows = read_chunks(limit=limit)
    print(f"  Chunks to index: {len(rows)}")

    embedder = Embedder()
    if skip_embedding:
        # Cache-only: never call the endpoint, but still produce vectors, or the
        # loader silently writes nothing. (This previously left `vectors` empty,
        # so zip(rows, vectors) yielded no rows and the graph came back with 0
        # chunks while entities loaded fine — a silent, total data loss.)
        print("  Loading vectors from cache only (no API calls) ...")
        vectors = [embedder._read_cache(_embedding_input(r)) or [] for r in rows]
        missing = sum(1 for v in vectors if not v)
        print(f"  Cached {len(rows) - missing}/{len(rows)}"
              + (f"  ({missing} missing — run without --skip-embedding)" if missing else ""))
        if missing == len(rows):
            raise RuntimeError("no cached vectors found — run without --skip-embedding")
    else:
        print(f"  Embedding with {embedder.model} ...")
        vectors = embedder.embed(_embedding_input(r) for r in rows)
        ok = sum(1 for v in vectors if v)
        print(f"  Embedded {ok}/{len(rows)}  "
              f"(new {embedder.stats['embedded']}, cached {embedder.stats['cached']}, "
              f"retries {embedder.stats['retries']}, failed {embedder.stats['failed']})")
        if ok == 0:
            raise RuntimeError("no embeddings produced — check the endpoint and key")

    if embed_only:
        return {"chunks": len(rows), "embedded": sum(1 for v in vectors if v),
                "written": 0, "links": 0}

    if not any(vectors):
        raise RuntimeError("no vectors to load — refusing to write an empty graph")

    store = GraphStore()
    try:
        info = store.check()
        print(f"  Neo4j: {info.get('name','?')} {info.get('version','?')} "
              f"({info.get('edition','?')})")
        store.apply_schema()
        print("  Schema, vector index and full-text index applied.")

        # Chunks arrive in document order, so seq is the reading order the
        # :NEXT chain needs for context expansion around a hit.
        seq_by_doc: dict[str, int] = {}
        written = 0
        payload = []
        for row, vector in zip(rows, vectors):
            if not vector:
                continue   # embedding failed; do not write a vector-less chunk
            seq = seq_by_doc.get(row["doc_id"], 0)
            seq_by_doc[row["doc_id"]] = seq + 1
            payload.append(_to_graph_row(row, vector, seq))

        for batch in _batches(payload, WRITE_BATCH):
            written += store.upsert(batch)
            print(f"    written {written}/{len(payload)}", end="\r", flush=True)
        print()

        links = store.link_sequential()
        loaded = sorted({row["doc_id"] for row in payload})
        staged = set(store.staged_documents())
        to_publish = [d for d in loaded if d in staged]
        release = None
        if stage_only:
            print(f"  {len(to_publish)} documents staged, not published "
                  f"(publish with --publish)")
        elif to_publish:
            release = store.publish(to_publish, note=note)
            print(f"  Published release {release}: {len(to_publish)} documents")
        counts = store.counts()
        print(f"  Graph: {counts.get('documents')} documents, "
              f"{counts.get('chunks')} chunks, {counts.get('embedded')} with vectors, "
              f"{links} :NEXT links")
        return {"chunks": len(rows), "embedded": len(payload),
                "written": written, "links": links, "release": release, **counts}
    finally:
        store.close()
