"""CLI for embedding and Neo4j indexing."""

from __future__ import annotations

import argparse
import sys

from .config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    NEO4J_DATABASE,
    NEO4J_URI,
    OPENAI_BASE_URL,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and load them into Neo4j.")
    parser.add_argument("--check", action="store_true",
                        help="Verify the embedding endpoint and Neo4j connection, then exit.")
    parser.add_argument("--embed-only", action="store_true",
                        help="Build the embedding cache without writing to Neo4j.")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Load into Neo4j using only already-cached vectors.")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Process only the first N chunks.")
    parser.add_argument("--search", metavar="QUERY",
                        help="Run a hybrid search against the loaded graph.")
    parser.add_argument("--top", type=int, default=8, help="Results for --search.")
    parser.add_argument("--as-of", type=int, metavar="RELEASE",
                        help="Search the graph as it was at this release.")
    parser.add_argument("--stage-only", action="store_true",
                        help="Load documents as staged; do not publish them.")
    parser.add_argument("--publish", metavar="DOC_IDS",
                        help="Publish staged documents (comma-separated) as one release.")
    parser.add_argument("--supersedes", action="append", default=[], metavar="NEW=OLD",
                        help="With --publish: NEW retires OLD in the same release.")
    parser.add_argument("--note", default="", help="Release note for --publish.")
    parser.add_argument("--releases", action="store_true",
                        help="List releases and staged documents.")
    args = parser.parse_args(argv)

    print("=" * 65)
    print("  Corpus indexing — embeddings + Neo4j")
    print(f"  Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIM}d)")
    print(f"  Endpoint:        {OPENAI_BASE_URL or 'api.openai.com'}")
    print(f"  Neo4j:           {NEO4J_URI} / {NEO4J_DATABASE}")
    print("=" * 65)

    if args.check:
        return _check()

    if args.search:
        return _search(args.search, args.top, args.as_of)

    if args.releases:
        return _releases()

    if args.publish:
        return _publish(args.publish, args.supersedes, args.note)

    from .load import build

    build(limit=args.limit, embed_only=args.embed_only,
          skip_embedding=args.skip_embedding, stage_only=args.stage_only)
    print("=" * 65)
    return 0


def _check() -> int:
    from .embedder import Embedder
    from .graph import GraphStore

    status = 0
    try:
        vector = Embedder().embed_one("postoperative urinary catheter removal")
        print(f"  ✓ embedding endpoint OK — {len(vector)} dimensions")
        if len(vector) != EMBEDDING_DIM:
            print(f"  ! EMBEDDING_DIM is {EMBEDDING_DIM}; set it to {len(vector)} "
                  f"before creating the vector index")
            status = 1
    except Exception as exc:
        print(f"  ✗ embedding endpoint failed: {type(exc).__name__}: {str(exc)[:160]}")
        status = 1

    store = GraphStore()
    try:
        info = store.check()
        print(f"  ✓ Neo4j reachable — {info.get('name','?')} {info.get('version','?')}")
        print(f"    counts: {store.counts()}")

        # Consistency checks. Neo4j answers queries against a half-loaded graph
        # without complaining — a missing property is a *warning*, and the
        # duplicate filter treats a missing `is_canonical` as canonical, so
        # dedup switches itself off silently. Both have happened here; fail loudly.
        problems = []
        with store.session() as session:
            def scalar(query: str) -> int:
                record = session.run(query).single()
                return record["n"] if record else 0

            chunks = scalar("MATCH (c:Chunk) RETURN count(c) AS n")
            vectors = scalar("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
                             "RETURN count(c) AS n")
            flagged = scalar("MATCH (c:Chunk) WHERE c.is_canonical IS NOT NULL "
                             "RETURN count(c) AS n")
            nexts = scalar("MATCH ()-[r:NEXT]->() RETURN count(r) AS n")
            entities = scalar("MATCH (e:Entity) RETURN count(e) AS n")
            mentions = scalar("MATCH ()-[m:MENTIONED_IN]->() RETURN count(m) AS n")

            if chunks == 0:
                problems.append("no Chunk nodes — the graph is empty")
            if chunks and vectors < chunks:
                problems.append(f"{chunks - vectors} chunks have no embedding")
            if chunks and flagged < chunks:
                problems.append(
                    f"{chunks - flagged} chunks lack `is_canonical` — duplicate "
                    "filtering is INACTIVE for them (NULL is treated as canonical)")
            if chunks and nexts == 0:
                problems.append("no :NEXT links — context expansion will not work")
            release = scalar("MATCH (r:Release) RETURN coalesce(max(r.seq), 0) AS n")
            staged = scalar("MATCH (d:Document) WHERE d.live_from IS NULL "
                            "RETURN count(d) AS n")
            print(f"    current release {release}, staged documents {staged}")
            if chunks and release == 0:
                problems.append("no release published — search returns nothing")
            if staged:
                problems.append(f"{staged} documents staged but not published — "
                                "invisible to search until --publish")
            if entities and mentions == 0:
                problems.append(
                    f"{entities} entities but no :MENTIONED_IN links — the KG "
                    "cannot ground traversals in chunks (were entities loaded "
                    "before chunks?)")

        print(f"    chunks {chunks}, with vectors {vectors}, dedup-flagged {flagged}, "
              f":NEXT {nexts}, entities {entities}, mentions {mentions}")
        if problems:
            status = 1
            print("  ✗ consistency problems:")
            for problem in problems:
                print(f"      - {problem}")
        else:
            print("  ✓ graph is consistent")
    except Exception as exc:
        print(f"  ✗ Neo4j unreachable: {type(exc).__name__}: {str(exc)[:160]}")
        print("    Start one with:")
        print("      docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\")
        print("        -e NEO4J_AUTH=neo4j/<password> neo4j:5.26")
        print("    then set NEO4J_PASSWORD in .env")
        status = 1
    finally:
        store.close()

    print("=" * 65)
    return status


def _search(query: str, top: int, as_of: int | None = None) -> int:
    import time

    from .embedder import Embedder
    from .graph import GraphStore

    store = GraphStore()
    try:
        started = time.perf_counter()
        results = store.search_text(query, Embedder(), limit=top, snapshot=as_of)
        elapsed = time.perf_counter() - started
        snap = results[0]["snapshot"] if results else (as_of or store.current_release())
        print(f"\n  Query: {query!r}   release {snap}   {elapsed:.2f}s\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [T{r['tier']} {r['source']} {r.get('year','')}] "
                  f"score={r['score']:.4f}")
            print(f"     {r['title'][:72]}")
            print(f"     {r['section'][:72]}  (p{r['page']})")
            print(f"     {r['text'][:180].replace(chr(10),' ')}...")
            print()
        if not results:
            print("  no results — is the graph loaded?")
        return 0
    finally:
        store.close()


def _releases() -> int:
    from .graph import GraphStore

    store = GraphStore()
    try:
        for r in store.releases():
            print(f"  release {r['seq']:>3}  {r['at'][:19]}  +{r['published']} "
                  f"-{r['retired']}  {r['note'] or ''}")
        staged = store.staged_documents()
        print(f"  staged (not searchable): {len(staged)}"
              + (f"  {', '.join(staged[:8])}" if staged else ""))
        return 0
    finally:
        store.close()


def _publish(doc_ids: str, supersedes: list[str], note: str) -> int:
    from .graph import GraphStore, PublishRefused

    pairs = []
    for item in supersedes:
        new, _, old = item.partition("=")
        if not (new and old):
            print(f"  ✗ --supersedes expects NEW=OLD, got {item!r}")
            return 2
        pairs.append((new.strip(), old.strip()))
    store = GraphStore()
    try:
        seq = store.publish([d.strip() for d in doc_ids.split(",") if d.strip()],
                            supersedes=pairs, note=note)
        print(f"  ✓ published release {seq}")
        return 0
    except PublishRefused as exc:
        print(f"  ✗ {exc}")
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
