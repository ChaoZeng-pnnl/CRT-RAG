"""Neo4j schema, indexes, and hybrid retrieval.

One store holds chunks, their vectors, and (later) the clinical relation graph,
so the semantic and relational branches join in Cypher instead of in
application code.

    (:Document {doc_id, title, source, tier, year, url, ...})
      -[:HAS_CHUNK]->
    (:Chunk {chunk_id, text, embedding, section, page, tier, ...})
      -[:NEXT]-> (:Chunk)          # reading order, for context expansion
    (:Release {seq, at, note}) -[:PUBLISHED]-> (:Document)
                               -[:RETIRED]->   (:Document)
    (:Document) -[:SUPERSEDES]-> (:Document)

Releases make updates atomic. A loaded document is STAGED (`live_from` is
null) and invisible to search. Publishing is one transaction that creates
Release N, stamps the new documents `live_from = N` and any documents they
replace `superseded_in = N`. A search reads N once and evaluates every
visibility test against it, so a publish that commits mid-search cannot change
what that search sees, and any past release can be searched again exactly.

Two indexes back the hybrid retriever:
  * a native VECTOR index (cosine) for the semantic branch
  * a Lucene FULLTEXT index for the lexical branch — drug names, dosages and
    device names are exactly what dense retrieval blurs
"""

from __future__ import annotations

import copy
import logging
from collections import Counter, OrderedDict
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from .config import (
    EMBEDDING_DIM,
    FULLTEXT_INDEX,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    VECTOR_INDEX,
    VECTOR_SIMILARITY,
)

# --- schema -----------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    # Two concurrent publishers cannot both create release N: the second
    # commit fails instead of interleaving.
    "CREATE CONSTRAINT release_seq IF NOT EXISTS "
    "FOR (r:Release) REQUIRE r.seq IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX chunk_tier IF NOT EXISTS FOR (c:Chunk) ON (c.tier)",
    "CREATE INDEX chunk_source IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
    "CREATE INDEX document_tier IF NOT EXISTS FOR (d:Document) ON (d.tier)",
    "CREATE INDEX document_year IF NOT EXISTS FOR (d:Document) ON (d.publication_year)",
]

VECTOR_INDEX_CYPHER = f"""
CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {EMBEDDING_DIM},
  `vector.similarity_function`: '{VECTOR_SIMILARITY}'
}}}}
"""

# Section and title are indexed alongside the body: a query naming a guideline
# section should match it even when the phrasing differs from the prose.
FULLTEXT_INDEX_CYPHER = f"""
CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.text, c.section, c.title]
"""

UPSERT_CHUNKS = """
UNWIND $rows AS row
MERGE (d:Document {doc_id: row.doc_id})
  ON CREATE SET d.title = row.title, d.source = row.source,
                d.organization = row.organization, d.tier = row.tier,
                d.trust_level = row.trust_level, d.topic = row.topic,
                d.document_type = row.document_type,
                d.publication_year = row.publication_year,
                d.url = row.url, d.filename = row.filename,
                d.file_format = row.file_format, d.sha256 = row.sha256
MERGE (c:Chunk {chunk_id: row.chunk_id})
  SET c.text = row.text, c.section = row.section, c.page = row.page,
      c.kind = row.kind, c.n_chars = row.n_chars, c.est_tokens = row.est_tokens,
      c.tier = row.tier, c.trust_level = row.trust_level,
      c.source = row.source, c.title = row.title, c.topic = row.topic,
      c.subtopic = row.subtopic, c.publication_year = row.publication_year,
      c.url = row.url, c.doc_id = row.doc_id, c.seq = row.seq,
      c.is_canonical = row.is_canonical, c.duplicate_of = row.duplicate_of,
      c.cluster_size = row.cluster_size
MERGE (d)-[:HAS_CHUNK]->(c)
WITH c, row
CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
RETURN count(c) AS written
"""

LINK_SEQUENTIAL = """
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WITH d, c ORDER BY c.seq
WITH d, collect(c) AS chunks
UNWIND range(0, size(chunks) - 2) AS i
WITH chunks[i] AS a, chunks[i + 1] AS b
MERGE (a)-[:NEXT]->(b)
RETURN count(*) AS links
"""

# --- hybrid retrieval --------------------------------------------------------

# Each branch is queried separately and fused in Python with Reciprocal Rank
# Fusion. Doing the fusion in Cypher needs window functions Neo4j does not have,
# and keeping it here makes the ranking testable without a database.

# Visibility at a snapshot, applied to every candidate in the query rather
# than after fusion, so a hidden chunk never occupies a top-k slot.
#
# 1. Its document must be live at $snapshot: published at or before it, and
#    not superseded at or before it. Staged documents (live_from null) never
#    pass.
# 2. Duplicates are hidden only while their canonical copy is itself visible
#    at $snapshot. Canonical election prefers the newest edition, so once WHO
#    2018 is loaded its 2016 twins are flagged duplicates; a search pinned to a
#    release before 2018 went live must still see the 2016 text. A static
#    is_canonical filter would silently drop it.
#    `is_canonical IS NULL` still counts as canonical, so a graph indexed
#    before the dedup pass never returns nothing.
# 3. A passage can be retired on its own (`node.superseded_in`): a partial
#    update such as CDC 2017, which replaces some 1999 recommendations and
#    explicitly leaves others as accepted practice, retires only the replaced
#    passages and keeps the rest of the older document searchable.
VISIBLE_AT_SNAPSHOT = """
MATCH (d:Document {doc_id: node.doc_id})
WHERE d.live_from IS NOT NULL AND d.live_from <= $snapshot
  AND (d.superseded_in IS NULL OR d.superseded_in > $snapshot)
  AND (node.superseded_in IS NULL OR node.superseded_in > $snapshot)
OPTIONAL MATCH (canon:Chunk {chunk_id: node.duplicate_of})
OPTIONAL MATCH (cd:Document {doc_id: canon.doc_id})
WITH node, score, cd, canon
WHERE $include_duplicates OR node.is_canonical IS NULL OR node.is_canonical = true
   OR cd IS NULL OR cd.live_from IS NULL OR cd.live_from > $snapshot
   OR (cd.superseded_in IS NOT NULL AND cd.superseded_in <= $snapshot)
   OR (canon.superseded_in IS NOT NULL AND canon.superseded_in <= $snapshot)
"""

# Conventional retrieval, for comparison (system A): one static index. Every
# published version stays searchable; nothing is ever retired; duplicates use
# the static flag. This is how an ordinary RAG index behaves after documents
# are simply added to it.
VISIBLE_STATIC = """
MATCH (d:Document {doc_id: node.doc_id})
WHERE d.live_from IS NOT NULL AND d.live_from <= $snapshot
WITH node, score
WHERE $include_duplicates OR node.is_canonical IS NULL OR node.is_canonical = true
"""

def _dense(visible: str) -> str:
    return ("CALL db.index.vector.queryNodes($vector_index, $overfetch, $embedding)\n"
            "YIELD node, score\n" + visible +
            "RETURN node.chunk_id AS chunk_id, score\nORDER BY score DESC\nLIMIT $k\n")


def _lexical(visible: str) -> str:
    return ("CALL db.index.fulltext.queryNodes($fulltext_index, $search_text, {limit: $overfetch})\n"
            "YIELD node, score\n" + visible +
            "RETURN node.chunk_id AS chunk_id, score\nORDER BY score DESC\nLIMIT $k\n")


DENSE_SEARCH, LEXICAL_SEARCH = _dense(VISIBLE_AT_SNAPSHOT), _lexical(VISIBLE_AT_SNAPSHOT)
DENSE_STATIC, LEXICAL_STATIC = _dense(VISIBLE_STATIC), _lexical(VISIBLE_STATIC)

# Candidates are over-fetched by this factor before the visibility filter.
# Superseded editions and duplicates can each remove a large share of the top
# hits for exactly the questions they cover (WHO 2016 and 2018 share 75% of
# their text), so 2x was not enough margin.
OVERFETCH = 4

CURRENT_RELEASE = "MATCH (r:Release) RETURN coalesce(max(r.seq), 0) AS seq"

FETCH_CHUNKS = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, c.text AS text, c.section AS section,
       c.page AS page, c.tier AS tier, c.source AS source, c.title AS title,
       c.url AS url, c.publication_year AS year, c.kind AS kind
"""


class _NotificationCollector(logging.Handler):
    """Aggregate Neo4j server notifications instead of drowning in them.

    The driver logs one multi-line warning per affected query. A pilot run emits
    thousands, so they get filtered out of the console — which is precisely how a
    missing `is_canonical` property (silently disabling duplicate filtering) went
    unnoticed through a whole evaluation. Collapse them by code and report the
    distinct set at the end of a run, where they cannot be scrolled past.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.counts: Counter = Counter()
        self.samples: dict[str, str] = {}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        # e.g. "The property `is_canonical` does not exist"
        key = message.split("Verify")[0].strip()[:160] or message[:160]
        self.counts[key] += 1
        self.samples.setdefault(key, message[:400])


NOTIFICATIONS = _NotificationCollector()
_neo4j_logger = logging.getLogger("neo4j")
_neo4j_logger.addHandler(NOTIFICATIONS)
_neo4j_logger.setLevel(logging.WARNING)
# Keep them off the console; the summary is what should be read.
_neo4j_logger.propagate = False


def notification_summary(reset: bool = True) -> list[tuple[str, int]]:
    """Distinct Neo4j notifications seen so far, most frequent first."""
    items = NOTIFICATIONS.counts.most_common()
    if reset:
        NOTIFICATIONS.counts.clear()
        NOTIFICATIONS.samples.clear()
    return items


def print_notification_summary() -> int:
    """Print distinct DB notifications. Returns how many distinct kinds there were."""
    items = notification_summary(reset=False)
    if not items:
        print("  Neo4j notifications: none")
        return 0
    print(f"  ⚠ Neo4j notifications ({len(items)} distinct):")
    for message, count in items:
        print(f"      [{count:>5}x] {message}")
    return len(items)


class GraphStore:
    """Thin wrapper over the Neo4j driver: schema, bulk upsert, hybrid search."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER,
                 password: str = NEO4J_PASSWORD, database: str = NEO4J_DATABASE,
                 cache_size: int = 256):
        self.uri, self.user, self.password, self.database = uri, user, password, database
        self._driver = None
        # Results cached per (query, parameters, release). The release number is
        # part of the key, so publishing invalidates every entry at once: a
        # cached answer can never outlive the snapshot it was computed on.
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    @contextmanager
    def session(self) -> Iterator[Any]:
        with self.driver.session(database=self.database) as session:
            yield session

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def check(self) -> dict:
        with self.session() as session:
            record = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            ).single()
            return dict(record) if record else {}

    # Note: parameters are passed as a dict, never as **kwargs — a parameter
    # named "query" would otherwise collide with Session.run's own argument.

    def apply_schema(self) -> None:
        with self.session() as session:
            for statement in [*CONSTRAINTS, *INDEXES]:
                session.run(statement)
            session.run(VECTOR_INDEX_CYPHER)
            session.run(FULLTEXT_INDEX_CYPHER)

    def upsert(self, rows: list[dict]) -> int:
        with self.session() as session:
            record = session.run(UPSERT_CHUNKS, {"rows": rows}).single()
            return record["written"] if record else 0

    def link_sequential(self) -> int:
        with self.session() as session:
            record = session.run(LINK_SEQUENTIAL).single()
            return record["links"] if record else 0

    # --- releases ------------------------------------------------------------

    def current_release(self, session=None) -> int:
        """The latest published release, 0 if nothing has been published."""
        if session is not None:
            return session.run(CURRENT_RELEASE).single()["seq"]
        with self.session() as s:
            return s.run(CURRENT_RELEASE).single()["seq"]

    def staged_documents(self) -> list[str]:
        """Documents loaded but not yet published, hence invisible to search."""
        with self.session() as s:
            return [r["doc_id"] for r in s.run(
                "MATCH (d:Document) WHERE d.live_from IS NULL "
                "RETURN d.doc_id AS doc_id ORDER BY doc_id")]

    def releases(self) -> list[dict]:
        with self.session() as s:
            return [dict(r) for r in s.run(
                "MATCH (r:Release) "
                "OPTIONAL MATCH (r)-[:PUBLISHED]->(p:Document) "
                "WITH r, count(p) AS published "
                "OPTIONAL MATCH (r)-[:RETIRED]->(o:Document) "
                "RETURN r.seq AS seq, toString(r.at) AS at, r.note AS note, "
                "       published, count(o) AS retired ORDER BY seq")]

    def publish(self, doc_ids: list[str],
                supersedes: list[tuple[str, str]] | None = None,
                note: str = "") -> int:
        """Make staged documents live, and retire the documents they replace,
        in ONE transaction. Returns the new release number.

        Refuses, changing nothing, if any document is missing, already live,
        has no chunks, or has chunks without an embedding or duplicate flags,
        or if a retired document is not currently live. Either the whole
        release becomes visible at once or none of it does.
        """
        supersedes = supersedes or []
        doc_ids = sorted(set(doc_ids))
        if not doc_ids:
            raise ValueError("nothing to publish")
        for new, _old in supersedes:
            if new not in doc_ids:
                raise ValueError(f"{new} supersedes something but is not being published")

        def work(tx) -> int:
            seq = tx.run(CURRENT_RELEASE).single()["seq"] + 1
            bad = tx.run("""
                UNWIND $docs AS did
                OPTIONAL MATCH (d:Document {doc_id: did})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH did, d, count(c) AS n, count(c.embedding) AS e,
                     count(c.is_canonical) AS f
                WHERE d IS NULL OR n = 0 OR e < n OR f < n OR d.live_from IS NOT NULL
                RETURN did, d IS NULL AS missing, n AS chunks, e AS embedded,
                       f AS flagged, d.live_from AS already_live_from""",
                {"docs": doc_ids}).data()
            if bad:
                raise PublishRefused(f"release {seq} refused: {bad}")
            if supersedes:
                bad = tx.run("""
                    UNWIND $pairs AS p
                    OPTIONAL MATCH (o:Document {doc_id: p[1]})
                    WITH p, o WHERE o IS NULL OR o.live_from IS NULL
                         OR o.superseded_in IS NOT NULL
                    RETURN p[1] AS old, o IS NULL AS missing""",
                    {"pairs": [list(x) for x in supersedes]}).data()
                if bad:
                    raise PublishRefused(f"release {seq} refused, cannot retire: {bad}")
            tx.run("CREATE (:Release {seq: $seq, at: datetime(), note: $note})",
                   {"seq": seq, "note": note})
            tx.run("""
                MATCH (r:Release {seq: $seq})
                UNWIND $docs AS did
                MATCH (d:Document {doc_id: did})
                SET d.live_from = $seq
                MERGE (r)-[:PUBLISHED]->(d)""", {"seq": seq, "docs": doc_ids})
            if supersedes:
                tx.run("""
                    MATCH (r:Release {seq: $seq})
                    UNWIND $pairs AS p
                    MATCH (n:Document {doc_id: p[0]}), (o:Document {doc_id: p[1]})
                    SET o.superseded_in = $seq
                    MERGE (n)-[:SUPERSEDES]->(o)
                    MERGE (r)-[:RETIRED]->(o)""",
                    {"seq": seq, "pairs": [list(x) for x in supersedes]})
            return seq

        with self.session() as s:
            return s.execute_write(work)

    def retire_passages(self, chunk_ids: list[str], replaced_by: dict[str, str] | None = None,
                        note: str = "") -> int:
        """Retire individual passages in ONE new release (a partial update).

        `replaced_by` maps each retired chunk to the chunk that replaces it, and
        is recorded as (:Chunk)-[:REPLACED_BY]->(:Chunk). Refuses, changing
        nothing, if a chunk is missing, not live, or already retired.
        """
        chunk_ids = sorted(set(chunk_ids))
        if not chunk_ids:
            raise ValueError("nothing to retire")

        def work(tx) -> int:
            seq = tx.run(CURRENT_RELEASE).single()["seq"] + 1
            bad = tx.run("""
                UNWIND $ids AS cid
                OPTIONAL MATCH (c:Chunk {chunk_id: cid})
                OPTIONAL MATCH (d:Document {doc_id: c.doc_id})
                WITH cid, c, d WHERE c IS NULL OR d.live_from IS NULL
                     OR c.superseded_in IS NOT NULL OR d.superseded_in IS NOT NULL
                RETURN cid""", {"ids": chunk_ids}).data()
            if bad:
                raise PublishRefused(f"release {seq} refused: {bad[:5]}")
            tx.run("CREATE (:Release {seq: $seq, at: datetime(), note: $note})",
                   {"seq": seq, "note": note})
            tx.run("""MATCH (r:Release {seq: $seq})
                      UNWIND $ids AS cid MATCH (c:Chunk {chunk_id: cid})
                      SET c.superseded_in = $seq MERGE (r)-[:RETIRED_PASSAGE]->(c)""",
                   {"seq": seq, "ids": chunk_ids})
            pairs = [[a, b] for a, b in (replaced_by or {}).items()]
            if pairs:
                tx.run("""UNWIND $pairs AS p
                          MATCH (o:Chunk {chunk_id: p[0]}), (n:Chunk {chunk_id: p[1]})
                          MERGE (o)-[:REPLACED_BY]->(n)""", {"pairs": pairs})
            return seq

        with self.session() as s:
            return s.execute_write(work)

    def counts(self) -> dict:
        with self.session() as session:
            record = session.run(
                "MATCH (d:Document) WITH count(d) AS documents "
                "MATCH (c:Chunk) WITH documents, count(c) AS chunks "
                "MATCH (c2:Chunk) WHERE c2.embedding IS NOT NULL "
                "RETURN documents, chunks, count(c2) AS embedded"
            ).single()
            return dict(record) if record else {}

    def search(self, query: str, embedding: list[float], limit: int = 10,
               k_dense: int = 50, k_lexical: int = 50, rrf_k: int = 60,
               tier_weights: tuple[float, float, float, float] = (1.30, 1.15, 1.05, 1.00),
               include_duplicates: bool = False,
               allowed_docs: set[str] | None = None,
               snapshot: int | None = None, versioned: bool = True) -> list[dict]:
        """Hybrid dense + lexical retrieval with a tier prior on the fused score.

        Tier enters as a multiplier on the fused score, not as a filter: a Tier 4
        chunk still wins when it is the only thing that answers the question, but
        authority breaks ties — which is the claim the paper is making.

        `allowed_docs` restricts results to a document set. It exists for
        coverage-controlled comparisons: a graph built over a subset of the corpus
        must not be compared against a retriever holding all of it, which is the
        confound that invalidated both our first relational pilot and the first
        global evaluation. Default None leaves normal retrieval untouched.
        Filtering is applied to the fused pool *before* truncation, so callers
        must also raise `k_dense`/`k_lexical` when the allowed set is a small
        fraction of the corpus, or the pool empties before it is ranked.

        `snapshot` pins the search to a release; by default the current one is
        read ONCE, here, and every row carries it as `snapshot`.
        """
        with self.session() as session:
            seq = self.current_release(session) if snapshot is None else snapshot
            if seq == 0:
                return []   # nothing published: staged documents are invisible

            lexical_q = LEXICAL_SEARCH if versioned else LEXICAL_STATIC
            dense_q = DENSE_SEARCH if versioned else DENSE_STATIC
            try:
                lexical = [r["chunk_id"] for r in session.run(lexical_q, {
                    "fulltext_index": FULLTEXT_INDEX,
                    "search_text": _lucene_escape(query),
                    "k": k_lexical,
                    "overfetch": k_lexical * OVERFETCH,
                    "include_duplicates": include_duplicates,
                    "snapshot": seq,
                })]
            except Exception:
                lexical = []   # a malformed Lucene query must not kill the search

            dense = [r["chunk_id"] for r in session.run(dense_q, {
                "vector_index": VECTOR_INDEX,
                "embedding": embedding,
                "k": k_dense,
                "overfetch": k_dense * OVERFETCH,
                "include_duplicates": include_duplicates,
                "snapshot": seq,
            })]

            fused: dict[str, float] = {}
            branches: dict[str, list[str]] = {}
            for name, ranking in (("dense", dense), ("lexical", lexical)):
                for rank, chunk_id in enumerate(ranking, start=1):
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                    branches.setdefault(chunk_id, []).append(name)

            if allowed_docs is not None:
                # chunk_id is "<doc_id>:<ordinal>".
                fused = {c: s for c, s in fused.items()
                         if c.split(":")[0] in allowed_docs}

            if not fused:
                return []

            weights = {1: tier_weights[0], 2: tier_weights[1],
                       3: tier_weights[2], 4: tier_weights[3]}
            top_ids = sorted(fused, key=fused.get, reverse=True)[:max(limit * 3, limit)]
            rows = {r["chunk_id"]: dict(r)
                    for r in session.run(FETCH_CHUNKS, {"chunk_ids": top_ids})}

        results = []
        for chunk_id in top_ids:
            row = rows.get(chunk_id)
            if row is None:
                continue
            row["score"] = fused[chunk_id] * weights.get(row.get("tier", 4), 1.0)
            row["branches"] = branches[chunk_id]
            row["snapshot"] = seq
            results.append(row)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def search_text(self, query: str, embedder, limit: int = 10,
                    use_cache: bool = True, **kwargs) -> list[dict]:
        """Search from raw text, with results cached per (query, parameters,
        release). Publishing a release changes the key, so a stale entry is
        never served. A hit costs about 1 ms against ~0.4-0.9 s for a miss.

        Overlapping the embedding call with the lexical branch was tried and
        measured (2026-11-20): hybrid search takes ~36 ms against a 0.35-0.9 s
        embedding, so there was nothing to overlap, and the thread cost more
        than it saved (median 0.42 s vs 0.39 s sequential). Removed.
        """
        snapshot = kwargs.pop("snapshot", None)
        if snapshot is None:
            snapshot = self.current_release()
        key = (" ".join(query.lower().split()), limit, snapshot,
               tuple(sorted((k, repr(v)) for k, v in kwargs.items())))
        if use_cache and key in self._cache:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return copy.deepcopy(self._cache[key])
        self.cache_misses += 1
        results = self.search(query, embedder.embed_one(query), limit=limit,
                              snapshot=snapshot, **kwargs)
        if use_cache:
            self._cache[key] = copy.deepcopy(results)
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return results


class PublishRefused(RuntimeError):
    """A release failed its pre-publication checks; nothing was changed."""


_LUCENE_SPECIAL = r'+-&|!(){}[]^"~*?:\/'


def _lucene_escape(query: str) -> str:
    """Escape Lucene operators so a clinical query cannot become a syntax error."""
    out = []
    for ch in query:
        if ch in _LUCENE_SPECIAL:
            out.append("\\")
        out.append(ch)
    return "".join(out)
