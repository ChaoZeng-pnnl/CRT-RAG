"""Load resolved triples into Neo4j alongside the chunk graph.

Entities and chunks live in one database, connected by :EVIDENCED_BY, so a
graph traversal ends at real passages the generator can quote and the citation
verifier can check.

    (:Entity {name, type})-[:REL {chunk_id, evidence, tier}]->(:Entity)
    (:Entity)-[:MENTIONED_IN]->(:Chunk)
"""

from __future__ import annotations

from corpus_index.graph import GraphStore

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
]
INDEXES = [
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
]

# Relation type is a parameter, so one statement covers the whole schema via
# apoc-free dynamic typing: a CASE per relation keeps it explicit and safe.
UPSERT = """
UNWIND $rows AS row
MERGE (s:Entity {name: row.subject, type: row.subject_type})
MERGE (o:Entity {name: row.object,  type: row.object_type})
MERGE (s)-[r:RELATES {relation: row.relation, chunk_id: row.chunk_id}]->(o)
  SET r.evidence = row.evidence, r.confidence = row.confidence,
      r.tier = row.tier, r.source = row.source, r.page = row.page,
      r.title = row.title, r.url = row.url, r.year = row.year,
      r.population = row.population, r.condition = row.condition
WITH s, o, row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MERGE (s)-[:MENTIONED_IN]->(c)
MERGE (o)-[:MENTIONED_IN]->(c)
RETURN count(*) AS written
"""


def load(triples: list[dict], batch: int = 500) -> dict:
    store = GraphStore()
    try:
        with store.session() as session:
            for statement in [*CONSTRAINTS, *INDEXES]:
                session.run(statement)

            written = 0
            for i in range(0, len(triples), batch):
                record = session.run(UPSERT, {"rows": triples[i:i + batch]}).single()
                written += record["written"] if record else 0
                print(f"    loaded {min(i + batch, len(triples))}/{len(triples)}",
                      end="\r", flush=True)
            print()

            counts = session.run("""
                MATCH (e:Entity) WITH count(e) AS entities
                MATCH ()-[r:RELATES]->() WITH entities, count(r) AS relations
                MATCH ()-[m:MENTIONED_IN]->() RETURN entities, relations,
                       count(m) AS mentions
            """).single()
            return {"written": written, **(dict(counts) if counts else {})}
    finally:
        store.close()
