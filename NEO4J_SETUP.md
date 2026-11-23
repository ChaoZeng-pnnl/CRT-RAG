# Neo4j setup

Chunks, their vectors, and (next) the clinical relation graph all live in one
Neo4j instance, so the semantic and relational branches join in Cypher rather
than in application code.

## Start the database

**The container lives on one node.** Rootless podman keeps images, containers
and volumes under `/scratch/$USER/podman`, and `/scratch` is local to each
cluster node. A container created on one node does not exist on another — log
in somewhere else and `bolt://localhost:7687` is simply refused. Check which
node you are on (`hostname`) before assuming the database is gone.

Current instance: node `deception04`, created 2026-11-20.

```bash
podman start nurse-neo4j          # same node, after a stop or reboot
podman ps                         # confirm it is Up
```

On a node without it, recreate it (data volumes are per node too, so the graph
must be reloaded — cheap, see below):

```bash
umask 077
python3 -c "
cfg={}
for l in open('.env'):
    l=l.strip()
    if '=' in l and not l.startswith('#'):
        k,_,v=l.partition('='); cfg[k.strip()]=v.strip().strip('\"').strip(\"'\")
print('NEO4J_AUTH=%s/%s' % (cfg['NEO4J_USER'], cfg['NEO4J_PASSWORD']))" > /tmp/neo4j.env
podman run -d --name nurse-neo4j \
  -p 127.0.0.1:7474:7474 -p 127.0.0.1:7687:7687 \
  -v nurse-neo4j-data:/data -v nurse-neo4j-logs:/logs \
  --env-file /tmp/neo4j.env \
  -e NEO4J_server_memory_heap_initial__size=2G \
  -e NEO4J_server_memory_heap_max__size=2G \
  -e NEO4J_server_memory_pagecache_size=2G \
  --restart unless-stopped \
  docker.io/library/neo4j@sha256:3cbb98bbdc5d3bdff9553f8346d9c3758ba65c80dd563d778f677ba97c0f4a83
rm -f /tmp/neo4j.env
```

Why each choice:

- **Ports bound to `127.0.0.1`.** Cluster nodes are shared; publishing on all
  interfaces would expose the database to every other user on the node.
- **Pinned by digest** (Neo4j 5.26.33 Community). The `5.26` tag is rebuilt
  upstream; the digest is what makes a rebuild reproducible.
- **Credentials from `.env` via a temporary file**, so the password never
  appears in shell history or `podman inspect` output of the command line.
- **Named volumes, not bind mounts.** Rootless podman cannot chown a bind
  mount, so bind-mounting `./neo4j/data` makes the container exit with a chown
  error.

**It stops when you log out.** `--restart unless-stopped` only takes effect if
your user services keep running after logout, which needs
`loginctl enable-linger $USER` (`Linger=no` today). That changes a system
setting and may need an administrator — check site policy before enabling it,
and before treating a compute node as a home for a long-running service at all.

Browser at http://localhost:7474 (reach it from your workstation with an SSH
tunnel: `ssh -L 7474:localhost:7474 -L 7687:localhost:7687 deception04`).

## Index the corpus

```bash
export LD_LIBRARY_PATH=/share/apps/python/3.11.13/lib:$LD_LIBRARY_PATH
./.venv/bin/python -m corpus_index --check         # endpoint + database
./.venv/bin/python -m corpus_index                 # embed + load
./.venv/bin/python -m corpus_index --search "when is the catheter removed"
```

`--embed-only` builds the cache without touching Neo4j; `--skip-embedding` loads
from cache alone. Vectors are cached under `chunks/embedding_cache/` keyed by
(model, text), so re-indexing after a schema change costs nothing.

## Releases: nothing is searchable until it is published

Loaded documents are **staged** and invisible to search. Publishing makes a set
of documents live, and retires the documents they replace, in one transaction;
every search reads the release number once and sees exactly that release. See
`manuscript/DESIGN_span_verified_answers.md` §20.

```bash
./.venv/bin/python -m corpus_index                       # load + publish as one release
./.venv/bin/python -m corpus_index --stage-only          # load, publish later
./.venv/bin/python -m corpus_index --publish NEW_ID --supersedes NEW_ID=OLD_ID --note "2nd edition"
./.venv/bin/python -m corpus_index --releases            # history + anything still staged
./.venv/bin/python -m corpus_index --search "..." --as-of 1   # the graph as it was at release 1
```

Current graph (`deception04`, 2026-11-20): release 1 = base corpus with WHO SSI
2016; release 2 = WHO SSI 2018 superseding it; release 3 = WHO caesarean
antisepsis 2021. 239 documents, 14,074 chunks, none staged.

## Graph shape

```
(:Document {doc_id, title, source, tier, trust_level, publication_year, url, sha256})
  -[:HAS_CHUNK]->
(:Chunk {chunk_id, text, embedding, section, page, kind, tier, source, seq})
  -[:NEXT]-> (:Chunk)
```

- `chunk_embedding` — native VECTOR index, 3072 dimensions, cosine.
- `chunk_fulltext` — Lucene index over text + section + title, the lexical half
  of hybrid retrieval. Drug names, dosages and device names are exactly what
  dense retrieval blurs.
- `:NEXT` chains chunks in reading order so a hit can be expanded with its
  neighbours.

## Embedding model

`gemini-embedding-001-project` via the PNNL LiteLLM proxy, **3072 dimensions**,
L2-normalised (so cosine is the right similarity). Two things the proxy forces:

- `encoding_format="float"` is mandatory — the OpenAI SDK defaults to base64,
  which Vertex AI rejects through LiteLLM.
- The dimension is fixed at index creation and cannot be altered afterwards. If
  you change embedding models, drop and recreate `chunk_embedding`.

## Retrieval

`GraphStore.search()` queries both indexes, fuses with Reciprocal Rank Fusion in
Python (Neo4j has no window functions for it), then applies a tier prior:

    Tier 1 x1.30, Tier 2 x1.15, Tier 3 x1.05, Tier 4 x1.00

Tier multiplies the fused score rather than filtering, so a Tier 4 chunk still
wins when it is the only thing that answers the question while authority breaks
ties. These weights are a starting point and are exactly what the reranker
ablation in the paper should tune.

## Deduplication

```bash
./.venv/bin/python -m corpus_dedup            # report only
./.venv/bin/python -m corpus_dedup --apply    # write flags to chunks.jsonl + Neo4j
```

Near-duplicates are found with MinHash + LSH over word-5-gram shingles
(Jaccard >= 0.80). Each cluster elects a canonical chunk — highest tier, then
newest, then fullest text — and the rest are **marked, not deleted**:

    (:Chunk {is_canonical: false, duplicate_of: "<chunk_id>", cluster_size: n})
      -[:DUPLICATE_OF]-> (:Chunk {is_canonical: true})

Keeping them makes the duplication rate reportable in the paper and lets the
retrieval ablation run both ways. `GraphStore.search(..., include_duplicates=True)`
restores the pre-dedup behaviour.

Current state: 730 of 13,462 chunks marked duplicate, 28 of them in cross-tier
(Tier 1 <-> Tier 4) clusters.
