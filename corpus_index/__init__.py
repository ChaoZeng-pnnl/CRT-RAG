"""Embedding + Neo4j indexing for the nursing RAG corpus.

    python3 -m corpus_index --check          # verify endpoint and Neo4j
    python3 -m corpus_index --embed-only     # build the embedding cache
    python3 -m corpus_index                  # embed + load into Neo4j
    python3 -m corpus_index --search "..."   # hybrid query against the graph
"""

from .embedder import Embedder
from .graph import GraphStore
from .load import build

__all__ = ["Embedder", "GraphStore", "build"]
