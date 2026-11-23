"""Ingestion: validated corpus files -> section-aware, provenance-carrying chunks.

Reads `corpus/metadata/corpus_metadata.json` (the audited manifest, never the
raw directory) so that only documents which passed content validation are ever
chunked, and every chunk inherits the tier/source/year that the tier-weighted
retrieval claim depends on.

    python3 -m corpus_ingest --stats      # what would be ingested
    python3 -m corpus_ingest              # extract + chunk -> chunks/chunks.jsonl
"""

from .pipeline import ingest

__all__ = ["ingest"]
