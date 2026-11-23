"""verified_rag: perioperative nursing answers whose every claim is checked
against the guideline text it cites.

Layers (each depends only on the ones above it):

    corpus_index   chunks, embeddings, Neo4j, atomic releases     (existing)
    span_select    candidate passages, read-back, forced choice   (existing)
    verified_rag
      clients/     chat model, Jev, embeddings
      retrieval    snapshot-pinned search, graph enrichment, population ranking
      answer/      synthesize -> verify -> repair -> assemble
      knowledge/   recommendations, slots and links in the graph
      store        audit log, nurse feedback, calibration
      pipeline     the end-to-end answer path

Nothing in this package writes text that reaches a nurse unchecked: generated
sentences pass three checks or are repaired, replaced by the source's exact
words, or withdrawn as "not stated".
"""

from .types import Claim, Response, Signals  # noqa: F401

__all__ = ["Claim", "Response", "Signals"]
