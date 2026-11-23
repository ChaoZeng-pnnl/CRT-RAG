"""The ablation grid — the paper's central results table.

The pilot compared two conditions that differed in three ways at once (lexical
branch, tier prior, dedup filter). It proved the metrics separate; it could not
attribute the gain. Each condition here removes exactly ONE component from the
full system, so the difference from `full` is that component's contribution.

Judge-free metrics (Recall@k, MRR, tier distribution) are reported alongside the
RAGAS ones throughout, because the clinician-proxy result showed LLM judges are
reliable for relative comparison and not for absolute acceptability — and a
reviewer who distrusts every judged number should still be able to read a result.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from corpus_index.embedder import Embedder
from corpus_index.graph import OVERFETCH, VECTOR_INDEX, VISIBLE_AT_SNAPSHOT, GraphStore

from .pipelines import ANSWER_PROMPT, Pipeline, Retrieved

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

# Single-branch conditions share the main retriever's visibility rule (release
# snapshot + snapshot-aware duplicates) and over-fetch before filtering. The
# earlier versions asked the index for exactly k and then filtered, which left
# 46 of 300 dense-only questions with fewer than 5 passages.
DENSE_ONLY = """
CALL db.index.vector.queryNodes($vector_index, $overfetch, $embedding)
YIELD node, score
""" + VISIBLE_AT_SNAPSHOT + """
RETURN node.chunk_id AS chunk_id, node.text AS text, node.section AS section,
       node.page AS page, node.tier AS tier, node.source AS source,
       node.title AS title, node.url AS url, node.publication_year AS year, score
ORDER BY score DESC
LIMIT $k
"""

LEXICAL_ONLY = """
CALL db.index.fulltext.queryNodes($fulltext_index, $search_text, {limit: $overfetch})
YIELD node, score
""" + VISIBLE_AT_SNAPSHOT + """
RETURN node.chunk_id AS chunk_id, node.text AS text, node.section AS section,
       node.page AS page, node.tier AS tier, node.source AS source,
       node.title AS title, node.url AS url, node.publication_year AS year, score
ORDER BY score DESC
LIMIT $k
"""

FLAT_TIERS = (1.0, 1.0, 1.0, 1.0)


class Ablation(Pipeline):
    """One retrieval configuration, defined by what it switches off."""

    def __init__(self, name: str, store, embedder, top_k: int = 5,
                 use_lexical: bool = True, use_tier_prior: bool = True,
                 use_dedup: bool = True, dense_only: bool = False,
                 lexical_only: bool = False):
        super().__init__(name, store, embedder, top_k=top_k)
        self.use_lexical = use_lexical
        self.use_tier_prior = use_tier_prior
        self.use_dedup = use_dedup
        self.dense_only = dense_only
        self.lexical_only = lexical_only

    def retrieve(self, question: str) -> Retrieved:
        from corpus_index.graph import FULLTEXT_INDEX, _lucene_escape

        if self.dense_only or self.lexical_only:
            query = DENSE_ONLY if self.dense_only else LEXICAL_ONLY
            params = {"k": self.top_k, "overfetch": self.top_k * OVERFETCH * 10,
                      "include_duplicates": not self.use_dedup}
            if self.dense_only:
                params |= {"vector_index": VECTOR_INDEX,
                           "embedding": self.embedder.embed_one(question)}
            else:
                params |= {"fulltext_index": FULLTEXT_INDEX,
                           "search_text": _lucene_escape(question)}
            with self.store.session() as session:
                params["snapshot"] = self.store.current_release(session)
                rows = [dict(r) for r in session.run(query, params)]
        else:
            rows = self.store.search(
                question, self.embedder.embed_one(question), limit=self.top_k,
                tier_weights=(1.30, 1.15, 1.05, 1.00) if self.use_tier_prior else FLAT_TIERS,
                include_duplicates=not self.use_dedup,
                k_lexical=50 if self.use_lexical else 0)
        return Retrieved(question, [r["text"] for r in rows], rows)


def build_grid(top_k: int = 5) -> dict[str, Ablation]:
    store, embedder = GraphStore(), Embedder()

    def make(name, **kw):
        return Ablation(name, store, embedder, top_k=top_k, **kw)

    return {
        # Full system, then one component removed at a time.
        "full":            make("full"),
        "no_lexical":      make("no_lexical", use_lexical=False),
        "no_tier_prior":   make("no_tier_prior", use_tier_prior=False),
        "no_dedup":        make("no_dedup", use_dedup=False),
        # Single-branch references, for the retrieval-architecture claim.
        "dense_only":      make("dense_only", dense_only=True),
        "lexical_only":    make("lexical_only", lexical_only=True),
    }


def judge_free(records: list[dict]) -> dict:
    """Metrics that need no LLM: recall of the generating chunk, MRR, tiers."""
    hits, rr = [], []
    tiers, top1 = Counter(), Counter()
    for record in records:
        gold = record["source_chunk_id"]
        got = [m.get("chunk_id") for m in record.get("meta", [])]
        hit = gold in got
        hits.append(1.0 if hit else 0.0)
        rr.append(1.0 / (got.index(gold) + 1) if hit else 0.0)
        for m in record.get("meta", []):
            tiers[m.get("tier")] += 1
        if record.get("meta"):
            top1[record["meta"][0].get("tier")] += 1
    n = len(records) or 1
    return {
        "recall_at_k": sum(hits) / n,
        "mrr": sum(rr) / n,
        "top1_tier1_rate": top1.get(1, 0) / n,
        "retrieved_tier_dist": {str(k): v for k, v in sorted(
            tiers.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "mean_depth": statistics.mean(len(r.get("meta", [])) for r in records),
    }
