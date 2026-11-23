"""Cross-document entity linkage — the corpus property that gates global GraphRAG.

WHY
---
The multi-scale sweep (`community_multiscale.sweep()`) showed the granularity knob
does not work on this graph: lowering Leiden resolution never produces
corpus-level communities. It saturates at ~127 communities with a median span of
ONE document, while the giant component collapses into a single blob covering
everything. There is no mesoscale — no resolution at which a community is "a
theme across bodies".

That is not a tuning failure. A community can only span documents if entities
span documents, and in this corpus they mostly do not: 15% of extracted entities
appear in more than one document. Community detection has nothing to find.

Before blaming the corpus, two alternative explanations must be excluded:

  (a) STRING VARIANCE — "low-molecular-weight heparin" vs "low molecular weight
      heparin" are separate nodes. Tested by aggressive normalisation.
  (b) SEMANTIC UNDER-MERGING — the resolver's 0.93 cosine threshold is strict, so
      genuinely equivalent concepts may stay apart. Tested by sweeping the
      threshold and watching linkage against over-merge damage.

Both are measured here. Neither rescues the graph, which is the finding: linkage
is a property of the corpus, not an artefact of the pipeline.

REMAINING CONFOUND, stated not solved: extraction granularity. The extractor
emits propositional noun phrases ("patient-to-patient transmission of infectious
agents") rather than canonical clinical concepts. Concept-normalised extraction
(UMLS/SNOMED, or a prompt constrained to canonical terms) would raise linkage by
an unknown amount. `linkage_ceiling` below is therefore a ceiling for THIS
extraction, not for the corpus in principle.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .community_multiscale import edges

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"

# Swept from the resolver's operating point down to obvious over-merge.
THRESHOLDS = (0.93, 0.90, 0.87, 0.84, 0.80, 0.75, 0.70)

# A cluster this large is not a concept; it is the graph collapsing. Used to mark
# thresholds as unusable rather than to pick one automatically.
OVERMERGE_CLUSTER = 50

_STOP = {"the", "a", "an", "of", "for", "in", "with", "and", "to",
         "patient", "patients"}


def _entity_documents(rows: list[dict]) -> dict[str, set[str]]:
    """Entity surface form -> set of documents it was extracted from."""
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc = (row.get("chunk_id") or "").split(":")[0]
        out[row["subject"]].add(doc)
        out[row["object"]].add(doc)
    return out


def _normalise(name: str) -> str:
    """Aggressive form-only normalisation: case, punctuation, plurals, order."""
    text = re.sub(r"\(.*?\)", "", name.lower())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w[:-1] if len(w) > 4 and w.endswith("s") else w
             for w in text.split() if w not in _STOP]
    return " ".join(sorted(words))


def _stats(clusters: list[set[str]], ent_docs: dict[str, set[str]]) -> dict:
    spans = [set().union(*(ent_docs[n] for n in c)) for c in clusters]
    return {
        "clusters": len(clusters),
        "cross_document_rate": round(
            sum(len(s) > 1 for s in spans) / len(spans), 4) if spans else 0.0,
        "entities_ge5_docs": sum(len(s) >= 5 for s in spans),
        "largest_cluster": max((len(c) for c in clusters), default=0),
    }


def analyse(thresholds=THRESHOLDS) -> dict:
    import numpy as np

    from corpus_index.embedder import Embedder

    rows = edges()
    ent_docs = _entity_documents(rows)
    names = sorted(ent_docs)
    print(f"  {len(names)} entity surface forms over "
          f"{len({d for s in ent_docs.values() for d in s})} documents")

    baseline = _stats([{n} for n in names], ent_docs)
    print(f"  as extracted            : cross-document {baseline['cross_document_rate']:.3f}"
          f"  |  >=5 docs {baseline['entities_ge5_docs']}")

    # (a) string variance
    buckets: dict[str, set[str]] = defaultdict(set)
    for name in names:
        buckets[_normalise(name)].add(name)
    string_only = _stats(list(buckets.values()), ent_docs)
    print(f"  + string normalisation  : cross-document {string_only['cross_document_rate']:.3f}"
          f"  |  >=5 docs {string_only['entities_ge5_docs']}")

    # (b) semantic merging, swept
    vectors = np.asarray(Embedder().embed(names, progress=False), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1.0)

    def merge_at(threshold: float) -> list[set[str]]:
        parent = list(range(len(names)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ii, jj = np.where(similarity >= threshold)
        for a, b in zip(ii, jj):
            if a < b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
        groups: dict[int, set[str]] = defaultdict(set)
        for i, name in enumerate(names):
            groups[find(i)].add(name)
        return list(groups.values())

    print(f"\n  {'threshold':>9} {'clusters':>9} {'x-doc':>7} {'>=5doc':>7} "
          f"{'largest':>8}  usable")
    sweep = []
    for threshold in thresholds:
        clusters = merge_at(threshold)
        row = _stats(clusters, ent_docs)
        row["threshold"] = threshold
        row["usable"] = row["largest_cluster"] < OVERMERGE_CLUSTER
        sweep.append(row)
        print(f"  {threshold:>9.2f} {row['clusters']:>9} "
              f"{row['cross_document_rate']:>7.3f} {row['entities_ge5_docs']:>7} "
              f"{row['largest_cluster']:>8}  {'yes' if row['usable'] else 'NO — over-merged'}")

    usable = [r for r in sweep if r["usable"]]
    ceiling = max(usable, key=lambda r: r["cross_document_rate"]) if usable else baseline

    result = {
        "entities": len(names),
        "documents": len({d for s in ent_docs.values() for d in s}),
        "as_extracted": baseline,
        "string_normalised": string_only,
        "semantic_sweep": sweep,
        "linkage_ceiling": {
            "threshold": ceiling.get("threshold"),
            "cross_document_rate": ceiling["cross_document_rate"],
            "entities_ge5_docs": ceiling["entities_ge5_docs"],
        },
        "note": ("Ceiling is for this extraction. Concept-normalised extraction "
                 "(UMLS/SNOMED or a canonical-term prompt) is an untested "
                 "confound and would raise linkage by an unknown amount."),
    }
    path = OUT_DIR / "linkage_analysis.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\n  as-extracted {baseline['cross_document_rate']:.3f} -> "
          f"ceiling {ceiling['cross_document_rate']:.3f} "
          f"at threshold {ceiling.get('threshold')}")
    print(f"  Wrote {path}")
    return result


if __name__ == "__main__":
    analyse()
