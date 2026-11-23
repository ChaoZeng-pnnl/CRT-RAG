"""Entity resolution.

"VTE", "venous thromboembolism" and "venous thrombo-embolism" must become one
node or the graph shatters into near-duplicates and no multi-hop path ever
connects — which is the usual reason an LLM-extracted KG underperforms.

Three stages, cheapest first:
  1. exact match on the normalised surface form
  2. a small alias table for the abbreviations this domain always produces
  3. embedding similarity within an entity type, reusing the cached embedder

Full UMLS/SNOMED normalisation would be better and is a project of its own;
this is the defensible pragmatic version, and the alias table plus threshold
are reportable in Methods.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"

# Abbreviations guidelines use interchangeably with their expansions. Extraction
# is told to prefer full terms, but passages that only ever say "VTE" force it.
ALIASES = {
    "vte": "venous thromboembolism",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "ssi": "surgical site infection",
    "cauti": "catheter-associated urinary tract infection",
    "uti": "urinary tract infection",
    "lmwh": "low molecular weight heparin",
    "ufh": "unfractionated heparin",
    "ted stockings": "graduated compression stockings",
    "ipc": "intermittent pneumatic compression",
    "scd": "intermittent pneumatic compression",
    "pph": "postpartum haemorrhage",
    "postpartum hemorrhage": "postpartum haemorrhage",
    "txa": "tranexamic acid",
    "eras": "enhanced recovery after surgery",
    "idc": "indwelling urinary catheter",
    "indwelling catheter": "indwelling urinary catheter",
    "urinary catheter": "indwelling urinary catheter",
    "foley": "indwelling urinary catheter",
    "thromboprophylaxis": "pharmacological thromboprophylaxis",
    "vte prophylaxis": "pharmacological thromboprophylaxis",
    "antibiotic prophylaxis": "antimicrobial prophylaxis",
}

SIMILARITY_THRESHOLD = 0.93   # deliberately strict: a wrong merge is worse than a miss


def _normalise(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip().lower())
    name = re.sub(r"[.,;:]+$", "", name)
    name = re.sub(r"\bprophylaxis\b", "prophylaxis", name)
    return ALIASES.get(name, name)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def resolve(triples: list[dict], use_embeddings: bool = True) -> tuple[list[dict], dict]:
    """Canonicalise entity names in place; return (triples, merge report)."""
    # Stage 1-2: normalisation and aliases.
    for triple in triples:
        triple["subject"] = _normalise(triple["subject"])
        triple["object"] = _normalise(triple["object"])

    # Collect surface forms per entity type — only merge within a type, so a
    # Drug never collapses into a Complication.
    by_type: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        by_type[triple["subject_type"]].add(triple["subject"])
        by_type[triple["object_type"]].add(triple["object"])

    merges: dict[str, str] = {}
    if use_embeddings:
        from corpus_index.embedder import Embedder

        embedder = Embedder()
        for entity_type, names in by_type.items():
            names = sorted(names)
            if len(names) < 2:
                continue
            vectors = embedder.embed(names, progress=False)
            for i, name in enumerate(names):
                if not vectors[i] or name in merges:
                    continue
                for j in range(i + 1, len(names)):
                    other = names[j]
                    if other in merges or not vectors[j]:
                        continue
                    if _cosine(vectors[i], vectors[j]) >= SIMILARITY_THRESHOLD:
                        # Keep the longer, more explicit name as canonical.
                        canonical, alias = (name, other) if len(name) >= len(other) else (other, name)
                        merges[alias] = canonical

    for triple in triples:
        triple["subject"] = merges.get(triple["subject"], triple["subject"])
        triple["object"] = merges.get(triple["object"], triple["object"])

    entities = {(t["subject"], t["subject_type"]) for t in triples}
    entities |= {(t["object"], t["object_type"]) for t in triples}

    report = {
        "triples": len(triples),
        "entities_after_resolution": len(entities),
        "embedding_merges": len(merges),
        "merge_examples": dict(list(merges.items())[:25]),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    (OUT_DIR / "resolution_report.json").write_text(json.dumps(report, indent=2))
    return triples, report
