"""Sampling for manual precision validation.

A paper claiming an LLM-extracted clinical knowledge graph without a precision
number will be asked for one, and "we did not measure" is a rejection. This
produces a stratified sample as a CSV a clinician can mark up in a spreadsheet:
one row per triple, with the verbatim evidence and the source citation, and an
empty `correct` column to fill with y/n.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"


def sample(triples: list[dict], n: int = 150, seed: int = 20260905) -> list[dict]:
    """Stratify by relation type so rare edges are actually checked."""
    rng = random.Random(seed)
    by_relation: dict[str, list[dict]] = defaultdict(list)
    for triple in triples:
        by_relation[triple["relation"]].append(triple)

    for bucket in by_relation.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    relations = sorted(by_relation)
    while len(picked) < n and any(by_relation[r] for r in relations):
        for relation in relations:
            if by_relation[relation] and len(picked) < n:
                picked.append(by_relation[relation].pop())
    rng.shuffle(picked)
    return picked


def write_sheet(triples: list[dict], out_file: Path | None = None) -> Path:
    out_file = out_file or (OUT_DIR / "validation_sample.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fields = ["id", "correct", "triple", "population", "condition", "evidence",
              "source", "page", "relation", "subject", "subject_type",
              "object", "object_type", "confidence", "chunk_id"]
    with out_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(triples, 1):
            writer.writerow({
                "id": i,
                "correct": "",          # mark y / n / unsure
                "triple": f"({t['subject']}) -[{t['relation']}]-> ({t['object']})",
                "population": t.get("population", ""),
                "condition": t.get("condition", ""),
                "evidence": t["evidence"],
                "source": f"{t['source']} {t.get('year','')} — {(t.get('title') or '')[:60]}",
                "page": t.get("page") or "",
                "relation": t["relation"],
                "subject": t["subject"], "subject_type": t["subject_type"],
                "object": t["object"], "object_type": t["object_type"],
                "confidence": t.get("confidence", ""),
                "chunk_id": t["chunk_id"],
            })
    return out_file


def score_sheet(path: Path | None = None) -> dict:
    """Read a marked-up sheet and report precision overall and by relation."""
    path = path or (OUT_DIR / "validation_sample.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    marked = [r for r in rows if (r.get("correct") or "").strip().lower() in
              ("y", "n", "yes", "no", "unsure", "u")]
    if not marked:
        return {"error": "no rows marked in the `correct` column yet",
                "rows": len(rows)}

    def is_yes(row):
        return (row["correct"] or "").strip().lower() in ("y", "yes")

    by_relation: dict[str, list[bool]] = defaultdict(list)
    for row in marked:
        by_relation[row["relation"]].append(is_yes(row))

    overall = sum(is_yes(r) for r in marked) / len(marked)
    report = {
        "marked": len(marked),
        "unmarked": len(rows) - len(marked),
        "precision_overall": overall,
        "precision_by_relation": {
            relation: {"precision": sum(v) / len(v), "n": len(v)}
            for relation, v in sorted(by_relation.items())
        },
        "unsure": sum(1 for r in marked
                      if (r["correct"] or "").strip().lower() in ("unsure", "u")),
    }
    (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report


def summarise(triples: list[dict]) -> dict:
    """Shape of the extracted graph — what goes in the paper's KG table."""
    entities = {(t["subject"], t["subject_type"]) for t in triples}
    entities |= {(t["object"], t["object_type"]) for t in triples}
    degree: Counter = Counter()
    for t in triples:
        degree[t["subject"]] += 1
        degree[t["object"]] += 1
    return {
        "triples": len(triples),
        "unique_entities": len(entities),
        "by_relation": dict(Counter(t["relation"] for t in triples).most_common()),
        "by_entity_type": dict(Counter(e[1] for e in entities).most_common()),
        "by_tier": dict(Counter(t["tier"] for t in triples)),
        "by_source": dict(Counter(t["source"] for t in triples).most_common()),
        "most_connected": degree.most_common(15),
        "mean_confidence": sum(t.get("confidence", 0) for t in triples) / len(triples)
        if triples else 0,
    }
