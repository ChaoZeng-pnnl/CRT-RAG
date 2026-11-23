"""Clinical KG pipeline: extract -> resolve -> load -> sample for validation.

    python3 -m clinical_kg --extract 300 --topics VTE,CAUTI
    python3 -m clinical_kg --resolve --load --sample 150
    python3 -m clinical_kg --score            # after marking the CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"
TRIPLES = OUT_DIR / "triples.jsonl"


def _load_triples() -> list[dict]:
    with TRIPLES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _save_triples(triples: list[dict]) -> None:
    with TRIPLES.open("w", encoding="utf-8") as handle:
        for triple in triples:
            handle.write(json.dumps(triple, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the clinical knowledge graph.")
    parser.add_argument("--extract", type=int, metavar="N",
                        help="Extract triples from N Tier 1-2 chunks.")
    parser.add_argument("--topics", help="Comma-separated topic filter, e.g. VTE,CAUTI")
    parser.add_argument("--resolve", action="store_true", help="Canonicalise entity names.")
    parser.add_argument("--load", action="store_true", help="Load triples into Neo4j.")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="Write an N-row CSV for manual precision validation.")
    parser.add_argument("--score", action="store_true",
                        help="Score a marked-up validation CSV.")
    parser.add_argument("--judge", type=int, nargs="?", const=0, metavar="N",
                        help="Adjudicate N triples with the high-cost model "
                             "(omit N for all). Automated screening, not a "
                             "substitute for clinician validation.")
    parser.add_argument("--stats", action="store_true", help="Summarise the graph.")
    args = parser.parse_args(argv)

    print("=" * 65)
    print("  Clinical knowledge graph")
    print("=" * 65)

    if args.extract:
        from .extract import run

        topics = tuple(t.strip() for t in args.topics.split(",")) if args.topics else None
        run(limit=args.extract, topics=topics)

    if args.resolve:
        from .resolve import resolve

        triples, report = resolve(_load_triples())
        _save_triples(triples)
        print(f"\n  Entities after resolution: {report['entities_after_resolution']}"
              f"  (embedding merges: {report['embedding_merges']})")
        for alias, canonical in list(report["merge_examples"].items())[:8]:
            print(f"    {alias!r} -> {canonical!r}")

    if args.load:
        from .load import load

        result = load(_load_triples())
        print(f"\n  Neo4j: {result.get('entities')} entities, "
              f"{result.get('relations')} relations, "
              f"{result.get('mentions')} entity->chunk mentions")

    if args.sample:
        from .validate import sample, write_sheet

        triples = _load_triples()
        path = write_sheet(sample(triples, n=args.sample))
        print(f"\n  Validation sample: {path}")
        print("  Mark the `correct` column y / n / unsure, then run --score.")

    if args.score:
        from .validate import score_sheet

        report = score_sheet()
        if "error" in report:
            print(f"\n  {report['error']}")
        else:
            print(f"\n  Precision: {report['precision_overall']:.3f} "
                  f"over {report['marked']} marked triples")
            for relation, values in report["precision_by_relation"].items():
                print(f"    {relation:22s} {values['precision']:.3f}  (n={values['n']})")

    if args.judge is not None:
        from .judge import HIGHCOST_MODEL, judge
        from .validate import sample

        triples = _load_triples()
        subset = sample(triples, n=args.judge) if args.judge else triples
        result = judge(subset, model=HIGHCOST_MODEL)

        print(f"\n  Judge: {result['judge_model']}")
        print(f"  Verdicts: {result['verdicts']}")
        print(f"  Precision (correct / [correct+incorrect]): "
              f"{result['precision_overall']:.3f}")
        print("\n  By relation:")
        for relation, v in result["precision_by_relation"].items():
            print(f"    {relation:22s} {v['precision']:.3f}  (n={v['n']})")
        print("\n  By tier:")
        for tier, v in result["precision_by_tier"].items():
            print(f"    tier {tier:2s} {' ':14s} {v['precision']:.3f}  (n={v['n']})")
        if result["failure_modes"]:
            print(f"\n  Failure modes: {result['failure_modes']}")

    if args.stats:
        from .validate import summarise

        stats = summarise(_load_triples())
        print(f"\n  Triples: {stats['triples']} | entities: {stats['unique_entities']}")
        print(f"  By relation: {stats['by_relation']}")
        print(f"  By entity type: {stats['by_entity_type']}")
        print(f"  By source: {stats['by_source']}")
        print(f"  Most connected: {stats['most_connected'][:8]}")
        (OUT_DIR / "graph_stats.json").write_text(json.dumps(stats, indent=2))

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
