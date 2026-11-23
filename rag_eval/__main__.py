"""Run the RAGAS pilot.

    python3 -m rag_eval --generate 50      # build the test set
    python3 -m rag_eval --run              # answer with each pipeline
    python3 -m rag_eval --score            # RAGAS + judge-free metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from corpus_index.config import CHAT_MODEL, FALLBACK_MODEL

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def _run_pipelines(top_k: int) -> dict[str, list[dict]]:
    from .evaluate import load_testset
    from .pipelines import build_pipelines

    testset = load_testset()
    pipelines = build_pipelines(top_k=top_k)
    outputs: dict[str, list[dict]] = {}

    for name, pipeline in pipelines.items():
        print(f"\n  Running {name} over {len(testset)} questions ...")
        records = []
        for i, item in enumerate(testset, 1):
            result = pipeline.answer(item["question"])
            records.append({**item, **result})
            print(f"    {i}/{len(testset)}", end="\r", flush=True)
        print()
        outputs[name] = records
        out = EVAL_DIR / f"answers_{name}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"    wrote {out}")
    return outputs


def _load_answers() -> dict[str, list[dict]]:
    outputs = {}
    for path in sorted(EVAL_DIR.glob("answers_*.jsonl")):
        name = path.stem.replace("answers_", "")
        with path.open(encoding="utf-8") as handle:
            outputs[name] = [json.loads(line) for line in handle]
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAGAS pilot for the nursing RAG corpus.")
    parser.add_argument("--generate", type=int, metavar="N",
                        help="Generate an N-question test set from Tier 1-2 chunks.")
    parser.add_argument("--run", action="store_true",
                        help="Answer the test set with each retrieval condition.")
    parser.add_argument("--score", action="store_true",
                        help="Score stored answers with RAGAS and judge-free metrics.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--in-domain", action="store_true",
                        help="With --relational: draw questions only from chunks the KG "
                             "covers. Reports KG coverage of the evaluation set — "
                             "without which a null result is uninterpretable.")
    parser.add_argument("--relational", type=int, nargs="?", const=30, metavar="N",
                        help="Generate N multi-hop questions and run the "
                             "relational pilot (KG vs :NEXT vs hybrid-only).")
    parser.add_argument("--judge", default=FALLBACK_MODEL,
                        help="Judge model; default is a different family from the generator.")
    args = parser.parse_args(argv)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  RAGAS pilot")
    print(f"  Generator: {CHAT_MODEL}   Judge: {args.judge}")
    print("=" * 65)

    if args.generate:
        from .testset import generate

        generate(n=args.generate)

    if args.run:
        _run_pipelines(args.top_k)

    if args.relational is not None:
        from .evaluate import retrieval_metrics
        from .pipelines import build_relational_pipelines
        from .relational import generate

        name = ("relational_indomain_testset.jsonl" if args.in_domain
                else "relational_testset.jsonl")
        testset_file = EVAL_DIR / name
        if not testset_file.exists():
            generate(n=args.relational, in_domain=args.in_domain, out_name=name)
        testset = [json.loads(l) for l in testset_file.open(encoding="utf-8")]
        print(f"\n  Relational pilot: {len(testset)} multi-hop questions"
              + (" (in-domain)" if args.in_domain else ""))

        # Coverage of the evaluation set by the KG. A relational pilot without
        # this number cannot be interpreted: the first run scored 0.000 for every
        # condition purely because no question's evidence was in the graph.
        from .relational import kg_covered_chunk_ids

        covered = kg_covered_chunk_ids()
        gold = [c for t in testset for c in t["source_chunk_ids"]]
        in_kg = sum(1 for c in gold if c in covered)
        both_in = sum(1 for t in testset
                      if all(c in covered for c in t["source_chunk_ids"]))
        print(f"  KG coverage of this set: {in_kg}/{len(gold)} gold chunks "
              f"({100*in_kg/len(gold):.0f}%), both-covered questions "
              f"{both_in}/{len(testset)}")

        pipelines = build_relational_pipelines(top_k=args.top_k)
        summary = {}
        for name, pipeline in pipelines.items():
            print(f"\n  Running {name} ...")
            records = []
            for i, item in enumerate(testset, 1):
                result = pipeline.answer(item["question"])
                records.append({**item, **result})
                print(f"    {i}/{len(testset)}", end="\r", flush=True)
            print()
            out = EVAL_DIR / (f"relational_indomain_{name}.jsonl" if args.in_domain
                              else f"relational_{name}.jsonl")
            with out.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Judge-free: did the retriever return BOTH source chunks?
            # `mean_depth` is reported because the conditions return different
            # numbers of chunks — comparing recall without it would credit a
            # condition for simply returning more.
            both = partial = 0
            gold_found = 0
            depths = []
            for record in records:
                retrieved = {m.get("chunk_id") for m in record.get("meta", [])}
                depths.append(len(retrieved))
                gold = set(record["source_chunk_ids"])
                hit = len(gold & retrieved)
                gold_found += hit
                both += 1 if hit == 2 else 0
                partial += 1 if hit >= 1 else 0
            n = len(records) or 1
            summary[name] = {
                "both_recall": both / n,
                "any_recall": partial / n,
                "gold_chunk_recall": gold_found / (2 * n),
                "mean_depth": sum(depths) / n,
                "n": len(records),
            }
            print(f"    both {both}/{n}={both/n:.3f} | any {partial/n:.3f} | "
                  f"gold-chunk {gold_found}/{2*n}={gold_found/(2*n):.3f} | "
                  f"depth {sum(depths)/n:.1f}")

        summary["_coverage"] = {"gold_chunks_in_kg": in_kg, "gold_chunks": len(gold),
                                "both_covered_questions": both_in,
                                "questions": len(testset)}
        results_name = ("relational_indomain_results.json" if args.in_domain
                        else "relational_results.json")
        (EVAL_DIR / results_name).write_text(json.dumps(summary, indent=2))
        # Surface every distinct database notification from the run. Filtering
        # these out of the console is how a disabled duplicate filter survived a
        # whole evaluation unnoticed.
        from corpus_index.graph import print_notification_summary

        print()
        n_kinds = print_notification_summary()
        if n_kinds:
            print("      ^ investigate these before trusting the numbers below")

        print("\n  Relational pilot summary (judge-free):")
        for name, v in summary.items():
            if name.startswith("_"):
                continue
            print(f"    {name:20s} both={v['both_recall']:.3f}  any={v['any_recall']:.3f}"
                  f"  gold-chunk={v['gold_chunk_recall']:.3f}  depth={v['mean_depth']:.1f}")

    if args.score:
        from .evaluate import ragas_scores, retrieval_metrics, separation_check

        outputs = _load_answers()
        if not outputs:
            print("  No answers found — run with --run first.")
            return 1

        summary = {}
        for name, records in outputs.items():
            print(f"\n  Scoring {name} ({len(records)} items) ...")
            judge_free = retrieval_metrics(records)
            print(f"    Recall@k {judge_free['recall_at_k']:.3f} | "
                  f"MRR {judge_free['mrr']:.3f} | "
                  f"tiers {judge_free['retrieved_tier_distribution']}")
            try:
                ragas = ragas_scores(records, judge_model=args.judge)
                for metric, values in ragas["metrics"].items():
                    print(f"    {metric:34s} {values['mean']:.3f} ± {values['sd']:.3f}")
            except Exception as exc:
                print(f"    ! RAGAS failed: {type(exc).__name__}: {str(exc)[:200]}")
                ragas = {"judge_model": args.judge, "metrics": {}}
            summary[name] = {"judge_free": judge_free, "ragas": ragas}

        separation = separation_check(summary)
        print("\n  Separation check (does any metric distinguish the conditions?)")
        for metric, finding in separation.items():
            flag = "SEPARATES" if finding["separates"] else (
                "SATURATED" if finding["saturated"] else "flat")
            values = "  ".join(f"{k}={v:.3f}" for k, v in finding["values"].items())
            print(f"    {metric:34s} {values}   spread={finding['spread']:.3f}  [{flag}]")

        out = EVAL_DIR / "pilot_results.json"
        out.write_text(json.dumps({"conditions": summary, "separation": separation},
                                  indent=2))
        print(f"\n  Wrote {out}")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
