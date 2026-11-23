"""Run the ablation grid: judge-free metrics for all conditions, then answers.

Sequenced deliberately. Retrieval and the judge-free metrics cost nothing and
are the table the paper leans on, so they run first and completely. Generation
is cheap. RAGAS judging is ~85% of the total cost and carries +/-0.10 of
judge-dependent uncertainty (see the clinician-proxy and judge-agreement
results), so it is a separate, explicit step rather than something that happens
by default.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import setting

from .ablation import build_grid, judge_free

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
GENERATOR = setting("GRID_GENERATOR_MODEL") or "gemini-3.7-flash-project"


def run(testset: str = "full_testset.jsonl", top_k: int = 5,
        generate_answers: bool = True, limit: int | None = None,
        concurrency: int = 10) -> dict:
    # Rows marked `excluded` stay in the file so every qid (= line index)
    # keeps its meaning; they are skipped here instead. See the row's
    # `excluded` field for why.
    rows = [json.loads(l) for l in (EVAL_DIR / testset).open(encoding="utf-8")]
    excluded = sum(1 for r in rows if r.get("excluded"))
    rows = [r for r in rows if not r.get("excluded")]
    if excluded:
        print(f"  skipping {excluded} excluded questions")
    if limit:
        rows = rows[:limit]
    grid = build_grid(top_k=top_k)
    print(f"  {len(rows)} questions x {len(grid)} conditions")
    print(f"  generator: {GENERATOR}\n")

    summary = {}
    for name, pipeline in grid.items():
        pipeline.model = GENERATOR

        # 6 conditions x 300 questions is 1,800 round-trips; sequential it is
        # ~95 minutes and the endpoint is latency-bound, not rate-bound.
        def one(item: dict) -> dict:
            retrieved = pipeline.retrieve(item["question"])
            record = {**item, "contexts": retrieved.contexts,
                      "meta": retrieved.meta, "pipeline": name}
            if generate_answers:
                record["answer"] = pipeline.generate(item["question"],
                                                     retrieved.contexts)
            return record

        records = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for record in pool.map(one, rows):
                records.append(record)
                if len(records) % 25 == 0 or len(records) == len(rows):
                    print(f"    {name:16s} {len(records)}/{len(rows)}",
                          end="\r", flush=True)
        print()

        out = EVAL_DIR / f"grid_{name}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = judge_free(records)
        summary[name] = metrics
        print(f"    {name:16s} Recall@{top_k} {metrics['recall_at_k']:.3f} | "
              f"MRR {metrics['mrr']:.3f} | "
              f"top1-Tier1 {metrics['top1_tier1_rate']:.3f} | "
              f"tiers {metrics['retrieved_tier_dist']}")

    (EVAL_DIR / "grid_judgefree_results.json").write_text(
        json.dumps({"generator": GENERATOR, "n_questions": len(rows),
                    "top_k": top_k, "conditions": summary}, indent=2))

    print("\n  === ABLATION (judge-free; no LLM in these numbers) ===")
    base = summary.get("full", {})
    for name, m in summary.items():
        delta = m["recall_at_k"] - base.get("recall_at_k", 0)
        marker = "" if name == "full" else f"   (delta vs full: {delta:+.3f})"
        print(f"    {name:16s} Recall@{top_k} {m['recall_at_k']:.3f}  "
              f"MRR {m['mrr']:.3f}  top1-T1 {m['top1_tier1_rate']:.3f}{marker}")
    return summary
