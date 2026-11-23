"""Pairwise judging of the coverage-controlled global answers.

Same protocol as `judge_global.py` — position-randomised, blake2b-seeded so the
flips are reproducible, headline metric `overall` because the sub-metrics
collapsed onto it in the first run — but over the four controlled conditions.

Pairings are chosen to answer three separate questions rather than to produce a
league table:

    graphrag_mapall      vs vector_large_k   does the method beat the baseline
                                             once coverage is equalised?
    graphrag_prefiltered vs vector_large_k   does the shortcut we originally ran
                                             beat the baseline?
    graphrag_mapall      vs graphrag_prefiltered
                                             does mapping over everything buy
                                             anything over the top-40 shortcut,
                                             and is it worth ~12x the calls?
    vector_large_k       vs long_context     baseline sanity check, carried over
                                             from the first run for continuity.

POWER. n=20 with a high tie rate resolved nothing last time: the same answers
gave 7-5 and then 5-5 on position assignment alone. Nothing here fixes that.
These numbers are directional and must be reported with the tie count and the
cost ratio; the durable fix is the element-recall metric over questions built
from known seed documents, which is continuous rather than a three-way tally.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .global_eval import JUDGE_MODEL, judge_pair

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

CONDITIONS = ["graphrag_mapall", "graphrag_prefiltered",
              "vector_large_k", "long_context"]
PAIRINGS = [("graphrag_mapall", "vector_large_k"),
            ("graphrag_prefiltered", "vector_large_k"),
            ("graphrag_mapall", "graphrag_prefiltered"),
            ("vector_large_k", "long_context")]
METRICS = ["comprehensiveness", "diversity", "empowerment", "overall"]
HEADLINE = "overall"


def run(answers_file: str = "global_answers_controlled.jsonl",
        out_prefix: str = "global_controlled",
        seed: int = 20260907, concurrency: int = 6) -> dict:
    rows = [json.loads(line) for line in (EVAL_DIR / answers_file).open()]
    print(f"  {len(rows)} questions | judge: {JUDGE_MODEL}")

    lengths = {c: statistics.median(len(r.get(c) or "") for r in rows)
               for c in CONDITIONS}
    print("  median answer length: "
          + ", ".join(f"{c}={int(v)}" for c, v in lengths.items()))

    jobs = [(i, row, a, b) for i, row in enumerate(rows) for a, b in PAIRINGS]

    def one(job):
        i, row, a, b = job
        digest = hashlib.blake2b(f"{seed}:{i}:{a}:{b}".encode(),
                                 digest_size=8).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        verdict = judge_pair(row["question"], a, row[a], b, row[b], rng)
        return {"question": row["question"], "topic": row.get("topic", ""),
                "pair": f"{a}_vs_{b}", **verdict}

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for result in pool.map(one, jobs):
            results.append(result)
            print(f"    judged {len(results)}/{len(jobs)}", end="\r", flush=True)
    print()

    with (EVAL_DIR / f"{out_prefix}_judgements.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tally: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in results:
        for metric in METRICS:
            tally[row["pair"]][metric][row[metric]] += 1

    cost_path = EVAL_DIR / "global_controlled_cost.json"
    cost = json.loads(cost_path.read_text())["cost"] if cost_path.exists() else {}

    flips = sum(1 for r in results if r["position_flipped"])
    collapsed = sum(1 for r in results if len({r[m] for m in METRICS}) == 1)
    summary = {"judge": JUDGE_MODEL, "questions": len(rows),
               "judgements": len(results),
               "coverage_controlled": True,
               "position_flipped": f"{flips}/{len(results)}",
               "metric_collapse": f"{collapsed}/{len(results)}",
               "median_answer_chars": {k: int(v) for k, v in lengths.items()},
               "cost": cost, "pairs": {}}

    print("\n  === COVERAGE-CONTROLLED GLOBAL RESULTS (headline: overall) ===")
    for a, b in PAIRINGS:
        pair = f"{a}_vs_{b}"
        counts = tally[pair][HEADLINE]
        decided = counts[a] + counts[b]
        summary["pairs"][pair] = {
            "a": a, "b": b, "a_wins": counts[a], "b_wins": counts[b],
            "ties": counts["tie"],
            "a_win_rate_of_decided": counts[a] / decided if decided else None,
        }
        rate = f"{counts[a]/decided:.2f}" if decided else "n/a"
        print(f"\n  {a} {counts[a]}  vs  {b} {counts[b]}   "
              f"(ties {counts['tie']})   win rate of decided: {rate} for {a}")
        if a in cost and b in cost:
            print(f"    calls/question: {a}={cost[a]['calls_per_question']}, "
                  f"{b}={cost[b]['calls_per_question']}")

    (EVAL_DIR / f"{out_prefix}_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  position flipped on {flips}/{len(results)} (expect ~50%)")
    print(f"  sub-metrics identical to `overall` in {collapsed}/{len(results)}")
    print(f"  NOTE n=20: tie counts and cost ratios carry this result, "
          f"not the win/loss margin.")
    return summary
