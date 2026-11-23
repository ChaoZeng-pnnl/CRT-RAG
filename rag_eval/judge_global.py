"""Pairwise judging of the global answers.

Three head-to-head pairings, each judged on comprehensiveness, diversity and
empowerment (Edge et al.). Every judgement is POSITION-RANDOMISED and the winner
resolved back through the flip, so presentation order cannot manufacture a win
rate.

Answer length is recorded alongside, because LLM judges are known to prefer
longer answers and the conditions here differ systematically in length
(GraphRAG median 8.7k chars, vector 7.6k, long-context 4.6k). A win that tracks
length is not the same as a win on substance, and the reader needs both numbers
to tell them apart.
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
CONDITIONS = ["graphrag_global", "vector_large_k", "long_context"]
PAIRINGS = [("graphrag_global", "vector_large_k"),
            ("graphrag_global", "long_context"),
            ("vector_large_k", "long_context")]
# The judge collapsed the sub-metrics: 56 of 60 judgements in the first run gave
# an identical verdict for comprehensiveness, diversity, empowerment and overall.
# They are not independent measurements, so `overall` is the headline and the
# others are reported only as an agreement diagnostic — presenting four columns
# would overstate the evidence roughly fourfold.
METRICS = ["comprehensiveness", "diversity", "empowerment", "overall"]
HEADLINE = "overall"


def run(seed: int = 20260905, concurrency: int = 6) -> dict:
    rows = [json.loads(l) for l in (EVAL_DIR / "global_answers.jsonl").open()]
    print(f"  {len(rows)} questions | judge: {JUDGE_MODEL}")

    lengths = {c: statistics.median(len(r[c]) for r in rows) for c in CONDITIONS}
    print(f"  median answer length: "
          + ", ".join(f"{c}={int(v)}" for c, v in lengths.items()))

    jobs = []
    for i, row in enumerate(rows):
        for a, b in PAIRINGS:
            jobs.append((i, row, a, b))

    def one(job):
        i, row, a, b = job
        # Python's hash() is salted per process, so a hash-derived seed makes
        # the position flips differ between runs and the result irreproducible.
        # Use a stable digest instead.
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

    with (EVAL_DIR / "global_judgements.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Tally
    tally: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in results:
        for metric in METRICS:
            tally[row["pair"]][metric][row[metric]] += 1

    flips = sum(1 for r in results if r["position_flipped"])
    summary = {"judge": JUDGE_MODEL, "questions": len(rows),
               "judgements": len(results),
               "position_flipped": f"{flips}/{len(results)}",
               "median_answer_chars": {k: int(v) for k, v in lengths.items()},
               "pairs": {}}

    collapsed = sum(1 for r in results
                    if len({r[m] for m in METRICS}) == 1)
    summary["metric_collapse"] = f"{collapsed}/{len(results)}"

    print("\n  === GLOBAL PAIRWISE RESULTS (headline: overall) ===")
    for pair, metrics in tally.items():
        a, b = pair.split("_vs_")
        counts = metrics[HEADLINE]
        total = sum(counts.values()) or 1
        decided = counts[a] + counts[b]
        summary["pairs"][pair] = {
            "a": a, "b": b, "a_wins": counts[a], "b_wins": counts[b],
            "ties": counts["tie"],
            "a_win_rate_of_decided": counts[a] / decided if decided else None,
            "sub_metrics": {m: dict(metrics[m]) for m in METRICS if m != HEADLINE},
        }
        print(f"\n  {a} {counts[a]}  vs  {b} {counts[b]}   (ties {counts['tie']})")
        if decided:
            print(f"    win rate of decided judgements: {counts[a]/decided:.2f} "
                  f"for {a}")

    (EVAL_DIR / "global_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  position flipped on {flips}/{len(results)} judgements "
          f"(expect ~50% if randomisation works)")
    print(f"  sub-metrics identical to `overall` in {collapsed}/{len(results)} "
          f"judgements — the judge is not scoring them independently, so only "
          f"`overall` is reported as a result")
    return summary
