"""Is a pairwise 'win' a judgement about quality, or about length?

WHY
---
The coverage-controlled global run gave GraphRAG a 10-5 win over vector retrieval
(0.67 of decided judgements). Before that can be reported, it has to survive the
known failure mode of pairwise LLM judging: judges prefer longer answers, and the
conditions differ systematically in length because map-reduce concatenates many
community summaries before the reduce step.

    graphrag_mapall       ~9,500 chars
    graphrag_prefiltered  ~10,000
    vector_large_k        ~5,800
    long_context          ~4,700

Two diagnostics, neither of which needs new LLM calls:

  1. HOW OFTEN DID THE LONGER ANSWER WIN, across every pairing? If this is near
     chance the metric is sound. If it is high, the metric is largely measuring
     length and every win rate in the run inherits that.

  2. WHAT HAPPENS ON LENGTH-COMPARABLE PAIRS? Restricting to judgements where the
     two answers are within a length ratio removes the bias by construction. If
     the advantage survives, it is real; if it vanishes or reverses, it was length.

The subsets are small and this is a diagnostic, not a hypothesis test — but a
metric that picks the longer answer 84% of the time cannot carry a 0.67 win rate,
and that conclusion does not need statistical power to reach.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

# A pair is "length-comparable" when the shorter answer is at least this fraction
# of the longer. 0.7 keeps enough judgements to look at; 0.8 is stricter and
# reported alongside so the choice is visible rather than tuned.
COMPARABLE_RATIOS = (0.8, 0.7)


def analyse(answers_file: str = "global_answers_controlled.jsonl",
            judgements_file: str = "global_controlled_judgements.jsonl",
            out_file: str = "global_controlled_length_bias.json") -> dict:
    answers = {r["question"]: r
               for r in map(json.loads, (EVAL_DIR / answers_file).open())}
    judgements = [json.loads(line)
                  for line in (EVAL_DIR / judgements_file).open()]

    def lengths(judgement) -> tuple[str, str, int, int]:
        a, b = judgement["pair"].split("_vs_")
        row = answers[judgement["question"]]
        return a, b, len(row[a]), len(row[b])

    # --- diagnostic 1: did length decide it? -------------------------------
    per_pair: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for judgement in judgements:
        if judgement["overall"] == "tie":
            continue
        a, b, la, lb = lengths(judgement)
        if la == lb:
            continue
        longer = a if la > lb else b
        per_pair[judgement["pair"]][1] += 1
        overall[1] += 1
        if judgement["overall"] == longer:
            per_pair[judgement["pair"]][0] += 1
            overall[0] += 1

    print("  longer answer won:")
    for pair, (won, decided) in per_pair.items():
        print(f"    {pair:<45} {won}/{decided} = {won/decided:.2f}")
    print(f"    {'OVERALL':<45} {overall[0]}/{overall[1]} = "
          f"{overall[0]/overall[1]:.2f}")

    # --- diagnostic 2: length-comparable subsets ---------------------------
    comparable: dict[str, dict] = {}
    print("\n  length-comparable subsets (advantage should survive if real):")
    for ratio in COMPARABLE_RATIOS:
        print(f"    ratio >= {ratio}")
        for pair in sorted(per_pair):
            a, b = pair.split("_vs_")
            wins_a = wins_b = ties = 0
            for judgement in judgements:
                if judgement["pair"] != pair:
                    continue
                _, _, la, lb = lengths(judgement)
                if min(la, lb) / max(la, lb) < ratio:
                    continue
                if judgement["overall"] == a:
                    wins_a += 1
                elif judgement["overall"] == b:
                    wins_b += 1
                else:
                    ties += 1
            total = wins_a + wins_b + ties
            comparable[f"{pair}@{ratio}"] = {
                "a": a, "b": b, "a_wins": wins_a, "b_wins": wins_b,
                "ties": ties, "n": total}
            print(f"      {pair:<45} {wins_a}-{wins_b} (ties {ties}) of {total}")

    result = {
        "longer_answer_won": {
            "overall": {"won": overall[0], "decided": overall[1],
                        "rate": round(overall[0] / overall[1], 3) if overall[1] else None},
            "per_pair": {p: {"won": w, "decided": d,
                             "rate": round(w / d, 3) if d else None}
                         for p, (w, d) in per_pair.items()},
        },
        "length_comparable": comparable,
        "median_answer_chars": {
            c: sorted(len(r[c]) for r in answers.values())[len(answers) // 2]
            for c in ("graphrag_mapall", "graphrag_prefiltered",
                      "vector_large_k", "long_context")
            if all(c in r for r in answers.values())},
        "verdict": ("Pairwise preference on these answers is dominated by answer "
                    "length; win rates from this metric do not support a quality "
                    "claim in either direction."
                    if overall[1] and overall[0] / overall[1] >= 0.70 else
                    "Length preference is within a tolerable range; win rates may "
                    "be interpreted with the length figures attached."),
    }
    path = EVAL_DIR / out_file
    path.write_text(json.dumps(result, indent=2))
    print(f"\n  {result['verdict']}")
    print(f"  Wrote {path}")
    return result


if __name__ == "__main__":
    analyse()
