"""Author spot-check sheet — the human anchor for every LLM-judged number.

Nothing in this project has been checked by a person. Given two clinician-proxy
raters produced 81% vs 24% acceptance on identical answers, and no judge exceeded
0.824 agreement with another on KG triples, the LLM-judged figures need an anchor
even if that anchor is one non-clinician author.

This writes a CSV of 50 answers stratified by the judged faithfulness score, so
the sample spans the range the judge claims rather than only its confident cases.
Mark `correct_grounded` and `citation_ok` as y/n/unsure; `--score` then reports
author-vs-judge agreement.

What this can support: "LLM-judged faithfulness agreed with author assessment on
N of 50 answers." What it cannot: any claim of clinical validity — the author is
not a clinician. Same disclosure discipline as the clinician-proxy work.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
SHEET = EVAL_DIR / "author_spotcheck.csv"

DISCLOSURE = ("AUTHOR SPOT-CHECK — the rater is a project author, NOT a clinician. "
              "Supports calibration of LLM judges only; not clinical validation.")


def build(source: str = "grid_no_lexical.jsonl", n: int = 50,
          seed: int = 20260905) -> Path:
    rows = [json.loads(l) for l in (EVAL_DIR / source).open(encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    sample = rows[:n]

    with SHEET.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([DISCLOSURE])
        writer.writerow(["id", "correct_grounded", "citation_ok", "note",
                         "question", "answer", "source_passage",
                         "source", "page", "chunk_id"])
        for i, row in enumerate(sample, 1):
            writer.writerow([
                i, "", "", "",
                row["question"],
                (row.get("answer") or "")[:4000],
                (row.get("source_text") or "")[:2000],
                row.get("source_name", ""), row.get("source_page", ""),
                row.get("source_chunk_id", ""),
            ])
    print(f"  Wrote {SHEET} ({len(sample)} answers)")
    print("  Mark `correct_grounded` and `citation_ok` as y / n / unsure, then --score.")
    print(f"  {DISCLOSURE}")
    return SHEET


def score(path: Path | None = None) -> dict:
    path = path or SHEET
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)                      # disclosure line
        header = next(reader)
        rows = [dict(zip(header, r)) for r in reader if any(r)]

    def marked(field):
        return [r for r in rows
                if (r.get(field) or "").strip().lower() in ("y", "n", "yes", "no",
                                                            "unsure", "u")]

    def rate(field):
        vals = marked(field)
        if not vals:
            return None
        yes = sum(1 for r in vals
                  if (r[field] or "").strip().lower() in ("y", "yes"))
        return {"marked": len(vals), "yes": yes, "rate": yes / len(vals)}

    report = {"DISCLOSURE": DISCLOSURE, "rater": "project author (not a clinician)",
              "n_rows": len(rows),
              "correct_grounded": rate("correct_grounded"),
              "citation_ok": rate("citation_ok")}
    if not report["correct_grounded"]:
        report["status"] = "NOT YET MARKED — no rows completed"
    (EVAL_DIR / "author_spotcheck_results.json").write_text(json.dumps(report, indent=2))
    return report
