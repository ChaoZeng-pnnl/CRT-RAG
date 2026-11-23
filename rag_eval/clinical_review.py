"""Clinician review sheet — for the nurse first author.

This is NOT the author spot-check. That instrument (`spotcheck.py`) asked a
non-clinician a text-comparison question: does the answer say what the guideline
says. It calibrated the automated *grounding* metric and nothing more.

This instrument asks a registered nurse the questions only a clinician can answer:
is this correct, is it safe, would you let a nurse on your unit act on it. Its
dimensions deliberately mirror the LLM clinician-proxy prompt, so proxy ratings
can finally be calibrated against a clinician — the comparison that turns
"two proxies disagreed 0.81 vs 0.24" into "and here is which of them, if either,
tracked a nurse's judgement."

Two sheets are produced:
  clinical_review_answers.csv   — agent answers (the primary instrument)
  clinical_review_questions.csv — a realism check on the generated questions,
                                  which addresses the circularity limitation:
                                  are these things a perioperative nurse would
                                  actually ask on shift?
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

HEADER_NOTE = ("CLINICAL REVIEW — to be completed by a registered nurse. "
               "Rate 1-5 unless stated. Leave blank if you cannot judge. "
               "Your ratings are the only clinician assessment in this study.")

ANSWER_COLS = [
    "id",
    "clinical_correctness_1to5",   # agrees with the source guideline
    "citation_adequacy_1to5",      # claims traceable to the cited passage
    "actionability_1to5",          # could a nurse act on this at the bedside
    "safety_1to5",                 # stays in scope, escalates appropriately
    "completeness_1to5",           # covers population/conditions the guideline attaches
    "acceptable_for_use_y_n",      # would you accept a nurse on your unit using this
    "concern_free_text",
    "question", "answer", "source_guideline_passage", "source", "page",
]

QUESTION_COLS = [
    "id",
    "realistic_y_n",               # would a perioperative nurse actually ask this
    "answerable_at_bedside_y_n",   # is this a point-of-care question
    "comment_free_text",
    "question", "question_type", "source", "source_page",
]


def build_answer_sheet(source: str = "grid_no_lexical.jsonl", n: int = 50,
                       seed: int = 20260906) -> Path:
    rows = [json.loads(l) for l in (EVAL_DIR / source).open(encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = EVAL_DIR / "clinical_review_answers.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([HEADER_NOTE])
        w.writerow(ANSWER_COLS)
        for i, r in enumerate(rows[:n], 1):
            w.writerow([i, "", "", "", "", "", "", "",
                        r["question"], (r.get("answer") or "")[:4000],
                        (r.get("source_text") or "")[:2000],
                        r.get("source_name", ""), r.get("source_page", "")])
    print(f"  {out}  ({n} answers)")
    return out


def build_question_sheet(source: str = "full_testset.jsonl", n: int = 60,
                         seed: int = 20260906) -> Path:
    rows = [json.loads(l) for l in (EVAL_DIR / source).open(encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = EVAL_DIR / "clinical_review_questions.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["CLINICAL REVIEW — are these questions ones a perioperative "
                    "gynecologic nurse would realistically ask during a shift? y/n"])
        w.writerow(QUESTION_COLS)
        for i, r in enumerate(rows[:n], 1):
            w.writerow([i, "", "", "", r["question"], r.get("question_type", ""),
                        r.get("source_name", ""), r.get("source_page", "")])
    print(f"  {out}  ({n} questions)")
    return out


def score(answers: Path | None = None, questions: Path | None = None) -> dict:
    """Score both sheets, and compare the clinician against the LLM proxies."""
    import statistics

    report: dict = {"rater": "registered nurse (first author)"}

    ap = answers or (EVAL_DIR / "clinical_review_answers.csv")
    if ap.exists():
        with ap.open(encoding="utf-8") as fh:
            r = csv.reader(fh); next(r); hdr = next(r)
            rows = [dict(zip(hdr, x)) for x in r if any(x)]
        dims = [c for c in ANSWER_COLS if c.endswith("_1to5")]
        scored = {}
        for d in dims:
            vals = [int(x[d]) for x in rows if (x.get(d) or "").strip().isdigit()]
            if vals:
                scored[d] = {"mean": statistics.mean(vals), "n": len(vals),
                             "low_1_2": sum(1 for v in vals if v <= 2)}
        acc = [x["acceptable_for_use_y_n"].strip().lower() for x in rows
               if (x.get("acceptable_for_use_y_n") or "").strip()]
        if acc:
            scored["acceptable_rate"] = {
                "rate": sum(1 for a in acc if a in ("y", "yes")) / len(acc),
                "n": len(acc)}
        report["answers"] = scored or {"status": "not yet marked"}

        # The comparison this study has been missing.
        proxy_file = EVAL_DIR / "clinician_proxy_results.json"
        if proxy_file.exists() and "acceptable_rate" in scored:
            proxy = json.loads(proxy_file.read_text())
            report["proxy_vs_clinician"] = {
                "clinician_acceptable_rate": scored["acceptable_rate"]["rate"],
                "proxy_acceptable_rates": {
                    m: v["acceptable_rate"] for m, v in proxy["per_model"].items()},
                "note": "Which proxy, if either, tracked the clinician? This is the "
                        "calibration the clinician-proxy result could not provide.",
            }

    qp = questions or (EVAL_DIR / "clinical_review_questions.csv")
    if qp.exists():
        with qp.open(encoding="utf-8") as fh:
            r = csv.reader(fh); next(r); hdr = next(r)
            rows = [dict(zip(hdr, x)) for x in r if any(x)]
        def rate(field):
            vals = [x[field].strip().lower() for x in rows if (x.get(field) or "").strip()]
            return {"rate": sum(1 for v in vals if v in ("y", "yes")) / len(vals),
                    "n": len(vals)} if vals else {"status": "not yet marked"}
        report["questions"] = {"realistic": rate("realistic_y_n"),
                               "answerable_at_bedside": rate("answerable_at_bedside_y_n")}

    (EVAL_DIR / "clinical_review_results.json").write_text(json.dumps(report, indent=2))
    return report
