"""Score selector predictions against nurse-annotated gold spans.

    python3 -m span_select.score_spans eval/annotation/gold_spans_*_CZ.jsonl \
            --pred eval/annotation/pred_gpt4o.jsonl

    # inter-annotator agreement on the overlap, no predictions needed
    python3 -m span_select.score_spans a.jsonl --second b.jsonl

Every number here is mechanical. No model judges anything; a human marked the
acceptable spans once, offline, and selections are compared to those marks by
integer equality. That is the point: the metric the paper reports for selection
quality cannot drift with a judge's prompt.

What is counted
---------------
Two outcomes are reported because they answer different questions.

* **strict** — the selector chose the span the annotator called best. This is
  the number to quote when the field has one right answer (a dose).
* **acceptable** — the selector chose any span the annotator would be content
  to show a nurse. This is the number to quote for `action`, where a chunk
  often restates the same recommendation in prose and in a table row.

Abstention is scored, not ignored, and the two error directions are kept apart:

* **false abstention** — the chunk states it, the selector declined. Costs
  coverage. Annoying.
* **false selection** — the chunk does NOT state it, the selector picked
  something anyway. This is the one that matters clinically: a verbatim span
  from a real guideline, correctly cited, that does not answer the question.
  The span guarantee cannot catch it, by construction, so it has to be measured.

Decisions the annotator marked `unclear`, and any whose annotation was made
after revealing the written reference answer, are excluded and counted
separately — the latter because the reference was generated from this chunk,
so an annotator who has read it is no longer an independent judge of which span
answers the question.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .candidates import normalize

Key = Tuple[str, str]  # (qid, field)


def candidate_texts() -> Dict[Key, List[str]]:
    """index -> normalized text, per (qid, field), from the corpus.

    Needed because two candidates can carry the SAME text at different offsets:
    in the 60-question study sample, 12 of 60 quantity pools and 8 of 60 action
    pools contain a duplicated text. Scoring by index alone would mark a
    selection wrong for pointing at an identical string one line further down,
    which is not an error anyone would recognise as one. The nurse sees text.
    """
    from .annotate import load, pools
    questions, chunks = load()
    out: Dict[Key, List[str]] = {}
    for q in questions:
        chunk = chunks.get(q["source_chunk_id"])
        if not chunk:
            continue
        for field, spans in pools(chunk).items():
            out[(q["qid"], field)] = [normalize(sp.text) for sp in spans]
    return out


def read(paths: List[str]) -> Dict[Key, dict]:
    rows: Dict[Key, dict] = {}
    for pattern in paths:
        for name in sorted(glob.glob(pattern)) or [pattern]:
            with open(name) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows[(row["qid"], row["field"])] = row
    return rows


def gold_set(g: dict) -> set:
    out = set(g.get("acceptable") or [])
    if g.get("best") is not None:
        out.add(g["best"])
    return out


def kappa(a: List[str], b: List[str]) -> float:
    """Cohen's kappa on the present/absent/unclear verdict."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    if not n or len(labels) < 2:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def compute(gold: Dict[Key, dict], pred: Dict[Key, dict],
            include_revealed: bool = False, by_index_only: bool = False,
            texts=None) -> Dict[str, tuple]:
    """{field or 'ALL': (stats, first disagreements)}. No printing."""
    if texts is None:
        texts = {} if by_index_only else candidate_texts()
    fields = sorted({f for _, f in gold})
    overall = {}
    for field in fields + ["ALL"]:
        keys = [k for k in gold if field == "ALL" or k[1] == field]
        s = dict(scored=0, strict=0, acceptable=0, false_abstention=0,
                 true_abstention=0, false_selection=0, present=0, absent=0,
                 skipped_unclear=0, skipped_revealed=0, skipped_nopred=0,
                 skipped_nocand=0, violation=0)
        wrong: List[str] = []
        for key in sorted(keys):
            g = gold[key]
            p = pred.get(key)
            if g.get("verdict") == "unclear" or not g.get("verdict"):
                s["skipped_unclear"] += 1
                continue
            if g.get("reference_revealed_first") and not include_revealed:
                s["skipped_revealed"] += 1
                continue
            if p is None:
                s["skipped_nopred"] += 1
                continue
            if p.get("n_candidates", 0) == 0:
                s["skipped_nocand"] += 1
                continue
            if p.get("violation"):
                s["violation"] += 1
            s["scored"] += 1
            chose = p.get("chosen")
            if g["verdict"] == "absent":
                s["absent"] += 1
                if chose is None:
                    s["true_abstention"] += 1
                else:
                    s["false_selection"] += 1
                    if len(wrong) < 6:
                        wrong.append("%s/%s  gold: NOT in chunk   selector chose [%d] %r"
                                     % (key[0], key[1], chose, (p.get("text") or "")[:62]))
                continue
            s["present"] += 1
            ok = gold_set(g)
            if chose is None:
                s["false_abstention"] += 1
                continue
            pool = texts.get(key) or []
            ok_texts = {pool[i] for i in ok if 0 <= i < len(pool)}
            same_text = bool(pool) and 0 <= chose < len(pool) and pool[chose] in ok_texts
            if chose in ok or same_text:
                s["acceptable"] += 1
                best = g.get("best")
                if best is not None and (
                        chose == best or
                        (pool and 0 <= chose < len(pool) and 0 <= best < len(pool)
                         and pool[chose] == pool[best])):
                    s["strict"] += 1
            elif len(wrong) < 6:
                wrong.append("%s/%s  gold %s   selector chose [%d] %r"
                             % (key[0], key[1], sorted(ok), chose,
                                (p.get("text") or "")[:62]))
        overall[field] = (s, wrong)
    return overall


def score(gold: Dict[Key, dict], pred: Dict[Key, dict],
          include_revealed: bool, by_index_only: bool = False) -> None:
    overall = compute(gold, pred, include_revealed, by_index_only)
    fields = sorted(k for k in overall if k != "ALL")

    def pct(num, den):
        return "   n/a" if not den else "%5.1f%%" % (100.0 * num / den)

    print("=" * 74)
    print("selection accuracy against nurse-annotated gold spans")
    print("=" * 74)
    head = "%-10s %7s %7s %8s %8s %8s %8s" % (
        "field", "scored", "strict", "accept.", "false-ab", "true-ab", "FALSE-SEL")
    print(head)
    print("-" * 74)
    for field in fields + ["ALL"]:
        s, _ = overall[field]
        print("%-10s %7d %7s %8s %8s %8s %8s" % (
            field, s["scored"],
            pct(s["strict"], s["present"]),
            pct(s["acceptable"], s["present"]),
            pct(s["false_abstention"], s["present"]),
            pct(s["true_abstention"], s["absent"]),
            pct(s["false_selection"], s["absent"])))
    s, _ = overall["ALL"]
    print("-" * 74)
    print("strict / acceptable are over the %d decisions where the chunk states it;"
          % s["present"])
    print("true-ab / FALSE-SEL are over the %d where it does not." % s["absent"])
    print("FALSE-SEL is the clinically important error: a real, correctly cited,")
    print("verbatim span that does not answer the question. The span guarantee")
    print("cannot detect it; only this column can.")
    print("\nA selection counts as correct when it points at a span the annotator")
    print("marked, OR at a different offset carrying the identical text. Pass")
    print("--by-index-only for the stricter offset-exact variant.")
    print("\nexcluded: %d unclear, %d annotated after revealing the reference,"
          " %d with no prediction, %d with no candidate"
          % (s["skipped_unclear"], s["skipped_revealed"],
             s["skipped_nopred"], s["skipped_nocand"]))
    if s["violation"]:
        print("contract violations among scored decisions: %d" % s["violation"])

    _, wrong = overall["ALL"]
    if wrong:
        print("\n--- first disagreements ---")
        for line in wrong:
            print("  " + line)


def agreement(a: Dict[Key, dict], b: Dict[Key, dict]) -> None:
    shared = sorted(set(a) & set(b))
    print("=" * 74)
    print("inter-annotator agreement on %d overlapping decisions" % len(shared))
    print("=" * 74)
    if not shared:
        print("no overlap between the two files")
        return
    va = [a[k].get("verdict", "") for k in shared]
    vb = [b[k].get("verdict", "") for k in shared]
    same = sum(1 for x, y in zip(va, vb) if x == y)
    print("verdict (present/absent/unclear): %d/%d agree (%.1f%%), kappa %.2f"
          % (same, len(shared), 100.0 * same / len(shared), kappa(va, vb)))

    both = [k for k in shared
            if a[k].get("verdict") == "present" == b[k].get("verdict")]
    if both:
        exact = sum(1 for k in both if a[k].get("best") == b[k].get("best"))
        overlap = sum(1 for k in both if gold_set(a[k]) & gold_set(b[k]))
        print("best span identical:            %d/%d (%.1f%%)"
              % (exact, len(both), 100.0 * exact / len(both)))
        print("acceptable sets intersect:      %d/%d (%.1f%%)"
              % (overlap, len(both), 100.0 * overlap / len(both)))
        print("\nThe gap between these two is the measurement to report: where")
        print("annotators pick different spans but both sets overlap, the chunk")
        print("restates one recommendation and 'strict' is the wrong metric for")
        print("that field.")
    for k in shared:
        if a[k].get("verdict") != b[k].get("verdict"):
            print("  disagree %s/%s: %s vs %s"
                  % (k[0], k[1], a[k].get("verdict"), b[k].get("verdict")))


def inter_model(paths: List[str], by_index_only: bool) -> None:
    """Agreement between selectors, which needs no annotation at all.

    Useful before the sheet comes back, for two reasons. It bounds what
    annotation can settle: decisions where every selector agrees are unlikely
    to be where the method lives or dies. And it locates the hard items, which
    is where a second annotator is worth paying for.

    It is NOT an accuracy measure. Three models agreeing on the wrong span is
    exactly the correlated failure this corpus should expect, since they share
    pretraining data and the same question wording.
    """
    runs: Dict[str, Dict[Key, dict]] = {}
    for pattern in paths:
        for name in sorted(glob.glob(pattern)) or [pattern]:
            label = Path(name).stem.replace("pred_", "")
            runs[label] = read([name])
    names = sorted(runs)
    if len(names) < 2:
        print("need at least two prediction files for agreement")
        return
    texts = {} if by_index_only else candidate_texts()

    def signature(key: Key, row: dict):
        if row.get("violation"):
            return "VIOLATION"
        if row.get("abstained"):
            return "ABSTAIN"
        pool = texts.get(key) or []
        chose = row.get("chosen")
        if pool and chose is not None and 0 <= chose < len(pool):
            return ("text", pool[chose])
        return ("index", chose)

    keys = [k for k in sorted(runs[names[0]])
            if all(k in runs[n] for n in names)
            and all(runs[n][k].get("n_candidates", 0) > 0 for n in names)]
    buckets: Dict[str, Dict[str, int]] = {}
    for key in keys:
        sigs = [signature(key, runs[n][key]) for n in names]
        top = max(sigs.count(x) for x in set(sigs))
        label = ("unanimous" if top == len(names)
                 else "majority" if top > 1 else "no two agree")
        for field in (key[1], "ALL"):
            buckets.setdefault(field, {}).setdefault(label, 0)
            buckets[field][label] += 1

    print("=" * 74)
    print("inter-selector agreement over %d decisions, %s"
          % (len(keys), ", ".join(names)))
    print("compared by %s" % ("offset" if by_index_only else "resolved text"))
    print("=" * 74)
    print("%-10s %7s %11s %10s %12s" % ("field", "n", "unanimous", "majority", "no two agree"))
    for field in sorted(buckets, key=lambda f: (f == "ALL", f)):
        b = buckets[field]
        n = sum(b.values())
        print("%-10s %7d %10.1f%% %9.1f%% %11.1f%%"
              % (field, n, 100.0 * b.get("unanimous", 0) / n,
                 100.0 * b.get("majority", 0) / n,
                 100.0 * b.get("no two agree", 0) / n))
    print("\nThis is agreement, not accuracy: selectors that share pretraining")
    print("data can agree on the same wrong span. Its use is to bound what")
    print("annotation can settle and to find the items worth annotating twice.")

    # abstention is the other axis, and it varies far more than agreement does
    print("\nabstention rate where candidates exist:")
    for n in names:
        parts = []
        for field in sorted({k[1] for k in keys}):
            ks = [k for k in keys if k[1] == field]
            ab = sum(1 for k in ks if runs[n][k].get("abstained"))
            parts.append("%s %d/%d" % (field, ab, len(ks)))
        print("  %-26s %s" % (n, "   ".join(parts)))


def calibration(gold: Dict[Key, dict], pred_paths: List[str],
                include_revealed: bool, bins: int = 5) -> None:
    """Does a stated probability mean what it says, against nurse gold?

    For every scored decision, the selector's would-be answer (`shadow_choice`,
    recorded even when it abstained) is checked against the annotator's
    acceptable set. The selector's score for showing that answer is
    P(something fits) x P(this option | something fits). Then:

    * reliability: within each score bin, how often was the answer right?
    * Brier score and ECE: one-number summaries of the same thing
    * AUROC: does a higher score at least RANK right answers above wrong ones?
      A selector can be badly calibrated yet well ranked, and a well-ranked one
      can be recalibrated from nurse feedback; a badly ranked one cannot.
    * risk-coverage: at each threshold, what fraction would be shown, how many
      of those are right, and how many are false selections (gold says the
      chunk does not state it). This is the curve the abstention threshold is
      chosen from — chosen from nurse data, not set by hand.
    """
    texts = candidate_texts()
    for path in pred_paths:
        for name in sorted(glob.glob(path)) or [path]:
            pred = read([name])
            rows = calibration_rows(gold, pred, include_revealed, texts)
            _print_calibration(name, rows, bins)


def calibration_rows(gold, pred, include_revealed=False, texts=None):
    """[(score, right, gold_says_absent)] for decisions with probability evidence."""
    if texts is None:
        texts = candidate_texts()
    rows = []
    for key, g in gold.items():
        p = pred.get(key)
        if not p or p.get("n_candidates", 0) == 0:
            continue
        if g.get("verdict") not in ("present", "absent"):
            continue
        if g.get("reference_revealed_first") and not include_revealed:
            continue
        shadow = p.get("shadow_choice")
        if shadow is None or p.get("p_present") is None:
            continue
        prob = p.get("prob_chosen")
        score = p["p_present"] * (prob if prob is not None else 1.0)
        if g["verdict"] == "absent":
            right = 0
        else:
            pool = texts.get(key) or []
            ok = gold_set(g)
            ok_t = {pool[i] for i in ok if 0 <= i < len(pool)}
            right = int(shadow in ok or
                        (0 <= shadow < len(pool) and pool[shadow] in ok_t))
        rows.append((score, right, g["verdict"] == "absent"))
    return rows




def _print_calibration(name, rows, bins):
    label = Path(name).stem.replace("pred_", "")
    print("=" * 74)
    print("calibration: %s   (%d decisions with probability evidence)"
          % (label, len(rows)))
    print("=" * 74)
    if len(rows) < 10:
        print("too few decisions carry probabilities to say anything")
        return
    n = len(rows)
    brier = sum((sc - r) ** 2 for sc, r, _ in rows) / n
    print("%-11s %5s %11s %10s" % ("score bin", "n", "mean score", "observed"))
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        cell = [(sc, r) for sc, r, _ in rows
                if lo <= sc < hi or (b == bins - 1 and sc == 1.0)]
        if not cell:
            print("%.1f-%.1f    %5d %11s %10s" % (lo, hi, 0, "-", "-"))
            continue
        ms = sum(sc for sc, _ in cell) / len(cell)
        ob = sum(r for _, r in cell) / len(cell)
        ece += len(cell) / n * abs(ms - ob)
        print("%.1f-%.1f    %5d %11.2f %10.2f" % (lo, hi, len(cell), ms, ob))
    pos = [sc for sc, r, _ in rows if r]
    neg = [sc for sc, r, _ in rows if not r]
    auroc = (sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
             / (len(pos) * len(neg))) if pos and neg else float("nan")
    print("Brier %.3f   ECE %.3f   AUROC %.3f   (base rate right %.2f)"
          % (brier, ece, auroc, len(pos) / n))

    print("\n%-9s %9s %12s %14s" % ("threshold", "coverage", "right|shown",
                                    "false-sel|shown"))
    for t in (0.0, 0.3, 0.5, 0.7, 0.9):
        shown = [(r, absent) for sc, r, absent in rows if sc >= t]
        if not shown:
            print("%-9.1f %8.0f%% %12s %14s" % (t, 0, "-", "-"))
            continue
        print("%-9.1f %8.0f%% %11.0f%% %13.0f%%"
              % (t, 100.0 * len(shown) / n,
                 100.0 * sum(r for r, _ in shown) / len(shown),
                 100.0 * sum(a for _, a in shown) / len(shown)))
    print("\nwith %d decisions, bins hold few items each: this can expose a"
          " grossly\nmiscalibrated selector, not certify a good one." % n)
    print()

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="span_select.score_spans")
    ap.add_argument("gold", nargs="*", help="annotator JSONL (globs allowed)")
    ap.add_argument("--pred", nargs="*", default=[],
                    help="prediction JSONL from `python3 -m span_select --out ...`")
    ap.add_argument("--second", nargs="*", default=[],
                    help="a second annotator's files, for agreement")
    ap.add_argument("--include-revealed", action="store_true",
                    help="keep decisions made after the reference was revealed")
    ap.add_argument("--by-index-only", action="store_true",
                    help="require the exact offset, not merely identical text")
    ap.add_argument("--calibration", nargs="*", default=None, metavar="PRED",
                    help="calibration of selector probabilities against gold")
    ap.add_argument("--inter-model", nargs="*", default=None, metavar="PRED",
                    help="agreement between prediction files; needs no gold")
    args = ap.parse_args(argv)

    if args.inter_model is not None:
        inter_model(args.inter_model or args.gold, args.by_index_only)
        return 0

    gold = read(args.gold)
    if any(g.get("mock") for g in gold.values()):
        who = sorted({g.get("annotator", "?") for g in gold.values() if g.get("mock")})
        print("#" * 74)
        print("#  MOCK GOLD from %s: model-generated, NOT nurse annotation." % ", ".join(who))
        print("#  Numbers below are agreement with that model, not clinical accuracy.")
        print("#" * 74)
    # Questions excluded from the test set (e.g. gold chunk removed from the
    # corpus) are dropped from scoring even if a sheet already covered them.
    testset = Path(__file__).resolve().parent.parent / "eval" / "full_testset.jsonl"
    excluded = {"q%03d" % i for i, line in enumerate(testset.open())
                if json.loads(line).get("excluded")}
    dropped = [k for k in gold if k[0] in excluded]
    for k in dropped:
        del gold[k]
    print("gold decisions: %d%s" % (len(gold), ("   (%d dropped: question excluded "
          "from the test set)" % len(dropped)) if dropped else ""))
    if args.calibration is not None:
        calibration(gold, args.calibration, args.include_revealed)
        return 0
    if args.pred:
        pred = read(args.pred)
        print("predictions:    %d" % len(pred))
        score(gold, pred, args.include_revealed, args.by_index_only)
    if args.second:
        agreement(gold, read(args.second))
    if not args.pred and not args.second:
        present = sum(1 for g in gold.values() if g.get("verdict") == "present")
        absent = sum(1 for g in gold.values() if g.get("verdict") == "absent")
        unclear = len(gold) - present - absent
        print("present %d   absent %d   unclear/blank %d" % (present, absent, unclear))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
