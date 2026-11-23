"""Run the span-selection loop over real questions and their gold chunks.

    python3 -m span_select            # lexical baseline, no network, 50 questions
    python3 -m span_select --llm -n 20

Selection is deliberately isolated from retrieval: each question is run against
the chunk it was written from, so the numbers describe the selection
architecture and not the retriever. Retrieval already has its own harness.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

from .candidates import (MAX_CHOICE_OPTIONS, enumerate_spans,
                         naive_match_would_fail, resolve)
from .selector import JevStyleSelector, LexicalSelector
from .jev import JevSelector

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "eval" / "full_testset.jsonl"
CHUNKS = ROOT / "chunks" / "chunks.jsonl"

FIELDS = [
    ("action", "the sentence stating what the nurse should do"),
    ("quantity", "the specific number and unit the question asks for"),
]


def _settings() -> Dict[str, str]:
    """Read .env without importing the project (its deps are not installed)."""
    out: Dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _load(n: int, qids=None):
    """Load questions. qid is the line index in the FULL test set, always, so a
    prediction dump joins to an annotation sheet of the same questions."""
    questions = [json.loads(line) for line in TESTSET.open()]
    for i, q in enumerate(questions):
        q["qid"] = "q%03d" % i
    if qids is not None:
        order = {qid: i for i, qid in enumerate(qids)}
        questions = sorted((q for q in questions if q["qid"] in order),
                           key=lambda q: order[q["qid"]])
    else:
        questions = questions[:n]
    wanted = {q["source_chunk_id"] for q in questions}
    chunks: Dict[str, dict] = {}
    for line in CHUNKS.open():
        row = json.loads(line)
        if row["chunk_id"] in wanted:
            chunks[row["chunk_id"]] = row
            if len(chunks) == len(wanted):
                break
    return questions, chunks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="span_select")
    parser.add_argument("-n", type=int, default=50, help="questions to run")
    parser.add_argument("--model", default="", help="override the chat model")
    parser.add_argument("--llm", action="store_true",
                        help="use the chat endpoint instead of the lexical baseline")
    parser.add_argument("--jev", action="store_true",
                        help="use TypeSafe Jev (needs TYPESAFE_API_KEY)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Jev: abstain when P(anything fits) is below this")
    parser.add_argument("--logprobs", action="store_true",
                        help="chat selector: request token log-probabilities")
    parser.add_argument("--qids", default="",
                        help="a manifest.json from span_select.annotate, or a "
                             "file of qids: run exactly those questions, in "
                             "that order, instead of the first -n")
    parser.add_argument("--out", default="",
                        help="write per-decision predictions here as JSONL, for "
                             "span_select.score_spans to join against gold spans")
    args = parser.parse_args(argv)

    qids = None
    if args.qids:
        raw = Path(args.qids).read_text()
        if args.qids.endswith(".json"):
            qids = json.loads(raw)["qids"]
        else:
            qids = [l.strip() for l in raw.split() if l.strip()]

    questions, chunks = _load(args.n, qids)
    print("questions: %d   gold chunks resolved: %d%s"
          % (len(questions), len(chunks),
             ("   (from %s)" % args.qids) if qids else ""))

    if args.jev:
        cfg = _settings()
        import os
        key = os.environ.get("TYPESAFE_API_KEY") or cfg.get("TYPESAFE_API_KEY")
        if not key:
            # No silent fallback: predictions labelled "jev" that came from the
            # lexical baseline would poison the calibration comparison.
            print("! TYPESAFE_API_KEY is not set; refusing to run --jev")
            return 2
        selector = JevSelector(key, model=args.model or "jev-latest",
                               threshold=args.threshold)
        print("selector: %s (choice + noul, abstain below P=%.2f)"
              % (selector.model, args.threshold))
    elif args.llm:
        cfg = _settings()
        model = args.model or cfg.get("OPENAI_LOWCOST_MODEL") or cfg.get("OPENAI_MODEL")
        if not (cfg.get("OPENAI_API_KEY") and cfg.get("OPENAI_BASE_URL")):
            print("! no endpoint configured; falling back to lexical")
            selector = LexicalSelector()
        else:
            selector = JevStyleSelector(cfg["OPENAI_BASE_URL"],
                                        cfg["OPENAI_API_KEY"], model,
                                        logprobs=args.logprobs)
            print("selector: %s (forced choice, one integer or NONE)" % model)
    else:
        selector = LexicalSelector()
        print("selector: lexical baseline (no network)")

    counts: List[int] = []
    over_cap = 0
    with_quantity = 0
    naive_fail_spans = 0
    total_spans = 0
    stats = {f: {"chosen": 0, "abstain": 0, "violation": 0, "recoverable": 0,
                 "error": 0, "no_cand": 0}
             for f, _ in FIELDS}
    invariant_violations = 0
    errors: List[str] = []
    examples: List[tuple] = []

    predictions: List[dict] = []

    for q in questions:
        qid = q["qid"]
        chunk = chunks.get(q["source_chunk_id"])
        if not chunk:
            continue
        text = chunk["text"]
        spans = enumerate_spans(chunk["chunk_id"], text, max_options=0)
        counts.append(len(spans))
        if len(spans) > MAX_CHOICE_OPTIONS:
            over_cap += 1
        spans = spans[:MAX_CHOICE_OPTIONS]
        total_spans += len(spans)
        naive_fail_spans += sum(1 for s in spans if naive_match_would_fail(s))
        quantities = [s for s in spans if s.kind == "quantity"]
        if quantities:
            with_quantity += 1

        for fname, fdesc in FIELDS:
            pool = quantities if fname == "quantity" else [s for s in spans
                                                           if s.kind == "segment"]
            row = {"qid": qid, "chunk_id": chunk["chunk_id"], "field": fname,
                   "selector": getattr(selector, "model", selector.name),
                   "n_candidates": len(pool), "chosen": None, "abstained": False,
                   "violation": False, "error": "", "text": "",
                   "start": None, "end": None}
            predictions.append(row)
            if not pool:
                stats[fname]["no_cand"] += 1
                row["error"] = "no candidates"
                continue
            result = selector.select(q["question"], fdesc, pool)
            for attr in ("shadow_choice", "prob_chosen", "p_present",
                         "confidence", "model_version"):
                value = getattr(result, attr, None)
                if value not in (None, ""):
                    row[attr] = value
            if result.error:
                row["error"] = result.error
                stats[fname]["error"] += 1
                if len(errors) < 3:
                    errors.append(result.error)
                continue
            if result.violation:
                row["violation"] = True
                stats[fname]["violation"] += 1
                if result.recoverable:
                    stats[fname]["recoverable"] += 1
                if len(errors) < 6:
                    errors.append("contract violation, model said: %r" % result.raw)
                continue
            if result.abstained or result.chosen is None:
                row["abstained"] = True
                stats[fname]["abstain"] += 1
                continue
            if not 0 <= result.chosen < len(pool):
                row["violation"] = True
                stats[fname]["violation"] += 1
                continue
            span = pool[result.chosen]
            try:
                recovered = resolve(span, text)          # the invariant
            except ValueError as exc:
                invariant_violations += 1
                errors.append(str(exc)[:160])
                continue
            row["chosen"] = result.chosen
            row["text"] = recovered
            row["start"], row["end"] = span.start, span.end
            stats[fname]["chosen"] += 1
            if fname == "quantity" and len(examples) < 5:
                examples.append((q["question"][:66], recovered,
                                 span.start, span.end, chunk.get("page")))

    n = len(counts) or 1
    print("\n--- candidate enumeration (deterministic, no model) ---")
    print("  candidates/chunk       mean %.1f   median %d   max %d"
          % (statistics.mean(counts), statistics.median(counts), max(counts)))
    print("  chunks over Jev's 255-option cap          %d / %d" % (over_cap, n))
    print("  chunks with >=1 quantity candidate        %d / %d" % (with_quantity, n))
    print("  spans whose text != normalize(text)       %d / %d"
          % (naive_fail_spans, total_spans))
    print("    (these are exactly the spans a generate-then-verify design")
    print("     would flag as unsupported despite being verbatim correct)")

    print("\n--- selection ---")
    for fname, _ in FIELDS:
        s = stats[fname]
        print("  %-9s chose %3d   abstained %3d   no candidates %3d   "
              "strict violations %3d (recoverable %d)   errors %3d"
              % (fname, s["chosen"], s["abstain"], s["no_cand"],
                 s["violation"], s["recoverable"], s["error"]))

    print("\n--- the guarantee ---")
    print("  invariant failures (span did not resolve): %d" % invariant_violations)
    print("  answer text produced by a model:            0 (by construction)")

    if examples:
        print("\n--- resolved quantity spans (text read from the chunk) ---")
        for question, got, start, end, page in examples:
            print("  Q: %s" % question)
            print("     -> %r  @ offsets (%d,%d), page %s" % (got, start, end, page))

    if isinstance(selector, JevSelector) and selector.calls:
        print("\n--- cost ---")
        print("  calls %d   input tokens %d   output tokens %d"
              % (selector.calls, selector.input_tokens, selector.output_tokens))
        print("  model versions that answered: %s" % selector.model_versions)

    if isinstance(selector, JevStyleSelector) and selector.calls:
        print("\n--- cost ---")
        print("  calls %d   input tokens %d   output tokens %d"
              % (selector.calls, selector.input_tokens, selector.output_tokens))
        if selector.calls:
            print("  output tokens per decision: %.0f  (the answer is one integer)"
                  % (selector.output_tokens / selector.calls))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in predictions:
                fh.write(json.dumps(row) + "\n")
        print("\npredictions -> %s  (%d decisions)" % (out, len(predictions)))

    if errors:
        print("\n--- first failures ---")
        for e in errors[:6]:
            print("  %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
