"""Invariants of the span-selection architecture, as runnable assertions.

    python3 -m span_select.selftest

Section 8 of the design document ranks "normalization defeat" second among the
risks: match rates disappoint, someone reaches for cosine similarity, and a
mechanical guarantee quietly becomes a statistical one while still being
described as a guarantee. The stated mitigation is to make the rule a design
invariant and write it into the test suite. This is that test suite.

It runs against the real corpus, not fixtures, because the properties worth
defending are properties of this corpus: non-breaking spaces inside doses,
line breaks inside quantities, the same number printed twice on one page.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .candidates import (MAX_CHOICE_OPTIONS, QUANTITY, enumerate_spans,
                         naive_match_would_fail, normalize, resolve)

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "chunks" / "chunks.jsonl"

FAILURES = []
CHECKS = [0]


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS[0] += 1
    if not condition:
        FAILURES.append("%s%s" % (name, (": " + detail) if detail else ""))


def corpus(limit: int = 1500):
    with CHUNKS.open() as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                return
            row = json.loads(line)
            if row.get("text"):
                yield row


def main() -> int:
    # --- 1. the one invariant the whole architecture rests on ----------------
    # Every span must read back byte-identically from the chunk it came from.
    # If this fails, no claim in the paper survives.
    spans_checked = 0
    for row in corpus():
        for span in enumerate_spans(row["chunk_id"], row["text"], max_options=0):
            spans_checked += 1
            try:
                got = resolve(span, row["text"])
            except ValueError as exc:
                check("resolve", False, str(exc)[:120])
                continue
            check("resolve is verbatim", got == span.text)
            check("offsets are ordered", 0 <= span.start < span.end <= len(row["text"]))
    check("corpus produced spans", spans_checked > 1000,
          "only %d spans" % spans_checked)

    # resolve() must REJECT a tampered span rather than return something near it
    row = next(iter(corpus(1)))
    span = enumerate_spans(row["chunk_id"], row["text"], max_options=0)[0]
    shifted = span.__class__(span.id, span.chunk_id, span.start + 1, span.end,
                             span.text, span.kind)
    try:
        resolve(shifted, row["text"])
        check("resolve rejects a shifted offset", False, "it returned instead")
    except ValueError:
        check("resolve rejects a shifted offset", True)

    # --- 2. normalization is lexical, and stays lexical ---------------------
    check("NFKC applied", normalize("ﬁve") == "five")
    check("non-breaking space collapsed", normalize("2 hours") == "2 hours")
    check("soft hyphen dropped", normalize("peri­operative") == "perioperative")
    check("line-break hyphen rejoined", normalize("anti-\nbiotic") == "antibiotic")
    check("newline collapsed", normalize("6\nweeks") == "6 weeks")
    check("idempotent", normalize(normalize("2 – 8 °C"))
          == normalize("2 – 8 °C"))
    # it must NOT paper over a clinical difference
    check("does not fold digits", normalize("500 ml") != normalize("5000 ml"))
    check("does not fold units", normalize("5 mg") != normalize("5 mcg"))
    check("does not lowercase", normalize("Mg") != normalize("mg"))
    # and it must not be reachable from a similarity score. Scan CODE only:
    # the module's docstring argues against embedding similarity by name, and a
    # test that cannot tell an argument from an implementation is worse than no
    # test, because it trains people to delete the prose.
    source = Path(__file__).with_name("candidates.py").read_text()
    code = re.sub(r'(?s)"""[\s\S]*?"""', "", source)
    code = re.sub(r"(?s)'''[\s\S]*?'''", "", code)
    code = re.sub(r"#.*", "", code)
    for banned in ("cosine", "embedding", "similarity", "difflib",
                   "SequenceMatcher", "fuzz", "sentence_transformers"):
        check("no similarity matching in candidates.py code (%s)" % banned,
              banned not in code)
    # the warning itself must survive, since it is the reason the rule exists
    check("the docstring still warns against similarity matching",
          "embedding similarity" in source)

    # --- 3. the ballot the selector is shown ------------------------------
    # No two options may render identically: the selector cannot choose between
    # them, and neither can the annotator.
    dup_questions = 0
    capped = 0
    for row in corpus(400):
        spans = enumerate_spans(row["chunk_id"], row["text"], max_options=0)
        if len(spans) > MAX_CHOICE_OPTIONS:
            capped += 1
        for kind in ("segment", "quantity"):
            rendered = [" ".join(s.text.split()) for s in spans if s.kind == kind]
            if len(rendered) != len(set(rendered)):
                dup_questions += 1
        # ids must be dense and in order, since the model answers with an index
        check("ids are dense and ordered",
              [s.id for s in spans] == list(range(len(spans))))
    check("no two options render identically", dup_questions == 0,
          "%d pools still contain an indistinguishable pair" % dup_questions)
    check("option cap respected",
          all(len(enumerate_spans(r["chunk_id"], r["text"])) <= MAX_CHOICE_OPTIONS
              for r in corpus(200)))

    # --- 4. the trap this design sidesteps, quantified ---------------------
    # Spans that are verbatim correct but would fail byte equality if a model
    # retyped them. Under generate-then-verify these are false alarms; here
    # they cannot arise. If this ever reaches zero, the argument is weaker and
    # the paper should say so rather than keep the sentence.
    total = affected = 0
    for row in corpus(400):
        for span in enumerate_spans(row["chunk_id"], row["text"], max_options=0):
            total += 1
            affected += naive_match_would_fail(span)
    check("the normalization trap is real in this corpus", affected > 0,
          "no span in 400 chunks would break naive matching")

    # a quantity may legitimately straddle a line break; that is the point
    check("quantity regex spans a line break",
          QUANTITY.search("give 6\nweeks of therapy") is not None)

    print("span_select self-test: %d checks" % CHECKS[0])
    print("  spans resolved verbatim: %d" % spans_checked)
    print("  spans needing normalization: %d / %d (%.1f%%) in 400 chunks"
          % (affected, total, 100.0 * affected / max(1, total)))
    print("  chunks over the %d-option cap: %d / 400" % (MAX_CHOICE_OPTIONS, capped))
    if FAILURES:
        print("\nFAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("\nall invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
