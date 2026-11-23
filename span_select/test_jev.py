"""Offline tests for the Jev selector and the chat selector's log-probs.

    python3 -m span_select.test_jev

No network. The transport is stubbed with responses shaped exactly as the
OpenAPI spec documents them, so a failure here is a bug in our client, not in
Jev. Run before spending a live call.
"""

from __future__ import annotations

import io
import json
import math
import sys
import urllib.error
from unittest import mock

from .candidates import enumerate_spans, resolve
from .jev import JevSelector
from .selector import _token_evidence

FAILURES = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(name + ((": " + detail) if detail else ""))


TEXT = ("Remove the urinary catheter within 24 hours of surgery. "
        "Avoid routine drains after hysterectomy. Give 2–8 mg if needed.")
SPANS = [s for s in enumerate_spans("c1", TEXT, max_options=0) if s.kind == "segment"]


def response(choice="option_0", noul=0.9, probs=None, model="jev-1.14.0"):
    probs = probs or {"option_%d" % i: (0.8 if i == 0 else 0.2 / (len(SPANS) - 1))
                      for i in range(len(SPANS))}
    return {"model": model,
            "answers": {"pick": {"type": "choice", "choice": choice,
                                 "confidence": 0.7, "probabilities": probs},
                        "fits": {"type": "noul", "noul": noul}},
            "usage": {"input_tokens": 321, "output_tokens": 2}}


def main() -> int:
    sel = JevSelector("test-key", threshold=0.5)

    # 1. request shape matches the documented SystemOneRequest
    body = sel.build_request("When do I remove the catheter?", "the action", SPANS)
    check("model present", body["model"] == "jev-latest")
    check("state carries question, need, options",
          set(body["state"]) == {"question", "need", "options"})
    q = body["questions"]
    check("one noul + one choice", q["fits"]["type"] == "noul" and q["pick"]["type"] == "choice")
    check("criteria are the options", q["pick"]["criteria"] == body["state"]["options"])
    check("option names are dense", list(q["pick"]["criteria"]) ==
          ["option_%d" % i for i in range(len(SPANS))])
    check("noul has no criteria (yes/no)", "criteria" not in q["fits"])
    check("no hidden NONE option", not any("none" in k.lower() for k in q["pick"]["criteria"]))
    json.dumps(body)  # must be serialisable
    try:
        sel.build_request("q", "f", SPANS * 200)
        check("over 255 options rejected", False, "it built the request")
    except ValueError:
        pass

    # 2. a confident answer resolves to the exact span
    r = sel.parse(response(), len(SPANS))
    check("chooses option_0", r.chosen == 0 and not r.abstained and not r.violation)
    check("shadow == chosen", r.shadow_choice == 0)
    check("prob_chosen read", abs((r.prob_chosen or 0) - 0.8) < 1e-9)
    check("probabilities ordered by option", len(r.probabilities) == len(SPANS)
          and abs(r.probabilities[0] - 0.8) < 1e-9)
    check("version recorded", r.model_version == "jev-1.14.0")
    check("usage counted", sel.input_tokens == 321)
    check("resolves verbatim", resolve(SPANS[r.chosen], TEXT) == SPANS[r.chosen].text)

    # 3. low noul abstains but keeps the shadow choice for later thresholding
    r = sel.parse(response(choice="option_1", noul=0.2), len(SPANS))
    check("abstains below threshold", r.abstained and r.chosen is None)
    check("shadow kept on abstain", r.shadow_choice == 1)
    check("p_present kept on abstain", abs(r.p_present - 0.2) < 1e-9)

    # 4. anything outside our vocabulary is a violation, never coerced
    r = sel.parse(response(choice="option_99"), len(SPANS))
    check("unknown option is a violation", r.violation and r.chosen is None)
    bad = response(); bad["answers"]["pick"]["probabilities"]["sneaky"] = 0.1
    check("unknown probability key is a violation", sel.parse(bad, len(SPANS)).violation)
    bad = response(); del bad["answers"]["fits"]
    check("missing noul is a violation", sel.parse(bad, len(SPANS)).violation)

    # 5. 429 is retried with backoff, then succeeds; 401 is not retried
    calls = {"n": 0}

    def flaky(req, timeout):
        calls["n"] += 1
        check("bearer auth header", req.get_header("Authorization") == "Bearer test-key")
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b"{}"))
        return io.BytesIO(json.dumps(response()).encode())

    with mock.patch("urllib.request.urlopen", flaky), mock.patch("time.sleep"):
        r = sel.select("q", "f", SPANS)
    check("retried after 429", calls["n"] == 2 and r.chosen == 0, str(calls))

    def unauthorised(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b'{"detail":"bad key"}'))

    calls["n"] = 0
    with mock.patch("urllib.request.urlopen", unauthorised), mock.patch("time.sleep"):
        r = sel.select("q", "f", SPANS)
    check("401 not retried", calls["n"] == 1 and r.error.startswith("HTTP 401"), r.error)

    # 6. chat log-probs: integer tokens split the option mass, NONE is abstention
    lp = lambda p: math.log(p)
    ev = _token_evidence({"logprobs": {"content": [{"top_logprobs": [
        {"token": "2", "logprob": lp(0.6)}, {"token": "0", "logprob": lp(0.2)},
        {"token": "NONE", "logprob": lp(0.2)}, {"token": "99", "logprob": lp(0.05)}]}]}}, 3)
    check("shadow = top integer", ev.get("shadow_choice") == 2)
    check("prob_chosen renormalised over options", abs(ev["prob_chosen"] - 0.75) < 1e-9)
    check("p_present excludes NONE mass", abs(ev["p_present"] - 0.8) < 1e-9)
    check("out-of-range token ignored", len(ev["probabilities"]) == 3)
    check("no logprobs -> no evidence", _token_evidence({"message": {}}, 3) == {})

    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("jev selector offline tests: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
