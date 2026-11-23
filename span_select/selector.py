"""Forced-choice selectors: question + candidates -> one integer, or abstain.

A selector is the only place a model appears in this pipeline, and its entire
output vocabulary is {0..n-1, NONE}. Two implementations:

* LexicalSelector  - no model, no network. A baseline, and a fallback when the
                     endpoint is unavailable.
* JevStyleSelector - a general chat model held to the Choice contract by prompt
                     and by strict parsing. This is what Jev claims to make
                     structurally impossible to violate; running it measures how
                     often an ordinary model breaks the contract, which is the
                     number that says whether Jev buys you anything here.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from .candidates import Span

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "is", "are", "be",
    "what", "when", "which", "how", "should", "after", "before", "and", "or",
    "with", "that", "this", "it", "as", "by", "from", "can", "do", "does",
}


@dataclass
class SelectionResult:
    """One selection.

    `violation` means the reply did not satisfy the strict contract (exactly an
    integer, or NONE). `recoverable` means a lenient parser could still pull a
    valid index out of it — e.g. the model wrote "[0]" or "Option 3.".

    The two are reported separately on purpose. A Jev-style Choice API returns
    one of the caller's keys and makes both impossible. An ordinary chat model
    needs a parser, and the size of `recoverable` relative to `violation` is
    what says whether the type guarantee is doing structural work or just
    saving you five lines of regex.
    """

    chosen: Optional[int] = None
    abstained: bool = False
    violation: bool = False
    recoverable: bool = False
    raw: str = ""
    error: str = ""
    # Probability evidence, when the selector exposes any. None means "this
    # selector gave none", never zero. These feed calibration and are never
    # displayed as the confidence a nurse sees.
    shadow_choice: Optional[int] = None   # what it would pick, even if abstaining
    prob_chosen: Optional[float] = None   # P(this option | something fits)
    p_present: Optional[float] = None     # P(any option provides the field)
    confidence: Optional[float] = None    # selector's own spread-based confidence
    probabilities: Optional[List[Optional[float]]] = None
    model_version: str = ""


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


class LexicalSelector:
    """Argmax token overlap with the question. Abstains on a zero-overlap field."""

    name = "lexical"

    def select(self, question: str, field_desc: str,
               spans: List[Span]) -> SelectionResult:
        want = _tokens(question) | _tokens(field_desc)
        if not spans:
            return SelectionResult(abstained=True)
        best, best_score = None, 0
        for i, span in enumerate(spans):
            score = len(want & _tokens(span.text))
            if score > best_score:
                best, best_score = i, score
        if best is None:
            return SelectionResult(abstained=True)
        return SelectionResult(chosen=best, raw=str(best))


_PROMPT = """You are selecting evidence for a perioperative nurse.

Question: {question}

Select the ONE numbered option below that best provides: {field_desc}

{options}

Reply with ONLY a bare integer: the number of the option you choose, with no
brackets and no other characters (for example: 7). Reply with the single word
NONE if no option provides it. No explanation, no punctuation, no other text."""

# Accepts exactly an integer or NONE. Anything else is a contract violation and
# is recorded as one rather than being salvaged by a looser parse.
_STRICT = re.compile(r"\A\s*(?:(\d{1,3})|(NONE))\s*\Z", re.IGNORECASE)

# What a tolerant caller would write: strip brackets, quotes, trailing prose.
_LENIENT = re.compile(r"\b(\d{1,3})\b")


class JevStyleSelector:
    """Choice over caller-supplied options via an OpenAI-compatible endpoint.

    Mirrors Jev's `choice` contract: options are supplied by the caller, the
    model returns one of their ids, and nothing it emits reaches the answer.
    """

    name = "llm"

    # 512 is not enough. A reasoning model spent 607 output tokens arriving at
    # the single word NONE, and at 512 it returned an EMPTY completion with
    # finish_reason=length — a hard contract violation caused entirely by the
    # budget, not by the model refusing the format. Billing is on tokens
    # actually emitted, so a generous ceiling costs a non-reasoning selector
    # nothing: those still answer in one token.
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: int = 120, max_option_chars: int = 300,
                 max_tokens: int = 2048, logprobs: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_option_chars = max_option_chars
        self.max_tokens = max_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._path = None  # discovered on first successful call
        # Token log-probabilities give a chat model the same kind of evidence
        # Jev returns natively, so the two can be calibrated side by side.
        # Reasoning models return none; endpoints that reject the parameter get
        # it switched off after the first refusal rather than failing the run.
        self.logprobs = logprobs

    def _endpoints(self) -> List[str]:
        if self._path:
            return [self._path]
        return [self.base_url + "/chat/completions",
                self.base_url + "/v1/chat/completions"]

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last = None
        for url in self._endpoints():
            request = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.api_key},
                method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    self._path = url
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = "HTTP %d" % exc.code
                if exc.code != 404:
                    detail = exc.read()[:200].decode("utf-8", "replace")
                    raise RuntimeError("%s: %s" % (last, detail))
            except Exception as exc:                      # noqa: BLE001
                last = "%s: %s" % (type(exc).__name__, str(exc)[:120])
                raise RuntimeError(last)
        raise RuntimeError(last or "no endpoint")

    def select(self, question: str, field_desc: str,
               spans: List[Span]) -> SelectionResult:
        if not spans:
            return SelectionResult(abstained=True)

        options = "\n".join(
            "[%d] %s" % (i, " ".join(s.text[:self.max_option_chars].split()))
            for i, s in enumerate(spans))
        prompt = _PROMPT.format(question=question, field_desc=field_desc,
                                options=options)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if self.logprobs:
            payload.update({"logprobs": True, "top_logprobs": 20})
        try:
            try:
                data = self._post(payload)
            except RuntimeError as exc:
                if self.logprobs and str(exc).startswith("HTTP 400"):
                    self.logprobs = False
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                    data = self._post(payload)
                else:
                    raise
            self.calls += 1
            usage = data.get("usage") or {}
            self.input_tokens += usage.get("prompt_tokens") or 0
            self.output_tokens += usage.get("completion_tokens") or 0
            choice = data["choices"][0]
            content = (choice["message"].get("content") or "")
            if not content.strip():
                return SelectionResult(
                    violation=True,
                    raw="<empty, finish_reason=%s>" % choice.get("finish_reason"))
        except Exception as exc:                          # noqa: BLE001
            return SelectionResult(error=str(exc)[:160])

        evidence = _token_evidence(choice, len(spans))

        match = _STRICT.match(content)
        if not match:
            loose = _LENIENT.search(content)
            salvaged = (loose and 0 <= int(loose.group(1)) < len(spans)
                        and content.count("\n") == 0 and len(content) <= 24)
            return SelectionResult(violation=True, recoverable=bool(salvaged),
                                   raw=content[:80])
        if match.group(2):
            return SelectionResult(abstained=True, raw=content.strip()[:80],
                                   **evidence)

        chosen = int(match.group(1))
        if not 0 <= chosen < len(spans):
            return SelectionResult(violation=True, raw=content.strip()[:80])
        if evidence.get("shadow_choice") is None:
            evidence["shadow_choice"] = chosen
        return SelectionResult(chosen=chosen, raw=content.strip()[:80], **evidence)


def _token_evidence(choice: dict, n: int) -> dict:
    """Probability evidence from the first output token's top log-probs.

    The answer is a bare integer or NONE, so the first token carries the whole
    decision. Mass on integer tokens is split across options; mass on NONE is
    the abstention. Mass outside the top 20 is unobserved, so these are lower
    bounds, and the field is left None when the endpoint returned nothing.
    """
    import math
    try:
        first = choice["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return {}
    by_option: dict = {}
    none_mass = 0.0
    for entry in first:
        token = (entry.get("token") or "").strip()
        p = math.exp(entry.get("logprob", -1e9))
        if token.isdigit() and 0 <= int(token) < n:
            by_option[int(token)] = by_option.get(int(token), 0.0) + p
        elif token.upper().startswith("NO"):
            none_mass += p
    option_mass = sum(by_option.values())
    if not option_mass and not none_mass:
        return {}
    out: dict = {"p_present": option_mass / (option_mass + none_mass)}
    if by_option:
        best = max(by_option, key=by_option.get)
        out["shadow_choice"] = best
        out["prob_chosen"] = by_option[best] / option_mass
        out["probabilities"] = [by_option.get(i, 0.0) / option_mass for i in range(n)]
    return out
