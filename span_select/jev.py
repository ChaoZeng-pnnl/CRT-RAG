"""Jev (TypeSafe AI) as a span selector: a forced choice that also returns
probabilities for every option.

Wire format, from https://docs.typesafe.ai/api/system-one.md (OpenAPI at
https://api.typesafe.ai/openapi.json), read 2026-11-20:

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    { "model": "jev-latest",
      "state": <context: string | object | array>,
      "questions": { "<id>": {"type": "choice", "instructions": ..., "criteria": {name: description}},
                     "<id>": {"type": "noul",   "instructions": ...} } }
    -> { "model": "jev-1.14.0",
         "answers": { "<id>": {"type": "choice", "choice": name, "confidence": c,
                               "probabilities": {name: p, ...}},
                      "<id>": {"type": "noul", "noul": p} },
         "usage": {"input_tokens": n, "output_tokens": n} }

Why two questions per decision
------------------------------
Jev's Choice page says a "none of the above" option "may not work well in many
use cases" and recommends instead a companion Noul in the same request that
checks whether anything fits at all, reading the Choice only when the Noul says
yes. That is exactly our NOT_STATED decision, so it is implemented the
documented way rather than as an extra option.

Both answers are ALWAYS recorded, including the choice the selector would have
made when it abstains (`shadow_choice`). That costs nothing — both come back in
one request — and it is what lets the abstention threshold be chosen later,
from nurse-annotated data, instead of being fixed now by guesswork.

What this does NOT give you
---------------------------
Jev's probabilities are the model's, not the nurse's. They are an input to the
calibration table (span_select.score_spans --calibration), never the number
shown on screen. The docs also say values "can change between Jev releases",
so every row records the concrete model version that answered, and a
calibration table is only valid for the version it was fitted on.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from .candidates import MAX_CHOICE_OPTIONS, Span
from .selector import SelectionResult

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

_CHOICE_INSTRUCTIONS = (
    "A perioperative nurse asked `question`. Choose the ONE option that best "
    "provides `need`. Judge only by what each option's text actually says.")
_NOUL_INSTRUCTIONS = (
    "A perioperative nurse asked `question`. Does any of `options` actually "
    "provide `need` for that question? Answer yes only if an option states it; "
    "a related or partial statement is not enough.")


def post(api_key: str, body: dict, endpoint: str = ENDPOINT, timeout: int = 60,
         retries: int = 4) -> dict:
    """POST one SystemOne request. Retries 429 and 529 (documented as
    retryable) with exponential backoff; any other HTTP error is raised.
    Shared by JevSelector and verified_rag.clients.jev."""
    data = json.dumps(body).encode("utf-8")
    delay = 1.0
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + api_key})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            if exc.code in (429, 529) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError("HTTP %d: %s" % (exc.code, detail))
    raise RuntimeError("retries exhausted")


def _name(i: int) -> str:
    return "option_%d" % i


class JevSelector:
    """Choice over caller-supplied spans, plus a Noul for 'anything fits?'.

    Same interface as the other selectors: question + field + spans in,
    SelectionResult out, and the answer is only ever an index into `spans`.
    """

    name = "jev"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 endpoint: str = ENDPOINT, threshold: float = 0.5,
                 timeout: int = 60, max_option_chars: int = 600,
                 retries: int = 4):
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.threshold = threshold
        self.timeout = timeout
        self.max_option_chars = max_option_chars
        self.retries = retries
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_versions: Dict[str, int] = {}

    # -- request ---------------------------------------------------------------
    def build_request(self, question: str, field_desc: str,
                      spans: List[Span]) -> dict:
        if len(spans) > MAX_CHOICE_OPTIONS:
            raise ValueError("%d options exceed Jev's %d-option Choice limit"
                             % (len(spans), MAX_CHOICE_OPTIONS))
        options = {_name(i): " ".join(s.text[:self.max_option_chars].split())
                   for i, s in enumerate(spans)}
        return {
            "model": self.model,
            "state": {"question": question, "need": field_desc, "options": options},
            "questions": {
                "fits": {"type": "noul", "instructions": _NOUL_INSTRUCTIONS},
                "pick": {"type": "choice", "instructions": _CHOICE_INSTRUCTIONS,
                         "criteria": options},
            },
        }

    def _post(self, body: dict) -> dict:
        return post(self.api_key, body, endpoint=self.endpoint,
                    timeout=self.timeout, retries=self.retries)

    # -- response --------------------------------------------------------------
    def parse(self, data: dict, n_options: int) -> SelectionResult:
        """Turn a Jev response into a SelectionResult. Strict: anything that is
        not one of our option names is a contract violation, never coerced."""
        version = data.get("model") or "?"
        self.model_versions[version] = self.model_versions.get(version, 0) + 1
        usage = data.get("usage") or {}
        self.input_tokens += usage.get("input_tokens") or 0
        self.output_tokens += usage.get("output_tokens") or 0

        answers = data.get("answers") or {}
        pick, fits = answers.get("pick") or {}, answers.get("fits") or {}
        names = [_name(i) for i in range(n_options)]

        raw_probs = pick.get("probabilities") or {}
        probabilities: List[Optional[float]] = [raw_probs.get(nm) for nm in names]
        chosen_name = pick.get("choice")
        p_present = fits.get("noul")

        result = SelectionResult(
            raw=json.dumps({"choice": chosen_name, "noul": p_present})[:80],
            model_version=version,
            p_present=p_present if isinstance(p_present, (int, float)) else None,
            confidence=pick.get("confidence"),
            probabilities=[p if isinstance(p, (int, float)) else None
                           for p in probabilities],
        )
        unknown = [k for k in raw_probs if k not in names]
        if chosen_name not in names or unknown or result.p_present is None:
            result.violation = True
            return result

        index = names.index(chosen_name)
        result.shadow_choice = index
        result.prob_chosen = probabilities[index]
        if result.p_present < self.threshold:
            result.abstained = True        # nothing fits: NOT_STATED
        else:
            result.chosen = index
        return result

    def select(self, question: str, field_desc: str,
               spans: List[Span]) -> SelectionResult:
        if not spans:
            return SelectionResult(abstained=True)
        try:
            data = self._post(self.build_request(question, field_desc, spans))
            self.calls += 1
        except Exception as exc:                          # noqa: BLE001
            return SelectionResult(error=str(exc)[:160])
        return self.parse(data, len(spans))
