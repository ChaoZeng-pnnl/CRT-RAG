"""Jev (TypeSafe) client: noul and choice questions, batched per request.

Jev returns probabilities, never text, so nothing it produces can reach a
nurse as an answer; it is used only to score and to point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from span_select.jev import post


@dataclass
class JevAnswers:
    answers: Dict[str, dict]
    model: str
    seconds: float

    def noul(self, key: str) -> Optional[float]:
        value = (self.answers.get(key) or {}).get("noul")
        return value if isinstance(value, (int, float)) else None

    def choice(self, key: str) -> tuple:
        a = self.answers.get(key) or {}
        return a.get("choice"), a.get("probabilities") or {}


class JevClient:
    def __init__(self, api_key: str, model: str = "jev-latest"):
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is not set")
        self.api_key, self.model = api_key, model
        self.calls = 0
        self.versions: Dict[str, int] = {}

    def ask(self, state, questions: Dict[str, dict]) -> JevAnswers:
        """One request carrying several questions over a shared `state`."""
        started = time.perf_counter()
        data = post(self.api_key, {"model": self.model, "state": state,
                                   "questions": questions})
        self.calls += 1
        version = data.get("model") or "?"
        self.versions[version] = self.versions.get(version, 0) + 1
        return JevAnswers(answers=data.get("answers") or {}, model=version,
                          seconds=time.perf_counter() - started)

    @staticmethod
    def noul_q(instructions: str) -> dict:
        return {"type": "noul", "instructions": instructions}

    @staticmethod
    def choice_q(instructions: str, options: Dict[str, str]) -> dict:
        return {"type": "choice", "instructions": instructions, "criteria": options}
