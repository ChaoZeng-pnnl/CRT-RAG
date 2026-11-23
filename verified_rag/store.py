"""Audit log, nurse feedback and calibration, in SQLite.

The design puts these in Postgres beside the event log; SQLite keeps the
prototype dependency-free and has the same tables. Every response is stored
exactly as shown, with its release snapshot, so "what did the system say on
3 March, and why" is a query. Questions may contain patient details: this file
needs the same access controls as any clinical record.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
  response_id TEXT PRIMARY KEY, at REAL, question TEXT, snapshot INTEGER,
  status TEXT, models TEXT, response TEXT);
CREATE TABLE IF NOT EXISTS feedback (
  response_id TEXT, claim_index INTEGER, verdict TEXT CHECK (verdict IN
  ('right','wrong','not_stated')), note TEXT, at REAL);
"""

# Calibration groups: raw support probability in bins, and the claim's origin,
# because a claim that passed only after repair is less trustworthy than one
# that passed first time and must not be pooled with it.
BINS = (0.0, 0.5, 0.7, 0.85, 0.95, 1.01)
MIN_N = 30


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def record(self, response) -> str:
        rid = response.response_id or uuid.uuid4().hex
        response.response_id = rid
        with self._conn() as c:
            c.execute("INSERT INTO responses VALUES (?,?,?,?,?,?,?)",
                      (rid, time.time(), response.question, response.snapshot, response.status,
                       json.dumps(response.models), json.dumps(response.to_dict(), default=str)))
        return rid

    def feedback(self, response_id: str, claim_index: int, verdict: str, note: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO feedback VALUES (?,?,?,?,?)",
                      (response_id, claim_index, verdict, note, time.time()))

    def calibration(self, raw: float, origin: str) -> Optional[Tuple[float, int]]:
        """Share of nurse 'right' verdicts among past claims in the same group,
        or None while the group has fewer than MIN_N verdicts."""
        lo = max(b for b in BINS if b <= raw)
        hi = min(b for b in BINS if b > raw)
        right = total = 0
        with self._conn() as c:
            for (resp_json, idx, verdict) in c.execute(
                    "SELECT r.response, f.claim_index, f.verdict FROM feedback f "
                    "JOIN responses r USING (response_id)"):
                claims = json.loads(resp_json)["claims"]
                if not (0 <= idx < len(claims)):
                    continue
                cl = claims[idx]
                if cl.get("origin") != origin or cl.get("support") is None:
                    continue
                if lo <= cl["support"] < hi:
                    total += 1
                    right += verdict == "right"
        return (right / total, total) if total >= MIN_N else None
