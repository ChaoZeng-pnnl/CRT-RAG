"""Typed objects that cross module boundaries. Plain dataclasses, JSON-able."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Source:
    """One retrieved passage, labelled S1..Sn for the writer to cite."""
    label: str
    chunk_id: str
    doc_id: str
    text: str
    title: str
    tier: int
    year: str
    page: Optional[int]
    snapshot: int
    current: bool = True                  # live at this snapshot
    superseded_note: str = ""             # e.g. "replaces WHO SSI 2016"
    refines: str = ""                     # population this source is specific to
    grade: str = ""                       # exact grade text of a recommendation stated here
    grade_change: str = ""                # e.g. "was 'Strong recommendation, ...' in the 2016 edition"
    # recommendations stated in this passage that a newer, live document has
    # replaced: [{old_text, new_text, new_chunk, new_label, new_year, new_start, new_end}]
    replaced: list = field(default_factory=list)


@dataclass
class Check:
    name: str            # "citation" | "details" | "support"
    passed: bool
    detail: str = ""
    score: Optional[float] = None


@dataclass
class Claim:
    text: str
    sources: List[str]                    # cited labels, e.g. ["S2"]
    origin: str = "synthesized"           # synthesized | repaired | verbatim | withdrawn
    checks: List[Check] = field(default_factory=list)
    support: Optional[float] = None       # Jev P(supported), raw
    calibrated: Optional[float] = None    # from nurse feedback, if enough exists
    span: Optional[dict] = None           # for verbatim claims: chunk_id/start/end
    history: List[dict] = field(default_factory=list)   # failed versions, for audit

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)


@dataclass
class Signals:
    """Four separate signals, never blended (design doc, response section)."""
    integrity: str          # "verified" | "partial" | "none"
    evidence: str           # e.g. "Tier 1, 2018, current"
    consensus: str          # e.g. "refined by WHO 2021 for caesarean section"
    relevance: str          # calibrated number, or "not yet calibrated (raw 0.85)"


@dataclass
class Response:
    question: str
    snapshot: int
    status: str                           # answered | partial | not_stated
    claims: List[Claim]
    sources: List[Source]
    signals: Signals
    timings: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    response_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
