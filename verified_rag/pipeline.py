"""The end-to-end answer path."""

from __future__ import annotations

import time
from typing import Optional

from .answer.assemble import assemble
from .answer.repair import repair
from .answer.synthesize import synthesize
from .answer.verify import verify_all
from .config import Settings, load
from .types import Response


class Pipeline:
    def __init__(self, settings: Optional[Settings] = None, store=None, embedder=None,
                 chat=None, jev=None, audit=None):
        self.settings = settings or load()
        s = self.settings
        if store is None:
            from corpus_index.graph import GraphStore
            store = GraphStore()
        if embedder is None:
            from corpus_index.embedder import Embedder
            embedder = Embedder()
        if chat is None:
            from .clients.chat import ChatClient
            chat = ChatClient(s.chat_base_url, s.chat_api_key, s.chat_model)
        if jev is None:
            from .clients.jev import JevClient
            jev = JevClient(s.jev_api_key, s.jev_model)
        if audit is None:
            from .store import Store
            audit = Store(s.audit_db)
        from .retrieval import Retriever
        self.store, self.chat, self.jev, self.audit = store, chat, jev, audit
        self.retriever = Retriever(store, embedder, jev, limit=s.retrieve_limit,
                                   population_threshold=s.population_threshold)
        # System A of the comparison: the same search over one static index.
        self.static_retriever = Retriever(store, embedder, jev, limit=s.retrieve_limit,
                                          versioned=False)

    def answer(self, question: str, snapshot: Optional[int] = None, mode: str = "C") -> Response:
        """mode C: the full system. For the comparison, B: versioned retrieval
        without sentence checks; A: conventional (static index, no checks).
        A and B answers are recorded with origin 'unchecked'."""
        s, t = self.settings, {}
        t0 = time.perf_counter()
        retriever = self.static_retriever if mode == "A" else self.retriever
        sources, seq, rt = retriever.retrieve(question, snapshot)
        t.update(rt)

        claims, gen = [], None
        if sources:
            t1 = time.perf_counter()
            claims, gen = synthesize(self.chat, question, sources, s.max_claims)
            t["synthesize_s"] = time.perf_counter() - t1

        if mode in ("A", "B"):
            for claim in claims:
                claim.origin = "unchecked"
        else:
            t2 = time.perf_counter()
            results = verify_all(self.jev, claims, sources, s.support_threshold)
            t["verify_s"] = time.perf_counter() - t2
            t3 = time.perf_counter()
            for claim, result in zip(claims, results):
                repair(self.chat, self.jev, claim, result, sources, s.support_threshold)
            t["repair_s"] = time.perf_counter() - t3

        response = assemble(question, seq, claims, sources, calibrate=self.audit.calibration)
        t["total_s"] = time.perf_counter() - t0
        response.timings = {k: round(v, 3) for k, v in t.items()}
        response.models = {"mode": mode, "chat": self.chat.model, "jev": sorted(self.jev.versions),
                           "chat_tokens": [gen.input_tokens, gen.output_tokens] if gen else None}
        self.audit.record(response)
        return response
