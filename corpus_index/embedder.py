"""Batched embedding with an on-disk cache.

Embedding 14k chunks is slow and metered, and the index will be rebuilt many
times while the schema settles. Every vector is therefore cached by a hash of
(model, text), so a re-run costs nothing for unchanged chunks and a crash
mid-run loses only the batch in flight.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

from .config import (
    CACHE_DIR,
    EMBED_BATCH_SIZE,
    EMBED_CONCURRENCY,
    EMBED_MAX_RETRIES,
    EMBEDDING_ENCODING_FORMAT,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL, cache_dir: Path = CACHE_DIR,
                 batch_size: int = EMBED_BATCH_SIZE, use_cache: bool = True,
                 concurrency: int = EMBED_CONCURRENCY):
        self.model = model
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.concurrency = max(1, concurrency)
        self._lock = threading.Lock()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self.stats = {"embedded": 0, "cached": 0, "retries": 0, "failed": 0}

    # -- client -------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Put it in .env or the environment."
                )
            self._client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
        return self._client

    # -- cache --------------------------------------------------------------

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        # Shard by prefix: 14k files in one directory is slow on network storage.
        shard = self.cache_dir / key[:2]
        shard.mkdir(exist_ok=True)
        return shard / f"{key}.json"

    def _read_cache(self, text: str) -> list[float] | None:
        if not self.use_cache:
            return None
        path = self._cache_path(self._key(text))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, text: str, vector: list[float]) -> None:
        if not self.use_cache:
            return
        try:
            self._cache_path(self._key(text)).write_text(json.dumps(vector))
        except OSError:
            pass   # a cache miss next time is not worth failing the run over

    # -- embedding ----------------------------------------------------------

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """One API call, with exponential backoff on transient failures."""
        delay = 2.0
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=list(texts),
                    encoding_format=EMBEDDING_ENCODING_FORMAT,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                message = str(exc)
                # A batch too large for the backend never succeeds on retry.
                if "too large" in message.lower() or "maximum" in message.lower():
                    if len(texts) == 1:
                        raise
                    mid = len(texts) // 2
                    return self._embed_batch(texts[:mid]) + self._embed_batch(texts[mid:])
                if attempt == EMBED_MAX_RETRIES - 1:
                    raise
                self.stats["retries"] += 1
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def embed(self, texts: Iterable[str], progress: bool = True) -> list[list[float]]:
        texts = list(texts)
        vectors: list[list[float] | None] = [None] * len(texts)

        pending: list[int] = []
        for i, text in enumerate(texts):
            cached = self._read_cache(text)
            if cached is not None:
                vectors[i] = cached
                self.stats["cached"] += 1
            else:
                pending.append(i)

        # Batches are independent, and the endpoint round-trip dominates, so
        # run several in parallel. The OpenAI client is thread-safe; the shared
        # counters and the vectors list are guarded.
        groups = [pending[i:i + self.batch_size]
                  for i in range(0, len(pending), self.batch_size)]
        done = 0

        def run(batch_idx: list[int]) -> tuple[list[int], list[list[float]] | None]:
            try:
                return batch_idx, self._embed_batch([texts[i] for i in batch_idx])
            except Exception as exc:
                print(f"\n    ! batch failed ({type(exc).__name__}: {str(exc)[:110]})")
                return batch_idx, None

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for batch_idx, results in pool.map(run, groups):
                with self._lock:
                    if results is None:
                        self.stats["failed"] += len(batch_idx)
                    else:
                        for i, vector in zip(batch_idx, results):
                            vectors[i] = vector
                            self._write_cache(texts[i], vector)
                            self.stats["embedded"] += 1
                    done += len(batch_idx)
                    if progress:
                        print(f"    embedded {done}/{len(pending)} "
                              f"(cached {self.stats['cached']}, "
                              f"retries {self.stats['retries']})",
                              end="\r", flush=True)

        if progress and pending:
            print()
        return [v if v is not None else [] for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text], progress=False)[0]
