"""OpenAI-compatible chat client (the PNNL LiteLLM proxy)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from corpus_index.config import sampling_kwargs


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    model: str


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        if not (base_url and api_key):
            raise ValueError("chat endpoint not configured (OPENAI_BASE_URL / OPENAI_API_KEY)")
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 600) -> ChatResult:
        started = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **sampling_kwargs(self.model))
        usage = resp.usage
        return ChatResult(text=resp.choices[0].message.content or "",
                          input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                          output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                          seconds=time.perf_counter() - started,
                          model=getattr(resp, "model", self.model))
