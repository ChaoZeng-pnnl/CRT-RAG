"""Settings, read once from the environment / .env via corpus_index.config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from corpus_index.config import setting

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    chat_base_url: str = field(default_factory=lambda: setting("OPENAI_BASE_URL"))
    chat_api_key: str = field(default_factory=lambda: setting("OPENAI_API_KEY"))
    # Non-reasoning by default: the synthesis step is latency-bound (1.2 s
    # measured) and a reasoning model spends hundreds of tokens per call.
    chat_model: str = field(default_factory=lambda: setting("VERIFIED_RAG_CHAT_MODEL",
                                                            "gpt-4o-project"))
    jev_api_key: str = field(default_factory=lambda: setting("TYPESAFE_API_KEY"))
    jev_model: str = field(default_factory=lambda: setting("VERIFIED_RAG_JEV_MODEL", "jev-latest"))

    retrieve_limit: int = 6          # passages handed to the writer
    max_claims: int = 5              # sentences the writer may produce
    support_threshold: float = 0.5   # Jev P(supported) below this fails check 3
    population_threshold: float = 0.5
    audit_db: Path = ROOT / "runs" / "verified_rag.sqlite"


def load() -> Settings:
    return Settings()
