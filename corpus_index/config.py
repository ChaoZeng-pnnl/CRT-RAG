"""Settings for embedding and the Neo4j graph store.

Credentials come from `.env` (git-ignored) or from real environment variables,
which take precedence so CI and shared runs never depend on a file on disk.
`.env` is the single credential file; the earlier `.config` fallback has been
removed so there is exactly one place secrets can live.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "chunks" / "chunks.jsonl"
CACHE_DIR = PROJECT_ROOT / "chunks" / "embedding_cache"


def _load_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser; tolerates quotes, spaces and comments."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV = _load_file(PROJECT_ROOT / ".env")


def setting(name: str, default: str = "") -> str:
    """Real environment first, then .env, then the default."""
    return (os.environ.get(name) or _ENV.get(name) or default).strip()


def sampling_kwargs(model: str) -> dict:
    """Per-family chat sampling parameters.

    The proxy rejects `temperature=0` for GPT-5 family models ("Only
    temperature=1 is supported"), returning HTTP 400. Callers that hard-code
    temperature=0 therefore fail completely the moment a GPT-5 model is
    configured — which is easy to miss, because the failure looks like the model
    being unusable rather than the request being malformed. Central so every
    call site inherits the fix.
    """
    name = model.lower()
    if "gpt-5" in name or "o1" in name or "o3" in name:
        return {}
    return {"temperature": 0}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

OPENAI_API_KEY   = setting("OPENAI_API_KEY")
OPENAI_BASE_URL  = setting("OPENAI_BASE_URL")
EMBEDDING_MODEL  = setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL       = setting("OPENAI_MODEL")
FALLBACK_MODEL   = setting("OPENAI_FALLBACK_MODEL")
TAVILY_API_KEY   = setting("Tavily_API_key")

# gemini-embedding-001 returns 3072 dimensions, L2-normalised. Verified against
# the live endpoint rather than assumed — the vector index DDL needs the exact
# number and Neo4j will not let you change it after creation.
EMBEDDING_DIM = int(setting("EMBEDDING_DIM", "3072"))

# The proxy is LiteLLM in front of Vertex AI, which rejects the OpenAI SDK's
# default base64 encoding. Always ask for floats.
EMBEDDING_ENCODING_FORMAT = "float"

EMBED_BATCH_SIZE = int(setting("EMBED_BATCH_SIZE", "32"))
EMBED_MAX_RETRIES = int(setting("EMBED_MAX_RETRIES", "5"))
# Batches are independent and latency-bound; a handful in flight cuts a 14k-chunk
# run from hours to minutes. Raise cautiously — the proxy is shared infrastructure.
EMBED_CONCURRENCY = int(setting("EMBED_CONCURRENCY", "8"))

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

NEO4J_URI      = setting("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = setting("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = setting("NEO4J_PASSWORD")
NEO4J_DATABASE = setting("NEO4J_DATABASE", "neo4j")

VECTOR_INDEX    = "chunk_embedding"
FULLTEXT_INDEX  = "chunk_fulltext"
# Cosine matches the normalised vectors the endpoint returns.
VECTOR_SIMILARITY = "cosine"
