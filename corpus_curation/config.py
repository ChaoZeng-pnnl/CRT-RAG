"""Paths, HTTP settings, and API credentials shared by every fetcher."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = PROJECT_ROOT / "corpus"
LOG_JSONL   = CORPUS_ROOT / "metadata" / "download_log.jsonl"
LOG_CSV     = CORPUS_ROOT / "metadata" / "download_log.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NurseRAG-Downloader/1.0; "
        "+https://github.com/placeholder/nurse-rag)"
    )
}


def _credential(env_var: str, cfg_key: str) -> str:
    """Environment variable first, then the git-ignored .env file."""
    value = os.environ.get(env_var, "").strip()
    # configparser lower-cases option names, so normalise before lookup.
    return value if value else _CFG.get(cfg_key.lower(), "").strip()


# `.env` is the single credential file and is optional: absent credentials simply
# disable the phases that need them. The earlier `.config` fallback is removed so
# secrets cannot linger in a second location.
_CFG: dict[str, str] = {}
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _CFG[_k.strip().lower()] = _v.strip().strip('"').strip("'")

WILEY_TDM_TOKEN  = _credential("WILEY_TDM_TOKEN", "Wiley_TDM_token")
ELSEVIER_API_KEY = _credential("ELSEVIER_API_KEY", "Elsevier_API_key")
UNPAYWALL_EMAIL  = _credential("UNPAYWALL_EMAIL", "UNPAYWALL_EMAIL")

WILEY_TDM_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"

# NCBI / PubMed Central
NCBI_API_KEY      = _credential("NCBI_API_KEY", "NCBI_API_key")
NCBI_BASE         = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_OA_BASE      = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
NCBI_DELAY        = 0.11 if NCBI_API_KEY else 0.4   # 10 req/s with key, 3/s without
PMC_DIR           = CORPUS_ROOT / "tier4" / "PubMed_PMC"
PMC_MAX_PER_QUERY = 8   # articles to retrieve per search topic

OPENALEX_BASE          = "https://api.openalex.org/works"
OPENALEX_DIR           = CORPUS_ROOT / "tier4" / "OpenAlex"
OPENALEX_MAX_PER_QUERY = 8

ELSEVIER_SEARCH_BASE   = "https://api.elsevier.com/content/search/sciencedirect"
ELSEVIER_FULLTEXT_BASE = "https://api.elsevier.com/content/article/pii"
ELSEVIER_DIR           = CORPUS_ROOT / "tier4" / "Elsevier"

REQUEST_DELAY = 1.5   # seconds between requests
TIMEOUT       = 30    # seconds per request

# ---------------------------------------------------------------------------
# Content-integrity thresholds
#
# An HTTP 200 is not evidence that a document was retrieved: DSpace landing
# pages, "preparing to download" interstitials and branded 404 pages all return
# 200 with a plausible content-type. Every saved file is checked against these
# thresholds before it is logged as a success.
# ---------------------------------------------------------------------------

MIN_PDF_BYTES  = 20_000   # smallest plausible real guideline PDF
MIN_TEXT_CHARS = 3_000    # extracted-text floor for HTML and XML documents

# Substrings that identify an error/interstitial page saved under any extension.
ERROR_PAGE_MARKERS = (
    "page not found",
    "404 not found",
    "access denied",
    "just a moment",
    "preparing to download",
    "<ds-app>",             # DSpace (WHO IRIS) JavaScript shell
    "enable javascript",
    "your request has been blocked",
    "subscribe to continue",
)
