"""WHO IRIS (DSpace 7) handle resolution.

The catalogue's legacy `apps.who.int/iris/bitstream/handle/...` URLs no longer
serve files: IRIS migrated to DSpace 7, and every one of those paths now returns
HTTP 200 with a 755-byte Angular shell. That is how nine empty "WHO guidelines"
entered the corpus.

The stable route is the REST API — resolve the handle to an item, walk its
ORIGINAL bundle, and pick the English PDF. Handles are durable identifiers, so
they keep working across future IRIS redesigns; bitstream UUIDs would not.
"""

from __future__ import annotations

import re

import requests

from ..config import HEADERS, TIMEOUT

IRIS_API = "https://iris.who.int/server/api"

_JSON_HEADERS = {**HEADERS, "Accept": "application/json"}


def _handle_from_url(url: str) -> str | None:
    """Extract `10665/44186` from any legacy or current IRIS URL."""
    match = re.search(r"/handle/(\d+/\d+)", url) or re.search(r"(10665/\d+)", url)
    return match.group(1) if match else None


def resolve_pdf_url(url_or_handle: str, prefer: str = "_eng") -> str | None:
    """Return a direct bitstream content URL for an IRIS item, or None.

    `prefer` selects the language suffix; the main document is chosen over
    supplementary files (checklists, annexes) by preferring the largest match.
    """
    handle = _handle_from_url(url_or_handle) or url_or_handle

    try:
        item = requests.get(f"{IRIS_API}/pid/find", params={"id": f"hdl:{handle}"},
                            headers=_JSON_HEADERS, timeout=TIMEOUT)
        if item.status_code != 200:
            return None
        uuid = item.json().get("uuid")
        if not uuid:
            return None

        bundles = requests.get(f"{IRIS_API}/core/items/{uuid}/bundles",
                               headers=_JSON_HEADERS, timeout=TIMEOUT).json()
        original = next(
            (b for b in bundles.get("_embedded", {}).get("bundles", [])
             if b.get("name") == "ORIGINAL"), None)
        if not original:
            return None

        listing = requests.get(f"{IRIS_API}/core/bundles/{original['uuid']}/bitstreams",
                               headers=_JSON_HEADERS, timeout=TIMEOUT).json()
        bitstreams = [b for b in listing.get("_embedded", {}).get("bitstreams", [])
                      if b.get("name", "").lower().endswith(".pdf")]
        if not bitstreams:
            return None
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None

    def rank(bitstream: dict) -> tuple[int, int, int]:
        name = bitstream.get("name", "").lower()
        return (
            1 if prefer in name else 0,            # right language
            0 if "checklist" in name or "annex" in name else 1,  # main document
            bitstream.get("sizeBytes", 0),         # then the fullest file
        )

    best = max(bitstreams, key=rank)
    return best["_links"]["content"]["href"]
