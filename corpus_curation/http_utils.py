"""Thin requests wrapper with shared headers and timeout."""

from __future__ import annotations


import requests

from .config import HEADERS, TIMEOUT


def _get(url: str, stream: bool = False, extra_headers: dict = None) -> requests.Response:
    hdrs = {**HEADERS, **(extra_headers or {})}
    return requests.get(url, headers=hdrs, timeout=TIMEOUT,
                        stream=stream, allow_redirects=True)


def _is_pdf_response(resp: requests.Response) -> bool:
    ct = resp.headers.get("content-type", "")
    return "pdf" in ct or resp.content[:4] == b"%PDF"
