"""One extractor per source format. Each returns a list of Section objects."""

from .base import Section, extract
from .jats import extract_jats
from .pdf import extract_pdf
from .html import extract_html

__all__ = ["Section", "extract", "extract_jats", "extract_pdf", "extract_html"]
