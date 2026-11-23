"""Span selection: the model points at text, it never writes text.

The guarantee this package exists to provide:

    every clinical specific in an answer is a verbatim substring of a named
    source chunk, because the only thing a model is ever allowed to emit is an
    integer index into a candidate list that *this code* built from that chunk.

Contrast with generate-then-verify, which is what the current `nurse_agent`
pipeline does: there a model writes prose, and a second model is asked whether
the prose is supported. That check is an opinion, and two such opinions in this
project's own data disagreed 0.81 vs 0.24 on acceptability.

Here there is nothing to check. Candidate spans carry (chunk_id, start, end),
the answer text is read back out of the chunk by those offsets, and a span that
is not in the corpus is not expressible.
"""

from .candidates import Span, enumerate_spans, normalize, resolve
from .selector import JevStyleSelector, LexicalSelector, SelectionResult

__all__ = [
    "Span", "enumerate_spans", "normalize", "resolve",
    "LexicalSelector", "JevStyleSelector", "SelectionResult",
]
