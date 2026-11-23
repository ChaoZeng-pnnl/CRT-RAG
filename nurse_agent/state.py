"""Agent state.

One dict threaded through every node. Each stage appends to it rather than
overwriting, so a finished run carries its own trace: which branch was chosen,
what each retriever returned, whether evidence was judged sufficient, what the
verifier struck out. The paper's ablations read these fields directly, so they
are part of the contract rather than debug output.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

QueryType = Literal["semantic", "relational", "freshness"]


def _keep_last(_old: Any, new: Any) -> Any:
    return new


def _extend(old: list, new: list) -> list:
    return (old or []) + (new or [])


class AgentState(TypedDict, total=False):
    # Input
    question: str

    # Classification
    query_type: Annotated[QueryType, _keep_last]
    classification_reason: str

    # Retrieval. `contexts` accumulates across retrieval branches and web
    # escalation, so it uses an append reducer. `ranked` is the reranker's
    # output and must REPLACE, not extend — appending it back onto its own
    # input silently duplicates every passage.
    contexts: Annotated[list[dict], _extend]
    ranked: Annotated[list[dict], _keep_last]
    retrieval_branches: Annotated[list[str], _extend]

    # Sufficiency / escalation
    sufficient: bool
    sufficiency_reason: str
    escalated: bool
    escalation_count: int

    # Generation and checks
    answer: str
    citations: list[dict]
    unsupported_claims: list[str]
    verification_passed: bool
    safety_flags: Annotated[list[str], _extend]
    safety_action: str

    # Bookkeeping
    trace: Annotated[list[dict], _extend]
    error: str
