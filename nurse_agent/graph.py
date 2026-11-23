"""The LangGraph pipeline.

                     classify
                        |
        +---------------+---------------+
     semantic        relational      freshness
        |               |                |
    hybrid RAG      next-chain        corpus-first
        +---------------+----------------+
                        |
                     rerank
                        |
                  sufficiency?
                   /         \\
              no  /           \\  yes
           web_search           |
                   \\           /
                    generate
                        |
               verify_citations
                        |
                  safety_check
                        |
                      END

The escalation loop is bounded: web_search runs at most once per question
(`escalation_count`), so a question the corpus cannot answer degrades to an
honest "I don't know" instead of looping.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    check_sufficiency,
    classify,
    generate,
    rerank,
    retrieve_freshness,
    retrieve_relational,
    retrieve_semantic,
    safety_check,
    verify_citations,
    web_search,
)
from .state import AgentState


def _route_by_type(state: AgentState) -> str:
    return {
        "semantic": "retrieve_semantic",
        "relational": "retrieve_relational",
        "freshness": "retrieve_freshness",
    }.get(state.get("query_type", "semantic"), "retrieve_semantic")


def _route_by_sufficiency(state: AgentState) -> str:
    """Escalate only when evidence is insufficient AND we have not already."""
    if state.get("sufficient", True):
        return "generate"
    if state.get("escalation_count", 0) >= 1:
        return "generate"      # answer honestly from what we have
    return "web_search"


def build_agent(enable_web: bool = True, enable_verification: bool = True,
                enable_safety: bool = True, enable_routing: bool = True):
    """Construct the graph.

    The flags exist for the ablation grid: each disabled component is removed
    from the graph rather than stubbed, so a run genuinely measures the pipeline
    without it.
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify)
    graph.add_node("retrieve_semantic", retrieve_semantic)
    graph.add_node("retrieve_relational", retrieve_relational)
    graph.add_node("retrieve_freshness", retrieve_freshness)
    graph.add_node("rerank", rerank)
    graph.add_node("check_sufficiency", check_sufficiency)
    graph.add_node("generate", generate)

    if enable_web:
        graph.add_node("web_search", web_search)
    if enable_verification:
        graph.add_node("verify_citations", verify_citations)
    if enable_safety:
        graph.add_node("safety_check", safety_check)

    graph.set_entry_point("classify")

    if enable_routing:
        graph.add_conditional_edges("classify", _route_by_type, {
            "retrieve_semantic": "retrieve_semantic",
            "retrieve_relational": "retrieve_relational",
            "retrieve_freshness": "retrieve_freshness",
        })
    else:
        # Ablation: always take the semantic branch.
        graph.add_edge("classify", "retrieve_semantic")

    for node in ("retrieve_semantic", "retrieve_relational", "retrieve_freshness"):
        graph.add_edge(node, "rerank")

    graph.add_edge("rerank", "check_sufficiency")

    if enable_web:
        graph.add_conditional_edges("check_sufficiency", _route_by_sufficiency, {
            "web_search": "web_search",
            "generate": "generate",
        })
        graph.add_edge("web_search", "rerank")
    else:
        graph.add_edge("check_sufficiency", "generate")

    tail = "generate"
    if enable_verification:
        graph.add_edge(tail, "verify_citations")
        tail = "verify_citations"
    if enable_safety:
        graph.add_edge(tail, "safety_check")
        tail = "safety_check"
    graph.add_edge(tail, END)

    return graph.compile()


def ask(question: str, **flags) -> dict:
    """Run one question end to end and return the final state."""
    agent = build_agent(**flags)
    return agent.invoke({"question": question, "escalation_count": 0})
