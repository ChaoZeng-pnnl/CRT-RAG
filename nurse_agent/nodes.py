"""Agent nodes.

Each node is a plain function on AgentState, so every stage can be unit-tested
and ablated without constructing the graph.
"""

from __future__ import annotations

import json
import re
import time

from corpus_index.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TAVILY_API_KEY
from corpus_index.embedder import Embedder
from corpus_index.graph import GraphStore

from .state import AgentState

_client = None
_store = None
_embedder = None


def llm():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def store() -> GraphStore:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _chat(prompt: str, model: str = CHAT_MODEL, temperature: float = 0.0,
          max_tokens: int = 4000) -> str:
    response = llm().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _json_from(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _step(name: str, start: float, **extra) -> dict:
    return {"node": name, "ms": int((time.time() - start) * 1000), **extra}


# --- 1. classification -------------------------------------------------------

CLASSIFY_PROMPT = """Classify this perioperative nursing question into one branch.

semantic   — asks what guidance says about a topic; answerable from one passage.
             e.g. "How long should the dressing stay on after a hysterectomy?"
relational — needs facts combined across entities: a patient factor plus a
             procedure, a drug interaction, a contraindication chain.
             e.g. "What VTE prophylaxis for a patient on apixaban having a lap hysterectomy?"
freshness  — asks for current/latest/changed guidance, or names a date or recent year.
             e.g. "Has the sepsis guidance changed this year?"

Question: {question}

Return JSON only: {{"query_type": "semantic|relational|freshness", "reason": "<8 words"}}"""


def classify(state: AgentState) -> AgentState:
    start = time.time()
    data = _json_from(_chat(CLASSIFY_PROMPT.format(question=state["question"]),
                            max_tokens=200))
    query_type = data.get("query_type")
    if query_type not in ("semantic", "relational", "freshness"):
        query_type = "semantic"      # the safe default: it always has a retriever
    return {
        "query_type": query_type,
        "classification_reason": data.get("reason", ""),
        "trace": [_step("classify", start, query_type=query_type)],
    }


# --- 2. retrieval ------------------------------------------------------------

# The 300-question ablation found the lexical branch actively harmful on this
# corpus: BM25 alone reaches 0.303 Recall@5, and RRF parity with dense costs the
# full system 0.083 (0.733 -> 0.650) and 0.114 of top-1 Tier-1 rate. Clinical
# guidelines are written in standardised professional language, so the drug names
# and dosages BM25 was included for are already captured by the biomedical
# embeddings. Retrieval therefore runs dense + tier prior, not hybrid.
USE_LEXICAL_BRANCH = False
K_LEXICAL = 50 if USE_LEXICAL_BRANCH else 0


def retrieve_semantic(state: AgentState) -> AgentState:
    """Dense retrieval with the tier prior — the configuration the ablation selected."""
    start = time.time()
    question = state["question"]
    rows = store().search(question, embedder().embed_one(question), limit=8,
                          k_lexical=K_LEXICAL)
    return {
        "contexts": rows,
        "retrieval_branches": ["semantic"],
        "trace": [_step("retrieve_semantic", start, n=len(rows))],
    }


def retrieve_relational(state: AgentState) -> AgentState:
    """Graph branch: link entities, traverse the clinical KG, ground in chunks.

    The traversal supplies the *relations* (with their population and condition
    qualifiers) and the chunk_ids that evidence them; hybrid retrieval is still
    run alongside, because a question usually needs the surrounding prose as
    well as the extracted facts.

    Falls back to :NEXT-chain expansion when the question links to no entity —
    an honest degradation rather than an empty branch.
    """
    start = time.time()
    question = state["question"]

    from clinical_kg.retrieve import format_paths, format_relations, traverse

    result = traverse(store(), question)
    anchors = result["anchors"]

    # Hybrid retrieval runs regardless: the graph gives facts, the chunks give
    # the guidance prose those facts were extracted from.
    hybrid = store().search(question, embedder().embed_one(question), limit=5,
                            k_lexical=K_LEXICAL)

    contexts: list[dict] = list(hybrid)
    seen = {c["chunk_id"] for c in contexts}
    for row in result["contexts"]:
        if row["chunk_id"] not in seen:
            seen.add(row["chunk_id"])
            contexts.append(row)

    if not anchors:
        return {
            **_next_chain_expansion(question, hybrid, seen),
            "trace": [_step("retrieve_relational", start, anchors=0,
                            fallback="next-chain")],
        }

    # The extracted relations are handed to the generator as a compact evidence
    # block, qualifications included — a bare triple is not safe to act on.
    relation_block = format_relations(result["relations"])
    path_block = format_paths(result["paths"])
    if relation_block or path_block:
        contexts.insert(0, {
            "chunk_id": "kg:relations",
            "text": ("Extracted clinical relations (each with the population and "
                     "condition the guideline attaches):\n" + relation_block
                     + (("\n\nTwo-step connections:\n" + path_block) if path_block else "")),
            "section": "clinical knowledge graph",
            "title": "Knowledge graph relations",
            "source": "graph",
            "tier": min((r.get("tier") or 4) for r in result["relations"])
            if result["relations"] else 1,
            "page": None,
            "url": "",
            "score": 1.0,
            "branches": ["graph"],
        })

    return {
        "contexts": contexts,
        "retrieval_branches": [
            f"relational(KG: {len(anchors)} anchors, "
            f"{len(result['relations'])} relations, {len(result['paths'])} paths)"],
        "trace": [_step("retrieve_relational", start,
                        anchors=len(anchors),
                        relations=len(result["relations"]),
                        paths=len(result["paths"]),
                        contexts=len(contexts))],
    }


def _next_chain_expansion(question: str, seeds: list[dict],
                          seen: set[str]) -> dict:
    """Pre-KG behaviour, kept as the relational ablation baseline.

    Expands each hit along the :NEXT chain to recover the surrounding
    recommendation context. This is what the graph has to beat.
    """
    expanded: list[dict] = []
    with store().session() as session:
        for row in seeds[:4]:
            for record in session.run(
                """
                MATCH (c:Chunk {chunk_id: $cid})
                OPTIONAL MATCH (c)-[:NEXT]->(nxt:Chunk)
                OPTIONAL MATCH (prv:Chunk)-[:NEXT]->(c)
                WITH [x IN [nxt, prv] WHERE x IS NOT NULL] AS ns
                UNWIND ns AS n
                RETURN n.chunk_id AS chunk_id, n.text AS text, n.section AS section,
                       n.page AS page, n.tier AS tier, n.source AS source,
                       n.title AS title, n.url AS url,
                       n.publication_year AS year
                """, {"cid": row["chunk_id"]}):
                item = dict(record)
                if item["chunk_id"] in seen:
                    continue
                seen.add(item["chunk_id"])
                item["score"] = row["score"] * 0.5
                item["branches"] = ["next_expand"]
                expanded.append(item)

    return {
        "contexts": list(seeds) + expanded,
        "retrieval_branches": ["relational(next-chain expansion; no KG match)"],
    }


def retrieve_freshness(state: AgentState) -> AgentState:
    """Corpus first, then the web — the corpus still holds the authoritative text."""
    start = time.time()
    question = state["question"]
    rows = store().search(question, embedder().embed_one(question), limit=5,
                          k_lexical=K_LEXICAL)
    return {
        "contexts": rows,
        "retrieval_branches": ["freshness(corpus-first)"],
        "trace": [_step("retrieve_freshness", start, n=len(rows))],
    }


# --- 3. rerank ---------------------------------------------------------------

def rerank(state: AgentState) -> AgentState:
    """Deduplicate, then order by tier-weighted score.

    A cross-encoder would go here; the tier-aware ordering is what the paper's
    reranker ablation compares against.
    """
    start = time.time()
    contexts = state.get("contexts", [])

    unique: dict[str, dict] = {}
    for row in contexts:
        key = row.get("chunk_id") or row.get("url", "") + row.get("text", "")[:60]
        if key not in unique or row.get("score", 0) > unique[key].get("score", 0):
            unique[key] = row

    ordered = sorted(unique.values(), key=lambda r: r.get("score", 0), reverse=True)[:8]
    return {
        "ranked": ordered,
        "trace": [_step("rerank", start, kept=len(ordered), from_n=len(contexts))],
    }


def _evidence(state: AgentState) -> list[dict]:
    """Reranked passages when available, else whatever retrieval produced."""
    return state.get("ranked") or state.get("contexts", [])


# --- 4. sufficiency ----------------------------------------------------------

SUFFICIENCY_PROMPT = """A perioperative nurse asked:

{question}

Retrieved evidence:
{context}

Can this question be answered accurately and safely from this evidence alone?
Answer "no" if the evidence is off-topic, or if it discusses the area but omits
the specific dose, timing or threshold the question asks for.

Return JSON only: {{"sufficient": true|false, "reason": "<12 words"}}"""


def check_sufficiency(state: AgentState) -> AgentState:
    start = time.time()
    contexts = _evidence(state)
    if not contexts:
        return {"sufficient": False, "sufficiency_reason": "no evidence retrieved",
                "trace": [_step("sufficiency", start, sufficient=False)]}

    # Keep this prompt small: it runs on every question and was the slowest
    # node in the pipeline by a wide margin.
    joined = "\n\n".join(
        f"[{i}] ({c.get('source','?')}, tier {c.get('tier','?')}) {c.get('text','')[:450]}"
        for i, c in enumerate(contexts[:5], 1))
    data = _json_from(_chat(SUFFICIENCY_PROMPT.format(
        question=state["question"], context=joined), max_tokens=300))

    sufficient = bool(data.get("sufficient", True))
    return {
        "sufficient": sufficient,
        "sufficiency_reason": data.get("reason", ""),
        "trace": [_step("sufficiency", start, sufficient=sufficient)],
    }


# --- 5. web escalation -------------------------------------------------------

def web_search(state: AgentState) -> AgentState:
    """Escalate to the web when corpus evidence is insufficient.

    Web results carry no provenance tier by construction, so they enter as
    tier 0 — explicitly *below* the curated corpus in the ranking and flagged in
    the answer. Letting unvetted pages inherit a tier would launder them into a
    system whose whole argument is provenance weighting.
    """
    start = time.time()
    escalations = state.get("escalation_count", 0)

    if not TAVILY_API_KEY:
        return {"escalated": False,
                "trace": [_step("web_search", start, skipped="no Tavily key")]}
    if escalations >= 1:
        return {"escalated": False,
                "trace": [_step("web_search", start, skipped="escalation cap reached")]}

    try:
        from tavily import TavilyClient

        results = TavilyClient(api_key=TAVILY_API_KEY).search(
            query=state["question"],
            max_results=4,
            search_depth="advanced",
            include_domains=["nice.org.uk", "rcog.org.uk", "acog.org", "who.int",
                             "cdc.gov", "ahrq.gov", "pubmed.ncbi.nlm.nih.gov",
                             "ncbi.nlm.nih.gov", "erassociety.org", "augs.org"],
        ).get("results", [])
    except Exception as exc:
        return {"escalated": False,
                "trace": [_step("web_search", start, error=str(exc)[:120])]}

    contexts = [{
        "chunk_id": f"web:{i}",
        "text": (row.get("content") or "")[:1500],
        "section": row.get("title", ""),
        "title": row.get("title", ""),
        "url": row.get("url", ""),
        "source": "web",
        "tier": 0,                     # unverified, retrieved live
        "trust_level": "Tier 0 — unverified, retrieved live",
        "page": None,
        "score": 0.01,
        "branches": ["web"],
    } for i, row in enumerate(results, 1)]

    return {
        "contexts": contexts,
        "retrieval_branches": ["web"],
        "escalated": True,
        "escalation_count": escalations + 1,
        "trace": [_step("web_search", start, n=len(contexts))],
    }


# --- 6. generation -----------------------------------------------------------

ANSWER_PROMPT = """You are assisting a perioperative gynecologic nurse during their shift.

Answer using ONLY the numbered evidence below. Rules:
- Cite every clinical claim with the passage number, like [2].
- Give specific doses, timings and thresholds when the evidence states them.
- If the evidence does not answer the question, say so plainly.
- Passages marked "tier 0" come from a live web search and are NOT from the
  vetted corpus; if you use one, say so in the answer.
- You are supporting a nurse's judgement, not replacing it. Do not prescribe.

Question: {question}

Evidence:
{context}

Answer:"""


def generate(state: AgentState) -> AgentState:
    start = time.time()
    contexts = _evidence(state)
    if not contexts:
        return {
            "answer": "I could not find evidence in the guideline corpus to answer this "
                      "question. Escalate to the surgical or nursing team.",
            "citations": [],
            "trace": [_step("generate", start, empty=True)],
        }

    joined = "\n\n".join(
        f"[{i}] ({c.get('source','?')}, tier {c.get('tier','?')}"
        f"{', ' + str(c.get('publication_year') or c.get('year')) if c.get('publication_year') or c.get('year') else ''}) "
        f"{c.get('title','')} — {c.get('section','')}\n{c.get('text','')[:1200]}"
        for i, c in enumerate(contexts, 1))

    answer = _chat(ANSWER_PROMPT.format(question=state["question"], context=joined))

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)
                    if 0 < int(n) <= len(contexts)})
    citations = [{
        "marker": n,
        "chunk_id": contexts[n - 1].get("chunk_id"),
        "source": contexts[n - 1].get("source"),
        "tier": contexts[n - 1].get("tier"),
        "title": contexts[n - 1].get("title"),
        "section": contexts[n - 1].get("section"),
        "page": contexts[n - 1].get("page"),
        "url": contexts[n - 1].get("url"),
    } for n in cited]

    return {
        "answer": answer,
        "citations": citations,
        "trace": [_step("generate", start, citations=len(citations))],
    }


# --- 7. citation verification ------------------------------------------------

VERIFY_PROMPT = """Check whether each clinical claim in the answer is supported by
the passage it cites.

Question: {question}

Evidence:
{context}

Answer:
{answer}

List only claims that are NOT supported by their cited passage, or that carry a
clinical specific (dose, timing, threshold) absent from the evidence.

Return JSON only:
{{"unsupported": ["<claim>", ...], "verdict": "pass|fail"}}"""


def verify_citations(state: AgentState) -> AgentState:
    """The node RAGAS faithfulness should move — see the pilot results."""
    start = time.time()
    contexts = _evidence(state)
    answer = state.get("answer", "")
    if not contexts or not answer:
        return {"verification_passed": True, "unsupported_claims": [],
                "trace": [_step("verify", start, skipped=True)]}

    joined = "\n\n".join(f"[{i}] {c.get('text','')[:900]}"
                         for i, c in enumerate(contexts, 1))
    data = _json_from(_chat(VERIFY_PROMPT.format(
        question=state["question"], context=joined, answer=answer), max_tokens=800))

    unsupported = [c for c in (data.get("unsupported") or []) if isinstance(c, str)]
    passed = data.get("verdict", "pass") == "pass" and not unsupported

    return {
        "unsupported_claims": unsupported,
        "verification_passed": passed,
        "trace": [_step("verify", start, unsupported=len(unsupported))],
    }


# --- 8. clinical safety ------------------------------------------------------

SAFETY_PROMPT = """Review this answer for a perioperative nurse.

Question: {question}
Answer: {answer}

Flag ONLY these problems:
- prescribing: tells the nurse to start/stop/change a medication on their own authority
- diagnosis: makes a diagnostic determination that is the clinician's to make
- missing_escalation: describes a deteriorating or unstable situation without
  saying to escalate to the surgical/medical team
- unsupported_specific: states a dose, rate or threshold with no citation
- out_of_scope: the question is not perioperative nursing practice

Return JSON only: {{"flags": ["..."], "action": "allow|annotate|refuse", "note": "<15 words"}}"""

_ESCALATION_NOTE = (
    "\n\n---\n**Clinical safety note:** This is decision support, not a directive. "
    "Confirm against local policy and escalate to the surgical or medical team for "
    "anything outside routine practice."
)


def safety_check(state: AgentState) -> AgentState:
    start = time.time()
    answer = state.get("answer", "")
    if not answer:
        return {"safety_action": "allow", "trace": [_step("safety", start, skipped=True)]}

    data = _json_from(_chat(SAFETY_PROMPT.format(
        question=state["question"], answer=answer), max_tokens=300))
    flags = [f for f in (data.get("flags") or []) if isinstance(f, str)]
    action = data.get("action", "allow")

    updated = answer
    if action == "refuse":
        updated = (
            "I can't answer this one safely from the guideline corpus — "
            f"{data.get('note', 'it falls outside perioperative nursing decision support')}. "
            "Please escalate to the surgical or medical team."
        )
    elif flags or state.get("unsupported_claims"):
        updated = answer + _ESCALATION_NOTE

    return {
        "answer": updated,
        "safety_flags": flags,
        "safety_action": action,
        "trace": [_step("safety", start, action=action, flags=len(flags))],
    }
