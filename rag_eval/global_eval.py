"""Global (corpus-level) evaluation — the query class GraphRAG actually claims.

Edge et al. (2024), "From Local to Global", argue that graph community summaries
help with *query-focused summarisation* — questions whose answer exists in no
single chunk — and concede that vector RAG is competitive on local lookup. Every
result so far in this project has been local, so the negative findings are scoped
to local retrieval and say nothing about the paper's actual claim.

BIAS CONTROL. Questions are generated from `corpus_metadata.json` — the topic
axes and the guideline bodies — and never from community titles, entities or
summaries. Seeding questions from the graph would let GraphRAG set its own exam:
every question would name a theme the partition had already found. Corpus
metadata is available to all three conditions equally.

Conditions:
  graphrag_global  map-reduce over community summaries (the claimed method)
  vector_large_k   hybrid retrieval at large k, concatenated
  long_context     top-N chunks stuffed into one long prompt

Metrics follow Edge et al.: pairwise LLM preference on comprehensiveness,
diversity and empowerment. Chunk recall is meaningless here — there is no single
gold chunk. Judgements are POSITION-RANDOMISED: pairwise LLM judging has a
documented position bias, and without randomisation a win rate can be produced by
presentation order alone.
"""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import (
    CHAT_MODEL,
    FALLBACK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    sampling_kwargs,
    setting,
)

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
COMMUNITIES = ROOT / "kg" / "communities.jsonl"
METADATA = ROOT / "corpus" / "metadata" / "corpus_metadata.json"

JUDGE_MODEL = setting("OPENAI_HIGHCOST_MODEL") or FALLBACK_MODEL

QUESTION_PROMPT = """You are writing evaluation questions for a system that answers
questions over a corpus of perioperative gynecologic nursing guidelines.

The corpus contains guidance from these bodies: {sources}
covering these clinical topics: {topics}

Write ONE question about the topic "{topic}" that requires synthesising across
MANY documents — a corpus-level question, not a lookup. It must be the kind of
question a nurse educator, quality-improvement lead or guideline author asks:
about themes, patterns, points of agreement or disagreement between bodies, or
the overall shape of the evidence.

It must NOT be answerable by finding one passage. Avoid asking for a single dose,
timing or threshold.

Good: "Across these guidelines, what different mechanisms are recommended for
preventing surgical site infection, and where do the bodies emphasise different
things?"
Bad: "What antibiotic should be given before hysterectomy?"

Return JSON only: {{"question": "...", "why_global": "<10 words>"}}"""

ANSWER_PROMPT = """Answer this corpus-level question about perioperative gynecologic
nursing guidelines, using only the material provided.

Question: {question}

Material:
{context}

Write a thorough synthesis. Name the guideline bodies and their positions, note
agreements and differences, and say what the material does not establish.
"""

MAP_PROMPT = """Question: {question}

Community summary from a clinical knowledge graph:
{summary}

Does this community contain anything relevant to the question? If yes, extract the
relevant points with their guideline sources. If not, reply exactly: NONE

Keep it under 120 words."""

JUDGE_PROMPT = """Two systems answered the same corpus-level question about
perioperative gynecologic nursing guidelines. Judge which answer is better.

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Judge on:
- comprehensiveness: covers the breadth the question asks for
- diversity: presents different perspectives, bodies and mechanisms
- empowerment: helps the reader understand and make judgements, with specifics
  (populations, conditions, sources) rather than generalities

Return JSON only:
{{"comprehensiveness": "A|B|tie", "diversity": "A|B|tie",
  "empowerment": "A|B|tie", "overall": "A|B|tie", "reason": "<20 words>"}}"""

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def _chat(prompt: str, model: str = CHAT_MODEL, max_tokens: int = 4000) -> str:
    try:
        response = client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, **sampling_kwargs(model))
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"__ERROR__ {type(exc).__name__}: {str(exc)[:100]}"


def _json_from(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# --- question generation (from corpus metadata only) -------------------------

def generate_questions(n: int = 20, seed: int = 20260905) -> list[dict]:
    records = json.loads(METADATA.read_text())
    sources = sorted({r["source"] for r in records if r.get("tier") != 4
                      and r.get("source")})
    if not sources:
        sources = sorted({r["source"] for r in records})[:8]
    topics = sorted({r["topic"].split("(")[0].strip() for r in records
                     if r.get("topic")})

    rng = random.Random(seed)
    chosen = [topics[i % len(topics)] for i in range(n)]
    rng.shuffle(chosen)

    def one(topic: str) -> dict | None:
        data = _json_from(_chat(QUESTION_PROMPT.format(
            sources=", ".join(sources), topics="; ".join(topics[:20]),
            topic=topic)))
        question = (data.get("question") or "").strip()
        if len(question) < 25:
            return None
        return {"question": question, "topic": topic,
                "why_global": data.get("why_global", "")}

    items = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(one, chosen):
            if result:
                items.append(result)

    out = EVAL_DIR / "global_testset.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {out} ({len(items)} global questions)")
    return items


# --- conditions ---------------------------------------------------------------

def graphrag_global(question: str, communities: list[dict],
                    top_communities: int = 40, concurrency: int = 10) -> str:
    """Map-reduce over community summaries.

    Communities are pre-filtered by embedding similarity to the question: mapping
    over all 486 costs 486 LLM calls per question, which is not a fair
    representation of the method's cost either.
    """
    from corpus_index.embedder import Embedder

    embedder = Embedder()
    qv = embedder.embed_one(question)
    texts = [f"{c['title']} {' '.join(c.get('key_entities') or [])}" for c in communities]
    cvs = embedder.embed(texts, progress=False)

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    ranked = sorted(zip(communities, cvs), key=lambda p: -cosine(qv, p[1]))
    selected = [c for c, _ in ranked[:top_communities]]

    # MAP
    def map_one(community: dict) -> str | None:
        result = _chat(MAP_PROMPT.format(
            question=question, summary=community["summary"][:2500]), max_tokens=400)
        if result.startswith("__ERROR__") or result.strip().upper().startswith("NONE"):
            return None
        return f"[{community['title']} — {', '.join(community['sources'])}]\n{result}"

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        mapped = [m for m in pool.map(map_one, selected) if m]

    if not mapped:
        return "No relevant material was found in the knowledge graph communities."

    # REDUCE
    return _chat(ANSWER_PROMPT.format(
        question=question, context="\n\n".join(mapped)[:60000]), max_tokens=4000)


def vector_large_k(question: str, store, embedder, k: int = 40) -> str:
    rows = store.search(question, embedder.embed_one(question), limit=k)
    context = "\n\n".join(
        f"[{r['source']} T{r['tier']}] {r.get('title','')} — {r.get('section','')}\n"
        f"{r['text'][:1200]}" for r in rows)
    return _chat(ANSWER_PROMPT.format(question=question, context=context[:60000]),
                 max_tokens=4000)


def long_context(question: str, store, embedder, k: int = 80) -> str:
    rows = store.search(question, embedder.embed_one(question), limit=k)
    context = "\n\n".join(f"[{r['source']}] {r['text'][:900]}" for r in rows)
    return _chat(ANSWER_PROMPT.format(question=question, context=context[:120000]),
                 max_tokens=4000)


# --- pairwise judging ----------------------------------------------------------

def judge_pair(question: str, name_a: str, answer_a: str,
               name_b: str, answer_b: str, rng: random.Random,
               model: str = JUDGE_MODEL) -> dict:
    """Position-randomised pairwise judgement."""
    flip = rng.random() < 0.5
    first, second = ((name_b, answer_b), (name_a, answer_a)) if flip else \
                    ((name_a, answer_a), (name_b, answer_b))

    data = _json_from(_chat(JUDGE_PROMPT.format(
        question=question, answer_a=first[1][:9000], answer_b=second[1][:9000]),
        model=model, max_tokens=600))

    def resolve(value: str) -> str:
        value = (value or "tie").strip().upper()
        if value not in ("A", "B"):
            return "tie"
        winner = first[0] if value == "A" else second[0]
        return winner

    return {
        "comprehensiveness": resolve(data.get("comprehensiveness")),
        "diversity": resolve(data.get("diversity")),
        "empowerment": resolve(data.get("empowerment")),
        "overall": resolve(data.get("overall")),
        "reason": data.get("reason", ""),
        "position_flipped": flip,
    }
