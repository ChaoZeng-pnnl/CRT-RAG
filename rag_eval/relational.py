"""Multi-hop question set for the relational pilot.

BIAS CONTROL — the point of this file.

Questions must not be generated from the knowledge graph. If they were, the KG
branch would win by construction: every question would be a verbalised traversal
of a path the graph already contains, and the pilot would measure nothing.

Instead, pairs of chunks are found by *embedding similarity across different
documents* — a signal available to every condition and derived from none of
them. A question is then written that needs BOTH chunks, and kept only if a
verifier confirms neither chunk answers it alone.

The judge-free metric is `both_recall`: did the retriever return both source
chunks? That is what multi-hop retrieval is for, and it needs no LLM.
"""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "chunks" / "chunks.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "eval"

# Related but not redundant: near-duplicates give a question one chunk answers,
# unrelated pairs give an incoherent question.
SIM_LOW, SIM_HIGH = 0.62, 0.86

PAIR_PROMPT = """Two passages from different clinical guidelines are below.

PASSAGE A ({source_a}, {title_a})
\"\"\"
{text_a}
\"\"\"

PASSAGE B ({source_b}, {title_b})
\"\"\"
{text_b}
\"\"\"

Write ONE question a perioperative gynecologic nurse would realistically ask that
requires information from BOTH passages to answer correctly. The question must be
unanswerable from either passage alone — it should force the two to be combined
(a patient factor from one plus a management step from the other; a risk from one
plus its prevention from the other).

If the two passages cannot be combined into a single realistic clinical question,
reply with exactly: SKIP

Never mention "passage", "guideline" or "document" in the question.

Return JSON only:
{{"question": "...", "answer": "<grounded in both passages>",
  "why_both": "<8 words on what each passage contributes>"}}"""

VERIFY_PROMPT = """Question: {question}

Passage:
\"\"\"
{text}
\"\"\"

Can this question be answered COMPLETELY and safely from this passage alone?
Return JSON only: {{"complete": true|false}}"""

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def _chat(prompt: str, max_tokens: int = 1200) -> str:
    response = client().chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=max_tokens)
    return (response.choices[0].message.content or "").strip()


def _json_from(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def kg_covered_chunk_ids() -> set[str]:
    """Chunks the knowledge graph actually extracted relations from."""
    triples_file = Path(__file__).resolve().parent.parent / "kg" / "triples.jsonl"
    if not triples_file.exists():
        return set()
    with triples_file.open(encoding="utf-8") as handle:
        return {json.loads(line)["chunk_id"] for line in handle}


def find_pairs(n_candidates: int = 400, max_pairs: int = 60,
               seed: int = 20260905,
               restrict_to: set[str] | None = None) -> list[tuple[dict, dict, float]]:
    """Cross-document Tier 1-2 chunk pairs, related but not redundant.

    `restrict_to` limits the candidate pool to specific chunk ids — used to draw
    questions from the chunks the KG actually covers. Note what this does and
    does not control for: the questions are still built from chunk *text* by
    embedding similarity, never from triples, so the KG gets no privileged role
    in constructing them. But it is a best-case test — it answers "does the graph
    help where it has coverage?", not "does it help in deployment?".
    """
    from corpus_index.embedder import Embedder

    rows = []
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if restrict_to is not None and row["chunk_id"] not in restrict_to:
                continue
            if (row.get("tier") in (1, 2) and row.get("is_canonical") is not False
                    and 100 <= row.get("est_tokens", 0) <= 500
                    and row.get("kind") != "table"):
                rows.append(row)

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_candidates]
    print(f"  Embedding {len(rows)} candidate chunks ...")
    vectors = Embedder().embed([r["text"] for r in rows], progress=False)

    pairs = []
    for i in range(len(rows)):
        if not vectors[i]:
            continue
        for j in range(i + 1, len(rows)):
            if not vectors[j] or rows[i]["doc_id"] == rows[j]["doc_id"]:
                continue
            score = _cosine(vectors[i], vectors[j])
            if SIM_LOW <= score <= SIM_HIGH:
                pairs.append((rows[i], rows[j], score))

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def _build_one(pair) -> dict | None:
    a, b, score = pair
    data = _json_from(_chat(PAIR_PROMPT.format(
        source_a=a.get("source", ""), title_a=(a.get("title") or "")[:70],
        text_a=a["text"][:2200],
        source_b=b.get("source", ""), title_b=(b.get("title") or "")[:70],
        text_b=b["text"][:2200])))
    question = (data.get("question") or "").strip()
    if not question or question.upper().startswith("SKIP") or len(question) < 20:
        return None

    # Keep only genuinely multi-hop questions: discard if either chunk alone
    # answers it completely.
    for chunk in (a, b):
        verdict = _json_from(_chat(VERIFY_PROMPT.format(
            question=question, text=chunk["text"][:2500]), max_tokens=200))
        if verdict.get("complete") is True:
            return None

    return {
        "question": question,
        "reference": (data.get("answer") or "").strip(),
        "why_both": data.get("why_both", ""),
        "source_chunk_ids": [a["chunk_id"], b["chunk_id"]],
        "source_docs": [a.get("title", ""), b.get("title", "")],
        "source_names": [a.get("source", ""), b.get("source", "")],
        "pair_similarity": round(score, 4),
    }


def generate(n: int = 30, concurrency: int = 6, in_domain: bool = False,
             out_name: str = "relational_testset.jsonl") -> list[dict]:
    restrict = kg_covered_chunk_ids() if in_domain else None
    if in_domain:
        print(f"  Restricting to {len(restrict)} KG-covered chunks (in-domain test)")
    pairs = find_pairs(restrict_to=restrict)
    print(f"  Candidate cross-document pairs: {len(pairs)}")

    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for result in pool.map(_build_one, pairs):
            if result:
                items.append(result)
                print(f"    kept {len(items)}/{n}", end="\r", flush=True)
            if len(items) >= n:
                break
    print()

    out = OUT_DIR / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Kept {len(items)} multi-hop questions (rejected pairs were "
          f"single-chunk answerable or incoherent)")
    print(f"  Wrote {out}")
    return items
