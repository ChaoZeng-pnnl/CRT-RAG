"""Coverage-controlled global evaluation.

WHY THIS REPLACES THE FIRST GLOBAL RUN
--------------------------------------
`global_eval.py` compared map-reduce over communities built from 36 documents
against `store.search(...)` over the full 237-document index. The graph held ~15%
of the corpus; the baselines held all of it. The resulting 5-5 tie measures the
coverage gap, not the method — the same confound that invalidated the first
relational pilot, repeated in the opposite direction.

Here every condition sees exactly the documents the knowledge graph was built
from. The allowed set is derived from the community file itself, so it cannot
drift from whatever graph is actually being evaluated.

A second correction: the original `graphrag_global` pre-filtered to the 40
communities most similar to the question, which turns map-reduce-over-everything
into large-k retrieval over summaries — a paraphrase of the vector baseline. Map
scope is now an explicit condition:

    graphrag_prefiltered   top-40 communities by embedding similarity (as before)
    graphrag_mapall        every community (the method as published)
    vector_large_k         hybrid retrieval, restricted to the same documents
    long_context           top-N chunks stuffed, restricted to the same documents

Cost is recorded per condition (LLM calls, wall-clock), because a quality tie at
40x the calls is a result about cost, and cost is the only number that separated
the conditions last time.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .global_eval import (
    ANSWER_PROMPT,
    EVAL_DIR,
    MAP_PROMPT,
    _chat,
)

ROOT = Path(__file__).resolve().parent.parent
KG_DIR = ROOT / "kg"

# Mapping over every community means thousands of requests per run. The first
# attempt issued ~29/second and the shared proxy answered 403; `map_one` treats a
# failed map as "this community is irrelevant", so the condition degraded to "the
# graph found nothing" on 12 of 20 questions and reported it as an answer. Rate
# limiting must therefore be retried, and residual failure must be counted and
# surfaced rather than absorbed.
MAP_RETRIES = 4
RETRY_BACKOFF = 3.0          # seconds, doubled each attempt
TRANSIENT = ("403", "429", "forbidden", "rate", "timeout", "overloaded",
             "unavailable", "connection")
# Above this share of failed map calls a question's answer is not a measurement.
MAX_MAP_FAILURE = 0.10


def _chat_retry(prompt: str, max_tokens: int, model: str | None = None,
                retries: int = MAP_RETRIES) -> str:
    """`_chat` with backoff on transient proxy failures."""
    kwargs = {"model": model} if model else {}
    delay = RETRY_BACKOFF
    result = ""
    for attempt in range(retries):
        result = _chat(prompt, max_tokens=max_tokens, **kwargs)
        if not result.startswith("__ERROR__"):
            return result
        if not any(t in result.lower() for t in TRANSIENT):
            return result           # a real error; retrying will not help
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
    return result


# ----------------------------------------------------------------- the KG scope

def load_communities(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def allowed_documents(communities: list[dict]) -> set[str]:
    """The document set the graph actually covers — the ceiling for every arm."""
    return {cid.split(":")[0]
            for c in communities for cid in c.get("chunk_ids", [])}


# -------------------------------------------------------------------- conditions

def _restricted_rows(store, embedder, question: str, k: int,
                     allowed: set[str]) -> list[dict]:
    """Hybrid retrieval confined to the graph's documents.

    The candidate pool is widened because the allowed set is a small share of the
    index: with default k_dense/k_lexical of 50, filtering to ~15% of documents
    would leave far fewer than k chunks and quietly handicap the baseline in the
    other direction.
    """
    return store.search(question, embedder.embed_one(question), limit=k,
                        k_dense=800, k_lexical=800, allowed_docs=allowed)


def vector_large_k(question: str, store, embedder, allowed: set[str],
                   k: int = 40) -> tuple[str, int, dict]:
    rows = _restricted_rows(store, embedder, question, k, allowed)
    context = "\n\n".join(
        f"[{r['source']} T{r['tier']}] {r.get('title','')} — {r.get('section','')}\n"
        f"{r['text'][:1200]}" for r in rows)
    answer = _chat_retry(ANSWER_PROMPT.format(question=question,
                                              context=context[:60000]),
                         max_tokens=4000)
    return answer, 1, {"chunks": len(rows)}


def long_context(question: str, store, embedder, allowed: set[str],
                 k: int = 80) -> tuple[str, int, dict]:
    rows = _restricted_rows(store, embedder, question, k, allowed)
    context = "\n\n".join(f"[{r['source']}] {r['text'][:900]}" for r in rows)
    answer = _chat_retry(ANSWER_PROMPT.format(question=question,
                                              context=context[:120000]),
                         max_tokens=4000)
    return answer, 1, {"chunks": len(rows)}


def graphrag(question: str, communities: list[dict], embedder,
             top_communities: int | None, concurrency: int = 4,
             map_model: str | None = None, reduce_model: str | None = None,
             map_max_tokens: int = 400) -> tuple[str, int, dict]:
    """Map-reduce over community summaries.

    `top_communities=None` maps over every community — the method as published.
    An integer restores the similarity pre-filter, kept as a condition so the
    cost/quality trade-off of the shortcut is measurable rather than assumed.
    """
    selected = communities
    if top_communities is not None:
        question_vector = embedder.embed_one(question)
        texts = [f"{c['title']} {' '.join(c.get('key_entities') or [])}"
                 for c in communities]
        vectors = embedder.embed(texts, progress=False)

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        selected = [c for c, _ in sorted(zip(communities, vectors),
                                         key=lambda p: -cosine(question_vector, p[1]))
                    ][:top_communities]

    def map_one(community: dict) -> tuple[str, str | None]:
        """Returns (outcome, text). Outcome distinguishes a community that had
        nothing to say from one we failed to ask — collapsing the two is what
        made an infrastructure outage look like a negative result."""
        result = _chat_retry(
            MAP_PROMPT.format(question=question,
                              summary=community["summary"][:2500]),
            max_tokens=map_max_tokens, model=map_model)
        if result.startswith("__ERROR__"):
            return "failed", None
        if result.strip().upper().startswith("NONE"):
            return "irrelevant", None
        return "mapped", (f"[{community['title']} — "
                          f"{', '.join(community['sources'])}]\n{result}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(map_one, selected))

    mapped = [text for outcome, text in outcomes if outcome == "mapped"]
    failed = sum(1 for outcome, _ in outcomes if outcome == "failed")
    calls = len(selected) + 1
    stats = {"communities_mapped": len(selected), "relevant": len(mapped),
             "failed": failed,
             "failure_rate": round(failed / len(selected), 3) if selected else 0.0}

    if stats["failure_rate"] > MAX_MAP_FAILURE:
        return (f"__INVALID__ {failed}/{len(selected)} map calls failed; "
                f"this answer is not a measurement"), calls, stats
    if not mapped:
        return ("No relevant material was found in the knowledge graph "
                "communities."), calls, stats

    answer = _chat_retry(ANSWER_PROMPT.format(
        question=question, context="\n\n".join(mapped)[:60000]), max_tokens=4000,
        model=reduce_model)
    if answer.startswith("__ERROR__"):
        answer = f"__INVALID__ reduce step failed: {answer[:120]}"
    return answer, calls, stats


# ------------------------------------------------------------------------- run

def run(community_file: str = "communities.jsonl",
        questions_file: str = "global_testset.jsonl",
        out_file: str = "global_answers_controlled.jsonl",
        map_model: str | None = None,
        prefilter: int = 40,
        concurrency: int = 4) -> dict:
    from corpus_index.embedder import Embedder
    from corpus_index.graph import GraphStore

    communities = load_communities(KG_DIR / community_file)
    allowed = allowed_documents(communities)
    questions = [json.loads(line) for line in (EVAL_DIR / questions_file).open()]

    print(f"  communities: {len(communities)} from {len(allowed)} documents")
    print(f"  questions:   {len(questions)}")
    print(f"  COVERAGE CONTROL: every condition restricted to those "
          f"{len(allowed)} documents")

    embedder = Embedder()
    store = GraphStore()
    rows, cost = [], {}
    try:
        # Sanity check: the restriction must actually bite, or the control is a
        # no-op and the whole comparison is invalid again.
        probe = _restricted_rows(store, embedder, questions[0]["question"], 40, allowed)
        outside = [r for r in probe if r["chunk_id"].split(":")[0] not in allowed]
        print(f"  restriction check: {len(probe)} chunks retrieved, "
              f"{len(outside)} outside the allowed set (must be 0)")
        if outside or len(probe) < 20:
            raise SystemExit("  ABORT: restriction failed or pool too small — "
                             "raise k_dense/k_lexical before trusting any result.")

        conditions = {
            "graphrag_mapall":
                lambda q: graphrag(q, communities, embedder, None, map_model=map_model),
            "graphrag_prefiltered":
                lambda q: graphrag(q, communities, embedder, prefilter, map_model=map_model),
            "vector_large_k":
                lambda q: vector_large_k(q, store, embedder, allowed),
            "long_context":
                lambda q: long_context(q, store, embedder, allowed),
        }

        for name, fn in conditions.items():
            started = time.time()
            calls = 0
            answers: dict[int, str] = {}

            def one(item):
                i, q = item
                return i, fn(q["question"])

            per_question: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for i, (answer, n, stats) in pool.map(one, list(enumerate(questions))):
                    answers[i] = answer
                    per_question[i] = stats
                    calls += n
                    print(f"    {name}: {len(answers)}/{len(questions)}",
                          end="\r", flush=True)
            print()

            invalid = sum(1 for a in answers.values() if a.startswith("__INVALID__"))
            empty = sum(1 for a in answers.values()
                        if a.startswith("No relevant material"))
            cost[name] = {"llm_calls": calls,
                          "seconds": round(time.time() - started, 1),
                          "calls_per_question": round(calls / len(questions), 1),
                          "invalid_answers": invalid,
                          "empty_answers": empty,
                          "usable": len(questions) - invalid}
            print(f"    {name}: {calls} calls, {cost[name]['seconds']}s, "
                  f"{invalid} invalid, {empty} empty")
            if invalid:
                print(f"    !! {name} produced {invalid}/{len(questions)} invalid "
                      f"answers — this condition CANNOT be judged as it stands")
            for i, q in enumerate(questions):
                if i >= len(rows):
                    rows.append({"question": q["question"], "topic": q.get("topic", "")})
                rows[i][name] = answers[i]
                rows[i][f"{name}__stats"] = per_question.get(i, {})
    finally:
        store.close()

    out = EVAL_DIR / out_file
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline = cost.get("vector_large_k", {}).get("llm_calls") or 1
    for name, values in cost.items():
        values["cost_ratio_vs_vector"] = round(values["llm_calls"] / baseline, 1)

    summary = {"communities": len(communities), "allowed_documents": len(allowed),
               "questions": len(questions), "coverage_controlled": True,
               "map_model": map_model, "prefilter": prefilter, "cost": cost}
    (EVAL_DIR / "global_controlled_cost.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {out}")
    print("  cost ratio vs vector_large_k: "
          + ", ".join(f"{k}={v['cost_ratio_vs_vector']}x" for k, v in cost.items()))
    return summary
