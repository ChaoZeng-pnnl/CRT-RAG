"""RAGAS evaluation of the retrieval conditions, plus judge-free metrics.

Two families of number are reported side by side, deliberately:

  * RAGAS metrics — faithfulness, answer relevancy, context precision/recall.
    LLM-judged, and the thing the paper leans on.
  * Judge-free metrics — Recall@k against the chunk each question was generated
    from, and the tier distribution of retrieved contexts. These need no LLM and
    are immune to the "you evaluated an LLM with an LLM" objection, so a
    sceptical reviewer can still read a result off the table.

The judge is deliberately a different model family from the generator, to blunt
self-preference bias.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from corpus_index.config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    FALLBACK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def load_testset(path: Path | None = None) -> list[dict]:
    path = path or (EVAL_DIR / "pilot_testset.jsonl")
    # Rows marked `excluded` stay in the file so every qid (= line index)
    # keeps its meaning; they are skipped here instead. See the row's
    # `excluded` field for why.
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [r for r in rows if not r.get("excluded")]


# --- judge-free metrics ------------------------------------------------------

def retrieval_metrics(records: list[dict]) -> dict:
    """Recall@k and tier statistics — computed from ids, no LLM involved."""
    hits, reciprocal_ranks = [], []
    tiers: Counter = Counter()
    top1_tiers: Counter = Counter()

    for record in records:
        gold = record["source_chunk_id"]
        retrieved = [m.get("chunk_id") for m in record.get("meta", [])]
        hit = gold in retrieved
        hits.append(1.0 if hit else 0.0)
        reciprocal_ranks.append(1.0 / (retrieved.index(gold) + 1) if hit else 0.0)
        for m in record.get("meta", []):
            tiers[m.get("tier")] += 1
        if record.get("meta"):
            top1_tiers[record["meta"][0].get("tier")] += 1

    n = len(records) or 1
    return {
        "recall_at_k": sum(hits) / n,
        "mrr": sum(reciprocal_ranks) / n,
        "retrieved_tier_distribution": {str(k): v for k, v in sorted(tiers.items(),
                                                                     key=lambda kv: (kv[0] is None, kv[0]))},
        "top1_tier_distribution": {str(k): v for k, v in sorted(top1_tiers.items(),
                                                                key=lambda kv: (kv[0] is None, kv[0]))},
        "mean_contexts": statistics.mean(len(r.get("contexts", [])) for r in records) if records else 0,
    }


# --- RAGAS -------------------------------------------------------------------

class _ProxyEmbeddings:
    """RAGAS embeddings backed by our own client.

    LangChain's OpenAIEmbeddings sends `encoding_format="base64"`, which the
    LiteLLM/Vertex proxy rejects with a 400 — that silently killed every
    answer-relevancy score in the first pilot run. Reusing `corpus_index.Embedder`
    keeps the float encoding and the on-disk cache, so scoring costs nothing on a
    re-run.
    """

    def __init__(self):
        from corpus_index.embedder import Embedder

        self._embedder = Embedder()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_one(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts, progress=False)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _ragas_components(judge_model: str):
    """Wrap the PNNL endpoint for RAGAS."""
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,
        temperature=0, max_retries=5, timeout=180,
    ))
    embeddings = LangchainEmbeddingsWrapper(_ProxyEmbeddings())
    return llm, embeddings


def ragas_scores(records: list[dict], judge_model: str = FALLBACK_MODEL) -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    llm, embeddings = _ragas_components(judge_model)

    dataset = EvaluationDataset.from_list([{
        "user_input": r["question"],
        "retrieved_contexts": r["contexts"],
        "response": r["answer"],
        "reference": r["reference"],
    } for r in records])

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics, llm=llm,
                      embeddings=embeddings, show_progress=True)
    frame = result.to_pandas()

    scores = {}
    for column in frame.columns:
        if column in ("user_input", "retrieved_contexts", "response", "reference"):
            continue
        series = frame[column].dropna()
        if len(series):
            scores[column] = {
                "mean": float(series.mean()),
                "sd": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "n": int(len(series)),
            }
    return {"judge_model": judge_model, "metrics": scores}


def separation_check(by_pipeline: dict[str, dict]) -> dict:
    """Do the metrics actually distinguish the conditions?

    A pilot whose metrics all sit at 0.95+ for every condition cannot support an
    ablation study, and that has to be discovered now rather than after the full
    grid has been paid for.
    """
    findings = {}
    names = list(by_pipeline)
    all_metrics: set[str] = set()
    for result in by_pipeline.values():
        all_metrics |= set(result.get("ragas", {}).get("metrics", {}))

    for metric in sorted(all_metrics):
        values = {name: by_pipeline[name]["ragas"]["metrics"].get(metric, {}).get("mean")
                  for name in names}
        present = {k: v for k, v in values.items() if v is not None}
        if len(present) < 2:
            continue
        spread = max(present.values()) - min(present.values())
        ceiling = min(present.values()) > 0.95
        findings[metric] = {
            "values": present,
            "spread": spread,
            "saturated": ceiling,
            "separates": spread >= 0.03 and not ceiling,
        }
    return findings
