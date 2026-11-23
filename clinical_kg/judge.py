"""LLM adjudication of extracted triples using the high-cost model.

This is a *screen*, not a substitute for clinician validation. The extractor and
the judge are both LLMs, so agreement partly measures shared bias rather than
correctness — the paper should report this number as automated screening and
keep a clinician-marked subset as the ground truth it is calibrated against.

Using a different and stronger model than the extractor (Fable 5.1 judging
Opus 5 output) at least removes same-model self-preference, and the judge sees
the FULL source passage rather than only the evidence span, so it can catch a
triple that is quoted accurately but asserted out of context.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import OPENAI_API_KEY, OPENAI_BASE_URL, setting

from .schema import EDGE_TYPES, NODE_TYPES

OUT_DIR = Path(__file__).resolve().parent.parent / "kg"
CHUNKS_FILE = Path(__file__).resolve().parent.parent / "chunks" / "chunks.jsonl"

HIGHCOST_MODEL = setting("OPENAI_HIGHCOST_MODEL") or setting("OPENAI_MODEL")

JUDGE_PROMPT = """You are auditing a clinical knowledge graph built for perioperative
gynecologic nursing decision support. Judge ONE extracted relation.

The relation was extracted from the source passage below and must be judged
against it — not against your own clinical knowledge.

RELATION
  ({subject} : {subject_type}) -[{relation}]-> ({object} : {object_type})
  population: {population}
  condition:  {condition}

RELATION MEANING
  {relation_meaning}

EVIDENCE SPAN THE EXTRACTOR CITED
  "{evidence}"

FULL SOURCE PASSAGE ({source}, tier {tier})
\"\"\"
{passage}
\"\"\"

Judge on four points:
1. asserted     — does the passage actually assert this relation? (not merely
                  mention both entities near each other)
2. direction    — is the subject/object order correct for this relation type?
3. typing       — are both entities labelled with the right entity type?
4. in_context   — is the relation true AS STATED, including its population and
                  condition fields? Judge the triple as a whole: a qualification
                  the passage attaches is only "dropped" if it appears in neither
                  the population nor the condition field. If those fields capture
                  it faithfully, this criterion passes.

Verdict:
  "correct"   — all four hold; safe to use in a clinical knowledge graph
  "incorrect" — any of 1-3 fails, or the dropped qualification makes it misleading
  "unsure"    — genuinely ambiguous from this passage alone

Return JSON only:
{{"verdict": "correct|incorrect|unsure", "failed": ["asserted"|"direction"|"typing"|"in_context"], "reason": "<20 words"}}"""

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def _chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    """Full passage text for each cited chunk."""
    texts: dict[str, str] = {}
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["chunk_id"] in chunk_ids:
                texts[row["chunk_id"]] = row["text"]
    return texts


def _sampling_kwargs(model: str) -> dict:
    """Per-family sampling parameters.

    The proxy rejects `temperature=0` for GPT-5 family models ("Only temperature=1
    is supported"), which silently turned every candidate judgement into an error
    and made a judge look unusable when it was merely mis-parameterised. Omit the
    parameter for those and accept their default.
    """
    name = model.lower()
    if "gpt-5" in name or "o1" in name or "o3" in name:
        return {}
    return {"temperature": 0}


def _relation_meaning(relation: str) -> str:
    if relation not in EDGE_TYPES:
        return relation
    subject, obj = EDGE_TYPES[relation]
    return f"{relation}: a ({subject}) {relation.lower().replace('_', ' ')} a ({obj})"


def _judge_one(triple: dict, passage: str, model: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        subject=triple["subject"], subject_type=triple["subject_type"],
        relation=triple["relation"], object=triple["object"],
        object_type=triple["object_type"],
        relation_meaning=_relation_meaning(triple["relation"]),
        population=triple.get("population") or "(none stated)",
        condition=triple.get("condition") or "(none stated)",
        evidence=triple["evidence"],
        source=f"{triple.get('source','')} {triple.get('year','')}",
        tier=triple.get("tier", "?"),
        passage=passage[:4000])
    try:
        response = client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=4000, **_sampling_kwargs(model))
        content = response.choices[0].message.content or ""
    except Exception as exc:
        return {"verdict": "error", "failed": [], "reason": f"{type(exc).__name__}: {str(exc)[:80]}"}

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {"verdict": "error", "failed": [], "reason": "unparseable judge output"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "error", "failed": [], "reason": "invalid JSON from judge"}

    verdict = (data.get("verdict") or "").strip().lower()
    if verdict not in ("correct", "incorrect", "unsure"):
        verdict = "unsure"
    return {
        "verdict": verdict,
        "failed": [f for f in (data.get("failed") or []) if isinstance(f, str)],
        "reason": (data.get("reason") or "")[:160],
    }


def judge(triples: list[dict], model: str = HIGHCOST_MODEL,
          concurrency: int = 6) -> dict:
    """Adjudicate every triple; returns a precision report."""
    passages = _chunk_texts({t["chunk_id"] for t in triples})
    print(f"  Judging {len(triples)} triples with {model} ...")

    results: list[dict] = []
    done = 0

    def run(triple: dict) -> dict:
        passage = passages.get(triple["chunk_id"], triple["evidence"])
        return {**triple, **_judge_one(triple, passage, model)}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for judged in pool.map(run, triples):
            results.append(judged)
            done += 1
            print(f"    {done}/{len(triples)}", end="\r", flush=True)
    print()

    out = OUT_DIR / "judged_triples.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return report(results, model, out)


def report(results: list[dict], model: str, out_path: Path) -> dict:
    from collections import Counter, defaultdict

    verdicts = Counter(r["verdict"] for r in results)
    scored = [r for r in results if r["verdict"] in ("correct", "incorrect")]
    precision = (sum(1 for r in scored if r["verdict"] == "correct") / len(scored)
                 if scored else 0.0)

    by_relation: dict[str, list[str]] = defaultdict(list)
    by_tier: dict[int, list[str]] = defaultdict(list)
    for r in scored:
        by_relation[r["relation"]].append(r["verdict"])
        by_tier[r.get("tier", 0)].append(r["verdict"])

    def _p(values: list[str]) -> dict:
        return {"precision": sum(1 for v in values if v == "correct") / len(values),
                "n": len(values)}

    failure_modes = Counter()
    for r in results:
        if r["verdict"] == "incorrect":
            failure_modes.update(r.get("failed") or ["unspecified"])

    data = {
        "judge_model": model,
        "judged": len(results),
        "verdicts": dict(verdicts),
        "precision_overall": precision,
        "precision_by_relation": {k: _p(v) for k, v in sorted(by_relation.items())},
        "precision_by_tier": {str(k): _p(v) for k, v in sorted(by_tier.items())},
        "failure_modes": dict(failure_modes.most_common()),
        "judged_file": str(out_path),
    }
    (OUT_DIR / "judge_report.json").write_text(json.dumps(data, indent=2))
    return data
