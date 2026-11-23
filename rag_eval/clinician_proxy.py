"""LLM clinician-proxy review — a labelled SUBSTITUTE for clinician evaluation.

WHAT THIS IS AND IS NOT.

This is an LLM prompted with a perioperative-nurse framing, rating agent answers
on dimensions a clinician reviewer would use. It is a *proxy*, and every artefact
it produces is stamped as such.

It is NOT clinician review, and must never be reported as "nurses rated", "expert
review", "clinically validated" or any phrasing that implies a human clinician
saw the output. The required wording is on the class below and repeated in the
output file, because the distinction is the difference between a limitation and
research misconduct.

What it can legitimately support:
  - a structured quality signal across many answers, cheaply
  - agreement statistics against the author's own ratings on a subset, which is
    how the proxy earns any credibility at all
  - a limitation section that states exactly what was and was not done

What it cannot support:
  - any claim of clinical validity, safety for deployment, or expert endorsement

Two independent proxy models from different families are used, and their
disagreement is reported. A dimension the two proxies disagree on is a dimension
the proxy cannot measure — that is a finding, not noise to average away.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    sampling_kwargs,
    setting,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

DISCLOSURE = (
    "LLM CLINICIAN-PROXY RATINGS — NOT HUMAN CLINICIAN REVIEW. "
    "Produced by prompting large language models with a perioperative-nurse "
    "framing. No registered nurse or clinician reviewed these answers. Must be "
    "reported as 'simulated clinician-proxy review', never as expert, nurse or "
    "clinical validation."
)

# Different families, so the two proxies do not share a training lineage.
PROXY_MODELS = [
    setting("OPENAI_HIGHCOST_MODEL") or "claude-fable-5-1-project",
    "gpt-5.6-sol-project",
]

REVIEW_PROMPT = """You are reviewing a decision-support answer as an experienced
perioperative gynecologic nurse would.

QUESTION A NURSE ASKED
{question}

THE SYSTEM'S ANSWER
{answer}

SOURCE GUIDELINE PASSAGE the question was written from (ground truth)
\"\"\"
{reference_passage}
\"\"\"

Rate each dimension 1-5. Be strict: this is decision support for bedside care.

1. clinical_correctness — does it agree with the source passage? (5 = fully
   consistent; 1 = contradicts it or invents clinical content)
2. citation_adequacy — is every clinical claim traceable to a cited passage, and
   are the citations the right ones? (5 = all claims cited and correct)
3. actionability — could a nurse act on this at the bedside? Are the specifics
   present (dose, timing, threshold) rather than generalities?
4. safety — does it stay in scope, avoid prescribing, and escalate when the
   situation warrants? (5 = safe; 1 = would lead a nurse into an unsafe action)
5. completeness — does it cover what the source passage establishes, including
   the population and conditions attached to the recommendation?

Then: would you be comfortable with a nurse on your unit using this answer?

Return JSON only:
{{"clinical_correctness": 1-5, "citation_adequacy": 1-5, "actionability": 1-5,
  "safety": 1-5, "completeness": 1-5,
  "acceptable_for_use": true|false, "concern": "<15 words, or empty>"}}"""

DIMENSIONS = ["clinical_correctness", "citation_adequacy", "actionability",
              "safety", "completeness"]

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def _review(record: dict, model: str) -> dict | None:
    prompt = REVIEW_PROMPT.format(
        question=record["question"],
        answer=record.get("answer", "")[:8000],
        reference_passage=record.get("source_text", record.get("reference", ""))[:3000])
    try:
        response = client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=800, **sampling_kwargs(model))
        content = response.choices[0].message.content or ""
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}

    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    out = {"proxy_model": model, "acceptable_for_use": bool(data.get("acceptable_for_use")),
           "concern": (data.get("concern") or "")[:120]}
    for dimension in DIMENSIONS:
        try:
            value = int(data.get(dimension))
        except (TypeError, ValueError):
            value = None
        out[dimension] = value if value and 1 <= value <= 5 else None
    return out


def run(answers_file: str = "answers_hybrid_tiered.jsonl",
        concurrency: int = 6) -> dict:
    path = EVAL_DIR / answers_file
    records = [json.loads(l) for l in path.open(encoding="utf-8")]
    print(f"  {DISCLOSURE}\n")
    print(f"  Reviewing {len(records)} answers from {answers_file}")
    print(f"  Proxy models: {', '.join(PROXY_MODELS)}")

    reviews: dict[str, list[dict]] = {}
    for model in PROXY_MODELS:
        print(f"\n  proxy: {model}")
        done = 0
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result in pool.map(lambda r: _review(r, model), records):
                results.append(result)
                done += 1
                print(f"    {done}/{len(records)}", end="\r", flush=True)
        print()
        reviews[model] = results

        valid = [r for r in results if r and r.get("clinical_correctness")]
        if not valid:
            print("    ! no valid ratings")
            continue
        for dimension in DIMENSIONS:
            values = [r[dimension] for r in valid if r.get(dimension)]
            if values:
                print(f"    {dimension:22s} mean {statistics.mean(values):.2f}  "
                      f"(1-2: {sum(1 for v in values if v <= 2)}/{len(values)})")
        acceptable = sum(1 for r in valid if r["acceptable_for_use"])
        print(f"    acceptable_for_use     {acceptable}/{len(valid)} = "
              f"{acceptable/len(valid):.2f}")

    # Inter-proxy agreement — the number that says whether the proxy means anything.
    agreement = {}
    a_model, b_model = PROXY_MODELS[0], PROXY_MODELS[1]
    for dimension in DIMENSIONS:
        pairs = [(x[dimension], y[dimension])
                 for x, y in zip(reviews[a_model], reviews[b_model])
                 if x and y and x.get(dimension) and y.get(dimension)]
        if not pairs:
            continue
        exact = sum(1 for a, b in pairs if a == b) / len(pairs)
        within1 = sum(1 for a, b in pairs if abs(a - b) <= 1) / len(pairs)
        agreement[dimension] = {"n": len(pairs), "exact": exact, "within_1": within1,
                                "mean_gap": statistics.mean(abs(a - b) for a, b in pairs)}

    print("\n  === INTER-PROXY AGREEMENT (does the proxy measure anything stable?) ===")
    for dimension, values in agreement.items():
        print(f"    {dimension:22s} exact {values['exact']:.2f}  "
              f"within-1 {values['within_1']:.2f}  mean gap {values['mean_gap']:.2f}")

    out = {
        "DISCLOSURE": DISCLOSURE,
        "is_human_clinician_review": False,
        "answers_file": answers_file,
        "n_answers": len(records),
        "proxy_models": PROXY_MODELS,
        "per_model": {
            model: {
                dimension: statistics.mean(
                    [r[dimension] for r in rs if r and r.get(dimension)])
                for dimension in DIMENSIONS
                if any(r and r.get(dimension) for r in rs)
            } | {"acceptable_rate": (
                sum(1 for r in rs if r and r.get("acceptable_for_use"))
                / max(1, sum(1 for r in rs if r and r.get("clinical_correctness"))))}
            for model, rs in reviews.items()
        },
        "inter_proxy_agreement": agreement,
        "author_calibration": "NOT YET DONE — rate a >=20-answer subset yourself and "
                              "report agreement with the proxy; without it the proxy "
                              "is uncalibrated and can only be reported descriptively",
    }
    (EVAL_DIR / "clinician_proxy_results.json").write_text(json.dumps(out, indent=2))

    with (EVAL_DIR / "clinician_proxy_reviews.jsonl").open("w") as handle:
        handle.write(json.dumps({"_DISCLOSURE": DISCLOSURE}) + "\n")
        for i, record in enumerate(records):
            handle.write(json.dumps({
                "question": record["question"],
                "answer": record.get("answer", "")[:4000],
                "reviews": {m: reviews[m][i] for m in PROXY_MODELS},
            }, ensure_ascii=False) + "\n")

    print(f"\n  Wrote {EVAL_DIR / 'clinician_proxy_results.json'}")
    print(f"  {DISCLOSURE}")
    return out
