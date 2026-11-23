"""Synthetic test-set generation from Tier 1-2 chunks.

Questions are generated only from Tier 1-2 (guidelines and nursing best
practice), never from Tier 4. A question synthesised from a primary study tests
recall of that study's findings, which is neither what a nurse asks nor what the
system should answer authoritatively — and it would make the tier-weighted
retrieval claim circular.

Each item records the chunk it came from, so `context_recall` and retrieval
Recall@k are computable without human gold labels.
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

GENERATION_PROMPT = """You are helping build an evaluation set for a clinical
decision-support system used by perioperative gynecologic nurses.

Below is a passage from an authoritative clinical guideline.

Source: {source} ({year}) — {title}
Section: {section}

Passage:
\"\"\"
{text}
\"\"\"

Write ONE question that a perioperative nurse would realistically ask during
their shift, and that this passage answers.

Rules:
- The question must be answerable from this passage alone.
- Write it the way a nurse would ask it — practical and specific.
  Good: "How soon after a laparoscopic hysterectomy should the catheter come out?"
  Bad:  "What does the passage state regarding catheter removal timing?"
- Never refer to "the passage", "the guideline", "the text" or "the document".
- Prefer questions about timing, dose, threshold, technique or escalation.
- If the passage is boilerplate, a reference list, or has no clinical content,
  reply with exactly: SKIP

Then give the answer, grounded strictly in the passage.

Return JSON only:
{{"question": "...", "answer": "...", "question_type": "timing|dose|threshold|technique|escalation|general"}}"""


def _client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)


# A passage worth asking a nurse question about states an action, a threshold or
# a timing. Without this filter the sampler feeds the generator fragments,
# contents pages and figure captions, and the model (correctly) skips them all.
_CLINICAL_SIGNAL = re.compile(
    r"\b(should|should not|must|recommend|recommended|consider|avoid|"
    r"administer|remove|insert|monitor|assess|ensure|offer|give|prescribe|"
    r"do not|indicated|contraindicated|prior to|within|every|per |mg|mcg|"
    r"ml|hours?|minutes?|days?|weeks?)\b", re.IGNORECASE)

# Numbers with units, or a numbered recommendation — the questions with a
# checkable answer live here.
_HAS_SPECIFIC = re.compile(
    r"\d+\s*(?:mg|mcg|g|ml|l|hours?|hrs?|h\b|minutes?|min\b|days?|weeks?|%|"
    r"mmhg|units?|degrees?)|^\s*\d+\.\d", re.IGNORECASE | re.MULTILINE)


def load_candidates(min_tokens: int = 80, max_tokens: int = 600,
                    tiers: tuple[int, ...] = (1, 2),
                    require_specific: bool = False) -> list[dict]:
    """Canonical, content-bearing Tier 1-2 chunks worth asking about."""
    rows = []
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("tier") not in tiers:
                continue
            if row.get("is_canonical") is False:
                continue
            if not (min_tokens <= row.get("est_tokens", 0) <= max_tokens):
                continue
            if row.get("kind") == "table":
                continue      # tables need a different question style
            text = row["text"]
            lowered = text[:200].lower()
            if any(marker in lowered for marker in
                   ("references", "your responsibility", "copyright", "acknowledg",
                    "contents", "list of tables", "abbreviations", "glossary")):
                continue
            # At least 3 clinical-signal terms, and enough prose to be a passage
            # rather than a heading stack.
            if len(_CLINICAL_SIGNAL.findall(text)) < 3:
                continue
            if text.count(".") < 3:
                continue
            if require_specific and not _HAS_SPECIFIC.search(text):
                continue
            rows.append(row)
    return rows


def _stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Spread the sample across topic axes so no axis dominates the pilot."""
    rng = random.Random(seed)
    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        by_topic.setdefault(row.get("topic") or row.get("source", "?"), []).append(row)

    for bucket in by_topic.values():
        rng.shuffle(bucket)

    picked: list[dict] = []
    topics = sorted(by_topic)
    while len(picked) < n and any(by_topic[t] for t in topics):
        for topic in topics:
            if by_topic[topic] and len(picked) < n:
                picked.append(by_topic[topic].pop())
    rng.shuffle(picked)
    return picked


def _parse(content: str) -> dict | None:
    content = content.strip()
    if content.upper().startswith("SKIP"):
        return None
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    # The model sometimes wraps its refusal in the JSON envelope it was asked for.
    if not question or not answer or question.upper() == "SKIP":
        return None
    if len(question) < 15 or len(answer) < 15:
        return None
    return {**data, "question": question, "answer": answer}


def _generate_one(row: dict, client, model: str) -> dict | None:
    prompt = GENERATION_PROMPT.format(
        source=row.get("source", ""), year=row.get("publication_year", ""),
        title=(row.get("title") or "")[:120], section=(row.get("section") or "")[:120],
        text=row["text"][:4000],
    )
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=4000,
        )
        return _parse(response.choices[0].message.content or "")
    except Exception as exc:
        print(f"    ! generation error: {type(exc).__name__}: {str(exc)[:90]}")
        return None


def generate(n: int = 50, seed: int = 20260905, model: str = CHAT_MODEL,
             out_file: Path | None = None, concurrency: int = 8) -> list[dict]:
    candidates = load_candidates()
    print(f"  Tier 1-2 candidate chunks: {len(candidates)}")

    # Over-sample: some passages will be skipped as boilerplate.
    sample = _stratified_sample(candidates, int(n * 1.6), seed)
    client = _client()

    items: list[dict] = []
    skipped = 0
    lock = threading.Lock()

    # One request per passage, several in flight: a 300-question set is otherwise
    # an hour of sequential round-trips.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for row, data in zip(sample, pool.map(
                lambda r: _generate_one(r, client, model), sample)):
            with lock:
                if len(items) >= n:
                    break
                if data is None:
                    skipped += 1
                    continue
                items.append(_item(row, data))
                print(f"    generated {len(items)}/{n}", end="\r", flush=True)

    print(f"\n  Generated {len(items)} items ({skipped} passages skipped as non-clinical)")

    out_file = out_file or (OUT_DIR / "pilot_testset.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {out_file}")
    return items


def _item(row: dict, data: dict) -> dict:
    return {
        "question": data["question"].strip(),
        "reference": data["answer"].strip(),
        "question_type": data.get("question_type", "general"),
        # Provenance of the gold passage: makes Recall@k judge-free.
        "source_chunk_id": row["chunk_id"],
        "source_doc_id": row["doc_id"],
        "source_text": row["text"],
        "source_tier": row["tier"],
        "source_name": row.get("source", ""),
        "source_title": row.get("title", ""),
        "source_section": row.get("section", ""),
        "source_page": row.get("page"),
        "source_year": row.get("publication_year", ""),
    }
