"""Step 1: the chat model writes the answer as separately citable claims."""

from __future__ import annotations

import json
import re
from typing import List

from ..types import Claim, Source

PROMPT = """You are answering a perioperative nurse's question using ONLY the numbered guideline passages below.

Question: {question}

Passages:
{passages}

Write the answer as at most {max_claims} short sentences. Rules:
- One clinical claim per sentence. Each sentence cites the passage(s) it relies on.
- Copy every number, unit, dose, duration and recommendation-strength word (strong, conditional, recommends, suggests) EXACTLY as the passage writes it.
- State the population a recommendation applies to when the passage states one.
- Do not add anything the passages do not say. If they answer only part of the question, answer that part. Return [] only if nothing in them bears on the question.

Return ONLY a JSON array, no prose:
[{{"text": "<sentence>", "sources": ["S1"]}}]"""


def render_passages(sources: List[Source], max_chars: int = 900) -> str:
    out = []
    for s in sources:
        tag = f"{s.title} ({s.year}), Tier {s.tier}" + (f", applies to: {s.refines}" if s.refines else "") \
            + (f", grade: {s.grade}" if s.grade else "")
        note = "".join(
            f"\nNOTE: this passage states an older recommendation that was REPLACED in "
            f"{x['new_year']}: \"{' '.join(x['old_text'].split())[:160]}\" -> current: "
            f"\"{' '.join(x['new_text'].split())[:200]}\"" + (f" (see [{x['new_label']}])" if x.get('new_label') else "")
            for x in s.replaced)
        out.append(f"[{s.label}] {tag}\n{' '.join(s.text.split())[:max_chars]}{note}")
    return "\n\n".join(out)


def parse_claims(text: str, max_claims: int) -> List[Claim]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    claims = []
    for item in items[:max_claims]:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        srcs = [str(x).strip() for x in (item.get("sources") or []) if str(x).strip()]
        claims.append(Claim(text=" ".join(str(item["text"]).split()), sources=srcs))
    return claims


def synthesize(chat, question: str, sources: List[Source], max_claims: int):
    # 4000: a reasoning model spends hundreds of tokens before the first word
    # of output, and a small ceiling returns an empty answer (design doc §14).
    result = chat.complete(PROMPT.format(question=question, passages=render_passages(sources),
                                         max_claims=max_claims), max_tokens=4000)
    return parse_claims(result.text, max_claims), result
