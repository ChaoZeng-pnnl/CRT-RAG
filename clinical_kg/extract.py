"""LLM extraction of schema-conforming triples from Tier 1-2 chunks.

Every triple carries the chunk it came from and a verbatim evidence span. A
graph that can traverse but cannot cite is unusable here: the relational branch
has to hand real passages to the generator and the citation verifier, so the
traversal finds the path and the chunk_ids fetch the text.

Triples that fail the schema type-check are dropped and counted, not repaired —
the drop rate is itself a reportable quality signal.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from corpus_index.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

from .schema import PHASES, schema_prompt_block, valid_edge

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "chunks" / "chunks.jsonl"
OUT_DIR = Path(__file__).resolve().parent.parent / "kg"

EXTRACTION_PROMPT = """Extract clinical relations from this guideline passage for a
perioperative gynecologic nursing knowledge graph.

{schema}

RULES
- Use ONLY the entity and relation types above. Drop anything that does not fit.
- Extract relations the passage ASSERTS. Do not add clinical knowledge of your own.
- `evidence` must be a VERBATIM span copied from the passage, under 200 characters.

- CRITICAL — capture the qualifications. Clinical guidance is almost never
  unconditional: it says "do X, for patients P, when condition C holds". A bare
  triple that drops P or C is not merely incomplete, it is misleading and unsafe.
    * `population` — who the relation applies to ("women with previous VTE",
      "patients receiving unfractionated heparin", "obstetric patients"). Empty
      string only if the passage genuinely states it for everyone.
    * `condition`  — the circumstance or constraint under which it holds
      ("providing bleeding risk is low", "until the lesion is secured",
      "for a minimum of 7 days", "if regional analgesia was not used").
  If the passage attaches a qualification you cannot fit into these fields,
  DROP the triple rather than emit it unqualified.
- If the relation holds only until something resolves, use
  TEMPORARILY_CONTRAINDICATED, not CONTRAINDICATED_WITH.
- If one thing merely determines WHEN another is given, use TIMING_DEPENDS_ON,
  not CONTRAINDICATED_WITH.

- Diagnostic tests DETECT; they do not MONITOR_FOR. Reserve MONITORS_FOR for
  ongoing or repeated observation over time.
- A test that DETECTS a risk factor detects THAT RISK FACTOR, not the downstream
  complication. Thrombophilia testing detects thrombophilia, not VTE.
- DO NOT extract incidental or alternative findings as relations. "CTPA can
  identify other pathology including pneumonia and rarely aortic dissection"
  describes differential diagnosis, not a clinical relation — emit nothing.
- DO NOT treat documentation, history-taking or record-keeping as monitoring or
  detection. "A careful history should be documented" asserts no relation.
- Give each entity a short canonical name in lower case, singular, no abbreviations
  where the full term is available ("venous thromboembolism", not "VTE"; but keep
  "lmwh" if that is all the passage gives).
- If the passage states no relation between two distinct entities, return an empty list.
  Boilerplate, references and background prose legitimately yield nothing.

Passage (from {source}, {year} — {title}):
\"\"\"
{text}
\"\"\"

Return JSON only:
{{"triples": [
  {{"subject": "...", "subject_type": "...", "relation": "...",
    "object": "...", "object_type": "...",
    "population": "<who it applies to, or \"\">",
    "condition": "<circumstance/constraint, or \"\">",
    "evidence": "<verbatim span>", "confidence": 0.0-1.0}}
]}}"""

_client = None
_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    return _client


def load_chunks(tiers: tuple[int, ...] = (1, 2), require_recommendation: bool = True,
                sources: tuple[str, ...] | None = None,
                topics: tuple[str, ...] | None = None) -> list[dict]:
    """Canonical Tier 1-2 chunks that assert something actionable."""
    recommendation = re.compile(
        r"\b(should|must|recommend|recommended|consider|avoid|do not|"
        r"contraindicat|indicated|offer|ensure)\b", re.IGNORECASE)

    rows = []
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("tier") not in tiers:
                continue
            if row.get("is_canonical") is False:
                continue
            if row.get("est_tokens", 0) < 60:
                continue
            if require_recommendation and not recommendation.search(row["text"]):
                continue
            if sources and row.get("source") not in sources:
                continue
            if topics and not any(t.lower() in (row.get("topic") or "").lower()
                                  for t in topics):
                continue
            rows.append(row)
    return rows


def _parse(content: str) -> list[dict]:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return data.get("triples") or []


def _normalise(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower()).strip(" .,;:")


def _clean(triple: dict, row: dict, stats: dict) -> dict | None:
    """Type-check and ground one triple; None if it fails the schema."""
    subject = _normalise(triple.get("subject"))
    obj = _normalise(triple.get("object"))
    relation = (triple.get("relation") or "").strip().upper()
    subject_type = (triple.get("subject_type") or "").strip()
    object_type = (triple.get("object_type") or "").strip()
    evidence = (triple.get("evidence") or "").strip()

    if not subject or not obj or subject == obj:
        stats["dropped_degenerate"] += 1
        return None
    if not valid_edge(relation, subject_type, object_type):
        stats["dropped_schema"] += 1
        return None
    if object_type == "Phase" and obj not in PHASES:
        stats["dropped_schema"] += 1
        return None

    # The evidence span must genuinely appear in the passage. This is the check
    # that catches a model paraphrasing instead of quoting, which would make
    # every downstream citation unverifiable.
    haystack = re.sub(r"\s+", " ", row["text"]).lower()
    needle = re.sub(r"\s+", " ", evidence).lower()
    if len(needle) < 15 or needle not in haystack:
        stats["dropped_evidence"] += 1
        return None

    return {
        "subject": subject, "subject_type": subject_type,
        "relation": relation,
        "object": obj, "object_type": object_type,
        # The qualifications that make the relation safe to act on. Retrieval
        # must surface these alongside the relation, never the bare triple.
        "population": (triple.get("population") or "").strip()[:200],
        "condition": (triple.get("condition") or "").strip()[:200],
        "evidence": evidence[:200],
        "confidence": float(triple.get("confidence") or 0.5),
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "tier": row["tier"],
        "source": row.get("source", ""),
        "title": row.get("title", ""),
        "section": row.get("section", ""),
        "page": row.get("page"),
        "url": row.get("url", ""),
        "year": row.get("publication_year", ""),
    }


def extract_one(row: dict, stats: dict, model: str = CHAT_MODEL) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(
        schema=schema_prompt_block(),
        source=row.get("source", ""), year=row.get("publication_year", ""),
        title=(row.get("title") or "")[:100], text=row["text"][:3500])
    try:
        response = client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=4000)
    except Exception as exc:
        with _lock:
            stats["errors"] += 1
        print(f"    ! {type(exc).__name__}: {str(exc)[:90]}")
        return []

    triples = []
    for raw in _parse(response.choices[0].message.content or ""):
        with _lock:
            stats["proposed"] += 1
            cleaned = _clean(raw, row, stats)
        if cleaned:
            triples.append(cleaned)
    return triples


def run(limit: int | None = 300, concurrency: int = 8,
        out_file: Path | None = None, **filters) -> dict:
    rows = load_chunks(**filters)
    if limit:
        rows = rows[:limit]
    print(f"  Extracting from {len(rows)} Tier 1-2 chunks ...")

    stats = {"proposed": 0, "dropped_schema": 0, "dropped_evidence": 0,
             "dropped_degenerate": 0, "errors": 0}
    all_triples: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for triples in pool.map(lambda r: extract_one(r, stats), rows):
            all_triples.extend(triples)
            done += 1
            print(f"    {done}/{len(rows)} chunks — {len(all_triples)} triples",
                  end="\r", flush=True)
    print()

    out_file = out_file or (OUT_DIR / "triples.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for triple in all_triples:
            handle.write(json.dumps(triple, ensure_ascii=False) + "\n")

    kept = len(all_triples)
    stats.update({"chunks": len(rows), "kept": kept,
                  "keep_rate": kept / stats["proposed"] if stats["proposed"] else 0.0})
    (OUT_DIR / "extraction_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  Kept {kept}/{stats['proposed']} proposed triples "
          f"(schema drops {stats['dropped_schema']}, "
          f"evidence drops {stats['dropped_evidence']})")
    print(f"  Wrote {out_file}")
    return stats
