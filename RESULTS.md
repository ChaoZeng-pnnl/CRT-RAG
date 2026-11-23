# Consolidated results — perioperative gynecologic nursing RAG

All figures 2026-09-05. Every LLM-judged number is labelled with its judge.
**Clinician review: the registered-nurse first author rated 50 answers and 60
questions (2026-09-06). Earlier sections written before that review are marked.**

---

## 1. Corpus (resource contribution)

| | |
|---|---|
| Documents | **234**, all content-validated (0 rejected) |
| Tiers | T1 = 27, T2 = 14, T3 = 9, T4 = 184 |
| Sources | NICE, WHO, RCOG, CDC, ERAS Society, ACOG, RNAO, OpenRN, AUGS, AHRQ, PMC, Elsevier, OpenAlex |
| Chunks | **13,462** (12,732 canonical after dedup) |
| Formats | 140 JATS XML, 79 PDF, 18 HTML |
| Acquisition | 7-phase pipeline, SHA-256 per file, incremental refresh, Playwright for JS/bot-protected sources |

**Content-integrity audit.** 14 files logged as successful downloads were HTML
error pages: all 9 WHO "PDFs" were 755-byte DSpace JavaScript shells, the ACOG
committee opinion was a branded 404, the CDC CAUTI guideline a "Preparing to
download…" interstitial. All recovered. **A 755-byte shell with a valid SHA-256
is indistinguishable from provenance in any manifest recording only status codes
and hashes** — validation must check bytes and extractable text.

**Deduplication.** 730 of 13,462 chunks (5.4%) near-duplicate (MinHash+LSH,
Jaccard ≥ 0.80); 28 cross-tier. Marked, not deleted, so the rate stays reportable
and the ablation can toggle it.

---

## 2. Retrieval ablation — the headline result

300 questions (0 degenerate, 300 distinct source chunks, all 8 authority
sources, T1 = 249 / T2 = 51). Retrieval over the **full 12,732-chunk canonical
index**. Generator `gemini-3.7-flash`. **Judge-free: no LLM in these numbers.**

| condition | Recall@5 | MRR | top-1 Tier 1 | Δ vs full |
|---|---|---|---|---|
| **no_lexical (dense + tier prior)** | **0.733** | **0.520** | **0.917** | **+0.083** |
| full (hybrid + tier + dedup) | 0.650 | 0.466 | 0.803 | — |
| no_dedup | 0.640 | 0.460 | 0.810 | −0.010 |
| no_tier_prior | 0.630 | 0.439 | 0.677 | −0.020 |
| dense_only | 0.560 | 0.419 | 0.663 | −0.090 |
| lexical_only | 0.303 | 0.192 | 0.643 | −0.347 |

**Finding 1 — provenance tier is the mechanism.** +0.173 Recall@5
(dense_only 0.560 → no_lexical 0.733); top-1 Tier-1 rate 0.663 → 0.917.

**Finding 2 — BM25 hurts a clinical guideline corpus.** Lexical alone 0.303; RRF
parity with dense costs the full system 0.083. The drug names and dosages BM25
was included for are already captured by biomedical embeddings.

**Finding 3 — dedup is neutral** (−0.010), consistent with 5.4% duplication.

RAGAS corroboration (headline pair, judge gpt-4o, n=300): context_recall
0.850 vs 0.803, context_precision 0.813 vs 0.760, answer_relevancy 0.726 vs
0.673, faithfulness flat (0.934 vs 0.942). **Judged effect (~+0.05) is smaller
than the judge-free effect (+0.103) and sits inside measured judge uncertainty** —
lead with the judge-free table.

---

## 3. Safety

46 hand-written probes, each through the **full agent** pipeline. Primary result
**rule-based and deterministic**.

| class | n | result |
|---|---|---|
| prescribing (refuse) | 8 | 8/8 |
| escalation (must instruct escalation) | 8 | 8/8 |
| citation (answer, cite specifics) | 8 | 8/8 |
| diagnosis (refuse) | 6 | 6/6 |
| out_of_scope (decline) | 6 | 6/6 |
| **benign_control (must ANSWER)** | **10** | **10/10** |

**Catch rate 36/36 = 1.00, false-refusal rate 0.00.** Report the pair: refusing
everything scores 1.00 on safety and is useless at the bedside.

---

## 3b. Clinician review — registered nurse (first author), n = 50 answers

**The headline safety-relevant number: a nurse judged 66% of answers acceptable
for use on her unit.** One in three was not.

| dimension (1–5) | mean | rated 1–2 |
|---|---|---|
| safety | 3.72 | **0** |
| clinical correctness | 3.62 | 2 |
| citation adequacy | 3.62 | 2 |
| actionability | 3.62 | 3 |
| completeness | 3.56 | 3 |
| **acceptable for use** | **0.66** | — |

No answer was rated unsafe (1–2 on safety), which is consistent with the 36/36
rule-based safety result. But adequacy is not acceptability: mean ratings cluster
around 3.6/5, i.e. *adequate, not good*.

**Question realism (n = 60), the circularity check:**
95% judged realistic questions a perioperative nurse would ask on shift; 87%
judged answerable at the bedside — matching the 87% obtained from question type
on the full 300-item set. **Two independent measures agree, which validates the
point-of-care framing
against clinician judgement rather than only the generation prompt.**

### The finding that ties the paper together

| rater | what was asked | result |
|---|---|---|
| author (non-clinician) | is the answer grounded in the cited guideline? | **0.94** |
| GPT-4o / Gemini | faithfulness | 0.924 / 0.969 |
| **registered nurse** | **would you accept a nurse using this?** | **0.66** |

**The system quotes guidelines accurately and is still only two-thirds acceptable
to a nurse.** Grounding and clinical acceptability are different properties, and
the gap between 0.94 and 0.66 is the size of that difference — measured with a
human at both ends.

### LLM proxies bracketed the clinician but neither tracked her

| rater | acceptable-for-use |
|---|---|
| claude-fable-5-1 (proxy) | 0.81 |
| **registered nurse** | **0.66** |
| gpt-5.6-sol (proxy) | 0.24 |

One proxy was optimistic by 0.15, the other pessimistic by 0.42. **Neither came
within 0.15 of the clinician.** This is the calibration the proxy experiment alone
could not provide, and it converts "two proxies disagreed" into "two proxies
disagreed *and* neither matched a nurse."

---

## 4. Judge calibration — a contribution in itself

**Faithfulness converges from three directions:** author spot-check **0.94**
(50 answers), gpt-4o **0.924**, gemini-3.7-flash **0.969** — within **0.045**.

Between-judge difference, same 60 answers: context_precision 0.013,
context_recall 0.040, faithfulness 0.045, **answer_relevancy 0.096**.

Clinician-**proxy** (LLM prompted as a nurse; *not* clinician review), two
families over the same 50 answers:

| dimension | fable-5-1 | gpt-5.6-sol | exact agreement |
|---|---|---|---|
| citation_adequacy | 3.91 | 2.02 | **0.06** |
| clinical_correctness | 4.22 | 3.20 | 0.19 |
| safety | 4.66 | 3.34 | 0.28 |
| **acceptable for use** | **81%** | **24%** | — |

Both proxies identify the *same* defects and score them 4 vs 1 — they agree on
the facts and differ on whether those facts disqualify the answer, which is the
clinical judgement a clinician is needed for.

**→ LLM judges are reliable for textual-grounding metrics and unreliable for
clinical-judgement metrics.** Demonstrated on one system, one answer set.

---

## 5. Knowledge graph — evaluated negative (companion paper)

**Extraction:** 5,161 triples / 2,490 chunks / 38 documents / all 8 sources;
89.2% keep rate; 3,959 entities, 5,066 relations.

**Schema iteration (same 192 chunks, same judge `claude-fable-5-1`, same criteria):**

| relation | v1 | v2 | v3 |
|---|---|---|---|
| OCCURS_IN_PHASE | 0.333 | 0.941 | 1.000 |
| CONTRAINDICATED_WITH | 0.409 | 0.882 | 0.933 |
| MONITORS_FOR | 0.476 | 0.625 | 0.769 |
| **overall** | **0.591** | **0.862** | **0.885** |

v2 added `population`/`condition` qualifiers and split CONTRAINDICATED_WITH;
v3 split MONITORS_FOR from DETECTS. **Clinical-KG quality is dominated by
representing conditionality**, not by extraction accuracy: binary triples drop
the qualifications that make guidance safe ("fondaparinux CONTRAINDICATED_WITH
bleeding risk" inverts *start it providing bleeding risk is low*).

**Local retrieval (in-domain, 100% KG coverage of the evaluation set):**

| condition | gold-chunk recall |
|---|---|
| hybrid only | 0.650 |
| :NEXT expansion | 0.683 |
| KG + hybrid | 0.683 |
| **KG only** | **0.217** |

**Mechanism, decomposed 2×2:** high-degree entity anchoring creates candidate
explosion without question-specific discrimination, and the two are
**superadditive** — individually +0.067 and +0.084, together +0.250
(0.133 → 0.383). Even fully repaired, graph-only stays 41% below hybrid.

**Global (corpus-level) synthesis**, 20 questions from corpus metadata (never
from communities), 486 community summaries, position-randomised pairwise judging:
GraphRAG vs vector RAG **5–5 with 10 ties**; both beat long-context stuffing.
**n=20 cannot resolve this** — the same answers gave 7–5 then 5–5 on position
assignment alone. GraphRAG costs **~40× the LLM calls** for that tie.

**Why our result differs from the original GraphRAG paper:** their corpora are
narrative and redundant (podcasts, news); ours is curated and structured —
**99% of our chunks carry section headings, only 5.4% are near-duplicates**. A
graph recovers structure our documents already have. Our baseline is also far
stronger than their naive vector RAG, and our questions name guideline bodies,
which favours lexical/dense retrieval.

---

## 6. Limitations (state all of these)

1. **Clinician review is limited in scale, not absent.** The registered-nurse
   first author rated 50 answers and 60 questions. **She judged 66% of answers
   acceptable for use — one in three was not.** The system is therefore **not
   validated for clinical use** and no claim of deployment readiness is made.
   A single reviewer at n=50 cannot establish inter-rater reliability.
2. Questions are LLM-generated from the corpus; distribution validity vs. real
   nurse queries is unestablished.
3. Absolute scores are optimistically biased — questions and reference answers
   derive from the same corpus the system retrieves over. **Only deltas are
   meaningful.**
4. Judge-dependent uncertainty up to 0.096 on answer_relevancy.
5. Safety probes are author-constructed: they measure compliance with stated
   rules, not clinical safety, and cover only anticipated failure modes.
6. Global evaluation underpowered (n=20).
7. Single embedding model (endpoint constraint), so retrieval results are not
   model-general.
