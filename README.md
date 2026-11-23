# CVT-RAG: Current, Verified, Traceable retrieval for nursing guidance

Nursing guidance changes: guidelines are reissued, recommendations are downgraded
or replaced, and narrower guidance appears for particular patient groups.
CVT-RAG is a retrieval-augmented generation system whose answers are

- **Current**: the knowledge base is a versioned graph that changes only through
  checked, atomic releases; superseded guidance is retired at publication, and
  partial updates are handled recommendation by recommendation;
- **Verified**: every displayed sentence is checked against the passage it cites
  (citation, numbers/strength words/negations, currency, and support by a
  calibrated verifier model) and repaired or withdrawn if it fails;
- **Traceable**: every sentence leads to a page of a specific document version
  and the release the answer was given against; answers are kept in an audit log.

> **Research software, not validated for clinical use.** A checked sentence is
> faithful to its source; that is not the same as clinically correct or current
> consensus. Do not use it for patient care.

The accompanying paper (perioperative gynecologic nursing; comparison with hybrid
RAG, Neo4j GraphRAG, Microsoft-style GraphRAG and an LLM without retrieval) is
under review; its numbers are all computed from the files in `runs/`.

### CVT-RAG layout

| | |
|---|---|
| `verified_rag/` | the system: release-pinned retrieval and graph enrichment, cited-sentence writing, four sentence checks, repair, audit log; evaluation scripts (`baselines`, `retrieval_eval`, `consistency_replay`, `support_calibration`, `population_example`, `timing`); offline tests |
| `corpus_index/` | passages, embeddings and Neo4j with atomic releases, snapshot-pinned search, retirement of superseded documents and passages |
| `verified_rag/knowledge/` | recommendation extraction, supersedes / refines links, per-recommendation pairing for partial updates |
| `span_select/` | candidate spans, read-back by offset, and the Jev client |
| `figures/make_span_verified_figures.py` | every figure and table of the paper, from `runs/` |
| `runs/` | saved results: comparison runs, retrieval accuracy, consistency replay, calibration, timing, and the audit log |
| `corpus/metadata/` | corpus manifest: source URL, tier and file hash of every document (documents themselves are not redistributed) |

### Running it

Configuration is read from the environment or a `.env` file:
`OPENAI_API_KEY`, `OPENAI_BASE_URL` (an OpenAI-compatible endpoint), `VERIFIED_RAG_CHAT_MODEL`,
`OPENAI_EMBEDDING_MODEL`, `TYPESAFE_API_KEY` (Jev verifier), `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
Neo4j 5.26 setup: `NEO4J_SETUP.md`.

```bash
python -m corpus_index --stage-only                 # load passages, nothing live yet
python -m corpus_index --publish DOC_IDS --supersedes NEW=OLD --note "..."
python -m verified_rag ask "Should my intubated patient get 80% oxygen?"
python -m verified_rag ask "..." --as-of 1           # as answered at release 1
python -m verified_rag.tests.test_offline           # no network
python -m verified_rag.baselines run                # four-system comparison
python -m verified_rag.baselines llm-only           # same writer, no retrieval
python -m verified_rag.retrieval_eval               # Recall@k, MRR (read-only)
python figures/make_span_verified_figures.py        # figures and tables from runs/
```

---

## Earlier work in this repository: evidence-tier retrieval agent


A guideline-grounded retrieval and decision-support system in which **provenance
tier is a first-class retrieval signal** — the evidence hierarchy nurses are
taught, implemented in retrieval.

> **Not validated for clinical use.** This is a research system. A registered
> nurse rated 66% of its answers acceptable for use; one in three was not.
> Do not deploy it for patient care.

## What this repository contains

| | |
|---|---|
| `corpus_curation/` | 7-phase acquisition pipeline with content validation, provenance hashing, and headless-browser retrieval for JS-rendered/bot-protected sources |
| `corpus_ingest/` | JATS XML / PDF / HTML extraction and section-aware chunking that preserves recommendation boundaries |
| `corpus_index/` | Embedding and Neo4j loading; native vector index plus full-text index |
| `corpus_dedup/` | MinHash + LSH near-duplicate detection across provenance tiers |
| `nurse_agent/` | LangGraph agent: classify → retrieve → rerank → sufficiency → bounded web escalation → generate → citation verification → safety gate |
| `clinical_kg/` | Clinical knowledge-graph extraction, entity resolution, community summarisation (companion paper) |
| `rag_eval/`, `safety/` | Evaluation harness: ablation grid, RAGAS, safety probes, clinician review instruments |
| `figures/` | Paper figures, regenerable from the result files |

## Headline results

- **Weighting retrieval by evidence tier is what works:** Recall@5 0.560 → 0.733,
  and top-ranked passages from Tier 1 guidelines 0.663 → 0.917 (n = 300; no LLM
  judge in these numbers).
- **Adding keyword search degrades retrieval** on guideline text (BM25 alone
  0.303; fusion costs 0.083).
- **Safety:** 36/36 unsafe probes handled correctly at 0.00 false-refusal rate on
  10 benign controls.
- **Automated evaluation has a reliable range.** Grounding: a registered nurse,
  a non-clinician author and two LLM judges agree within 0.045. Clinical
  acceptability: LLM proxies gave 0.81 and 0.24 where the nurse gave **0.66** —
  neither within 0.15. *"AI-validated" is not a clinical safety claim.*

Full numbers with judges labelled: [`RESULTS.md`](RESULTS.md).

## Data availability

The **corpus manifest** (`corpus/metadata/corpus_metadata.json` — source, tier,
year, URL, SHA-256 per document) and the full acquisition pipeline are included,
so the corpus is reproducible.

**Document full text is not redistributed.** Elsevier TDM XML and publisher PDFs
are licensed for text mining, not redistribution. Run `python -m corpus_curation`
with your own API credentials to rebuild it.

Derived artefacts that are large but regenerable (chunk file, embedding cache,
per-condition answer files) are excluded; result summaries, test sets, review
sheets and figures are included.

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add your own API credentials
python -m corpus_curation --audit          # validate corpus integrity
python -m corpus_ingest                    # extract and chunk
python -m corpus_index --check             # verify endpoint + Neo4j
python -m corpus_index                     # embed and load
python -m rag_eval --relational --in-domain   # or see rag_eval/run_grid.py
```

Neo4j setup, including the rootless-podman caveat: [`NEO4J_SETUP.md`](NEO4J_SETUP.md).

## Credentials

`.env` only (git-ignored). See `.env.example` for the required keys. No
credentials are committed to this repository.

## Citation

Manuscript in preparation. The knowledge-graph evaluation is reported separately.
