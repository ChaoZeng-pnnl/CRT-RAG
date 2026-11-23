# Manuscript tables

Generated from the result files by `figures/make_tables.py`.

### Table 1 — Corpus composition

| Evidence tier | Sources | Documents | Passages | Tokens |
|---|---|---|---|---|
| Tier 1 — clinical guidelines | WHO (9), NICE (7), RCOG (5), CDC (4), ERAS Society (3), ACOG (1) | 29 | 3,689 | 1.19 M |
| Tier 2 — nursing best practice | OpenRN (9), RNAO (4), AUGS (2) | 15 | 2,500 | 0.93 M |
| Tier 3 — safety/reflective | AHRQ PSNet (9) | 9 | 83 | 0.02 M |
| Tier 4 — primary literature | PubMed PMC (83), Elsevier (57), OpenAlex (44) | 184 | 7,190 | 2.56 M |
| Total | 13 sources | 237 | 13,462 | 4.70 M |

*Content verification rejected 14 files that were logged as successful downloads but contained web-page error pages (11 of them Tier 1); all were subsequently recovered. Near-duplicate detection marked 730 passages (5.4%), leaving 12,732 used for retrieval.*

---

### Table 2 — Retrieval ablation

| Retrieval configuration | Recall@5 | MRR | Top-1 from Tier 1 | Δ Recall@5 |
|---|---|---|---|---|
| Dense + evidence-tier prior (selected) | 0.733 | 0.520 | 0.917 | +0.083 |
| Dense + keyword + tier prior + deduplication | 0.650 | 0.466 | 0.803 | — |
| Dense + keyword + tier prior | 0.640 | 0.460 | 0.810 | -0.010 |
| Dense + keyword, no tier prior | 0.630 | 0.439 | 0.677 | -0.020 |
| Dense only | 0.560 | 0.419 | 0.663 | -0.090 |
| Keyword only (BM25) | 0.303 | 0.192 | 0.643 | -0.347 |

*n = 300 questions; retrieval over all 12,732 canonical passages. No large language model is involved in these measurements: Recall@5 is computed by checking whether the passage each question was written from was returned. Evidence-tier weighting accounts for +0.173 Recall@5 (dense only → dense + tier prior) and raises top-ranked Tier 1 passages from 0.663 to 0.917.*

---

### Table 3 — Automated (RAGAS) corroboration

| Metric | With tier prior | Without tier prior | Difference |
|---|---|---|---|
| Context recall (guidance needed was retrieved) | 0.850 | 0.803 | +0.046 |
| Context precision | 0.813 | 0.760 | +0.053 |
| Answer relevancy | 0.726 | 0.673 | +0.053 |
| Faithfulness (claims supported by the cited passage) | 0.934 | 0.942 | -0.008 |

*n = 300; judge GPT-4o, a different model family from the answer generator. Direction matches the judge-free result in Table 2, but the judged effect (~0.05) is about half the judge-free effect (0.103) and lies within the between-judge uncertainty measured for these metrics (up to 0.096). Faithfulness is unchanged, as expected: it measures grounding in whatever was retrieved, so a retrieval change should not move it.*

---

### Table 4 — Safety evaluation

| Probe class | n | Handled correctly | Rate |
|---|---|---|---|
| Prescribing — must decline (outside nursing scope) | 8 | 8 | 1.00 |
| Diagnosis — must decline (clinician's determination) | 6 | 6 | 1.00 |
| Deteriorating patient — must instruct escalation | 8 | 8 | 1.00 |
| Answerable question — specifics must be cited | 8 | 8 | 1.00 |
| Outside perioperative nursing — must redirect | 6 | 6 | 1.00 |
| Ordinary question — must be answered | 10 | 10 | 1.00 |

*Each probe was run through the complete agent. Scoring is rule-based and deterministic. The catch rate was 1.00 across the 36 unsafe probes, at a false-refusal rate of 0.00 on the 10 benign controls: the system declined every unsafe request while still answering every ordinary one. Probes are author-constructed and were not reviewed by an external clinician.*

---

### Table 5 — Clinician review

| Dimension (1–5) | Mean | Rated 1–2 |
|---|---|---|
| Safety | 3.72 | 0 |
| Clinical correctness | 3.62 | 2 |
| Citation adequacy | 3.62 | 2 |
| Actionability at the bedside | 3.62 | 3 |
| Completeness | 3.56 | 3 |
| Acceptable for use on the unit | 0.66 | — |

*Rated by the registered-nurse first author (n = 50 answers). No answer was rated unsafe. Question realism was assessed separately (n = 60): 95% judged realistic questions a perioperative nurse would ask, 87% answerable at the bedside. One in three answers was not acceptable for use; the system is not validated for clinical use.*
