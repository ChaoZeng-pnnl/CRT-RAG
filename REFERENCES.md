# Related work — with verification status

**✅ VERIFIED** = DOI/PMID confirmed against Crossref or PubMed during this session.
**⚠ UNVERIFIED** = identifier not confirmed here — a book, or a version question.
**Check each ⚠ before submission.** All journal articles, conference papers and
arXiv preprints below have been verified against Crossref, PubMed, OpenAlex or the
arXiv API during this session.

---

## 1. Evidence hierarchy / EBP models — the paper's theoretical anchor

Use these to frame tier-weighted retrieval as *the evidence hierarchy made
machine-actionable*. This is what makes the work read as nursing scholarship.

- ✅ **Iowa Model Collaborative (2017).** Iowa Model of Evidence-Based Practice:
  Revisions and Validation. *Worldviews on Evidence-Based Nursing.*
  `10.1111/wvn.12223`
- ✅ **Jordan Z, Lockwood C, Munn Z, Aromataris E (2018).** Redeveloping the JBI
  Model of Evidence Based Healthcare. *Int J Evid Based Healthc.*
  `10.1097/xeb.0000000000000139`
- ✅ **Pearson A, Wiechula R, Court A, Lockwood C (2005).** The JBI model of
  evidence-based healthcare. *Int J Evid Based Healthc.*
  `10.1097/01258363-200509000-00001`
- ✅ **Guidelines for evidence-based clinical practice (2000).** *Nursing Outlook.*
  `10.1016/s0029-6554(00)90003-4`
- ⚠ **Melnyk BM, Fineout-Overholt E.** *Evidence-Based Practice in Nursing &
  Healthcare* (book) — for the levels-of-evidence pyramid nurses are taught.

## 2. Failure to rescue & nursing surveillance — anchors the safety section

Frame the escalation probe class as *failure to rescue*, a nursing-sensitive
quality indicator, rather than as generic LLM safety.

- ✅ **Kelly L, Vincent D (2011).** The Impact of Nursing Surveillance on Failure
  to Rescue. *Research and Theory for Nursing Practice.* `10.1891/1541-6577.25.2.107`
- ✅ **Kelly L, Vincent D (2010).** The dimensions of nursing surveillance: a
  concept analysis. *Journal of Advanced Nursing.* `10.1111/j.1365-2648.2010.05525.x`
- ✅ **Failure-to-Rescue After Acute Myocardial Infarction (2018).** *Medical Care.*
  PMID `29578952`
- ✅ **Lasater KB et al. (2021).** Valuing hospital investments in nursing:
  multistate matched-cohort study. *BMJ Quality & Safety.* PMID `32220938`
- ✅ **Silber JH, Williams SV, Krakauer H, Schwartz JS (1992).** Hospital and
  patient characteristics associated with death after surgery: a study of failure
  to rescue. *Medical Care* 30(7):615-29. PMID `1614231` — **the origin of the
  concept**; cite when introducing the escalation probe class.
- ✅ **Silber JH et al. (1995).** Evaluation of the complication rate as a measure
  of quality of care in coronary artery bypass graft surgery. *JAMA* 274(4):317-23.
  PMID `7609261`
- ✅ **Aiken LH, Clarke SP, Sloane DM, Sochalski J, Silber JH (2002).** Hospital
  nurse staffing and patient mortality, nurse burnout, and job dissatisfaction.
  *JAMA* 288(16):1987-93. `10.1001/jama.288.16.1987` (PMID `12387650`)
- ✅ **Aiken LH et al. (2003).** Educational levels of hospital nurses and surgical
  patient mortality. *JAMA* 290(12):1617-23. `10.1001/jama.290.12.1617` (PMID `14506121`)

## 3. Clinical RAG — positions the system

- ✅ **Zakka C et al. (2023).** Almanac: Retrieval-Augmented Language Models for
  Clinical Medicine. Preprint `10.21203/rs.3.rs-2883198/v1`.
  ⚠ *Published version appeared in NEJM AI (2024) — cite the journal version.*
- ✅ **Xiong G, Jin Q, Lu Z, Zhang A (2024).** Benchmarking Retrieval-Augmented
  Generation for Medicine (MedRAG / MIRAGE). *Findings of ACL 2024.*
  `10.18653/v1/2024.findings-acl.372` (arXiv:2402.13178, verified) — the closest prior work
  on clinical RAG evaluation; position our nursing-specific corpus against it.
- ✅ **Lewis P, Perez E, et al. (2020).** Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks. *NeurIPS 2020.* (indexed via UCL Discovery;
  arXiv:2005.11401, verified) — the RAG origin paper.

## 4. Evaluation methodology — supports the judge-calibration contribution

- ✅ **Es S, James J, Espinosa-Anke L, Schockaert S (2024).** RAGAs: Automated
  Evaluation of Retrieval Augmented Generation. *EACL 2024 (demos).*
  `10.18653/v1/2024.eacl-demo.16`
- ✅ **Zheng L, Chiang W-L, et al. (2023).** Judging LLM-as-a-Judge with MT-Bench
  and Chatbot Arena. *NeurIPS Datasets & Benchmarks Track.*
  `10.48550/arXiv.2306.05685` (arXiv:2306.05685, verified) — **the key citation for position bias and
  self-preference**, and the justification for our position-randomised pairwise
  judging and cross-family judge design.
- ✅ **Gao Y, Xiong Y, et al. (2023).** Retrieval-Augmented Generation for Large
  Language Models: A Survey. `10.48550/arXiv.2312.10997` — general RAG background.
- ⚠ Optionally pair our 0.81-vs-0.24 proxy result with a clinical LLM-judge
  reliability study; the *Nursing Inquiry* paper in section 8 already serves this
  role from the nursing side.

## 5. GraphRAG — for the companion paper

- ✅ **Edge D, Trinh H, Cheng N, et al. (2024).** From Local to Global: A Graph RAG
  Approach to Query-Focused Summarization. arXiv:2404.16130 (submitted 2024-04-24)
  — verified by direct arXiv `id_list` lookup. **Our divergence argument is stated
  against this paper:** GraphRAG's demonstrated advantage is *global* sensemaking
  over narrative, redundant corpora (podcast transcripts, news), whereas clinical
  guidelines are curated and structured — 99% of our chunks carry section headings
  and only 5.4% are near-duplicates, so the graph recovers structure the documents
  already have.
- ✅ Related recent work found in Crossref, useful for showing the area is active:
  Context-Aware Fine-Grained Graph RAG for Query-Focused Summarization (2025)
  `10.1145/3746252.3760935`

## 6. Domain guidance already in the corpus — cite as sources, not related work

NICE NG180 (perioperative care), NICE NG125 (SSI), NICE NG89 (VTE), WHO Global
Guidelines for SSI Prevention, WHO Surgical Safety Checklist, RCOG GTG-52 (PPH),
RCOG GTG-37a/b (thromboembolism), CDC CAUTI (2009), CDC SSI (2017), ERAS Society
gynecologic/oncology guidelines (2016 Parts I & II; 2019 update),
AUGS-IUGA ERAS statement (2022). Full metadata with DOIs/URLs in
`corpus/metadata/corpus_metadata.json`.

---

## 7. Nursing information needs at the point of care — supports the 92% framing

Cite these so the "perioperative nursing questions are point-of-care" claim rests
on prior evidence, not only on our own generated test set.

- ✅ **The information needs of perioperative staff: a preparatory study for a
  proposed specialist library (2005).** *Health Information and Libraries Journal.*
  `10.1111/j.1471-1842.2005.00535.x` (PMID `15810930`) — **directly on point**:
  perioperative staff information needs.
- ✅ **Preventing surgical site infections: facilitators and barriers to nurses'
  adherence to clinical practice guidelines (2019).** *Journal of Clinical Nursing.*
  `10.1111/jocn.14766` (PMID `30589979`) — evidence that guideline *access and
  adherence* at the bedside is a real problem, which is the gap the system addresses.

## 8. AI in nursing practice — anchors the governance implication

Use these so "AI-validated is not a clinical safety claim" lands in the nursing
literature rather than only the ML literature.

- ✅ **Competing Framings: Variability in AI-Generated Health Policy Guidance
  (2026).** *Nursing Inquiry.* `10.1111/nin.70162` — variability in AI output for
  nursing-relevant guidance; pairs directly with our 0.81-vs-0.24 proxy result.
- ✅ **Evaluating the Accuracy, Empathy, and Readability of Generative AI Versus
  Registered Nurses (2026).** *Nursing Open.* `10.1002/nop2.70777` — a nursing
  evaluation of generative AI against nurses; the comparator our work lacks.
- ✅ **Artificial Intelligence Technology Enhancing Emergency Department Triage:
  An Integrative Review (2026).** *CIN: Computers, Informatics, Nursing.*
  `10.1097/CIN.0000000000001612` — establishes CIN as a venue publishing this
  class of work.
- ✅ **Nursing educators' perspectives on artificial intelligence: a rapid review
  (2026).** *Nurse Education Today.* `10.1016/j.nedt.2026.107382` — for the
  education implication.

## Gaps to fill before submission

1. ~~Verify Edge et al.~~ — **done.** All four CS citations verified by direct
   arXiv `id_list` lookup (the search-syntax queries had failed; `id_list` needs no
   search syntax). Only the Melnyk & Fineout-Overholt textbook and the Almanac
   journal version remain to confirm manually.
2. Confirm the **Almanac** journal version (NEJM AI 2024) rather than the preprint
   DOI listed above.
3. ~~Nursing informatics AI governance~~ — **done**, section 8.
4. ~~Perioperative nursing information needs~~ — **done**, section 7.
