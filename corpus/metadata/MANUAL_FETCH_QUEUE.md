# Manual fetch queue — Tier 1 backfill links

Corpus status 2026-09-05: **234 files, 234 valid, 0 rejected.**
Tier 1 = 27, Tier 2 = 14, Tier 3 = 9, Tier 4 = 184.

Everything reachable by automation has been fetched. What follows needs a
person with institutional journal access. All links below were tested on
2026-09-05 — the failure mode for each is stated, so you know what to expect.

After saving any file, run:

```bash
python3 -m corpus_curation --audit
```

That verifies the file is a real document (not a login wall saved as a PDF) and
regenerates `corpus_metadata.json`.

---

## 1. ERAS Society gynecologic/oncology guidelines — 2019 update  ⭐ highest value

The single most important document for this specialty. Unpaywall reports it as
OA at the publisher, but BMJ serves 403 to every automated client including a
full headless browser; Europe PMC lists it as "subscription required" with no
PMCID.

- **DOI:** https://doi.org/10.1136/ijgc-2019-000356
- **Publisher PDF (try from a browser on an institutional network):**
  https://ijgc.bmj.com/content/ijgc/29/4/651.full.pdf
- **Article page:** https://ijgc.bmj.com/content/29/4/651
- **PubMed record (for interlibrary loan):** https://pubmed.ncbi.nlm.nih.gov/30877144/
- **Citation:** Nelson G, et al. *Int J Gynecol Cancer.* 2019;29(4):651-668.

**Save to:** `corpus/tier1/ERAS/ERAS_GynOnc_2019_Update.pdf`

> Note: the 2016 Parts I and II are **already in the corpus** (fetched from the
> UNIL open repository), so the specialty is covered even without the 2019
> update. Adding the update is worthwhile because your paper's freshness /
> recency claim benefits from the newest version of its flagship guideline.

## 2. RCOG — Enhanced Recovery in Gynaecology (Scientific Impact Paper 36)

RCOG returns 403 to automated clients, and both historical PDF paths are dead
(`/globalassets/documents/guidelines/scientific-impact-papers/sip_36.pdf`
returns an HTML page; `/media/pkjbdmm0/sip_36.pdf` returns nothing).

- **Guideline hub (navigate from here in a browser):**
  https://www.rcog.org.uk/guidance/browse-all-guidance/scientific-impact-papers/
- **Green-top listing:**
  https://www.rcog.org.uk/guidance/browse-all-guidance/green-top-guidelines/

**Save to:** `corpus/tier1/RCOG/RCOG_SIP36_Enhanced_Recovery_Gynaecology.pdf`

## 3. AUGS documents — all paywalled in *Urogynecology* (LWW)

AUGS moved its clinical guidance to journal publication. The 9 PDFs still on
augs.org are mesh/sling and patient-facing documents — none is the voiding trial
or catheter guidance. Unpaywall confirms `is_oa: false` for each of these.

The most relevant one for your topic axes is actually the ERAS statement, which
was not on the original list:

- **AUGS-IUGA Joint Clinical Consensus Statement on Enhanced Recovery After
  Urogynecologic Surgery (2022)** — https://doi.org/10.1097/spv.0000000000001252
  → `corpus/tier2/AUGS/AUGS_IUGA_ERAS_Urogynecologic_Surgery_2022.pdf`
- **AUGS Best Practice Statement: Evaluation and Counseling of Patients with
  Recurrent UTI (2018)** — https://doi.org/10.1097/spv.0000000000000566
  → `corpus/tier2/AUGS/AUGS_Recurrent_UTI_Best_Practice_2018.pdf`
- AUGS clinical guidance index (for browsing what exists):
  https://www.augs.org/clinical-patient-resources/clinical-guidance-documents/

---

## Already recovered — no action needed

| Document | How |
|---|---|
| 9 WHO guidelines | IRIS DSpace 7 REST API (`fetchers/who_iris.py`) |
| CDC CAUTI 2009 | Browser download; `www.cdc.gov` 403s `requests` but serves Chromium |
| ACOG Committee Opinion 750 | Rendered article HTML, 53 K chars |
| RNAO Transitions in Care | Full Chromium + in-page click past Cloudflare |
| AHRQ TeamSTEPPS Pocket Guide | Inline PDF captured by content type |
| ERAS gyn/onc Parts I & II (2016) | UNIL open repository via `--sources browser` |

## Known limitation of automated validation

`AUGS_Clinical_Guidance_Documents_Index.html` passes validation — it is a real
page with 10.6 K characters — but it is a **table of links, not a guideline**.
Validation proves a file is not an error page; it cannot prove the file is the
document you wanted. Screen Tier 1–2 (41 documents) by hand once before freezing
the corpus.
