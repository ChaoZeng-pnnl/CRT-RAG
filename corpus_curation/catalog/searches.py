"""Search queries and DOI lists for the discovery phases."""

from __future__ import annotations


from ..config import OPENALEX_MAX_PER_QUERY, PMC_MAX_PER_QUERY


# ---------------------------------------------------------------------------
# Scraping targets — pages to spider for additional PDFs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PubMed Central (PMC) open-access search queries
# Phase 3 searches PMC with these queries and downloads free full-text PDFs.
# ---------------------------------------------------------------------------

PMC_SEARCHES = [
    {
        "query": '"enhanced recovery" gynecol* surgery[tiab]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "ERAS - Enhanced Recovery in Gynecologic Surgery",
        "subtopic": "ERAS protocol, perioperative optimization, laparoscopy",
    },
    {
        "query": '"surgical site infection" gynecol* surgery',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "SSI Prevention in Gynecologic Surgery",
        "subtopic": "Surgical site infection, wound care, antibiotic prophylaxis",
    },
    {
        "query": 'thromboembolism prophylaxis gynecol* surgery',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "VTE Prophylaxis in Gynecologic Surgery",
        "subtopic": "VTE prophylaxis, DVT, heparin, sequential compression",
    },
    {
        "query": 'postoperative pain management gynecolog* multimodal',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Postoperative Pain Management in Gynecology",
        "subtopic": "Multimodal analgesia, opioid reduction, NSAIDs, perioperative pain",
    },
    {
        "query": 'perioperative nursing gynecologic surgery patient outcomes "open access"[filter]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Perioperative Nursing in Gynecologic Surgery",
        "subtopic": "Nursing assessment, intraoperative care, patient safety",
    },
    {
        "query": 'catheter associated urinary tract infection prevention surgery[tiab] "open access"[filter]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "CAUTI Prevention in Surgical Patients",
        "subtopic": "CAUTI prevention, urinary catheter management, indwelling catheter",
    },
    {
        "query": 'postpartum haemorrhage prevention uterotonic management[tiab] "open access"[filter]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Postpartum Haemorrhage Prevention and Management",
        "subtopic": "PPH, uterotonics, oxytocin, uterine atony, haemostasis",
    },
    {
        "query": 'maternal sepsis recognition management bundle[tiab] "open access"[filter]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Maternal Sepsis Recognition and Management",
        "subtopic": "Maternal sepsis, early recognition, sepsis six, sepsis bundle",
    },
    {
        "query": '"perioperative fluid" surgery gynecol*',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Perioperative Fluid Management in Gynecologic Surgery",
        "subtopic": "Goal-directed fluid therapy, intraoperative fluid balance",
    },
    {
        "query": 'patient safety handoff communication perioperative operating room[tiab] "open access"[filter]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Patient Safety and Handoff Communication in Perioperative Care",
        "subtopic": "SBAR, handoff, clinical communication, care transitions",
    },
    {
        "query": 'wound care management postoperative gynecologic surgery[tiab] "open access"[filter]',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Postoperative Wound Care in Gynecologic Surgery",
        "subtopic": "Wound assessment, dressing, drain management, healing",
    },
    {
        "query": 'preoperative assessment gynecolog* risk stratification',
        "max_results": PMC_MAX_PER_QUERY,
        "topic": "Preoperative Assessment and Risk Stratification in Gynecology",
        "subtopic": "Preoperative workup, ASA classification, risk assessment",
    },
]

SCRAPE_TARGETS = [
    {
        "source": "RCOG",
        "tier": 1,
        "url": "https://www.rcog.org.uk/guidance/browse-all-guidance/green-top-guidelines/",
        "dest_subdir": "tier1/RCOG",
        "keywords": [
            "hysterectomy", "perioperative", "laparoscop", "gynaecolog",
            "thromboprophylaxis", "haemorrhage", "endometriosis", "ovarian",
        ],
    },
    {
        "source": "RNAO",
        "tier": 2,
        "url": "https://rnao.ca/bpg/guidelines",
        "dest_subdir": "tier2/RNAO",
        "keywords": [
            "pain", "wound", "infection", "perioperative", "catheter",
            "urinary", "venous", "thromboembolism", "nursing practice",
        ],
    },
]

# ---------------------------------------------------------------------------
# OpenAlex / Elsevier search queries
# Used by Phase 4 (OpenAlex PDF download) and Phase 5 (Elsevier XML).
# ---------------------------------------------------------------------------

OPENALEX_SEARCHES = [
    {
        "query": "enhanced recovery after surgery ERAS gynecologic oncology guideline protocol",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "ERAS in Gynecologic/Oncology Surgery",
        "subtopic": "ERAS protocol, perioperative optimization, multimodal analgesia",
    },
    {
        "query": "perioperative nursing care gynecologic laparoscopic surgery patient outcomes",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Perioperative Nursing in Gynecologic Surgery",
        "subtopic": "Nursing assessment, intraoperative nursing, patient safety",
    },
    {
        "query": "postpartum hemorrhage prevention management uterotonic oxytocin guideline",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Postpartum Haemorrhage Prevention and Management",
        "subtopic": "PPH, uterotonics, oxytocin, uterine atony, tranexamic acid",
    },
    {
        "query": "venous thromboembolism VTE prophylaxis prevention gynecologic surgery hysterectomy",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "VTE Prophylaxis in Gynecologic Surgery",
        "subtopic": "VTE prevention, DVT, LMWH, sequential compression device",
    },
    {
        "query": "surgical site infection SSI prevention gynecologic surgery antibiotic prophylaxis",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "SSI Prevention in Gynecologic Surgery",
        "subtopic": "Antibiotic prophylaxis, wound infection, aseptic technique",
    },
    {
        "query": "multimodal analgesia perioperative pain management gynecologic abdominal surgery",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Perioperative Pain Management in Gynecology",
        "subtopic": "Multimodal analgesia, opioid reduction, NSAIDs, ketamine, TAP block",
    },
    {
        "query": "catheter associated urinary tract infection CAUTI prevention indwelling catheter surgical",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "CAUTI Prevention in Surgical Patients",
        "subtopic": "CAUTI prevention, urinary catheter bundle, early removal",
    },
    {
        "query": "maternal sepsis obstetric infection recognition management clinical bundle",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Maternal Sepsis Recognition and Management",
        "subtopic": "Sepsis bundle, early recognition, sepsis six, antibiotics",
    },
    {
        "query": "perioperative intraoperative fluid management goal directed therapy surgery",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Perioperative Fluid Management and Goal-Directed Therapy",
        "subtopic": "Goal-directed fluid therapy, fluid balance, crystalloids, cardiac output",
    },
    {
        "query": "patient safety handoff communication SBAR TeamSTEPPS perioperative operating room",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Patient Safety and Handoff Communication in Perioperative Care",
        "subtopic": "SBAR, TeamSTEPPS, communication tools, handoff, care transitions",
    },
    {
        "query": "postoperative wound care dressing management surgical patients nursing",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Postoperative Wound Care Management",
        "subtopic": "Wound assessment, dressing change, drain management, healing",
    },
    {
        "query": "preoperative nursing assessment risk stratification elective surgery preparation",
        "max_results": OPENALEX_MAX_PER_QUERY,
        "topic": "Preoperative Nursing Assessment and Preparation",
        "subtopic": "Preoperative workup, ASA classification, patient education",
    },
]

# ---------------------------------------------------------------------------
# Phase 6 — Unpaywall DOI list
# Key society guidelines fetched via the Unpaywall API (OA version lookup).
# ---------------------------------------------------------------------------

UNPAYWALL_DOIS = [
    {
        "doi":          "10.1136/ijgc-2019-000356",
        "source":       "ERAS_Society",
        "tier":         1,
        "year":         2019,
        "topic":        "ERAS Guidelines for Gynecologic/Oncology Surgery (2019 Update)",
        "title":        "Guidelines for perioperative care in gynecologic/oncology: Enhanced Recovery After Surgery (ERAS) Society recommendations — 2019 update",
        "subtopic":     "ERAS protocol, gynecologic oncology, perioperative optimization, multimodal analgesia",
        "document_type": "consensus guideline",
        "trust_level":  "Tier 1 — Highest Authority",
        "retrieval_priority": "critical",
        "filename":     "ERAS_Society_Gynecologic_Oncology_Guidelines_2019.pdf",
        "dest_subdir":  "tier1/ERAS",
    },
    {
        "doi":          "10.1097/IGC.0000000000000883",
        "source":       "ERAS_Society",
        "tier":         1,
        "year":         2016,
        "topic":        "ERAS Guidelines for Gynecologic/Oncology Surgery (2016)",
        "title":        "Guidelines for Perioperative Care in Gynecologic/Oncology Surgery: Enhanced Recovery After Surgery (ERAS) Society Recommendations",
        "subtopic":     "ERAS protocol, gynecologic oncology, perioperative optimization, recovery",
        "document_type": "consensus guideline",
        "trust_level":  "Tier 1 — Highest Authority",
        "retrieval_priority": "critical",
        "filename":     "ERAS_Society_Gynecologic_Oncology_Guidelines_2016.pdf",
        "dest_subdir":  "tier1/ERAS",
    },
    {
        "doi":          "10.1111/1471-0528.14013",
        "source":       "ERAS_Society",
        "tier":         1,
        "year":         2016,
        "topic":        "ERAS Recommendations for Enhanced Recovery after Laparoscopic Gynaecological Surgery",
        "title":        "ERAS Society recommendations for enhanced recovery after laparoscopic gynaecological surgery",
        "subtopic":     "Laparoscopic hysterectomy, ERAS, laparoscopy, perioperative care, recovery",
        "document_type": "consensus guideline",
        "trust_level":  "Tier 1 — Highest Authority",
        "retrieval_priority": "critical",
        "filename":     "ERAS_Society_Laparoscopic_Gynecology_2016.pdf",
        "dest_subdir":  "tier1/ERAS",
    },
]
