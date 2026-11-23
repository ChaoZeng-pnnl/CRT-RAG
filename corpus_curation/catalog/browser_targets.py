"""Documents that require a headless browser.

Each entry failed plain-HTTP retrieval for a specific, verified reason recorded
in `why`. These are not guesses: every URL here was confirmed to download a
valid document through Chromium on 2026-09-05.
"""

from __future__ import annotations

BROWSER_TARGETS = [
    {
        "source": "CDC",
        "tier": 1,
        "year": 2009,
        "topic": "CDC Guideline for Prevention of Catheter-Associated Urinary Tract Infections",
        "title": "Guideline for Prevention of Catheter-Associated Urinary Tract Infections (2009)",
        "document_type": "clinical guideline",
        "url": "https://www.cdc.gov/infection-control/media/pdfs/Guideline-CAUTI-H.pdf",
        "referer": "https://www.cdc.gov/infection-control/hcp/cauti/index.html",
        "filename": "CDC_CAUTI_Prevention_Guidelines_2009.pdf",
        "dest_subdir": "tier1/CDC",
        "why": "www.cdc.gov returns HTTP 403 to requests; serves normally to a browser.",
    },
    {
        "source": "ACOG",
        "tier": 1,
        "year": 2018,
        "topic": "ACOG Committee Opinion 750 - Perioperative Pathways: Enhanced Recovery After Surgery",
        "title": "Perioperative Pathways: Enhanced Recovery After Surgery (Committee Opinion 750)",
        "document_type": "clinical guideline",
        "url": "https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2018/09/perioperative-pathways-enhanced-recovery-after-surgery",
        "filename": "ACOG_ERAS_Committee_Opinion_2018.html",
        "dest_subdir": "tier1/ACOG",
        "save_rendered_html": True,
        "why": "ACOG serves a branded 404 to non-browser clients. The article's own "
               "'Download PDF' link returns an HTML viewer wrapper, so the rendered "
               "article HTML is the usable full text.",
    },
    {
        "source": "RNAO",
        "tier": 2,
        "year": 2023,
        "topic": "RNAO BPG - Transitions in Care and Services",
        "title": "Transitions in Care and Services (Best Practice Guideline)",
        "document_type": "clinical practice guideline",
        "url": "https://rnao.ca/media/7525/download",
        "referer": "https://rnao.ca/bpg/guidelines/transitions-in-care",
        "filename": "RNAO_BPG_Care_Transitions.pdf",
        "dest_subdir": "tier2/RNAO",
        "click_selector": 'a[href="/media/7525/download"]',
        "why": "Download endpoint sits behind a Cloudflare interstitial; requires full "
               "Chromium and an in-page click, not direct navigation.",
    },
    {
        "source": "AHRQ_PSNet",
        "tier": 3,
        "year": 2023,
        "topic": "AHRQ TeamSTEPPS 3.0 Pocket Guide",
        "title": "TeamSTEPPS 3.0 Pocket Guide",
        "document_type": "educational curriculum",
        "url": "https://www.ahrq.gov/sites/default/files/wysiwyg/teamstepps-program/teamstepps-pocket-guide.pdf",
        "referer": "https://www.ahrq.gov/teamstepps-program/resources/pocket-guide/index.html",
        "filename": "AHRQ_TeamSTEPPS_Pocket_Guide.pdf",
        "dest_subdir": "tier3/AHRQ_PSNet",
        "why": "AHRQ restructured for TeamSTEPPS 3.0; the PDF link only appears in the "
               "client-rendered page, and Chromium serves it inline.",
    },
    {
        "source": "AHRQ_PSNet",
        "tier": 3,
        "year": 2023,
        "topic": "AHRQ TeamSTEPPS Module 1 - Communication (SBAR)",
        "title": "TeamSTEPPS Module 1: Communication",
        "document_type": "educational curriculum",
        "url": "https://www.ahrq.gov/teamstepps-program/curriculum/communication/index.html",
        "filename": "AHRQ_TeamSTEPPS_SBAR_Communication_Module.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
        "save_rendered_html": True,
        "why": "Client-rendered curriculum page; the standalone SBAR module PDF no "
               "longer exists under TeamSTEPPS 3.0.",
    },
    {
        "source": "ERAS_Society",
        "tier": 1,
        "year": 2016,
        "topic": "ERAS Guidelines for Pre- and Intra-operative Care in Gynecologic/Oncology Surgery (Part I)",
        "title": "Guidelines for pre- and intra-operative care in gynecologic/oncology surgery: "
                 "Enhanced Recovery After Surgery (ERAS) Society recommendations - Part I",
        "document_type": "clinical guideline",
        "subtopic": "ERAS protocol, preoperative optimization, intraoperative care",
        "url": "https://iris.unil.ch/bitstreams/519735e8-66b9-4709-8296-e1d1da843b15/download",
        "filename": "ERAS_GynOnc_PartI_Pre_Intraoperative_2016.pdf",
        "dest_subdir": "tier1/ERAS",
        "why": "Elsevier/publisher PDF is bot-blocked; the UNIL institutional repository "
               "hosts the OA copy (DOI 10.1016/j.ygyno.2015.11.015).",
    },
    {
        "source": "ERAS_Society",
        "tier": 1,
        "year": 2016,
        "topic": "ERAS Guidelines for Postoperative Care in Gynecologic/Oncology Surgery (Part II)",
        "title": "Guidelines for postoperative care in gynecologic/oncology surgery: "
                 "Enhanced Recovery After Surgery (ERAS) Society recommendations - Part II",
        "document_type": "clinical guideline",
        "subtopic": "ERAS protocol, postoperative care, early mobilization, analgesia",
        "url": "https://iris.unil.ch/bitstreams/564ee3ef-7cbe-4c78-b602-575b0e8da295/download",
        "filename": "ERAS_GynOnc_PartII_Postoperative_2016.pdf",
        "dest_subdir": "tier1/ERAS",
        "why": "Elsevier/publisher PDF is bot-blocked; the UNIL institutional repository "
               "hosts the OA copy (DOI 10.1016/j.ygyno.2015.12.019).",
    },
]
