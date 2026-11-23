"""Curated Tier 1-3 document catalogue."""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Curated document catalogue
# Each entry:
#   source, tier, year, topic, url, filename, dest_subdir
# ---------------------------------------------------------------------------

DOCUMENTS = [

    # ── NICE ─────────────────────────────────────────────────────────────
    {
        "source": "NICE", "tier": 1, "year": 2020,
        "topic": "Perioperative Care in Adults (NG180)",
        "url": "https://www.nice.org.uk/guidance/ng180/resources/"
               "perioperative-care-in-adults-pdf-66142014963397",
        "filename": "NICE_NG180_Perioperative_Care_Adults.pdf",
        "dest_subdir": "tier1/NICE",
    },
    {
        "source": "NICE", "tier": 1, "year": 2019,
        "topic": "Surgical Site Infections Prevention and Treatment (NG125)",
        "url": "https://www.nice.org.uk/guidance/ng125/resources/"
               "surgical-site-infections-prevention-and-treatment-pdf-66141660564421",
        "filename": "NICE_NG125_Surgical_Site_Infections.pdf",
        "dest_subdir": "tier1/NICE",
    },
    {
        "source": "NICE", "tier": 1, "year": 2019,
        "topic": "VTE Prevention in Adults (NG89)",
        "url": "https://www.nice.org.uk/guidance/ng89/resources/"
               "venous-thromboembolism-in-over-16s-reducing-the-risk-of-hospital-acquired-"
               "deep-vein-thrombosis-or-pulmonary-embolism-pdf-1837703092165",
        "filename": "NICE_NG89_VTE_Prevention.pdf",
        "dest_subdir": "tier1/NICE",
    },
    {
        "source": "NICE", "tier": 1, "year": 2018,
        "topic": "NICE NG94 - Emergency and Acute Medical Care in Adults",
        "url": "https://www.nice.org.uk/guidance/ng94/resources/"
               "emergency-and-acute-medical-care-in-over-16s-service-delivery-and-organisation-pdf-1837755160261",
        "filename": "NICE_NG94_Emergency_Acute_Medical_Care.pdf",
        "dest_subdir": "tier1/NICE",
    },

    # ── WHO ──────────────────────────────────────────────────────────────
    {
        "source": "WHO", "tier": 1, "year": 2009,
        "topic": "WHO Surgical Safety Checklist Implementation Guide",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/44186/9789241598590_eng.pdf",
        "filename": "WHO_Surgical_Safety_Checklist_Implementation.pdf",
        "dest_subdir": "tier1/WHO",
    },
    {
        "source": "WHO", "tier": 1, "year": 2016,
        "topic": "WHO Global Guidelines for Prevention of Surgical Site Infection",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/250680/9789241549882-eng.pdf",
        "filename": "WHO_Global_SSI_Prevention_Guidelines.pdf",
        "dest_subdir": "tier1/WHO",
    },
    {
        "source": "WHO", "tier": 1, "year": 2018,
        "topic": "WHO Recommendations for Uterotonics for Prevention of PPH",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/277276/9789241550420-eng.pdf",
        "filename": "WHO_Uterotonics_PPH_Prevention.pdf",
        "dest_subdir": "tier1/WHO",
    },

    # ── CDC ──────────────────────────────────────────────────────────────
    {
        "source": "CDC", "tier": 1, "year": 2017,
        "topic": "CDC Guideline for the Prevention of Surgical Site Infection 2017",
        "url": "https://stacks.cdc.gov/view/cdc/79361/cdc_79361_DS1.pdf",
        "filename": "CDC_SSI_Prevention_Guideline_2017.pdf",
        "dest_subdir": "tier1/CDC",
    },
    {
        "source": "CDC", "tier": 1, "year": 1999,
        "topic": "CDC Guideline for the Prevention of Surgical Site Infection 1999",
        "url": "https://stacks.cdc.gov/view/cdc/7160/cdc_7160_DS1.pdf",
        "filename": "CDC_SSI_Prevention_Guideline_1999.pdf",
        "dest_subdir": "tier1/CDC",
    },
    {
        "source": "CDC", "tier": 1, "year": 2009,
        "topic": "IDSA/SHEA Guidelines for Prevention of CAUTI (PMC open access)",
        # Gould et al. 2010 CAUTI guidelines, open access via PMC
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5869427/pdf/kiy215.pdf",
        "filename": "CDC_CAUTI_Prevention_Guidelines_2009.pdf",
        "dest_subdir": "tier1/CDC",
    },
    {
        "source": "CDC", "tier": 1, "year": 2002,
        "topic": "Guideline for Hand Hygiene in Health-Care Settings",
        "url": "https://www.cdc.gov/mmwr/pdf/rr/rr5116.pdf",
        "filename": "CDC_Hand_Hygiene_Guideline_2002.pdf",
        "dest_subdir": "tier1/CDC",
    },

    # ── RCOG ─────────────────────────────────────────────────────────────
    {
        "source": "RCOG", "tier": 1, "year": 2015,
        "topic": "GTG37a - Reducing Risk of Thrombosis and Embolism during Pregnancy",
        "url": "https://www.rcog.org.uk/media/m4mbpjwi/gtg-no37a-2015_amended-2023.pdf",
        "filename": "RCOG_GTG37a_Thrombosis_Embolism.pdf",
        "dest_subdir": "tier1/RCOG",
    },
    {
        "source": "RCOG", "tier": 1, "year": 2016,
        "topic": "GTG52 - Prevention and Management of Postpartum Haemorrhage",
        # Wiley BJOG journal; use TDM API with institutional token
        "url": "https://obgyn.onlinelibrary.wiley.com/doi/10.1111/1471-0528.14178",
        "wiley_doi": "10.1111/1471-0528.14178",
        "filename": "RCOG_GTG52_PPH_Prevention_Management.pdf",
        "dest_subdir": "tier1/RCOG",
    },
    {
        "source": "RCOG", "tier": 1, "year": 2018,
        "topic": "GTG37b - Thrombosis and Embolism during Pregnancy Acute Management",
        "url": "https://www.rcog.org.uk/media/wj2lpco5/gtg-37b-1.pdf",
        "filename": "RCOG_GTG37b_Thrombosis_Acute_Management.pdf",
        "dest_subdir": "tier1/RCOG",
    },
    {
        "source": "RCOG", "tier": 1, "year": 2024,
        "topic": "GTG64 - Identification and Management of Maternal Sepsis",
        # Wiley BJOG journal; use TDM API with institutional token
        "url": "https://obgyn.onlinelibrary.wiley.com/doi/10.1111/1471-0528.18009",
        "wiley_doi": "10.1111/1471-0528.18009",
        "filename": "RCOG_GTG64_Maternal_Sepsis.pdf",
        "dest_subdir": "tier1/RCOG",
    },

    # ── ACOG ─────────────────────────────────────────────────────────────
    # Most ACOG PBs require membership; these are freely available pages/summaries
    {
        "source": "ACOG", "tier": 1, "year": 2018,
        "topic": "ACOG Committee Opinion - Enhanced Recovery After Surgery",
        "url": "https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2018/12/enhanced-recovery-after-surgery",
        "filename": "ACOG_ERAS_Committee_Opinion_2018.html",
        "dest_subdir": "tier1/ACOG",
    },

    # ── RNAO ─────────────────────────────────────────────────────────────
    {
        "source": "RNAO", "tier": 2, "year": 2025,
        "topic": "Pain: Prevention, Assessment and Management (4th ed) - Fact Sheet",
        # Full BPG PDF requires clicking DOWNLOAD button; use fact sheet and page HTML instead
        "url": "https://rnao.ca/media/8382/download?attachment",
        "filename": "RNAO_BPG_Pain_4th_ed_FactSheet.pdf",
        "dest_subdir": "tier2/RNAO",
    },
    {
        "source": "RNAO", "tier": 2, "year": 2025,
        "topic": "Pain: Prevention, Assessment and Management (4th ed) - BPG Overview Page",
        "url": "https://rnao.ca/bpg/guidelines/assessment-and-management-pain",
        "filename": "RNAO_BPG_Pain_4th_ed_Overview.html",
        "dest_subdir": "tier2/RNAO",
    },
    {
        "source": "RNAO", "tier": 2, "year": 2016,
        "topic": "Care Transitions (Transitions in Care)",
        "url": "https://rnao.ca/sites/rnao-ca/files/Care_Transitions_BPG.pdf",
        "filename": "RNAO_BPG_Care_Transitions.pdf",
        "dest_subdir": "tier2/RNAO",
    },
    {
        "source": "RNAO", "tier": 2, "year": 2011,
        "topic": "Nursing Management of Hypertension",
        "url": "https://rnao.ca/sites/rnao-ca/files/Nursing_Management_of_Hypertension.pdf",
        "filename": "RNAO_BPG_Nursing_Management_Hypertension.pdf",
        "dest_subdir": "tier2/RNAO",
    },

    # ── OpenRN ───────────────────────────────────────────────────────────
    {
        "source": "OpenRN", "tier": 2, "year": 2023,
        "topic": "OpenRN Nursing Fundamentals 2e - Full Textbook (PDF)",
        "url": "https://wtcs.pressbooks.pub/nursingfundamentals/open/download?type=pdf",
        "filename": "OpenRN_Nursing_Fundamentals_2e.pdf",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2023,
        "topic": "OpenRN Nursing Fundamentals - Comfort and Pain Chapter",
        "url": "https://wtcs.pressbooks.pub/nursingfundamentals/chapter/11-4-pain-management/",
        "filename": "OpenRN_Nursing_Fundamentals_Pain_Management.html",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2023,
        "topic": "OpenRN Nursing Fundamentals - Infection Prevention Chapter",
        "url": "https://wtcs.pressbooks.pub/nursingfundamentals/chapter/9-6-preventing-infection/",
        "filename": "OpenRN_Nursing_Fundamentals_Infection_Prevention.html",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2023,
        "topic": "OpenRN Nursing Fundamentals - Elimination and Urinary Chapter",
        "url": "https://wtcs.pressbooks.pub/nursingfundamentals/chapter/16-5-urinary-retention/",
        "filename": "OpenRN_Nursing_Fundamentals_Urinary_Retention.html",
        "dest_subdir": "tier2/OpenRN",
    },

    # ── AUGS ─────────────────────────────────────────────────────────────
    {
        "source": "AUGS", "tier": 2, "year": 2019,
        "topic": "AUGS/SUFU Consensus Statement - Postoperative Voiding Trial (page)",
        "url": "https://www.augs.org/clinical-patient-resources/clinical-guidance-documents/",
        "filename": "AUGS_Clinical_Guidance_Documents_Index.html",
        "dest_subdir": "tier2/AUGS",
    },

    # ── AHRQ PSNet ───────────────────────────────────────────────────────
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2019,
        "topic": "AHRQ TeamSTEPPS Pocket Guide - Team Communication / SBAR",
        "url": "https://www.ahrq.gov/teamstepps/instructor/essentials/pocketguide.pdf",
        "filename": "AHRQ_TeamSTEPPS_Pocket_Guide.pdf",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2020,
        "topic": "AHRQ TeamSTEPPS 2.0 Course Management Guide",
        "url": "https://www.ahrq.gov/sites/default/files/wysiwyg/teamstepps/instructor/fundamentals/module6/slsbar.pdf",
        "filename": "AHRQ_TeamSTEPPS_SBAR_Module.pdf",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2018,
        "topic": "AHRQ Preventing Surgical Site Infections - Patient Safety Primer",
        "title": "Patient Safety Primer: Surgical Site Infections",
        "subtopic": "SSI prevention, postoperative wound infection",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "high",
        "url": "https://psnet.ahrq.gov/primer/surgical-site-infections",
        "filename": "AHRQ_PSNet_SSI_Patient_Safety_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # EXPANSION BATCH — added for corpus coverage breadth
    # ═══════════════════════════════════════════════════════════════════════

    # ── NICE expansion ────────────────────────────────────────────────────
    {
        "source": "NICE", "tier": 1, "year": 2025,
        "topic": "NG253 - Suspected Sepsis in Adults ≥16: Recognition, Assessment and Early Management",
        "title": "Suspected Sepsis in People Aged 16 or Over: Recognition, Assessment and Early Management",
        "subtopic": "Sepsis recognition, early warning, escalation of care",
        "document_type": "clinical guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "critical",
        "url": ("https://www.nice.org.uk/guidance/ng253/resources/"
                "suspected-sepsis-in-people-aged-16-or-over-recognition-assessment-and-"
                "early-management-pdf-66144015386053"),
        "filename": "NICE_NG253_Suspected_Sepsis_Adults.pdf",
        "dest_subdir": "tier1/NICE",
    },
    {
        "source": "NICE", "tier": 1, "year": 2025,
        "topic": "NG255 - Suspected Sepsis in Pregnant/Recently Pregnant People",
        "title": "Suspected Sepsis in Pregnant or Recently Pregnant People: Recognition, Diagnosis and Early Management",
        "subtopic": "Maternal sepsis, obstetric sepsis, postpartum infection",
        "document_type": "clinical guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "critical",
        "url": ("https://www.nice.org.uk/guidance/ng255/resources/"
                "suspected-sepsis-in-pregnant-or-recently-pregnant-people-recognition-"
                "diagnosis-and-early-management-pdf-66144018745285"),
        "filename": "NICE_NG255_Suspected_Sepsis_Pregnant.pdf",
        "dest_subdir": "tier1/NICE",
    },
    {
        "source": "NICE", "tier": 1, "year": 2016,
        "topic": "NG45 - Routine Preoperative Tests for Elective Surgery",
        "title": "Routine Preoperative Tests for Elective Surgery",
        "subtopic": "Preoperative assessment, ASA grading, preoperative testing",
        "document_type": "clinical guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "high",
        "url": ("https://www.nice.org.uk/guidance/ng45/resources/"
                "routine-preoperative-tests-for-elective-surgery-pdf-1837454508997"),
        "filename": "NICE_NG45_Routine_Preoperative_Tests.pdf",
        "dest_subdir": "tier1/NICE",
    },

    # ── WHO expansion ─────────────────────────────────────────────────────
    {
        "source": "WHO", "tier": 1, "year": 2011,
        "topic": "WHO Patient Safety Curriculum Guide: Multi-professional Edition",
        "title": "Patient Safety Curriculum Guide: Multi-professional Edition",
        "subtopic": "Patient safety education, teamwork, communication, human factors",
        "document_type": "educational curriculum",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "high",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/44641/9789241501958_eng.pdf",
        "filename": "WHO_Patient_Safety_Curriculum_Guide.pdf",
        "dest_subdir": "tier1/WHO",
    },
    {
        "source": "WHO", "tier": 1, "year": 2009,
        "topic": "WHO Safe Surgery Saves Lives - Second Global Patient Safety Challenge",
        "title": "Safe Surgery Saves Lives: The Second Global Patient Safety Challenge",
        "subtopic": "Surgical safety, WHO Surgical Safety Checklist background and evidence",
        "document_type": "clinical guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "high",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/44185/9789241598552_eng.pdf",
        "filename": "WHO_Safe_Surgery_Saves_Lives.pdf",
        "dest_subdir": "tier1/WHO",
    },

    # ── OpenRN Nursing Skills 2e ──────────────────────────────────────────
    {
        "source": "OpenRN", "tier": 2, "year": 2024,
        "topic": "OpenRN Nursing Skills 2e - Full Textbook (PDF)",
        "title": "Nursing Skills 2e",
        "subtopic": "Clinical skills: aseptic technique, wound care, catheter care, IV therapy",
        "document_type": "open-access textbook",
        "trust_level": "Tier 2 — Nursing Best Practice",
        "retrieval_priority": "high",
        "url": "https://wtcs.pressbooks.pub/nursingskills/open/download?type=pdf",
        "filename": "OpenRN_Nursing_Skills_2e.pdf",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2024,
        "topic": "OpenRN Nursing Skills 2e - Ch 4: Aseptic Technique",
        "title": "Nursing Skills 2e — Chapter 4: Aseptic Technique",
        "subtopic": "Hand hygiene, sterile fields, PPE, sterile gloves",
        "document_type": "open-access textbook chapter",
        "trust_level": "Tier 2 — Nursing Best Practice",
        "retrieval_priority": "high",
        "url": "https://wtcs.pressbooks.pub/nursingskills/chapter/4-1-asceptic-technique-introduction/",
        "filename": "OpenRN_Nursing_Skills_Aseptic_Technique.html",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2024,
        "topic": "OpenRN Nursing Skills 2e - Ch 20: Wound Care",
        "title": "Nursing Skills 2e — Chapter 20: Wound Care",
        "subtopic": "Wound assessment, dressing change, drain management, suture removal, staple removal",
        "document_type": "open-access textbook chapter",
        "trust_level": "Tier 2 — Nursing Best Practice",
        "retrieval_priority": "high",
        "url": "https://wtcs.pressbooks.pub/nursingskills/chapter/20-1-wound-care-introduction/",
        "filename": "OpenRN_Nursing_Skills_Wound_Care.html",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2024,
        "topic": "OpenRN Nursing Skills 2e - Ch 21: Facilitation of Elimination / Catheter Care",
        "title": "Nursing Skills 2e — Chapter 21: Facilitation of Elimination",
        "subtopic": "Foley catheter insertion, CAUTI prevention, catheter removal, ostomy care",
        "document_type": "open-access textbook chapter",
        "trust_level": "Tier 2 — Nursing Best Practice",
        "retrieval_priority": "high",
        "url": "https://wtcs.pressbooks.pub/nursingskills/chapter/21-1-facilitation-of-elimination-introduction/",
        "filename": "OpenRN_Nursing_Skills_Catheter_Care.html",
        "dest_subdir": "tier2/OpenRN",
    },
    {
        "source": "OpenRN", "tier": 2, "year": 2024,
        "topic": "OpenRN Nursing Skills 2e - Ch 23: IV Therapy Management",
        "title": "Nursing Skills 2e — Chapter 23: IV Therapy Management",
        "subtopic": "IV therapy, fluid management, IV site assessment, complications",
        "document_type": "open-access textbook chapter",
        "trust_level": "Tier 2 — Nursing Best Practice",
        "retrieval_priority": "medium",
        "url": "https://wtcs.pressbooks.pub/nursingskills/chapter/23-1-iv-therapy-management-introduction/",
        "filename": "OpenRN_Nursing_Skills_IV_Therapy.html",
        "dest_subdir": "tier2/OpenRN",
    },

    # ── AHRQ PSNet expansion ──────────────────────────────────────────────
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2025,
        "topic": "AHRQ PSNet Primer - Failure to Rescue",
        "title": "Patient Safety Primer: Failure to Rescue",
        "subtopic": "Postoperative deterioration, nursing surveillance, rapid response, patient safety",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "high",
        "url": "https://psnet.ahrq.gov/primer/failure-rescue",
        "filename": "AHRQ_PSNet_Failure_to_Rescue_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2024,
        "topic": "AHRQ PSNet Primer - Handoffs and Signouts",
        "title": "Patient Safety Primer: Handoffs and Signouts",
        "subtopic": "Clinical handoffs, SBAR, I-PASS, perioperative care transitions",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "high",
        "url": "https://psnet.ahrq.gov/primer/handoffs-and-signouts",
        "filename": "AHRQ_PSNet_Handoffs_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2023,
        "topic": "AHRQ PSNet Primer - Safety Culture",
        "title": "Patient Safety Primer: Safety Culture",
        "subtopic": "Safety culture, just culture, non-punitive reporting, perioperative environment",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "high",
        "url": "https://psnet.ahrq.gov/primer/culture-safety",
        "filename": "AHRQ_PSNet_Safety_Culture_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2023,
        "topic": "AHRQ PSNet Primer - Medication Errors and Adverse Drug Events",
        "title": "Patient Safety Primer: Medication Errors and Adverse Drug Events",
        "subtopic": "Medication safety, perioperative medications, adverse drug events",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "medium",
        "url": "https://psnet.ahrq.gov/primer/medication-errors-and-adverse-drug-events",
        "filename": "AHRQ_PSNet_Medication_Errors_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2023,
        "topic": "AHRQ PSNet Primer - Patient Falls",
        "title": "Patient Safety Primer: Patient Falls",
        "subtopic": "Falls prevention, postoperative mobility, inpatient safety",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "medium",
        "url": "https://psnet.ahrq.gov/primer/falls",
        "filename": "AHRQ_PSNet_Patient_Falls_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },
    {
        "source": "AHRQ_PSNet", "tier": 3, "year": 2024,
        "topic": "AHRQ PSNet Primer - Rapid Response Systems",
        "title": "Patient Safety Primer: Rapid Response Systems",
        "subtopic": "Rapid response teams, deterioration response, early warning systems",
        "document_type": "patient safety primer",
        "trust_level": "Tier 3 — Reflective Learning & Safety",
        "retrieval_priority": "high",
        "url": "https://psnet.ahrq.gov/primer/rapid-response-systems",
        "filename": "AHRQ_PSNet_Rapid_Response_Systems_Primer.html",
        "dest_subdir": "tier3/AHRQ_PSNet",
    },

    # ── WHO additional guidelines ─────────────────────────────────────────
    {
        "source": "WHO", "tier": 1, "year": 2012,
        "topic": "WHO Recommendations for Prevention and Treatment of Postpartum Haemorrhage",
        "title": "WHO Recommendations for the Prevention and Treatment of Postpartum Haemorrhage",
        "subtopic": "PPH prevention, active management of third stage of labour, uterotonics, tranexamic acid",
        "document_type": "clinical practice guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "critical",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/75411/9789241548502_eng.pdf",
        "filename": "WHO_PPH_Prevention_Treatment_Recommendations_2012.pdf",
        "dest_subdir": "tier1/WHO",
    },
    {
        "source": "WHO", "tier": 1, "year": 2016,
        "topic": "WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience",
        "title": "WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience",
        "subtopic": "Antenatal care, obstetric assessment, prenatal screening, pregnancy surveillance",
        "document_type": "clinical practice guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "high",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/250796/9789241550031-eng.pdf",
        "filename": "WHO_ANC_Recommendations_2016.pdf",
        "dest_subdir": "tier1/WHO",
    },
    {
        "source": "WHO", "tier": 1, "year": 2018,
        "topic": "WHO Recommendations: Intrapartum Care for a Positive Childbirth Experience",
        "title": "WHO Recommendations: Intrapartum Care for a Positive Childbirth Experience",
        "subtopic": "Intrapartum care, labour management, pain relief in labour, active management third stage",
        "document_type": "clinical practice guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "high",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/272447/9789241550215-eng.pdf",
        "filename": "WHO_Intrapartum_Care_Recommendations_2018.pdf",
        "dest_subdir": "tier1/WHO",
    },
    {
        "source": "WHO", "tier": 1, "year": 2018,
        "topic": "WHO Recommendations on Maternal Health: Guidelines Approved by WHO Guidelines Review Committee",
        "title": "WHO Recommendations on Maternal Health",
        "subtopic": "Maternal health, obstetric complications, haemorrhage, sepsis, hypertension, summary",
        "document_type": "clinical practice guideline",
        "trust_level": "Tier 1 — Highest Authority",
        "retrieval_priority": "high",
        "url": "https://apps.who.int/iris/bitstream/handle/10665/259268/WHO-RHR-18.01-eng.pdf",
        "filename": "WHO_Maternal_Health_Recommendations_2018.pdf",
        "dest_subdir": "tier1/WHO",
    },
]
