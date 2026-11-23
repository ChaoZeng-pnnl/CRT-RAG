"""Closed schema for the clinical knowledge graph.

Deliberately small. Open-ended extraction produces a graph that cannot be
evaluated — precision has no meaning against an undefined vocabulary, and
reviewers will ask for a precision number. Anything outside these types is
dropped rather than invented.
"""

from __future__ import annotations

NODE_TYPES = {
    "Procedure":      "A surgical or clinical procedure (laparoscopic hysterectomy, caesarean section)",
    "Complication":   "An adverse outcome or condition (surgical site infection, VTE, urinary retention)",
    "Intervention":   "A clinical action taken (early mobilisation, catheter removal, skin antisepsis)",
    "Drug":           "A medicinal product or class (oxytocin, LMWH, tranexamic acid, cefazolin)",
    "Device":         "Equipment or material (indwelling catheter, compression stockings, drain)",
    "RiskFactor":     "A patient or situational factor raising risk (obesity, prolonged surgery, diabetes)",
    "Phase":          "Perioperative phase (preoperative, intraoperative, postoperative)",
}

# Directed, with the semantics the retriever will rely on.
#
# v2: CONTRAINDICATED_WITH was the second most dangerous relation in the v1
# pilot (precision 0.409) because it conflated three clinically different
# things — an absolute contraindication, a temporary one, and a mere timing
# dependency. A nurse acting on "LMWH contraindicated with regional analgesia"
# would withhold a drug that is only time-sequenced against it. Split.
EDGE_TYPES = {
    "PREVENTS":            ("Intervention|Drug|Device", "Complication"),
    "TREATS":              ("Intervention|Drug", "Complication"),
    "INCREASES_RISK_OF":   ("RiskFactor|Procedure|Device|Drug", "Complication"),
    "CONTRAINDICATED_WITH": ("Drug|Intervention", "Drug|Intervention|Procedure|RiskFactor"),
    "TEMPORARILY_CONTRAINDICATED": ("Drug|Intervention", "Drug|Intervention|Procedure|RiskFactor"),
    "TIMING_DEPENDS_ON":   ("Drug|Intervention", "Drug|Intervention|Procedure"),
    "MONITORS_FOR":        ("Intervention|Device", "Complication"),
    "DETECTS":             ("Intervention|Device", "Complication|RiskFactor"),
    "OCCURS_IN_PHASE":     ("Intervention|Procedure", "Phase"),
    "APPLIES_TO":          ("Intervention|Drug|Device", "Procedure"),
}

# v2: the corpus is heavily obstetric (RCOG contributed 647 of 1,041 v1 triples),
# and obstetric guidance is phased antenatal / intrapartum / postpartum. Mapping
# "postpartum" onto "postoperative" was the single worst defect in v1
# (OCCURS_IN_PHASE precision 0.333) — delivery is not necessarily surgical.
SURGICAL_PHASES = {"preoperative", "intraoperative", "postoperative", "perioperative"}
OBSTETRIC_PHASES = {"antenatal", "intrapartum", "postpartum", "peripartum"}
PHASES = SURGICAL_PHASES | OBSTETRIC_PHASES


def valid_edge(edge_type: str, subject_type: str, object_type: str) -> bool:
    """Type-check a triple against the schema."""
    if edge_type not in EDGE_TYPES:
        return False
    allowed_subject, allowed_object = EDGE_TYPES[edge_type]
    return (subject_type in allowed_subject.split("|")
            and object_type in allowed_object.split("|"))


RELATION_NOTES = {
    # v3: MONITORS_FOR was the last weak relation (0.625) because it absorbed
    # three different things — ongoing surveillance, one-off diagnostic testing,
    # and incidental findings on an unrelated investigation. A test that
    # identifies a condition now is not surveillance for it developing.
    "MONITORS_FOR": "ONGOING or REPEATED observation over time to catch a complication "
                    "as it develops (serial platelet counts, daily wound inspection). "
                    "NOT a one-off diagnostic test",
    "DETECTS": "a test or investigation IDENTIFIES a condition or risk factor when "
               "performed (thrombophilia testing detects thrombophilia; CTPA detects "
               "pulmonary embolism). Use this, not MONITORS_FOR, for diagnostic tests",
    "CONTRAINDICATED_WITH": "an ABSOLUTE contraindication — never combine",
    "TEMPORARILY_CONTRAINDICATED": "contraindicated only until some condition resolves "
                                   "(e.g. 'until the lesion is secured')",
    "TIMING_DEPENDS_ON": "not a contraindication — the timing of one depends on the other "
                         "(e.g. first LMWH dose timed relative to regional analgesia)",
    "OCCURS_IN_PHASE": "use obstetric phases for obstetric guidance and surgical phases for "
                       "surgical guidance; never map postpartum onto postoperative",
}


def schema_prompt_block() -> str:
    nodes = "\n".join(f"  {name}: {desc}" for name, desc in NODE_TYPES.items())
    edges = []
    for name, (subj, obj) in EDGE_TYPES.items():
        note = RELATION_NOTES.get(name)
        edges.append(f"  {name}: ({subj}) -> ({obj})"
                     + (f"\n      NOTE: {note}" if note else ""))
    phases = ("  surgical: " + ", ".join(sorted(SURGICAL_PHASES))
              + "\n  obstetric: " + ", ".join(sorted(OBSTETRIC_PHASES)))
    return (f"ENTITY TYPES:\n{nodes}\n\nRELATION TYPES:\n" + "\n".join(edges)
            + f"\n\nALLOWED Phase VALUES:\n{phases}")
