"""Offline tests: every step with stubbed models and a stubbed graph.

    python -m verified_rag.tests.test_offline
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from verified_rag.answer.details import check_details, quantities
from verified_rag.answer.repair import repair
from verified_rag.answer.synthesize import parse_claims
from verified_rag.answer.verify import verify_all
from verified_rag.answer.assemble import assemble
from verified_rag.clients.jev import JevAnswers
from verified_rag.config import Settings
from verified_rag.pipeline import Pipeline
from verified_rag.store import Store
from verified_rag.types import Claim, Source

FAIL = []


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(f"{name} {detail}")


OXY18 = ("Recommendation The panel suggests that adult patients undergoing general anaesthesia with "
         "tracheal intubation for surgical procedures should receive an 80% fraction of inspired oxygen "
         "(FiO2) intraoperatively and, if feasible, in the immediate postoperative period for 2-6 hours "
         "to reduce the risk of SSI. (Conditional recommendation, moderate quality of evidence)")
DRAIN = "Routine use of wound drains is not recommended after hysterectomy."


def src(label, text, doc="d1", refines=""):
    return Source(label, f"{doc}:0001", doc, text, "WHO SSI", 1, "2018", 111, 2, refines=refines)


# ---- details: pure code --------------------------------------------------
c = check_details("WHO conditionally recommends 80% FiO2 for 2-6 hours.", OXY18)
check("details: faithful claim passes", c.passed, c.detail)
check("details: 'strongly' caught", not check_details("80% FiO2 is strongly recommended.", OXY18).passed)
check("details: wrong percentage caught", not check_details("Give 40% FiO2.", OXY18).passed)
check("details: wrong duration caught", not check_details("Continue for 24 hours.", OXY18).passed)
check("details: thousands separator is not a decimal",
      check_details("Absolute effect: 30 fewer deaths per 1,000.", "30 fewer per 1000 (95% CI 12 to 41)").passed)
check("details: real quantity mismatch still caught",
      not check_details("Give 2,000 mg.", "Give 1,000 mg once.").passed)
check("details: dropped negation caught", not check_details("Wound drains are recommended after hysterectomy.", DRAIN).passed)
check("details: kept negation passes", check_details("Drains are not recommended.", DRAIN).passed)
check("details: hours unit synonym", ("2-6", "h") in quantities("for 2–6 hrs"))
LONG = ("Hexachlorophene has residual activity for several hours after use. "
        "In 1972 the FDA warned that hexachlorophene should no longer be used routinely for bathing infants.")
check("details: unrelated 'should no longer be used' does not flag", check_details(
    "Hexachlorophene has residual activity for several hours after use.", LONG).passed)
MIX = ("Antibiotics are recommended for women with preterm prelabour rupture of membranes. "
       "Routine antibiotic administration is not recommended for women in preterm labour with intact membranes.")
check("details: negation from a different population does not flag", check_details(
    "Antibiotics are recommended for women with preterm prelabour rupture of membranes.", MIX).passed)
check("details: negation in the claim's own sentence still flags", not check_details(
    "Routine antibiotic administration is recommended for women in preterm labour with intact membranes.", MIX).passed)
check("details: citation tags are not quantities", check_details("Give 80% FiO2 [S1, S6].", OXY18).passed)
check("details: guideline years are not quantities", check_details("In the 2018 guideline, 80% FiO2 is suggested.", OXY18).passed)
check("details: a real wrong number still fails", not check_details("Give 40% FiO2 [S1].", OXY18).passed)
check("details: nbsp handled", check_details("store at 2-8 °C", "store at 2–8 °C").passed)

# ---- parse ---------------------------------------------------------------
cl = parse_claims('noise [{"text": "A.", "sources": ["S1"]}, {"text": " ", "sources": []}] tail', 5)
check("parse: drops empty claims", len(cl) == 1 and cl[0].sources == ["S1"])
check("parse: garbage -> no claims", parse_claims("no json here", 5) == [])


# ---- stub Jev: support decided by whether details pass --------------------
class StubJev:
    versions = {"stub": 1}

    def __init__(self, any_p=0.9):
        self.any_p, self.calls = any_p, 0

    noul_q = staticmethod(lambda i: {"type": "noul", "instructions": i})
    choice_q = staticmethod(lambda i, o: {"type": "choice", "instructions": i, "criteria": o})

    def ask(self, state, questions):
        self.calls += 1
        ans = {}
        for k, q in questions.items():
            if q["type"] == "noul":
                if k == "support":
                    ok = check_details(state["claim"], state["cited_passage"]).passed
                    ans[k] = {"noul": 0.9 if ok and self.any_p >= 0.5 and "children" not in state["claim"] else 0.2}
                elif k == "any":
                    ans[k] = {"noul": self.any_p}
                else:
                    ans[k] = {"noul": 0.9}
            else:
                first = next(iter(q["criteria"]))
                ans[k] = {"choice": first, "probabilities": {first: 0.8}}
        return JevAnswers(ans, "stub", 0.0)


class StubChat:
    model = "stub-chat"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, prompt, max_tokens=600):
        self.calls += 1
        from verified_rag.clients.chat import ChatResult
        out = self.text if self.calls == 1 else "Adults undergoing intubation should receive 80% FiO2 (conditional)."
        return ChatResult(out, 100, 20, 0.0, self.model)


sources = [src("S1", OXY18)]
claims = [Claim("WHO conditionally recommends 80% FiO2 for intubated adults.", ["S1"]),
          Claim("80% FiO2 is strongly recommended.", ["S1"]),
          Claim("Give 80% FiO2 to intubated adults and children.", ["S1"]),
          Claim("Give 80% FiO2.", ["S9"]),
          ]
jev = StubJev()
results = verify_all(jev, claims, sources, 0.5)
check("verify: faithful claim passes all three", claims[0].passed)
check("verify: strength error fails details", not claims[1].passed and
      not next(c for c in claims[1].checks if c.name == "details").passed)
check("verify: bad citation fails citation", not next(c for c in claims[3].checks if c.name == "citation").passed)

chat = StubChat("unused")
for cl_, r in zip(claims, results):
    repair(chat, jev, cl_, r, sources, 0.5)
check("repair: faithful claim untouched", claims[0].origin == "synthesized")
check("repair: strength error -> verbatim source sentence", claims[1].origin == "verbatim"
      and "Conditional recommendation" in claims[1].text and claims[1].span is not None, claims[1].origin)
check("repair: over-broad claim rewritten once", claims[2].origin == "repaired", claims[2].origin)
check("repair: bad citation re-cited", claims[3].origin in ("repaired", "verbatim"), claims[3].origin)
check("repair: history kept for audit", all(c.history for c in claims[1:]))

lonely = [Claim("Give magnesium.", ["S1"])]
r2 = verify_all(StubJev(any_p=0.1), lonely, sources, 0.5)
repair(chat, StubJev(any_p=0.1), lonely[0], r2[0], sources, 0.5)
check("repair: unsupported anywhere -> withdrawn, never rewritten", lonely[0].origin == "withdrawn")

resp = assemble("q", 2, claims + lonely, sources)
check("assemble: partial when something withdrawn", resp.status == "partial")
check("assemble: relevance says uncalibrated", "not yet calibrated" in resp.signals.relevance)
check("assemble: evidence names release", "release 2" in resp.signals.evidence)


# ---- store + calibration -------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    st = Store(Path(d) / "a.sqlite")
    rid = st.record(resp)
    check("store: response recorded", bool(rid))
    check("store: no calibration below MIN_N", st.calibration(0.9, "synthesized") is None)
    for i in range(40):
        st.feedback(rid, 0, "right" if i < 30 else "wrong")
    cal = st.calibration(0.9, "synthesized")
    check("store: calibration from nurse verdicts", cal is not None and abs(cal[0] - 0.75) < 1e-9
          and cal[1] == 40, str(cal))
    check("store: repaired claims calibrated separately", st.calibration(0.9, "repaired") is None)


# ---- pipeline with stubs, including population ranking ---------------------
class StubStore:
    def __init__(self, rows):
        self.rows = rows

    def search_text(self, q, emb, limit=10, snapshot=None, **kw):
        return self.rows[:limit]

    def current_release(self):
        return 3

    class _S:
        def __init__(self, data):
            self.data_ = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def run(self, query, *a, **k):
            data = [] if "Recommendation" in query else self.data_

            class R:
                def data(self_inner):
                    return data
            return R()

    def session(self):
        return self._S([{"doc_id": "cs", "replaces": [], "populations": ["caesarean section"]},
                        {"doc_id": "ssi", "replaces": ["WHO SSI (2016)"], "populations": []}])


rows = [{"chunk_id": "cs:1", "doc_id": "cs", "text": "Chlorhexidine for caesarean.", "title": "WHO CS",
         "tier": 1, "year": "2021", "page": 10, "snapshot": 3},
        {"chunk_id": "ssi:1", "doc_id": "ssi", "text": OXY18, "title": "WHO SSI", "tier": 1,
         "year": "2018", "page": 111, "snapshot": 3}]


class PopJev(StubJev):
    def ask(self, state, questions):
        if all(k.startswith("p") for k in questions):     # population question
            return JevAnswers({k: {"noul": 0.05} for k in questions}, "stub", 0.0)
        return super().ask(state, questions)


with tempfile.TemporaryDirectory() as d:
    pipe = Pipeline(Settings(audit_db=Path(d) / "a.sqlite", retrieve_limit=2),
                    store=StubStore(rows), embedder=object(),
                    chat=StubChat('[{"text": "WHO suggests 80% FiO2 for intubated adults (conditional).", "sources": ["S1"]}]'),
                    jev=PopJev(), audit=Store(Path(d) / "a.sqlite"))
    out = pipe.answer("oxygen for hysterectomy?")
    check("pipeline: population-specific doc demoted below general", out.sources[0].doc_id == "ssi",
          [s.doc_id for s in out.sources])
    check("pipeline: answered and verified", out.status == "answered" and out.signals.integrity.startswith("verified"),
          out.status + " " + out.signals.integrity)
    check("pipeline: supersession surfaced in evidence", "replaces WHO SSI (2016)" in out.signals.evidence,
          out.signals.evidence)
    check("pipeline: timings recorded", "total_s" in out.timings)

# ---- check 4 (current) and its repair --------------------------------------
from verified_rag.answer.details import rests_on_replaced
OLD99 = "For high-risk cesarean section, administer the prophylactic antimicrobial agent immediately after the umbilical cord is clamped."
NEW17 = "Administer the appropriate parenteral prophylactic antimicrobial agents before skin incision in all cesarean section procedures."
CH99 = "Category IA. " + OLD99 + " Category IA. Other text that remains current practice."
CH17 = "Intro sentence. " + NEW17 + " (Category IA-strong recommendation.)"
rep = [{"old_text": OLD99, "new_text": NEW17, "new_chunk": "c17:1", "new_label": "S2", "new_year": "2017",
        "new_start": CH17.index(NEW17), "new_end": CH17.index(NEW17) + len(NEW17)}]
check("current: old cord-clamping claim is caught",
      rests_on_replaced("For high-risk cesarean section, give the prophylactic antibiotic after the umbilical cord is clamped.", rep) is not None)
check("current: claim stating the new rule passes",
      rests_on_replaced("Give prophylactic antibiotics before skin incision in all cesarean section procedures.", rep) is None)
s99 = Source("S1", "c99:1", "c99", CH99, "CDC SSI", 1, "1999", 3, 4, replaced=rep)
s17 = Source("S2", "c17:1", "c17", CH17, "CDC SSI", 1, "2017", 5, 4)
old_claim = Claim("For high-risk cesarean section, administer the prophylactic antimicrobial agent immediately after the umbilical cord is clamped.", ["S1"])
res = verify_all(StubJev(), [old_claim], [s99, s17], 0.5)
check("current: check 4 fails on the old rule", not next(c for c in old_claim.checks if c.name == "current").passed)
repair(StubChat("unused"), StubJev(), old_claim, res[0], [s99, s17], 0.5)
check("current: repaired by quoting the 2017 replacement verbatim",
      old_claim.origin == "verbatim" and old_claim.text == NEW17 and old_claim.sources == ["S2"],
      old_claim.origin + " | " + old_claim.text[:60])

if FAIL:
    print("FAILED (%d):" % len(FAIL))
    for f in FAIL:
        print("  - " + f)
    sys.exit(1)
print("verified_rag offline tests: all passed")
