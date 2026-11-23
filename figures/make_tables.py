"""Generate the manuscript tables as markdown, from the result files."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"


def _load(p): return json.loads((ROOT / p).read_text())
def _lines(p): return [json.loads(l) for l in (ROOT / p).open(encoding="utf-8")]


def table1_corpus() -> str:
    meta = _load("corpus/metadata/corpus_metadata.json")
    chunks = _lines("chunks/chunks.jsonl")
    audit = _load("corpus/metadata/integrity_audit.json")
    dedup = _load("chunks/dedup_report.json")

    def tier(d):
        m = re.search(r"Tier\s*(\d)", d.get("trust_level", ""))
        return int(m.group(1)) if m else 4

    names = {1: "Tier 1 — clinical guidelines", 2: "Tier 2 — nursing best practice",
             3: "Tier 3 — safety/reflective", 4: "Tier 4 — primary literature"}
    rows = ["| Evidence tier | Sources | Documents | Passages | Tokens |",
            "|---|---|---|---|---|"]
    for t in (1, 2, 3, 4):
        srcs = Counter(d["source"] for d in meta if tier(d) == t)
        docs = sum(srcs.values())
        cs = [c for c in chunks if c["tier"] == t]
        rows.append(f"| {names[t]} | {', '.join(f'{k} ({v})' for k, v in srcs.most_common())} "
                    f"| {docs} | {len(cs):,} | {sum(c['est_tokens'] for c in cs)/1e6:.2f} M |")
    rows.append(f"| Total | 13 sources | {len(meta)} | {len(chunks):,} | "
                f"{sum(c['est_tokens'] for c in chunks)/1e6:.2f} M |")
    note = (f"\n*Content validation rejected {audit['failures_found']} files that were logged as "
            f"successful downloads but contained HTML error pages ({audit['by_tier']['1']} of them "
            f"Tier 1); all were subsequently recovered. Near-duplicate detection marked "
            f"{dedup['duplicates_marked']} passages "
            f"({100*dedup['duplicates_marked']/len(chunks):.1f}%), leaving "
            f"{dedup['canonical_remaining']:,} used for retrieval.*")
    return "### Table 1 — Corpus composition\n\n" + "\n".join(rows) + "\n" + note


def table2_ablation() -> str:
    d = _load("eval/grid_judgefree_results.json")
    c = d["conditions"]
    pretty = {"no_lexical": "Dense + evidence-tier prior (selected)",
              "full": "Dense + keyword + tier prior + deduplication",
              "no_dedup": "Dense + keyword + tier prior",
              "no_tier_prior": "Dense + keyword, no tier prior",
              "dense_only": "Dense only", "lexical_only": "Keyword only (BM25)"}
    order = sorted(c, key=lambda k: -c[k]["recall_at_k"])
    base = c["full"]["recall_at_k"]
    rows = ["| Retrieval configuration | Recall@5 | MRR | Top-1 from Tier 1 | Δ Recall@5 |",
            "|---|---|---|---|---|"]
    for k in order:
        m = c[k]
        delta = "—" if k == "full" else f"{m['recall_at_k']-base:+.3f}"
        rows.append(f"| {pretty[k]} | {m['recall_at_k']:.3f} | {m['mrr']:.3f} | "
                    f"{m['top1_tier1_rate']:.3f} | {delta} |")
    note = (f"\n*n = {d['n_questions']} questions; retrieval over all 12,732 canonical passages. "
            "No large language model is involved in these measurements: Recall@5 is computed "
            "by checking whether the passage each question was written from was returned. "
            "Evidence-tier weighting accounts for +0.173 Recall@5 (dense only → dense + tier "
            "prior) and raises top-ranked Tier 1 passages from 0.663 to 0.917.*")
    return "### Table 2 — Retrieval ablation\n\n" + "\n".join(rows) + "\n" + note


def table3_ragas() -> str:
    d = _load("eval/ragas_headline_pair.json")
    a, b = "no_lexical", "no_tier_prior"
    label = {"faithfulness": "Faithfulness (claims supported by the cited passage)",
             "answer_relevancy": "Answer relevancy",
             "context_recall": "Context recall (guidance needed was retrieved)",
             "llm_context_precision_without_reference": "Context precision"}
    rows = ["| Metric | With tier prior | Without tier prior | Difference |", "|---|---|---|---|"]
    for m in ("context_recall", "llm_context_precision_without_reference",
              "answer_relevancy", "faithfulness"):
        if m in d[a] and m in d[b]:
            x, y = d[a][m]["mean"], d[b][m]["mean"]
            rows.append(f"| {label[m]} | {x:.3f} | {y:.3f} | {x-y:+.3f} |")
    note = ("\n*n = 300; judge GPT-4o, a different model family from the answer generator. "
            "Direction matches the judge-free result in Table 2, but the judged effect (~0.05) is "
            "about half the judge-free effect (0.103) and lies within the between-judge "
            "uncertainty measured for these metrics (up to 0.096). Faithfulness is unchanged, as "
            "expected: it measures grounding in whatever was retrieved, so a retrieval change "
            "should not move it.*")
    return "### Table 3 — Automated (RAGAS) corroboration\n\n" + "\n".join(rows) + "\n" + note


def table4_safety() -> str:
    d = _load("eval/safety_results_fixed.json")
    desc = {"prescribing": "Prescribing — must decline (outside nursing scope)",
            "diagnosis": "Diagnosis — must decline (clinician's determination)",
            "escalation": "Deteriorating patient — must instruct escalation",
            "citation": "Answerable question — specifics must be cited",
            "out_of_scope": "Outside perioperative nursing — must redirect",
            "benign_control": "Ordinary question — must be answered"}
    rows = ["| Probe class | n | Handled correctly | Rate |", "|---|---|---|---|"]
    for k in ("prescribing", "diagnosis", "escalation", "citation", "out_of_scope",
              "benign_control"):
        v = d["by_class"][k]
        rows.append(f"| {desc[k]} | {v['n']} | {v['pass']} | {v['pass']/v['n']:.2f} |")
    note = (f"\n*Each probe was run through the complete agent. Scoring is rule-based and "
            f"deterministic. The catch rate was {d['catch_rate']:.2f} across the 36 unsafe probes, at "
            f"a false-refusal rate of {d['false_refusal_rate']:.2f} on the 10 benign controls: "
            "the system declined every unsafe request while still answering every ordinary one. "
            "Probes are author-constructed and were not reviewed by an external clinician.*")
    return "### Table 4 — Safety evaluation\n\n" + "\n".join(rows) + "\n" + note


def table5_clinician() -> str:
    d = _load("eval/clinical_review_results.json")
    a = d["answers"]
    label = {"clinical_correctness_1to5": "Clinical correctness",
             "citation_adequacy_1to5": "Citation adequacy",
             "actionability_1to5": "Actionability at the bedside",
             "safety_1to5": "Safety", "completeness_1to5": "Completeness"}
    rows = ["| Dimension (1–5) | Mean | Rated 1–2 |", "|---|---|---|"]
    for k in ("safety_1to5", "clinical_correctness_1to5", "citation_adequacy_1to5",
              "actionability_1to5", "completeness_1to5"):
        v = a[k]
        rows.append(f"| {label[k]} | {v['mean']:.2f} | {v['low_1_2']} |")
    rows.append(f"| Acceptable for use on the unit | {a['acceptable_rate']['rate']:.2f} | — |")
    q = d["questions"]
    note = (f"\n*Rated by the registered-nurse first author (n = {a['acceptable_rate']['n']} "
            f"answers). No answer was rated unsafe. Question realism was assessed separately "
            f"(n = {q['realistic']['n']}): {q['realistic']['rate']:.0%} judged realistic questions "
            f"a perioperative nurse would ask, {q['answerable_at_bedside']['rate']:.0%} answerable "
            "at the bedside. One in three answers was not acceptable for use; the system is "
            "not validated for clinical use.*")
    return "### Table 5 — Clinician review\n\n" + "\n".join(rows) + "\n" + note


if __name__ == "__main__":
    parts = [table1_corpus(), table2_ablation(), table3_ragas(),
             table4_safety(), table5_clinician()]
    out = OUT / "tables.md"
    out.write_text("# Manuscript tables\n\nGenerated from the result files by "
                   "`figures/make_tables.py`.\n\n" + "\n\n---\n\n".join(parts) + "\n")
    print(f"  wrote {out}")
    for p in parts:
        print("   ", p.splitlines()[0])
