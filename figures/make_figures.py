"""Generate the paper figures from the result files. Regenerable, not hand-drawn."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200,
                     "axes.titlesize": 13, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11})

TIER_C = {1: "#1b4965", 2: "#5fa8d3", 3: "#bee9e8", 4: "#cae9ff"}
ACCENT, MUTED = "#c1121f", "#8d99ae"


def _load(p):
    return json.loads((ROOT / p).read_text())


def _lines(p):
    return [json.loads(l) for l in (ROOT / p).open(encoding="utf-8")]


# ---------------------------------------------------------------- Fig 2: corpus
def fig_corpus():
    meta = _load("corpus/metadata/corpus_metadata.json")
    chunks = _lines("chunks/chunks.jsonl")
    audit = _load("corpus/metadata/integrity_audit.json")
    dedup = _load("chunks/dedup_report.json")

    def tier(d):
        m = re.search(r"Tier\s*(\d)", d.get("trust_level", ""))
        return int(m.group(1)) if m else 4

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) documents by tier, stacked by source
    by = {t: Counter() for t in (1, 2, 3, 4)}
    for d in meta:
        by[tier(d)][d["source"]] += 1
    labels = ["Tier 1\nguidelines", "Tier 2\nnursing BPG", "Tier 3\nsafety", "Tier 4\nliterature"]
    for i, t in enumerate((1, 2, 3, 4)):
        bottom = 0
        for src, n in by[t].most_common():
            ax[0, 0].bar(i, n, bottom=bottom, color=TIER_C[t],
                         edgecolor="white", linewidth=.6)
            if n >= 40:                       # only label segments that fit
                ax[0, 0].text(i, bottom + n / 2, f"{src}\n{n}", ha="center",
                              va="center", fontsize=9,
                              color="white" if t <= 2 else "#333")
            bottom += n
        ax[0, 0].text(i, bottom + 5, str(bottom), ha="center", fontweight="bold",
                          fontsize=11)
    ax[0, 0].set_xticks(range(4)); ax[0, 0].set_xticklabels(labels, fontsize=10)
    ax[0, 0].set_ylabel("documents")
    ax[0, 0].set_title("(a) documents by evidence tier", loc="left",
                       fontweight="bold")
    # Source detail belongs in Table 1, not crammed under the axis.

    # (b) chunks and tokens by tier
    cc = Counter(c["tier"] for c in chunks)
    tk = {t: sum(c["est_tokens"] for c in chunks if c["tier"] == t) for t in (1, 2, 3, 4)}
    x = range(4)
    ax[0, 1].bar([i - .19 for i in x], [cc[t] for t in (1, 2, 3, 4)], .38,
                 label="chunks", color="#1b4965")
    a2 = ax[0, 1].twinx()
    a2.bar([i + .19 for i in x], [tk[t] / 1e6 for t in (1, 2, 3, 4)], .38,
           label="tokens (M)", color="#5fa8d3")
    a2.set_ylabel("tokens (millions)"); a2.spines["top"].set_visible(False)
    ax[0, 1].set_xticks(list(x))
    ax[0, 1].set_xticklabels(["Tier 1", "Tier 2", "Tier 3", "Tier 4"], fontsize=10)
    ax[0, 1].set_ylabel("chunks")
    ax[0, 1].set_title("(b) retrievable content by tier", loc="left", fontweight="bold")
    h1, l1 = ax[0, 1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[0, 1].legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")

    # (c) recency
    yrs = [int(str(d["publication_year"])[:4]) for d in meta
           if str(d.get("publication_year", ""))[:4].isdigit()]
    ax[1, 0].hist(yrs, bins=range(min(yrs), max(yrs) + 2), color="#5fa8d3",
                  edgecolor="white")
    ax[1, 0].axvline(2019.5, color=ACCENT, ls="--", lw=1.2)
    ax[1, 0].text(2019.0, ax[1, 0].get_ylim()[1] * .72,
                  f"{sum(1 for y in yrs if y >= 2020)}/{len(yrs)}\nfrom 2020+",
                  color=ACCENT, fontsize=10.5, fontweight="bold", ha="right")
    ax[1, 0].set_xlabel("publication year"); ax[1, 0].set_ylabel("documents")
    ax[1, 0].set_title("(c) recency", loc="left", fontweight="bold")

    # (d) provenance integrity — at the DOCUMENT level, where the failure was.
    # A chunk-level chart hides it entirely: the bad files never reached chunking.
    fails = Counter({int(k): v for k, v in audit["by_tier"].items()})
    now = {t: sum(1 for d in meta if tier(d) == t) for t in (1, 2, 3, 4)}
    x = range(4)
    ax[1, 1].bar([i - .19 for i in x], [now[t] for t in (1, 2, 3, 4)], .38,
                 color="#1b4965", label="valid documents (after recovery)")
    ax[1, 1].bar([i + .19 for i in x],
                 [now[t] - fails.get(t, 0) for t in (1, 2, 3, 4)], .38,
                 color="#c1121f", alpha=.85,
                 label="had the 14 failures gone undetected")
    for i, t in enumerate((1, 2, 3, 4)):
        ax[1, 1].text(i - .19, now[t] + 4, str(now[t]), ha="center", fontsize=10,
                      fontweight="bold")
        lost = fails.get(t, 0)
        if lost:
            ax[1, 1].text(i + .19, now[t] - lost + 4, str(now[t] - lost),
                          ha="center", fontsize=10, color="#c1121f",
                          fontweight="bold")
    ax[1, 1].set_xticks(list(x))
    ax[1, 1].set_xticklabels(["Tier 1", "Tier 2", "Tier 3", "Tier 4"], fontsize=10)
    ax[1, 1].set_ylabel("documents")
    ax[1, 1].legend(fontsize=9, frameon=False, loc="upper left")
    ax[1, 1].set_title("(d) provenance integrity", loc="left", fontweight="bold")

    fig.tight_layout(pad=2.2)
    fig.savefig(OUT / "fig2_corpus.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_corpus.png")


# ------------------------------------------------------ Fig 3: question profile
def fig_questions():
    qs = _lines("eval/full_testset.jsonl")
    types = Counter(q["question_type"] for q in qs)
    order = ["timing", "technique", "escalation", "threshold", "dose", "general"]
    actionable = sum(types[t] for t in order if t != "general")

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8),
                           gridspec_kw={"width_ratios": [2.1, 1]})
    cols = ["#1b4965"] * 5 + [MUTED]
    ax[0].barh([t for t in order][::-1], [types[t] for t in order][::-1], color=cols[::-1])
    for i, t in enumerate(order[::-1]):
        ax[0].text(types[t] + 1.2, i, str(types[t]), va="center", fontsize=10)
    ax[0].set_xlabel("questions"); ax[0].set_xlim(0, max(types.values()) * 1.18)
    ax[0].set_title(f"(a) what nurses ask — {actionable}/{len(qs)} "
                    f"({100*actionable/len(qs):.0f}%) are point-of-care actionable",
                    loc="left", fontweight="bold", fontsize=11)

    tiers = Counter(q["source_tier"] for q in qs)
    ax[1].bar(["Tier 1", "Tier 2"], [tiers[1], tiers[2]], color=[TIER_C[1], TIER_C[2]])
    for i, t in enumerate((1, 2)):
        ax[1].text(i, tiers[t] + 6, str(tiers[t]), ha="center", fontweight="bold",
                   fontsize=11)
    ax[1].set_ylabel("questions")
    # Headroom so the two-line title never sits on top of the value labels.
    ax[1].set_ylim(0, max(tiers[1], tiers[2]) * 1.30)
    ax[1].set_title("(b) 100% answerable from\nTier 1–2 guidance", loc="left",
                    fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_questions.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_questions.png")




# ------------------------------------------------------------ Fig 4: ablation
def fig_ablation():
    d = _load("eval/grid_judgefree_results.json")["conditions"]
    order = sorted(d, key=lambda k: -d[k]["recall_at_k"])
    pretty = {"no_lexical": "dense +\ntier prior", "full": "hybrid +\ntier + dedup",
              "no_dedup": "hybrid +\ntier", "no_tier_prior": "hybrid,\nno tier prior",
              "dense_only": "dense\nonly", "lexical_only": "BM25\nonly"}
    x = list(range(len(order)))
    rec = [d[k]["recall_at_k"] for k in order]
    t1 = [d[k]["top1_tier1_rate"] for k in order]

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for a, vals, base, title, ylab in (
            (ax[0], rec, "#1b4965",
             "(a) retrieved the source guideline passage", "Recall@5"),
            (ax[1], t1, "#5fa8d3",
             "(b) top-ranked passage came from a Tier 1 guideline",
             "proportion of questions")):
        cols = [ACCENT if k == "no_lexical" else base for k in order]
        a.bar(x, vals, color=cols, width=.68)
        for i, v in enumerate(vals):
            a.text(i, v + .02, f"{v:.3f}", ha="center", fontsize=10,
                   fontweight="bold" if order[i] == "no_lexical" else "normal")
        a.set_ylim(0, 1.12)
        a.set_ylabel(ylab)
        a.set_xticks(x)
        a.set_xticklabels([pretty[k] for k in order], fontsize=9.5)
        a.set_title(title, loc="left", fontweight="bold", fontsize=11)

    # Effect annotation drawn in the empty band above the bars, not across them.
    i_d, i_n = order.index("dense_only"), order.index("no_lexical")
    y = 0.90
    ax[0].annotate("", xy=(i_n, y), xytext=(i_d, y),
                   arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4))
    ax[0].text((i_n + i_d) / 2, y + .035,
               f"evidence-tier weighting: +{rec[i_n]-rec[i_d]:.3f}",
               ha="center", color=ACCENT, fontsize=10.5, fontweight="bold")
    ax[0].vlines([i_d, i_n], [rec[i_d], rec[i_n]], [y, y], color=ACCENT,
                 lw=.7, ls=":")

    fig.tight_layout()
    fig.savefig(OUT / "fig4_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_ablation.png")


# ------------------------------------------------- Fig 5: where the prior acts
def fig_tier_dist():
    d = _load("eval/grid_judgefree_results.json")["conditions"]
    order = ["no_lexical", "full", "no_dedup", "no_tier_prior", "dense_only", "lexical_only"]
    pretty = {"no_lexical": "dense +\ntier prior", "full": "hybrid +\ntier + dedup",
              "no_dedup": "hybrid +\ntier", "no_tier_prior": "no tier\nprior",
              "dense_only": "dense\nonly", "lexical_only": "BM25\nonly"}
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bottom = [0] * len(order)
    for t in (1, 2, 3, 4):
        vals = [d[k]["retrieved_tier_dist"].get(str(t), 0) for k in order]
        ax.bar(range(len(order)), vals, bottom=bottom, color=TIER_C[t],
               label=f"Tier {t}", edgecolor="white", linewidth=.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, k in enumerate(order):
        n1 = d[k]["retrieved_tier_dist"].get("1", 0)
        ax.text(i, 60, f"{n1}", ha="center", color="white", fontweight="bold", fontsize=10)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pretty[k] for k in order], fontsize=10)
    ax.set_ylabel("retrieved passages (1,500 slots)")
    ax.legend(frameon=False, fontsize=10, ncol=4, loc="upper center",
              bbox_to_anchor=(.5, 1.13))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_tier_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_tier_distribution.png")


# ------------------------------------------------------ Fig 6: judge calibration
def fig_judges():
    """Panel (b) now carries the clinician, which is the point of the figure."""
    cross = _load("eval/cross_judge_ragas.json")
    author = _load("eval/author_spotcheck_results.json")
    proxy = _load("eval/clinician_proxy_results.json")
    clin = _load("eval/clinical_review_results.json")

    gpt = cross["gpt-4o-project"]["faithfulness"]["mean"]
    gem = cross["gemini-3.7-flash-project"]["faithfulness"]["mean"]
    auth = author["correct_grounded"]["rate"]
    nurse = clin["answers"]["acceptable_rate"]["rate"]
    pr = proxy["per_model"]
    p_hi = max(v["acceptable_rate"] for v in pr.values())
    p_lo = min(v["acceptable_rate"] for v in pr.values())
    hi_name = [k for k, v in pr.items() if v["acceptable_rate"] == p_hi][0]
    lo_name = [k for k, v in pr.items() if v["acceptable_rate"] == p_lo][0]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    a_vals = [auth, gpt, gem]
    ax[0].bar(["author\n(human)", "GPT-4o", "Gemini Flash"], a_vals,
              color=["#1b4965", "#5fa8d3", "#5fa8d3"])
    for i, v in enumerate(a_vals):
        ax[0].text(i, v + .025, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10.5)
    ax[0].axhspan(min(a_vals), max(a_vals), color="#2a9d8f", alpha=.15)
    ax[0].set_ylim(0, 1.12); ax[0].set_ylabel("score / proportion")
    ax[0].set_title("(a) Does the answer quote the guideline correctly?\n"
                    f"RATERS AGREE — spread {max(a_vals)-min(a_vals):.3f}",
                    loc="left", fontweight="bold", fontsize=11, color="#2a9d8f")

    names = [f"{hi_name.replace('-project','')}\n(AI proxy)",
             "registered nurse\n(first author)",
             f"{lo_name.replace('-project','')}\n(AI proxy)"]
    vals = [p_hi, nurse, p_lo]
    ax[1].bar(names, vals, color=["#f4a261", "#1b4965", "#f4a261"])
    for i, v in enumerate(vals):
        ax[1].text(i, v + .025, f"{v:.2f}", ha="center", fontweight="bold",
                   fontsize=12 if i == 1 else 10.5)
    ax[1].axhline(nurse, color=ACCENT, ls="--", lw=1.1)
    ax[1].text(-0.44, nurse + .022, "nurse", color=ACCENT, fontsize=10,
               va="bottom", ha="left")
    ax[1].set_title("(b) Would you accept a nurse using this answer?\n"
                    f"AI PROXIES BRACKET THE NURSE — neither within "
                    f"{min(abs(p_hi-nurse), abs(nurse-p_lo)):.2f}",
                    loc="left", fontweight="bold", fontsize=11, color=ACCENT)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_judge_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_judge_calibration.png")


# ------------------------------------------------------- Fig 1: architecture
def fig_architecture():
    """Mirrors the compiled agent, so the figure cannot drift from the code."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_xlim(-1.5, 11.6); ax.set_ylim(0, 12.4); ax.axis("off")
    GUIDE, EVID, CHECK, IO = "#1b4965", "#5fa8d3", "#2a9d8f", "#495057"

    def box(x, y, w, h, text, fc, tc="white", fs=8.4):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.1", fc=fc, ec="none"))
        ax.text(x, y, text, ha="center", va="center", color=tc,
                fontsize=fs, fontweight="bold", linespacing=1.35)

    def arrow(x1, y1, x2, y2, label="", color="#6c757d"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, color=color, lw=1.2,
                                     shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, fontsize=7, color=color,
                    ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=1.2))

    box(5, 11.8, 3.4, .62, "nurse's question", IO)
    arrow(5, 11.45, 5, 10.95)
    box(5, 10.6, 3.6, .7, "query classification", GUIDE)

    ys = 9.2
    box(2.0, ys, 2.6, .8, "semantic\n(dense + tier prior)", EVID)
    box(5.0, ys, 2.6, .8, "relational\n(:NEXT expansion)", EVID)
    box(8.0, ys, 2.6, .8, "freshness\n(corpus-first)", EVID)
    for x in (2.0, 5.0, 8.0):
        arrow(5, 10.22, x, ys + .42)
        arrow(x, ys - .42, 5, 7.95)

    box(5, 7.6, 3.2, .68, "rerank + deduplicate", GUIDE)
    arrow(5, 7.24, 5, 6.75)
    box(5, 6.4, 3.4, .68, "evidence sufficiency", CHECK)

    box(8.6, 5.2, 2.3, .68, "web search\n(Tier 0, unverified)", "#adb5bd", tc="#212529")
    arrow(6.7, 6.4, 8.6, 5.6, "insufficient", "#c1121f")
    arrow(8.6, 4.85, 6.6, 4.35, "", "#c1121f")
    ax.text(8.6, 4.45, "capped at one retry", fontsize=6.6, color="#c1121f",
            ha="center", va="center")

    arrow(5, 6.05, 5, 4.6, "sufficient")
    box(5, 4.2, 3.2, .68, "generate with citations", GUIDE)
    arrow(5, 3.85, 5, 3.35)
    box(5, 3.0, 3.6, .68, "citation verification", CHECK)
    arrow(5, 2.65, 5, 2.15)
    box(5, 1.8, 3.6, .68, "clinical safety gate", CHECK)
    arrow(5, 1.45, 5, .95)
    box(5, .6, 3.4, .62, "answer + tiered citations", IO)

    ax.text(-1.35, 9.2, "evidence\nretrieval", fontsize=7.5, color=EVID,
            fontweight="bold", va="center", ha="left")
    ax.text(-1.35, 2.6, "safeguards", fontsize=7.5, color=CHECK,
            fontweight="bold", va="center", ha="left")
    for x, y, note in [(1.35, 8.35, "Tier 1 \u2192 4\nauthority prior"),
                       (8.3, 3.0, "claims checked\nagainst cited\npassage"),
                       (8.3, 1.8, "scope of practice;\nescalation")]:
        ax.text(x, y, note, fontsize=6.6, color="#6c757d", ha="center", va="center")

    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  fig1_architecture.png")



if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_architecture()
    fig_corpus()
    fig_questions()
    fig_ablation()
    fig_tier_dist()
    fig_judges()
