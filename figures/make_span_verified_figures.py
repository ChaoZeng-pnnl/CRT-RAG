"""Figures for the CVT-RAG paper. Every number is read from a saved run file;
nothing is typed in. Writes PNG only (600 dpi) to figures/span_verified/, named
by paper number; tables (Markdown/CSV) go to manuscript/tables/.

    PYTHONPATH=. python figures/make_span_verified_figures.py

Style: journal two-column width (7.2 in), Arial-metric sans 6-7 pt, bold panel
letters, Okabe-Ito colour-blind-safe palette, 95% CIs on comparisons; captions
(and every explanatory note) live in manuscript/FIGURE_PLAN.md, not in the image.

Inputs                                      Figure
  (code)                                      fig1_architecture
  eval/annotation/pred_*.jsonl                fig2_selectors
  chunks/chunks.jsonl, runs/knowledge_pilot   fig3_editions
  runs/bench.json                             fig4_outcomes, fig5_latency

Palette: reference categorical slots 1-3 (blue, orange, aqua), which validate
on every pair in light mode, plus neutral gray. Print figures are light mode.
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/span_verified"
TABLES = ROOT / "manuscript/tables"
W1, W2 = 3.5, 7.2   # single / double column, inches
BLUE, ORANGE, AQUA = "#0072B2", "#D55E00", "#009E73"          # Okabe-Ito
GRAY, LIGHT, INK, INK2, SURFACE = "#8c8c8c", "#cccccc", "#000000", "#404040", "#f7f7f7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "text.color": "black", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 600, "legend.frameon": False, "hatch.linewidth": 0.5, "lines.linewidth": 1.0,
    "mathtext.default": "regular", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.bf": "Liberation Sans:bold", "mathtext.it": "Liberation Sans:italic",
})


# function name -> file name (paper numbering). Anything not listed is not written.
FILE = {  # main text
        "fig_corpus": "fig1_corpus", "fig_workflow": "fig2_workflow", "fig_update": "fig3_update_example",
        "fig_screen": "fig4_nurse_screen", "fig_baselines": "fig5_comparison",
        # supplement
        "fig_procedure": "figS1_procedure_steps", "fig_population": "figS2_population",
        "fig_consistency": "figS3_consistency", "fig_change_grid": "figS4_change_questions",
        "fig_checks": "figS5_checks", "fig_calibration": "figS6_support_calibration"}
# Not written (content moved to tables or text): fig_timeline -> Table 2, fig_timing / fig5 -> Table 3,
# fig_comparison / fig_stability -> Table S1, fig4 / fig_mismatch_types -> merged into fig_checks.


def save(fig, name):
    if name not in FILE:
        plt.close(fig); print("  skipped (not a paper figure):", name); return
    fig.savefig(OUT / f"{FILE[name]}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote", FILE[name])



# --- fig 1: architecture -----------------------------------------------------

# --- fig 2: selectors on the study ballot -------------------------------------
def load_preds():
    out = {}
    for f in sorted(glob.glob(str(ROOT / "eval/annotation/pred_*.jsonl"))):
        name = Path(f).stem.replace("pred_", "").replace("-project", "")
        out[name] = [json.loads(l) for l in open(f)]
    return out


def fig2():
    preds = load_preds()
    order = [n for n in ("gpt-4o", "claude-opus-5", "gemini-3.7-flash", "jev") if n in preds]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    w = 0.36
    for i, field, color in ((0, "action", BLUE), (1, "quantity", ORANGE)):
        rates = []
        for n in order:
            rows = [r for r in preds[n] if r["field"] == field and r["n_candidates"] > 0]
            rates.append(100 * sum(r.get("abstained", False) for r in rows) / len(rows))
        xs = [k + (i - 0.5) * w for k in range(len(order))]
        bar_round(a, xs, rates, w, color)
        for x, v in zip(xs, rates):
            a.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    a.set_xticks(range(len(order)), [{"gpt-4o": "gpt-4o", "claude-opus-5": "claude-\nopus-5",
                                     "gemini-3.7-flash": "gemini-\n3.7-flash", "jev": "Jev"}[n] for n in order])
    a.set_ylabel("declined to answer (%)")
    a.set_title("Same ballot, different selectors", loc="left", color=INK)
    a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
             labels=["action (n=60)", "quantity (n=30)"], loc="upper left")
    a.set_ylim(0, 100); a.grid(axis="x", visible=False)

    width = 0.042
    for i, (n, color, lab) in enumerate((("gpt-4o", BLUE, "gpt-4o (log-probs)"), ("jev", AQUA, "Jev"))):
        vals = [r["p_present"] for r in preds.get(n, []) if r.get("p_present") is not None]
        counts = [sum(1 for v in vals if k / 10 <= v < (k + 1) / 10 or (k == 9 and v == 1.0))
                  for k in range(10)]
        xs = [k / 10 + 0.05 + (i - 0.5) * width for k in range(10)]
        b.bar(xs, counts, width, color=color, edgecolor="white", linewidth=1.0,
              label=f"{lab} (n={len(vals)})")
    b.set_xlim(0, 1)
    b.set_xlabel("P(any candidate states the answer)")
    b.set_ylabel("decisions")
    b.set_title("gpt-4o is all-or-nothing; Jev is graded", loc="left", color=INK)
    b.legend(loc="upper center"); b.grid(axis="x", visible=False)
    save(fig, "fig2_selectors")


# --- fig 3: WHO SSI 2016 -> 2018 ----------------------------------------------

# --- fig 4 + 5: pipeline outcomes and latency ---------------------------------
def load_bench():
    p = ROOT / "runs/bench.json"
    return json.loads(p.read_text()) if p.exists() else None




# --- answer screen and release comparison, from the audit log ------------------
OXY_Q = "Should my intubated adult patient get 80% oxygen during surgery to prevent wound infection?"


def audit_response(question, snapshot):
    """Latest recorded response to `question` at `snapshot`, verbatim."""
    import sqlite3
    con = sqlite3.connect(ROOT / "runs/verified_rag.sqlite")
    row = con.execute("SELECT response FROM responses WHERE question=? AND snapshot=? "
                      "ORDER BY at DESC LIMIT 1", (question, snapshot)).fetchone()
    return json.loads(row[0]) if row else None


def _short(title, n=48):
    t = " ".join(title.split())
    return t if len(t) <= n else t[:t.rfind(" ", 0, n)] + "…"


def _wrap(text, width):
    import textwrap
    return textwrap.fill(" ".join(text.split()), width)


SCREEN_ID = "b06100e3"   # pinned audit-log response shown in Figure 5A (release 3)
REPAIR_ID = "1125c066"   # pinned audit-log response for the repair example (Figure 5B)


def _audit_by_id(prefix):
    import sqlite3
    row = sqlite3.connect(ROOT / "runs/verified_rag.sqlite").execute(
        "SELECT response FROM responses WHERE response_id LIKE ?", (prefix + "%",)).fetchone()
    return json.loads(row[0]) if row else None


def _marker(ax, x, y, n, **kw):
    ax.text(x, y, str(n), ha="center", va="center", fontsize=5.6, color="white", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.18", fc="black", ec="none"), zorder=5, **kw)


def fig_screen():
    """Figure 5. A: the answer screen as the nurse sees it (verbatim from the
    audit log) with one sentence traced to passage, page, document version and
    release. B: an overstated sentence corrected before display."""
    import hashlib, textwrap
    r, rp = _audit_by_id(SCREEN_ID), _audit_by_id(REPAIR_ID)
    if not (r and rp):
        print("  pinned audit records missing for Figure 5"); return
    man = json.loads((ROOT / "corpus/metadata/corpus_metadata.json").read_text())
    sha = {hashlib.sha1((x.get("sha256") or x.get("filename", "")).encode()).hexdigest()[:12]: x.get("sha256", "")
           for x in man}
    by = {x["label"]: x for x in r["sources"]}
    claims = [c for c in r["claims"] if c["origin"] != "withdrawn"]
    traced = next(i for i, c in enumerate(claims) if any(by[l].get("grade_change") for l in c["sources"] if l in by))
    sig = r["signals"]
    grade = sig["evidence"].split("grade: ")[-1].split("; replaces")[0] if "grade: " in sig["evidence"] else ""

    WRAP, LINE = 96, 0.108                      # characters per line; inches per line unit
    items = [("q", _wrap("Q: " + r["question"], 84))]
    for i, c in enumerate(claims):
        src = "; ".join(f"{_short(by[l]['title'], 40)} {by[l]['year']}, p.{by[l]['page']}, tier {by[l]['tier']}"
                        for l in c["sources"] if l in by)
        items.append(("claim", (i, _wrap(c["text"], WRAP))))
        items.append(("check", (i, f"Checked: cited, details match, P = {c['support']:.2f}", src)))
    if "was '" in grade:
        items.append(("changed", _wrap("Guidance changed: " + grade, 104)))
    for name, val in (("Integrity", sig["integrity"]), ("Evidence", sig["evidence"].split("; grade:")[0]),
                      ("Consensus", sig["consensus"]), ("Relevance", sig["relevance"])):
        items.append(("signal", (name, _wrap(val, 84))))
    items.append(("foot", f"Release {r['snapshot']} \u00b7 response {r['response_id'][:8]} \u00b7 reproduced verbatim from the audit log"))

    def nl(t):
        return t.count("\n") + 1

    def height(kind, val):
        return {"q": lambda: 1.15 * nl(val) + 0.7, "claim": lambda: 1.08 * nl(val[1]) + 0.12,
                "check": lambda: 1.35, "changed": lambda: 1.08 * nl(val) + 0.9,
                "signal": lambda: 1.08 * nl(val[1]) + 0.25, "foot": lambda: 1.1}[kind]()
    total = sum(height(k, v) for k, v in items) + 1.4
    screen_h, b_h = total * LINE, 1.55
    fig_h = screen_h + b_h + 0.35
    fig = plt.figure(figsize=(W2, fig_h))
    ax = fig.add_axes([0.0, (b_h + 0.3) / fig_h, 0.63, screen_h / fig_h]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, total)
    ax.add_patch(plt.Rectangle((0.5, 0.2), 99, total - 0.4, fc="white", ec="#7f7f7f", lw=0.7))
    panel(ax, "A", x=-0.005, y=1.0)
    y, marks = total - 0.9, {}
    for kind, val in items:
        h = height(kind, val)
        if kind == "q":
            ax.text(5, y, val, fontsize=6.6, va="top", fontweight="bold")
        elif kind == "claim":
            i, txt = val
            ax.text(5, y, txt, fontsize=6.1, va="top", linespacing=1.15,
                    fontweight="bold" if i == traced else "normal")
            if i == traced:
                marks[1] = y - 0.5
        elif kind == "check":
            i, status, src = val
            ax.text(5, y - 0.15, status, fontsize=5.4, va="top", color="#006b4f")
            ax.text(43, y - 0.15, src, fontsize=5.4, va="top", color="#006b4f")
            if i == traced:
                marks[2] = (y - 0.6, 3.0); marks[3] = (y - 0.6, 41.2)
        elif kind == "changed":
            ax.add_patch(plt.Rectangle((3.5, y - h + 0.35), 93, h - 0.45, fc="#fbe6d9", ec=ORANGE, lw=0.6))
            ax.text(5, y - 0.3, val, fontsize=5.8, va="top", color="#8a3300", linespacing=1.15)
            marks[4] = y - h / 2 + 0.1
        elif kind == "signal":
            name, v = val
            ax.text(5, y, name, fontsize=5.8, va="top", fontweight="bold")
            ax.text(19, y, v, fontsize=5.8, va="top", color=INK2, linespacing=1.15)
        else:
            ax.text(5, y - 0.15, val, fontsize=5.2, va="top", color=GRAY)
        y -= h
    _marker(ax, 2.6, marks[1], 1)
    _marker(ax, marks[2][1], marks[2][0], 2)
    _marker(ax, marks[3][1], marks[3][0], 3)
    _marker(ax, 2.0, marks[4], 4)

    # provenance chain for the traced sentence
    c = claims[traced]; s0 = by[c["sources"][0]]
    pas = " ".join(s0["text"].split())
    k = pas.find("The panel suggests")
    excerpt = pas[k if k >= 0 else 0:][:150].rsplit(" ", 1)[0] + "\u2026"
    steps = [("Sentence shown", f"Checked against its cited passage: citation valid, numbers and strength words match, "
                                f"support P = {c['support']:.2f}."),
             ("Passage", f"{s0['chunk_id']}: \u201c{excerpt}\u201d"),
             ("Document version and page", f"{s0['title']}, {s0['year']}, p.{s0['page']}; "
                                            f"file SHA-256 {sha.get(s0['doc_id'], '')[:12]}\u2026; {s0['grade']}."),
             ("Release", f"Answer pinned to release {s0['snapshot']}; {s0['superseded_note']}; "
                         f"{s0['grade_change']}.")]
    ch = fig.add_axes([0.655, (b_h + 0.3) / fig_h, 0.345, screen_h / fig_h]); ch.set_axis_off()
    ch.set_xlim(0, 1); ch.set_ylim(0, 1)
    ch.text(0.0, 0.985, "Tracing sentence 1", fontsize=6.6, fontweight="bold", va="top")
    n = len(steps); top, gap = 0.93, 0.035
    bh = (top - 0.02 - gap * (n - 1)) / n
    for j, (head, body) in enumerate(steps):
        yt = top - j * (bh + gap)
        ch.add_patch(plt.Rectangle((0.0, yt - bh), 1.0, bh, fc="white", ec="black", lw=0.6))
        _marker(ch, 0.055, yt - 0.03, j + 1)
        ch.text(0.11, yt - 0.03, head, fontsize=6.1, fontweight="bold", va="center")
        ch.text(0.03, yt - 0.065, textwrap.fill(body, 58), fontsize=5.5, va="top", linespacing=1.15)
        if j < n - 1:
            ch.annotate("", xy=(0.5, yt - bh - gap + 0.003), xytext=(0.5, yt - bh - 0.003),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black", mutation_scale=6))

    # B: repair before display
    t = fig.add_axes([0.0, 0.02, 1.0, b_h / fig_h]); t.set_axis_off()
    panel(t, "B", x=-0.005, y=1.0)
    cc = next(c for c in rp["claims"] if c["origin"] == "repaired")
    first = cc["history"][0]
    p_first = next((k["score"] for k in first.get("checks", []) if k["name"] == "support"), None)
    if p_first is None:
        p_first = float(next(k["detail"] for k in first["checks"] if k["name"] == "support").split("=")[-1])
    s1 = {x["label"]: x for x in rp["sources"]}[cc["sources"][0]]
    src_txt = " ".join(s1["text"].split())
    i0 = src_txt.find("As a result, in 1972")
    src_sent = src_txt[i0: src_txt.find(".", i0) + 1] if i0 >= 0 else ""
    t.text(0.012, 0.86, f"Question: \u201c{rp['question']}\u201d", fontsize=6.1, style="italic", transform=t.transAxes)
    _rule(t, 0.78, x0=0.012, x1=0.99, lw=0.8)
    rows = [("Writer\u2019s first draft", f"rejected: support P = {p_first:.2f}", first["text"], None),
            ("After one rewrite (shown)", f"passed: support P = {cc['support']:.2f}", cc["text"], "for bathing infants"),
            ("Source passage", f"{_short(s1['title'], 52)}, {s1['year']}, p.{s1['page']}", src_sent, "for bathing infants")]
    yy = 0.68

    def bold_part(txt, hl):
        """Inline bold for the words that make the claim true (mathtext, same font)."""
        if not hl or hl not in txt:
            return txt.replace("$", r"\$")
        pre, post = txt.split(hl, 1)
        return pre + r"$\bf{" + hl.replace(" ", r"\ ") + "}$" + post
    for head, status, txt, hl in rows:
        t.text(0.012, yy, head, fontsize=6.0, fontweight="bold", va="top", transform=t.transAxes)
        if status:
            t.text(0.012, yy - 0.12, status, fontsize=5.6, va="top", transform=t.transAxes,
                   color="#a63f00" if status.startswith("rejected") else
                   ("#006b4f" if status.startswith("passed") else "#404040"))
        t.text(0.27, yy, bold_part(txt, hl), fontsize=6.1, va="top", transform=t.transAxes)
        yy -= 0.25
    _rule(t, yy + 0.06, x0=0.012, x1=0.99, lw=0.8)
    save(fig, "fig_screen")

# --- selection accuracy and calibration (MOCK gold until nurse data exists) ----
def fig_accuracy(gold_path=ROOT / "eval/annotation/mock/gold_spans_MOCK_claude-opus-5-project.jsonl"):
    import sys
    sys.path.insert(0, str(ROOT))
    from span_select.score_spans import calibration_rows, candidate_texts, compute, read
    if not Path(gold_path).exists():
        print("  no gold file for fig_accuracy"); return
    gold = read([str(gold_path)])
    mock = any(g.get("mock") for g in gold.values())
    mock_model = next((g["annotator"].replace("MOCK-", "").replace("-project", "")
                       for g in gold.values() if g.get("mock")), "")
    texts = candidate_texts()
    preds = {n: read([str(ROOT / f"eval/annotation/pred_{f}.jsonl")]) for n, f in
             (("gpt-4o", "gpt-4o-project"), ("claude-opus-5", "claude-opus-5-project"),
              ("gemini-3.7-flash", "gemini-3.7-flash-project"), ("Jev", "jev"))}
    stats = {n: compute(gold, p, texts=texts)["ALL"][0] for n, p in preds.items()}

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    names = list(preds)
    w = 0.36
    for i, (key, lab, col) in enumerate((("strict", "best passage", BLUE), ("acceptable", "any acceptable passage", AQUA))):
        vals = [100 * stats[n][key] / max(1, stats[n]["present"]) for n in names]
        xs = [k + (i - 0.5) * w for k in range(len(names))]
        bars = a.bar(xs, vals, w, color=col, edgecolor="white", linewidth=1.2, label=lab)
        for x, v, n, bar in zip(xs, vals, names, bars):
            a.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
            if mock and n.startswith(mock_model.split("-")[0]):
                bar.set_hatch("///"); bar.set_edgecolor("white")
    a.set_xticks(range(len(names)), [n.replace("-3.7-", "-\n3.7-").replace("-opus", "-\nopus") for n in names])
    a.set_ylim(0, 110); a.set_ylabel("chose a correct passage (%)"); a.grid(axis="x", visible=False)
    a.legend(loc="upper right", fontsize=8.5)
    a.set_title("Selection agreement", loc="left", color=INK)
    if mock:
        a.text(0.0, -0.3, f"Hatched: the gold was generated by the same model ({mock_model}); circular.",
               transform=a.transAxes, fontsize=7.6, color=INK2)

    b.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="perfect calibration")
    for n, col, mk in (("Jev", AQUA, "o"), ("gpt-4o", BLUE, "s")):
        rows = calibration_rows(gold, preds[n], texts=texts)
        xs, ys, ns = [], [], []
        for k in range(5):
            cell = [(sc, r) for sc, r, _ in rows if k / 5 <= sc < (k + 1) / 5 or (k == 4 and sc == 1.0)]
            if cell:
                xs.append(sum(c[0] for c in cell) / len(cell)); ys.append(sum(c[1] for c in cell) / len(cell))
                ns.append(len(cell))
        b.plot(xs, ys, color=col, lw=2, marker=mk, ms=7, mec="white", mew=1.5, label=f"{n} (n={len(rows)})")
    b.set_xlim(0, 1); b.set_ylim(0, 1.05)
    b.set_xlabel("stated probability (bin mean)"); b.set_ylabel("observed share correct")
    b.set_title("Does the probability mean what it says?", loc="left", color=INK)
    b.legend(loc="lower right", fontsize=8.5)
    if mock:
        fig.suptitle(f"MOCK: gold passages generated by {mock_model}, not nurse annotation. "
                     "Replace before submission.", x=0.01, ha="left", fontsize=9.5, color=ORANGE, y=1.03)
    save(fig, "fig_accuracy_MOCK" if mock else "fig_accuracy")


# --- Table 1: the corpus, computed from the manifest and chunks ------------------
TIER_NAME = {1: "Clinical guidelines", 2: "Nursing best practice", 3: "Patient safety",
             4: "Primary research"}


def corpus_table():
    import hashlib
    man = json.loads((ROOT / "corpus/metadata/corpus_metadata.json").read_text())
    chunks, canon = Counter(), Counter()
    for line in open(ROOT / "chunks/chunks.jsonl"):
        r = json.loads(line)
        chunks[r["doc_id"]] += 1
        canon[r["doc_id"]] += bool(r["is_canonical"])
    did = lambda r: hashlib.sha1((r.get("sha256") or r.get("filename", "")).encode()).hexdigest()[:12]
    rows = []
    for r in man:
        d = did(r)
        if d not in chunks or d in UPDATE_DOCS:
            continue
        tier = int(r["trust_level"].split("Tier")[1].strip()[0]) if "Tier" in r.get("trust_level", "") else 4
        y = str(r.get("publication_year") or "")[:4]
        rows.append((tier, r, d, int(y) if y.isdigit() else None))
    lines = ["| Tier | Documents | Passages (unique) | Issuing body or source (documents) | Years (median) | Published 2020 or later |",
             "|---|---|---|---|---|---|"]
    csv = ["tier,label,documents,passages,unique_passages,year_min,year_max,year_median,since_2020"]
    for t in (1, 2, 3, 4):
        R = [x for x in rows if x[0] == t]
        ys = sorted(y for *_, y in R if y)
        orgs = Counter((x[1].get("organization") or x[1].get("source")).split(" — ")[0].split(" / ")[0] for x in R)
        pubs = "; ".join(f"{o} ({n})" for o, n in orgs.most_common(6))
        npass, nuniq = sum(chunks[x[2]] for x in R), sum(canon[x[2]] for x in R)
        med = statistics.median(ys)
        lines.append(f"| {t}. {TIER_NAME[t]} | {len(R)} | {npass:,} ({nuniq:,}) | {pubs} | "
                     f"{ys[0]}–{ys[-1]} ({med:.0f}) | {sum(y >= 2020 for y in ys)} |")
        csv.append(f"{t},{TIER_NAME[t]},{len(R)},{npass},{nuniq},{ys[0]},{ys[-1]},{med:.0f},{sum(y >= 2020 for y in ys)}")
    keep = {x[2] for x in rows}
    tot, uniq = sum(chunks[d] for d in keep), sum(canon[d] for d in keep)
    lines.append(f"| **Total** | **{len(rows)}** | **{tot:,} ({uniq:,})** | | | |")
    note = (f"\nPassages: median {statistics.median([json.loads(l)['est_tokens'] for l in open(ROOT / 'chunks/chunks.jsonl')]):.0f} "
            f"tokens. Duplicates flagged: {tot - uniq:,} of {tot:,} ({100 * (tot - uniq) / tot:.1f}%); the newest, "
            f"most authoritative copy is kept searchable. Base corpus (release 1). Not included: the documents added "
            f"later as updates, the test cases: " + "; ".join(UPDATE_DOCS.values()) + ".\n")
    (TABLES / "table1_corpus.md").write_text("\n".join(lines) + "\n" + note)
    (TABLES / "table1_corpus.csv").write_text("\n".join(csv) + "\n")
    print("  wrote table1_corpus")


# --- the procedure: two loops around a shared versioned store -------------------
UPDATE_STEPS = [("U1", "Receive", "file hash, source, language recorded", False),
                ("U2", "Version", "new version; unchanged passages reused", False),
                ("U3", "Extract", "recommendations kept as positions in text", True),
                ("U4", "Link", "supersedes or refines; grade changes found", False),
                ("U5", "Stage and check", "loaded invisibly, checked complete", False),
                ("U6", "Publish", "one transaction: all live at once", False),
                ("U7", "Record", "release logged; older answers retired", False)]
QUESTION_STEPS = [("Q1", "Pin", "answer against one release", False),
                  ("Q2", "Find", "search, then graph: current? which patients?", True),
                  ("Q3", "Write", "short sentences, each citing its source", True),
                  ("Q4", "Check each sentence", "source cited · details match · verifier", True),
                  ("Q5", "Repair or fall back", "re-cite · rewrite once · quote · not stated", True),
                  ("Q6", "Show", "page, version, grade, change notes", False),
                  ("Q7", "Record", "audit log; nurse feedback", False)]


def fig_procedure():
    fig = plt.figure(figsize=(W2, 7.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 111)
    LX, RX, W, H, TOP, STEP = 2, 64, 34, 9.2, 96, 12.4
    code_fc, code_ec = "white", "#000000"
    model_fc, model_ec = "#deebf7", "#0072B2"

    def arrow(path, dashed=False, color=INK2):
        xs, ys = zip(*path)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=0.9,
                ls=(0, (3, 2)) if dashed else "-", solid_capstyle="round")
        ax.annotate("", xy=path[-1], xytext=path[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, shrinkA=0, shrinkB=0))

    def step_box(x, y_top, code, title, sub, model):
        fc, ec = (model_fc, model_ec) if model else (code_fc, code_ec)
        ax.add_patch(FancyBboxPatch((x, y_top - H), W, H, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + 1.6, y_top - 2.6, code, fontsize=8.2, color=INK, weight="bold", va="center")
        ax.text(x + 6.4, y_top - 2.6, title, fontsize=8.2, color=INK, weight="bold", va="center")
        ax.text(x + 1.6, y_top - 6.4, sub, fontsize=6.9, color=INK2, va="center")
        if model:
            ax.text(x + W - 1.4, y_top - 2.6, "model", fontsize=6.2, color="#0072B2",
                    style="italic", va="center", ha="right")

    for col_x, head, steps, inp in ((LX, "A   When guidance changes", UPDATE_STEPS,
                                     "new edition, policy, or measurement"),
                                    (RX, "B   When a nurse asks", QUESTION_STEPS, "a nurse's question")):
        letter, title = head.split("   ")
        ax.text(col_x - 1.5, 108, letter, fontsize=9, weight="bold", va="center")
        ax.text(col_x + 2.5, 108, title, fontsize=7.6, weight="bold", va="center")
        ax.text(col_x + W / 2, 102.4, inp, fontsize=7.6, color=INK2, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc="white", ec=GRAY, lw=0.7))
        arrow([(col_x + W / 2, 100.6), (col_x + W / 2, TOP + 0.2)])
        for i, (code, title, sub, model) in enumerate(steps):
            y = TOP - i * STEP
            step_box(col_x, y, code, title, sub, model)
            if i < len(steps) - 1:
                arrow([(col_x + W / 2, y - H), (col_x + W / 2, y - STEP + 0.2)])

    # shared stores
    gx, gw, g_top, g_bot = 40, 20, 70, 39
    ax.add_patch(FancyBboxPatch((gx, g_bot), gw, g_top - g_bot, boxstyle="square,pad=0",
                                fc="#f0f0f0", ec="#7f7f7f", lw=0.7))
    ax.text(gx + gw / 2, g_top - 3.2, "Versioned\nknowledge graph", fontsize=8.2, color=INK,
            weight="bold", ha="center", va="top")
    ax.text(gx + gw / 2, g_top - 9.6,
            "document versions\npassages, pages\n\nrecommendations\ngrades, populations\n\n"
            "supersedes, refines\n\nreleases: one live\nat a time",
            fontsize=6.6, color=INK2, ha="center", va="top", linespacing=1.25)
    a_top, a_bot = 24, 10
    ax.add_patch(FancyBboxPatch((gx, a_bot), gw, a_top - a_bot, boxstyle="square,pad=0",
                                fc="#f0f0f0", ec="#7f7f7f", lw=0.7))
    ax.text(gx + gw / 2, a_top - 3.0, "Audit log", fontsize=8.2, color=INK, weight="bold", ha="center", va="top")
    ax.text(gx + gw / 2, a_top - 7.6, "every answer, its\nrelease; nurse feedback", fontsize=6.6,
            color=INK2, ha="center", va="top", linespacing=1.25)

    ymid = lambda i: TOP - i * STEP - H / 2
    arrow([(LX + W, ymid(3)), (gx, ymid(3))])                                   # U4 -> graph
    arrow([(LX + W, ymid(5)), (gx + 4, ymid(5)), (gx + 4, g_bot)])               # U6 -> graph
    ax.text(gx + 5.2, ymid(5) + 1.6, "release N", fontsize=6.4, color=INK2, va="bottom")
    arrow([(gx + gw - 4, g_top), (gx + gw - 4, ymid(1)), (RX, ymid(1))])         # graph -> Q2
    ax.text(gx + gw - 3, ymid(1) + 1.2, "current\nrelease", fontsize=6.4, color=INK2, va="bottom")
    arrow([(RX, ymid(6)), (gx + gw, ymid(6))])                                   # Q7 -> audit
    arrow([(gx + gw - 4, a_top), (gx + gw - 4, ymid(5) - 1.2), (RX, ymid(5) - 1.2)], dashed=True)
    ax.text(gx + gw - 3, ymid(5) - 0.6, "calibration", fontsize=6.4, color=INK2, va="bottom")

    # legend: colour AND text tag
    for i, (fc, ec, lab) in enumerate(((code_fc, code_ec, "code step (rule-based, checkable)"),
                                       (model_fc, model_ec, "involves a model (marked 'model')"),
                                       ("#f0f0f0", "#7f7f7f", "shared store"))):
        x = 2 + i * 33
        ax.add_patch(FancyBboxPatch((x, 2.2), 3.2, 2.6, boxstyle="square,pad=0",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + 4.4, 3.5, lab, fontsize=7, color=INK2, va="center")
    save(fig, "fig_procedure")


# --- Figure 1: guidance changes in this corpus (timeline + what changed) ------

# --- corpus distribution: clinical area, year, issuing body, by evidence tier ---
AREAS = [  # (label, pattern) - first match wins; order matters
    ("Postpartum haemorrhage", r"pph|haemorrhage|hemorrhage|uterotonic|oxytocin"),
    ("Sepsis", r"\bsepsis\b|\bseptic\b"),
    ("Catheter-associated UTI", r"cauti|catheter|urinary"),
    ("Surgical site infection, antisepsis", r"\bssi\b|surgical site|antisep|antibiotic prophylaxis|hand hygiene|wound infection"),
    ("VTE prophylaxis", r"\bvte\b|thrombo|embol"),
    ("Enhanced recovery (ERAS)", r"\beras\b|enhanced recovery|perioperative pathways"),
    ("Fluid management", r"fluid"),
    ("Pain management", r"pain|analges"),
    ("Wound care", r"wound"),
    ("Preoperative assessment", r"preoperative|risk strat"),
    ("General perioperative nursing", r"perioperative nursing in|intraoperative nursing|nursing fundamentals"),
    ("Safety, handoff, communication", r"safety|handoff|checklist|transition|communication|sbar"),
    ("General perioperative nursing", r".*"),
]
# Documents ingested later AS UPDATES (releases 2 and 3): the test cases, shown
# in the Results (fig_timeline), not part of the base corpus figure/table.
UPDATE_DOCS = {"7c3781feaf22": "WHO SSI guideline, 2nd edition (2018)",
               "f3e61e207cc2": "WHO caesarean skin antisepsis (2021)"}
TIER_SEQ = {1: "#08306b", 2: "#2171b5", 3: "#6baed6", 4: "#c6dbef"}   # ColorBrewer Blues (ordinal)
BODY_NAME = {"PubMed_PMC": "PubMed Central", "OpenAlex": "OpenAlex", "Elsevier": "Elsevier journals",
             "AHRQ_PSNet": "AHRQ PSNet", "ERAS_Society": "ERAS Society", "OpenRN": "OpenRN"}


def corpus_rows():
    import hashlib, re
    man = json.loads((ROOT / "corpus/metadata/corpus_metadata.json").read_text())
    chunks = Counter(json.loads(l)["doc_id"] for l in open(ROOT / "chunks/chunks.jsonl"))
    did = lambda r: hashlib.sha1((r.get("sha256") or r.get("filename", "")).encode()).hexdigest()[:12]
    out = []
    for r in man:
        d = did(r)
        if d not in chunks or d in UPDATE_DOCS:
            continue
        tier = int(r["trust_level"].split("Tier")[1].strip()[0]) if "Tier" in r.get("trust_level", "") else 4
        text = " ".join(str(r.get(k, "")) for k in ("topic", "subtopic")).lower()  # catalogued topic, not article titles
        area = next(l for l, pat in AREAS if re.search(pat, text))
        y = str(r.get("publication_year") or "")[:4]
        out.append({"doc": d, "tier": tier, "area": area, "year": int(y) if y.isdigit() else None,
                    "body": BODY_NAME.get(r["source"], r["source"]), "passages": chunks[d],
                    "type": r.get("document_type", "")})
    return out



# --- comparison: conventional RAG (A), + versioning (B), full system (C) --------

# --- four systems: quality, and time (ingestion + per question) -----------------


def timing_report():
    """Combine ingestion timing and per-question timing into runs/timing.json
    and a readable table (manuscript/tables/tableS8_timing.md)."""
    ing_p, base_p = ROOT / "runs/timing_ingest.json", ROOT / "runs/baselines.json"
    if not (ing_p.exists() and base_p.exists()):
        print("  timing inputs missing"); return None
    ing, base = json.loads(ing_p.read_text()), json.loads(base_p.read_text())
    out = {"ingestion_event": ing["event"], "systems": {}}
    lines = ["| System | Ingest one new edition (WHO SSI 2018, 508 passages) | Model calls at ingestion | "
             "Per question, median (90th pct) | Model calls per question | Old edition still searchable after update? |",
             "|---|---|---|---|---|---|"]
    for key, label in SYS:
        si = ing["systems"][key]
        rows = [r for r in base["rows"] if r["system"] == key and "error" not in r]
        sec = sorted(r["seconds"] for r in rows)
        med, p90 = sec[len(sec) // 2], sec[int(0.9 * (len(sec) - 1))]
        calls = sum(r["llm_calls"] for r in rows) / len(rows)
        extrap = any(ing["steps"][st]["kind"] == "extrapolated" for st in si["steps"])
        out["systems"][key] = {"ingest_seconds": si["seconds"], "ingest_extrapolated": extrap,
                               "ingest_llm_calls": si["llm_calls"], "ingest_jev_calls": si["jev_calls"],
                               "ingest_steps": {st: round(ing["steps"][st]["seconds"], 1) for st in si["steps"]},
                               "question_median_s": med, "question_p90_s": p90,
                               "question_llm_calls_mean": round(calls, 1), "questions_timed": len(rows)}
        ing_txt = (f"{si['seconds'] / 60:.1f} min" if si["seconds"] >= 120 else f"{si['seconds']:.0f} s") + (" *" if extrap else "")
        mc = f"{si['llm_calls']} LLM" + (f" + {si['jev_calls']} Jev" if si["jev_calls"] else "")
        lines.append(f"| {label.replace(chr(10), ' ')} | {ing_txt} | {mc} | {med:.1f} s ({p90:.1f} s) | "
                     f"{calls:.1f} | {'no: retired at publication' if key == 'ours' else 'yes'} |")
    note = ("\n\\* extrapolated from a timed sample (triple extraction: 24 passages; community summaries: "
            "8 of 487), scaled at the pipeline's own parallelism. Writer for all systems: "
            f"{base['writer']}, a reasoning model, which dominates question time. CVT-RAG embeds only the "
            "125 changed passages; a hybrid pipeline given the same embedding reuse would take about 25 s.\n")
    (ROOT / "runs/timing.json").write_text(json.dumps(out, indent=2))
    pass  # superseded by Table 3 (tables())
    print("  wrote runs/timing.json and table_timing")
    return out





# ==== journal-style figures (inserted by restyle, 2026-11-23) ==================

def panel(ax, letter, x=-0.02, y=1.02, fig_coords=False):
    """Bold panel letter at the top-left, outside the plotting area."""
    kw = dict(fontsize=9, fontweight="bold", va="bottom", ha="left")
    if fig_coords:
        ax.text(x, y, letter, **kw)
    else:
        ax.text(x, y, letter, transform=ax.transAxes, **kw)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def boot_ratio(rows, num, den, reps=2000, seed=11):
    """Ratio sum(num)/sum(den) with a 95% CI from a bootstrap over answers
    (sentences are clustered within answers)."""
    import random
    rng = random.Random(seed)
    a = [(num(r), den(r)) for r in rows]
    tot = sum(d for _, d in a)
    est = sum(n for n, _ in a) / tot if tot else float("nan")
    bs = []
    for _ in range(reps):
        s = [a[rng.randrange(len(a))] for _ in a]
        d = sum(x for _, x in s)
        if d:
            bs.append(sum(x for x, _ in s) / d)
    bs.sort()
    return est, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]


# ---- Figure 2: the state-update workflow (compact) --------------------------------
KIND_STYLE = {"code": ("white", "black"), "jev": ("#e5f3ed", "#009E73"), "llm": ("#deebf7", "#0072B2"),
              "state": ("#f0f0f0", "#595959"), "io": ("#f7f7f7", "#8c8c8c")}


def fig_workflow():
    fig = plt.figure(figsize=(W2, 3.25))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 52)

    def box(x, y, w, h, head, sub="", kind="code", italic=False):
        fc, ec = KIND_STYLE[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=fc, ec=ec, lw=0.8 if kind != "io" else 0.6,
                                    ls="-" if kind != "io" else (0, (3, 1.5))))
        if sub and kind == "state":
            ax.text(x + w / 2, y + h - 2.1, head, ha="center", va="center", fontsize=6.9, fontweight="bold")
            ax.text(x + w / 2, y + h - 4.6, sub, ha="center", va="top", fontsize=5.9, linespacing=1.25)
        elif sub:
            ax.text(x + w / 2, y + h - 2.1, head, ha="center", va="center", fontsize=6.9, fontweight="bold",
                    style="italic" if italic else "normal")
            ax.text(x + w / 2, y + (h - 3.2) / 2, sub, ha="center", va="center", fontsize=5.9, linespacing=1.2)
        else:
            ax.text(x + w / 2, y + h / 2, head, ha="center", va="center", fontsize=6.6,
                    style="italic" if italic else "normal", fontweight="normal" if italic else "bold")

    def arrow(pts, label=None, lpos=None, dashed=False, ha="center", color="black"):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=0.8, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))
        if label:
            ax.text(*lpos, label, fontsize=6.0, ha=ha, va="center", color=color)

    # shared state
    box(1, 17, 17, 19, "Knowledge state $R_n$",
        "one live release of the\nversioned graph:\n\npassages and pages,\nrecommendations, grades,\npopulations,\nsupersedes / refines links", "state")
    # --- update loop (top) ---
    ax.text(20, 50.2, "Update loop: when guidance changes", fontsize=6.9, fontweight="bold", va="center")
    box(20, 36, 10, 8, "New\ndocument $d$", kind="io", italic=True)
    box(33, 35, 15.5, 10, "Code \u00b7 Stage", "hash, version;\nreuse unchanged passages", "code")
    box(51.5, 35, 15.5, 10, "Jev \u00b7 Link", "supersedes or refines?\ngrade changed?", "jev")
    box(70, 35, 15.5, 10, "Code \u00b7 Check", "complete, consistent?\nPASS / FAIL", "code")
    arrow([(30, 40), (33, 40)]); arrow([(48.5, 40), (51.5, 40)]); arrow([(67, 40), (70, 40)])
    arrow([(77.75, 45), (77.75, 47.5), (9.5, 47.5), (9.5, 36)])
    ax.text(77.0, 48.4, "PASS: update state $R_n \\rightarrow R_{n+1}$ (one-step publish)",
            fontsize=6.0, ha="right", va="bottom")
    arrow([(85.5, 40), (89, 40)])
    ax.text(89.6, 40, "FAIL: refused;\n$R_n$ unchanged", fontsize=6.0, va="center", ha="left")
    # --- question loop (bottom) ---
    ax.text(20, 31.2, "Question loop: when a nurse asks", fontsize=6.9, fontweight="bold", va="center")
    box(20, 3, 12, 6, "Question $q$", kind="io", italic=True)
    box(22, 14, 15.5, 10, "Code + Jev \u00b7 Find", "search, then graph:\ncurrent? which patients?", "jev")
    box(40.5, 14, 15.5, 10, "LLM \u00b7 Write", "short sentences $c_i$,\neach citing a passage", "llm")
    box(59, 14, 16.5, 10, "Code + Jev \u00b7 Verify", "per $c_i$: citation, details,\ncurrent, support $p_i$", "jev")
    box(81, 14, 17, 10, "Answer shown", "page, version, grade,\nchange note per sentence", "code")
    arrow([(18, 19), (22, 19)], "pin $R_n$", (20, 21.2))
    arrow([(26, 9), (26, 14)])
    arrow([(37.5, 19), (40.5, 19)]); arrow([(56, 19), (59, 19)])
    arrow([(75.5, 19), (81, 19)], "PASS", (78.25, 21.0))
    box(44.5, 1.5, 18, 8.5, "LLM \u00b7 Rewrite once", "from failed check $f_i$; fails again\n\u2192 quote source or withdraw", "llm")
    arrow([(63.5, 14), (63.5, 5.75), (62.5, 5.75)])
    ax.text(64.2, 9.2, "FAIL + $f_i$", fontsize=6.0, ha="left", va="center")
    arrow([(53.5, 10), (53.5, 12), (60.5, 12), (60.5, 14)])
    ax.text(57.0, 11.3, "re-check", fontsize=6.0, ha="center", va="top")
    box(81, 1.5, 17, 8.5, "Audit log", "answer, its release,\nnurse feedback", "state")
    arrow([(89.5, 14), (89.5, 10)])
    arrow([(81, 5.75), (72, 5.75), (72, 14)], dashed=True, color="#595959")
    ax.text(76.5, 4.4, "calibrate $p_i$", fontsize=6.0, ha="center", va="top", color="#595959")
    # legend
    for i, (kind, lab) in enumerate((("code", "rule-based code"), ("jev", "Jev (fast verifier model)"),
                                     ("llm", "LLM (writer)"))):
        fc, ec = KIND_STYLE[kind]
        y0 = 11.2 - i * 4.0
        ax.add_patch(FancyBboxPatch((1, y0), 2.6, 2.4, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=0.8))
        ax.text(4.4, y0 + 1.2, lab, fontsize=5.8, va="center")
    save(fig, "fig_workflow")


# ---- Figure 1: the base corpus -------------------------------------------------
def fig_corpus():
    R = corpus_rows()
    fig = plt.figure(figsize=(W2, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1.15], wspace=0.42, hspace=0.62,
                          left=0.215, right=0.99, top=0.86, bottom=0.17)
    a, b, c = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    tiers = (1, 2, 3, 4)
    labels = list(dict.fromkeys(l for l, _ in AREAS))
    labels.sort(key=lambda l: sum(x["area"] == l for x in R))
    left = [0] * len(labels)
    for t in tiers:
        v = [sum(x["area"] == l and x["tier"] == t for x in R) for l in labels]
        a.barh(range(len(labels)), v, 0.7, left=left, color=TIER_SEQ[t], edgecolor="white", linewidth=0.5,
               label=f"{t}  {TIER_NAME[t]}")
        left = [p + q for p, q in zip(left, v)]
    for i, n in enumerate(left):
        a.text(n + 0.3, i, str(n), va="center", fontsize=6.3)
    a.set_yticks(range(len(labels)), labels)
    a.tick_params(axis="y", length=0)
    a.set_xlabel("Documents"); a.set_xlim(0, max(left) * 1.1); a.set_ylim(-0.6, len(labels) - 0.4)
    panel(a, "A", x=-0.56)
    years = list(range(min(x["year"] for x in R if x["year"]), max(x["year"] for x in R if x["year"]) + 1))
    bottom = [0] * len(years)
    for t in tiers:
        v = [sum(x["year"] == y and x["tier"] == t for x in R) for y in years]
        b.bar(years, v, 0.85, bottom=bottom, color=TIER_SEQ[t], edgecolor="white", linewidth=0.3)
        bottom = [p + q for p, q in zip(bottom, v)]
    b.set_ylabel("Documents"); b.set_xlabel("Publication year")
    b.set_xlim(years[0] - 1, years[-1] + 1)
    panel(b, "B", x=-0.2)
    bodies = Counter((x["body"], x["tier"]) for x in R)
    order = sorted(bodies, key=lambda k: (k[1], -bodies[k]))
    c.bar(range(len(order)), [bodies[k] for k in order], 0.75, color=[TIER_SEQ[k[1]] for k in order],
          edgecolor="white", linewidth=0.3)
    for i, k in enumerate(order):
        c.text(i, bodies[k] + 1.5, str(bodies[k]), ha="center", fontsize=5.8)
    c.set_xticks(range(len(order)), [k[0] for k in order], rotation=50, ha="right", fontsize=5.8)
    c.set_ylabel("Documents"); c.set_ylim(0, max(bodies.values()) * 1.2)
    c.set_xlim(-0.6, len(order) - 0.4)
    panel(c, "C", x=-0.2)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.215, 0.995), ncol=4, handlelength=1.0,
               columnspacing=1.4, title="Evidence tier", title_fontsize=6.5, alignment="left")
    save(fig, "fig_corpus")
    lines = ["| Clinical area | Documents | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Passages |", "|---|---|---|---|---|---|---|"]
    for l in reversed(labels):
        X = [x for x in R if x["area"] == l]
        lines.append(f"| {l} | {len(X)} | " + " | ".join(str(sum(x['tier'] == t for x in X)) for t in tiers)
                     + f" | {sum(x['passages'] for x in X):,} |")
    (TABLES / "table_corpus_areas.md").write_text("\n".join(lines) + "\n")


# ---- Figure 3: test cases ---------------------------------------------------------
CHANGE_COL = {"partial": "#D55E00", "edition": "#0072B2", "narrow": "#009E73"}


def _rule(ax, y, x0=0.0, x1=1.0, lw=0.6):
    ax.plot([x0, x1], [y, y], color="black", lw=lw, transform=ax.transAxes, clip_on=False)


def fig_timeline():
    cdc = json.loads((ROOT / "runs/knowledge_cdc.json").read_text())
    pil = json.loads((ROOT / "runs/knowledge_pilot.json").read_text())
    n99, n_cdc = cdc["documents"]["99fccebb0aaa"]["graded"], len(cdc["pairs"])
    n_pairs, n_chg = pil["editions"]["pairs"], len(pil["editions"]["grade_changed"])
    fig = plt.figure(figsize=(W2, 4.0))
    ax = fig.add_axes([0.255, 0.72, 0.70, 0.24])
    lanes = [("CDC, surgical site infection", 2), ("WHO, surgical site infection", 1),
             ("WHO, caesarean skin antisepsis", 0)]
    for _, y in lanes:
        ax.axhline(y, color="#d9d9d9", lw=0.6, zorder=0)
    ax.set_yticks([y for _, y in lanes], [n for n, _ in lanes])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(1997, 2023); ax.set_ylim(-0.55, 2.6); ax.set_xlabel("Year of publication")
    for x, y, lab, ha in ((1999, 2, "1999", "center"), (2017, 2, "2017", "center"),
                          (2016, 1, "2016 (1st ed.)", "right"), (2018, 1, "2018 (2nd ed.)", "left"),
                          (2021, 0, "2021", "left2")):
        ax.plot([x], [y], "o", ms=4.2, color="black", zorder=3)
        if ha == "left2":
            ax.text(x + 0.45, y, lab, ha="left", va="center", fontsize=6.3)
        else:
            ax.text(x + {"right": 0.3, "left": -0.3, "center": 0}[ha], y + 0.22, lab, ha=ha, fontsize=6.3)

    def arrow(x0, y0, x1, y1, col, rad=0.0):
        ax.annotate("", xy=(x1 - 0.2, y1), xytext=(x0 + 0.2, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, shrinkA=3, shrinkB=3,
                                    mutation_scale=7, connectionstyle=f"arc3,rad={rad}"))
    arrow(1999, 2, 2017, 2, CHANGE_COL["partial"])
    ax.text(2008, 2.12, "partial update", ha="center", va="bottom", fontsize=6.3, color=CHANGE_COL["partial"])
    arrow(2016, 1, 2018, 1, CHANGE_COL["edition"])
    ax.text(2017, 0.72, "new edition", ha="center", va="top", fontsize=6.3, color=CHANGE_COL["edition"])
    arrow(2018, 1, 2021, 0, CHANGE_COL["narrow"], rad=-0.25)
    ax.text(2019.3, 0.05, "narrower guidance", ha="right", va="center", fontsize=6.3, color=CHANGE_COL["narrow"])
    panel(ax, "A", x=-0.36, y=1.04)

    t = fig.add_axes([0.02, 0.0, 0.96, 0.57]); t.set_axis_off()
    panel(t, "B", x=0.0, y=1.04)
    cols = [(0.0, "Kind of change"), (0.24, "Recommendation"), (0.46, "Before"), (0.73, "After")]
    rows = [
        ("partial", f"Partial update\nCDC 1999 → 2017\n{n_cdc} of {n99} graded\nrecommendations replaced",
         [("Antibiotic prophylaxis,\ncaesarean section", "After cord clamping\n(high-risk caesarean only)",
           "Before skin incision,\nevery caesarean"),
          ("Perioperative glucose", "Control hyperglycaemia\n(diabetic patients)",
           "Target <200 mg/dL,\nwith or without diabetes"),
          ("Skin preparation", "An appropriate antiseptic", "Alcohol-based antiseptic"),
          ("Preoperative bathing", "Antiseptic agent required", "Soap or antiseptic agent")]),
        ("edition", f"New edition\nWHO 2016 → 2018\n{n_chg} of {n_pairs} paired\nrecommendations changed",
         [("80% inspired oxygen,\nintubated adults", "Strong (“the panel recommends”)",
           "Conditional (“the panel suggests”)")]),
        ("narrow", "Narrower guidance\nWHO 2018 → 2021\nno recommendation retired",
         [("Chlorhexidine–alcohol\nskin preparation", "All surgical patients (WHO 2018)",
           "Stated for elective or\nemergency caesarean section")]),
    ]
    top, line = 0.95, 0.052
    _rule(t, top + 0.04, lw=0.8)
    for x, h in cols:
        t.text(x + 0.005, top, h, fontsize=6.6, fontweight="bold", va="center", transform=t.transAxes)
    _rule(t, top - 0.04, lw=0.5)
    y = top - 0.08
    for key, kind, items in rows:
        y0 = y
        y_min = y - (kind.count("\n") + 1) * line
        for i, (rec, before, after) in enumerate(items):
            nl = max(s.count("\n") + 1 for s in (rec, before, after))
            for (x, _), txt in zip(cols[1:], (rec, before, after)):
                t.text(x + 0.005, y, txt, fontsize=6.3, va="top", transform=t.transAxes, linespacing=1.2)
            y -= nl * line + 0.018
        y = min(y, y_min - 0.012)
        t.add_patch(plt.Rectangle((0.0, y + 0.012), 0.006, y0 - y - 0.012, transform=t.transAxes,
                                  color=CHANGE_COL[key], lw=0))
        t.text(0.014, y0, kind, fontsize=6.3, va="top", transform=t.transAxes, linespacing=1.2)
        if key != "narrow":
            _rule(t, y + 0.006, lw=0.3)
    _rule(t, y + 0.006, lw=0.8)
    save(fig, "fig_timeline")


# ---- Figure 4: one guideline update, end to end --------------------------------------
def fig_update():
    OLD, NEW, CS = "e7ad0c398aa2", "7c3781feaf22", "f3e61e207cc2"
    rows = [json.loads(l) for l in open(ROOT / "chunks/chunks.jsonl")]
    by_doc = Counter(r["doc_id"] for r in rows)
    old = [r for r in rows if r["doc_id"] == OLD]
    new = [r for r in rows if r["doc_id"] == NEW]
    old_texts = {" ".join(r["text"].split()) for r in old}
    new_same = sum(1 for r in new if " ".join(r["text"].split()) in old_texts)
    old_dup = sum(1 for r in old if (r.get("duplicate_of") or "").startswith(NEW))
    ed = json.loads((ROOT / "runs/knowledge_pilot.json").read_text())["editions"]
    n_pairs, chg = ed["pairs"], ed["grade_changed"]
    base_docs = len({r["doc_id"] for r in rows}) - len(UPDATE_DOCS)
    base_pass = len(rows) - sum(by_doc[d] for d in UPDATE_DOCS)

    fig = plt.figure(figsize=(W2, 4.4))
    # A: releases
    a = fig.add_axes([0.03, 0.82, 0.94, 0.14]); a.set_axis_off(); a.set_xlim(0, 1); a.set_ylim(0, 1)
    panel(a, "A", x=-0.02, y=1.0)
    pts = [(0.12, "Release 1", f"base corpus\n{base_docs} documents, {base_pass:,} passages"),
           (0.50, "Release 2", f"+ WHO SSI 2018 ({by_doc[NEW]} passages)\n− WHO SSI 2016 ({by_doc[OLD]}) retired"),
           (0.86, "Release 3", f"+ WHO caesarean antisepsis 2021 ({by_doc[CS]})\nnothing retired (refines 2018)")]
    a.plot([0.12, 0.86], [0.62, 0.62], color="black", lw=0.6)
    for x, h, sub in pts:
        a.plot([x], [0.62], "o", ms=5, color="black")
        a.text(x, 0.86, h, ha="center", fontsize=6.8, fontweight="bold")
        a.text(x, 0.40, sub, ha="center", va="top", fontsize=6.3, linespacing=1.2)
    # B: what the update touched
    b = fig.add_axes([0.255, 0.50, 0.50, 0.24])
    panel(b, "B", x=-0.33, y=1.04)
    bars = [(f"WHO 2018 passages (n={len(new)})", new_same, len(new) - new_same,
             "identical to a 2016 passage", "changed or new"),
            (f"WHO 2016 passages (n={len(old)})", old_dup, len(old) - old_dup,
             "duplicate of a 2018 passage", "no 2018 duplicate"),
            (f"Paired recommendations (n={n_pairs})", n_pairs - len(chg), len(chg),
             "grade unchanged", "grade changed")]
    for i, (lab, same, diff, ls, ld) in enumerate(bars):
        y = len(bars) - 1 - i
        tot = same + diff
        b.barh(y, 100 * same / tot, 0.6, color="#bdbdbd", edgecolor="white", linewidth=0.5)
        b.barh(y, 100 * diff / tot, 0.6, left=100 * same / tot, color="#D55E00", edgecolor="white", linewidth=0.5)
        b.text(1.5, y, f"{same} {ls}", fontsize=6.1, va="center")
        b.text(102, y, f"{diff} {ld}", fontsize=6.1, va="center", ha="left", color="#a63f00", clip_on=False)
    b.set_yticks(range(len(bars)), [x[0] for x in bars][::-1])
    b.tick_params(axis="y", length=0); b.spines["left"].set_visible(False)
    b.set_xlim(0, 100); b.set_ylim(-0.5, len(bars) - 0.5); b.set_xlabel("Percent of each set")
    # C: the answer before and after
    t = fig.add_axes([0.03, 0.02, 0.94, 0.33]); t.set_axis_off()
    panel(t, "C", x=-0.02, y=1.0)
    r1, r3 = audit_response(OXY_Q, 1), audit_response(OXY_Q, 3)

    def first_who(r):
        by = {x["label"]: x for x in r["sources"]}
        for c in r["claims"]:
            for l in c["sources"]:
                if l in by and "WHO Global" in by[l]["title"]:
                    return c, by[l]
        return r["claims"][0], by.get(r["claims"][0]["sources"][0])

    t.text(0.0, 0.93, f"Question: “{OXY_Q}”", fontsize=6.6, style="italic", transform=t.transAxes)
    _rule(t, 0.86, lw=0.8)
    for x, head in ((0.0, "Answered at release 1 (2016 edition live)"), (0.51, "Answered at release 3 (2018 edition live)")):
        t.text(x, 0.80, head, fontsize=6.6, fontweight="bold", va="center", transform=t.transAxes)
    _rule(t, 0.74, lw=0.5)
    for x, r in ((0.0, r1), (0.51, r3)):
        if not r:
            continue
        c, src = first_who(r)
        ev = r["signals"]["evidence"]
        grade = ev.split("grade: ")[-1].split(" (")[0] if "grade: " in ev else "no grade recorded"
        t.text(x, 0.70, _wrap(c["text"], 78), fontsize=6.1, va="top", transform=t.transAxes, linespacing=1.2)
        t.text(x, 0.30, f"Grade shown: {grade.split(',')[0]}", fontsize=6.3, va="top", fontweight="bold",
               color="#D55E00" if "Conditional" in grade else "black", transform=t.transAxes)
        note = "; change note: was ‘Strong’ in 2016" if "was '" in ev else ""
        t.text(x, 0.17, f"{_short(src['title'], 40)} {src['year']}, p.{src['page']}{note}", fontsize=5.9,
               va="top", color="#404040", transform=t.transAxes)
    _rule(t, 0.06, lw=0.8)
    save(fig, "fig_update")


# ---- Figures 6 and 7: four systems ------------------------------------------------
SYS = [("hybrid_rag", "Hybrid RAG"), ("neo4j_graphrag", "Neo4j GraphRAG"),
       ("graphrag", "Microsoft GraphRAG"), ("ours", "CVT-RAG")]
SYS_COL = {"hybrid_rag": "#E69F00", "neo4j_graphrag": "#56B4E9", "graphrag": "#CC79A7", "ours": "#0072B2",
           "llm_only": "#7f7f7f"}
# Systems in the answer comparison (Figure 5, Figure S4): the four retrieval systems plus the
# same writer with no retrieval. Timing (Table 3) and run stability (Table S1) keep SYS.
SYS_CMP = [("llm_only", "LLM alone (no retrieval)")] + SYS
NO_TEXT_NOTE = {"graphrag": "not checkable: sentences rest on\nmodel-written summaries",
                "llm_only": "not checkable: no sources cited"}


def _sys_axis(ax, keys, labels=True):
    ys = list(range(len(keys)))[::-1]
    ax.set_yticks(ys, [dict(SYS_CMP)[k] for k in keys] if labels else [""] * len(keys))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    for y in ys:
        ax.axhline(y, color="#ececec", lw=0.5, zorder=0)
    return dict(zip(keys, ys))


def fig_baselines():
    p = ROOT / "runs/baselines.json"
    if not p.exists():
        print("  runs/baselines.json missing"); return
    D = json.loads(p.read_text())
    rows = [r for r in D["rows"] if "error" not in r]
    keys = [k for k, _ in SYS_CMP]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1.25, 1], "wspace": 0.12})
    stats = {}
    # A: superseded guidance on change questions
    a = axs[0]; Y = _sys_axis(a, keys)
    for k in keys:
        R = [r for r in rows if r["system"] == k and r["kind"] == "change"]
        p_, lo, hi = wilson(sum(bool(r["uses_superseded"]) for r in R), len(R))
        a.errorbar(100 * p_, Y[k], xerr=[[100 * (p_ - lo)], [100 * (hi - p_)]], fmt="o", ms=4,
                   color=SYS_COL[k], ecolor=SYS_COL[k], elinewidth=0.9, capsize=2)
        a.text(100 * hi + 3, Y[k] + 0.12, f"{sum(bool(r['uses_superseded']) for r in R)}/{len(R)}", fontsize=6,
               va="bottom", ha="left", color="#404040")
        stats[k] = {"superseded": (p_, lo, hi, len(R))}
    a.set_xlim(-3, 100); a.set_xlabel("Change questions answered with\nsuperseded guidance (%)")
    panel(a, "A", x=-0.05, y=1.03)
    # B: per 100 sentences
    b = axs[1]; _sys_axis(b, keys, labels=False)
    marks = [("detail_mismatch", "Detail does not match source", "s", 0.18, True),
             ("unsupported", "Not supported by cited text", "o", 0.0, True),
             ("unsupported_strict", "Not supported, stricter wording", "o", -0.18, False)]
    for k in keys:
        R = [r for r in rows if r["system"] == k]
        if all(r["traceable"] == 0 for r in R):
            b.text(1, Y[k], NO_TEXT_NOTE.get(k, "not checkable"), fontsize=5.9,
                   va="center", color="#404040", linespacing=1.15)
            continue
        for key, lab, m, dy, filled in marks:
            est, lo, hi = boot_ratio(R, lambda r: r[key], lambda r: r["n"])
            b.errorbar(100 * est, Y[k] + dy, xerr=[[100 * (est - lo)], [100 * (hi - est)]], fmt=m, ms=3.4,
                       color=SYS_COL[k], mfc=SYS_COL[k] if filled else "white", mec=SYS_COL[k], mew=0.8,
                       ecolor=SYS_COL[k], elinewidth=0.8, capsize=1.6)
            stats[k][key] = (est, lo, hi)
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker=m, ls="", color="#404040", mfc="#404040" if f else "white", ms=3.6, mew=0.8)
                      for _, _, m, _, f in marks], labels=[lab for _, lab, *_ in marks],
             loc="lower right", handletextpad=0.3, borderaxespad=0.2, fontsize=5.9)
    b.set_xlim(-1, 40); b.set_xlabel("Sentences shown (per 100)")
    panel(b, "B", x=-0.02, y=1.03)
    # C: traceable
    c = axs[2]; _sys_axis(c, keys, labels=False)
    for k in keys:
        R = [r for r in rows if r["system"] == k]
        est, lo, hi = boot_ratio(R, lambda r: r["traceable"], lambda r: r["n"])
        c.errorbar(100 * est, Y[k], xerr=[[100 * (est - lo)], [100 * (hi - est)]], fmt="o", ms=4,
                   color=SYS_COL[k], ecolor=SYS_COL[k], elinewidth=0.9, capsize=2)
        stats[k]["traceable"] = (est, lo, hi)
    c.set_xlim(-3, 103); c.set_xlabel("Sentences traceable to a\nguideline passage (%)")
    panel(c, "C", x=-0.02, y=1.03)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.93, bottom=0.27)
    save(fig, "fig_baselines")
    (ROOT / "runs/baselines_ci.json").write_text(json.dumps(stats, indent=1))


def fig_timing(t):
    if not t:
        return
    keys = [k for k, _ in SYS]
    base = json.loads((ROOT / "runs/baselines.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 1.95), gridspec_kw={"wspace": 0.1})
    Y = _sys_axis(a, keys)
    for k in keys:
        s = t["systems"][k]
        v = s["ingest_seconds"] / 60
        a.barh(Y[k], v, 0.55, color=SYS_COL[k], hatch="/////" if s["ingest_extrapolated"] else None,
               edgecolor="white" if not s["ingest_extrapolated"] else SYS_COL[k], fill=not s["ingest_extrapolated"],
               linewidth=0.8)
        txt = f"{v:.1f} min" if v >= 1 else f"{s['ingest_seconds']:.0f} s"
        a.text(v * 1.12, Y[k], txt + (" (est.)" if s["ingest_extrapolated"] else ""), va="center", fontsize=6.1)
    a.set_xscale("log"); a.set_xlim(0.2, 200)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [0.5, 1, 2, 5, 10, 20, 50, 100]
    a.xaxis.set_major_locator(FixedLocator(tk)); a.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in tk]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xlabel("Time to ingest one new edition (min, log scale)")
    panel(a, "A", x=-0.02, y=1.03)
    _sys_axis(b, keys, labels=False)
    data = [[r["seconds"] for r in base["rows"] if r["system"] == k and "error" not in r] for k in keys]
    bp = b.boxplot(data, positions=[Y[k] for k in keys], vert=False, widths=0.5, patch_artist=True,
                   showfliers=True, flierprops=dict(marker="o", ms=1.8, mec="#7f7f7f", mfc="none", mew=0.4),
                   medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                   boxprops=dict(lw=0.6))
    for patch, k in zip(bp["boxes"], keys):
        patch.set_facecolor(SYS_COL[k]); patch.set_alpha(0.85)
    _sys_axis(b, keys, labels=False)
    for k, d in zip(keys, data):
        b.text(max(d) + 1.2, Y[k], f"median {statistics.median(d):.1f} s (n={len(d)})", va="center", fontsize=6.0)
    b.set_xlim(0, 75); b.set_xlabel("Time to answer one question (s)")
    panel(b, "B", x=-0.02, y=1.03)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.22)
    save(fig, "fig_timing")


# ---- Supplement ---------------------------------------------------------------------
def fig4(bench):
    qs = bench["questions"]
    origins = Counter(c["origin"] for q in qs for c in q["claims"])
    order = [("synthesized", "Passed on first check", "#0072B2"), ("repaired", "Repaired once", "#009E73"),
             ("verbatim", "Replaced by source text", "#E69F00"), ("withdrawn", "Withdrawn", "#7f7f7f")]
    total = sum(origins.values())
    fails = Counter(n for q in qs for c in q["claims"] for n in c["failed_checks"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 1.7), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.35})
    left = 0
    for key, lab, col in order:
        n = origins.get(key, 0)
        a.barh(0, n, 0.35, left=left, color=col, edgecolor="white", linewidth=0.5, label=f"{lab} ({n})")
        left += n
    a.set_xlim(0, total); a.set_ylim(-0.5, 0.5); a.set_yticks([]); a.spines["left"].set_visible(False)
    a.set_xlabel(f"Claims ({total} in {len(qs)} answers)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, handlelength=1.0)
    panel(a, "A", x=-0.02, y=1.05)
    names = ["citation", "details", "support"]
    vals = [fails.get(n, 0) for n in names]
    b.bar(range(3), vals, 0.55, color="#0072B2")
    for i, v in enumerate(vals):
        b.text(i, v + max(vals + [1]) * 0.03, str(v), ha="center", fontsize=6.2)
    b.set_xticks(range(3), ["Citation\n(code)", "Exact details\n(code)", "Support\n(Jev)"])
    b.set_ylabel("Claims failing\non first check")
    from matplotlib.ticker import MaxNLocator
    b.yaxis.set_major_locator(MaxNLocator(integer=True))
    panel(b, "B", x=-0.3, y=1.05)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.42)
    save(fig, "fig4_outcomes")


def fig5(bench):
    qs = bench["questions"]
    stages = [("search_s", "Search (embedding cached)"), ("graph_s", "Graph lookups"),
              ("population_s", "Population check"), ("synthesize_s", "Writer model"),
              ("verify_s", "Sentence checks"), ("repair_s", "Repair")]
    med = [statistics.median([q["timings"].get(k, 0) for q in qs]) for k, _ in stages]
    fig, a = plt.subplots(figsize=(W1, 1.9))
    y = list(range(len(stages)))[::-1]
    a.barh(y, med, 0.55, color="#0072B2")
    for yi, v in zip(y, med):
        a.text(v + 0.05, yi, f"{v:.2f} s" if v >= 0.01 else "<0.01 s", va="center", fontsize=6.2)
    a.set_yticks(y, [lab for _, lab in stages]); a.tick_params(axis="y", length=0)
    a.set_xlabel(f"Median seconds per answer (n={len(qs)})"); a.set_xlim(0, max(med) * 1.3)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.97, bottom=0.22)
    save(fig, "fig5_latency")


def fig_comparison():
    path = ROOT / "runs/comparison.json"
    if not path.exists():
        print("  runs/comparison.json missing"); return
    S = json.loads(path.read_text())["summary"]
    systems = [("A", "A  Conventional RAG"), ("B", "B  + versioned releases"), ("C", "C  CVT-RAG (+ sentence checks)")]
    col = {"A": "#7f7f7f", "B": "#56B4E9", "C": "#0072B2"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 1.7), gridspec_kw={"wspace": 0.1})
    ys = {m: 2 - i for i, (m, _) in enumerate(systems)}
    for ax, lab in ((a, True), (b, False)):
        ax.set_yticks(list(ys.values()), [l for _, l in systems] if lab else ["", "", ""])
        ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, 2.6)
        for y in ys.values():
            ax.axhline(y, color="#ececec", lw=0.5, zorder=0)
    for m, _ in systems:
        n = S[f"change:{m}"]["answers"]; k = S[f"change:{m}"]["uses_superseded"]
        p_, lo, hi = wilson(k, n)
        a.errorbar(100 * p_, ys[m], xerr=[[100 * (p_ - lo)], [100 * (hi - p_)]], fmt="o", ms=4, color=col[m],
                   elinewidth=0.9, capsize=2)
        a.text(100 * hi + 3, ys[m] + 0.12, f"{k}/{n}", fontsize=6, va="bottom")
    a.set_xlim(-3, 100); a.set_xlabel("Change questions answered with\nsuperseded guidance (%)")
    panel(a, "A", x=-0.05, y=1.04)
    for key, mk, dy in (("detail_mismatch", "s", 0.15), ("verifier_unsupported", "o", -0.15)):
        for m, _ in systems:
            sh = S[f"change:{m}"]["sentences_shown"] + S[f"general:{m}"]["sentences_shown"]
            bad = S[f"change:{m}"][key] + S[f"general:{m}"][key]
            p_, lo, hi = wilson(bad, sh)
            b.errorbar(100 * p_, ys[m] + dy, xerr=[[100 * (p_ - lo)], [100 * (hi - p_)]], fmt=mk, ms=3.4,
                       color=col[m], elinewidth=0.8, capsize=1.6)
    from matplotlib.lines import Line2D
    b.legend(handles=[Line2D([], [], marker="s", ls="", color="#404040", ms=3.4),
                      Line2D([], [], marker="o", ls="", color="#404040", ms=3.4)],
             labels=["Detail does not match source", "Not supported (verifier)"], loc="lower right", fontsize=5.9)
    b.set_xlim(-0.15, 6); b.set_xlabel("Sentences shown (per 100)")
    panel(b, "B", x=-0.02, y=1.04)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.3)
    save(fig, "fig_comparison")



# ---- Supplement S10, S12-S14 (2026-11-23) ------------------------------------------
EVENT_NAME = {"99fccebb0aaa:R036": "CDC: caesarean antibiotic timing", "99fccebb0aaa:R018": "CDC: glucose target",
              "99fccebb0aaa:R023": "CDC: skin preparation", "99fccebb0aaa:R021": "CDC: preoperative bathing",
              "e7ad0c398aa2:R087": "WHO: 80% inspired oxygen"}


def fig_calibration():
    p = ROOT / "runs/support_calibration.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    cases = d.get("per_case")
    if not cases:
        print("  per-case calibration scores missing"); return
    kinds = [("verbatim", "True sentence\n(should pass)"), ("number_changed", "Number changed"),
             ("negated", "Negation added"), ("unrelated", "Unrelated sentence")]
    forms = [("p_cited_passage", "Verifier wording (cited_passage)", "#0072B2"),
             ("p_cited", "Stricter wording (cited)", "#E69F00")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.3})
    import random
    rng = random.Random(3)
    for i, (k, lab) in enumerate(kinds):
        for j, (f, flab, col) in enumerate(forms):
            v = [c[f] for c in cases if c["kind"] == k]
            x0 = i + (j - 0.5) * 0.34
            a.scatter([x0 + rng.uniform(-0.09, 0.09) for _ in v], v, s=2.5, color=col, alpha=0.55, lw=0,
                      label=flab if i == 0 else None)
            q = sorted(v)
            med = q[len(q) // 2]
            a.plot([x0 - 0.13, x0 + 0.13], [med, med], color="black", lw=1.0)
            ok = sum((x >= 0.5) == (k == "verbatim") for x in v) / len(v)
            a.text(x0, 1.04, f"{100 * ok:.0f}%", ha="center", va="bottom", fontsize=5.6, color=col)
    a.axhline(0.5, color="#7f7f7f", lw=0.6, ls=(0, (3, 2)))
    a.set_xticks(range(4), [l for _, l in kinds]); a.tick_params(axis="x", length=0)
    a.set_ylim(-0.03, 1.12); a.set_ylabel("Jev P(supported)")
    panel(a, "A", x=-0.09, y=1.03)
    for f, flab, col in forms:
        pos = sorted((c[f] for c in cases if c["label"] == 1), reverse=True)
        neg = [c[f] for c in cases if c["label"] == 0]
        ths = sorted({c[f] for c in cases} | {0.0, 1.01}, reverse=True)
        tpr = [sum(x >= t for x in pos) / len(pos) for t in ths]
        fpr = [sum(x >= t for x in neg) / len(neg) for t in ths]
        auc = sum((p_ > n_) + 0.5 * (p_ == n_) for p_ in pos for n_ in neg) / (len(pos) * len(neg))
        b.plot(fpr, tpr, color=col, lw=1.1, label=f"{flab}\nAUC {auc:.3f}")
        t5 = sum(x >= 0.5 for x in pos) / len(pos); f5 = sum(x >= 0.5 for x in neg) / len(neg)
        b.plot([f5], [t5], "o", ms=3.5, color=col)
    b.plot([0, 1], [0, 1], color="#bdbdbd", lw=0.6, ls=(0, (3, 2)))
    b.set_xlim(-0.01, 1); b.set_ylim(0, 1.01)
    b.set_xlabel("False-positive rate (controls passed)"); b.set_ylabel("True-positive rate")
    b.legend(loc="lower right", fontsize=5.8, handlelength=1.2)
    panel(b, "B", x=-0.22, y=1.03)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.2)
    save(fig, "fig_calibration")


def fig_change_grid():
    D = json.loads((ROOT / "runs/baselines.json").read_text())
    qmeta = {q["question"]: q for q in map(json.loads, open(ROOT / "eval/comparison/change_questions.jsonl"))}
    keys = [k for k, _ in SYS_CMP]
    rows = [r for r in D["rows"] if r["kind"] == "change" and "error" not in r]
    qs = list(dict.fromkeys(r["question"] for r in rows))
    order0 = list(qs)
    qs = sorted(qs, key=lambda q: (list(EVENT_NAME).index(qmeta[q]["old_rec"]), order0.index(q)))
    cell = {(r["question"], r["system"]): r for r in rows}
    state_col = {"superseded": "#D55E00", "current": "#0072B2", "neither": "#e0e0e0"}
    fig, ax = plt.subplots(figsize=(W2, 3.3))
    for i, q in enumerate(qs):
        y = len(qs) - 1 - i
        for j, k in enumerate(keys):
            r = cell.get((q, k))
            if not r:
                continue
            st = "superseded" if r["uses_superseded"] else ("current" if r["states_current"] else "neither")
            ax.add_patch(plt.Rectangle((j + 0.06, y + 0.08), 0.88, 0.84, color=state_col[st], lw=0))
            if st == "superseded" and r["states_current"]:
                ax.add_patch(plt.Rectangle((j + 0.06, y + 0.08), 0.88, 0.84, fill=False, hatch="////",
                                           edgecolor="#0072B2", lw=0))
    ax.set_xlim(0, len(keys)); ax.set_ylim(0, len(qs))
    ax.set_xticks([j + 0.5 for j in range(len(keys))], [("LLM alone\n(no retrieval)" if k == "llm_only" else dict(SYS)[k].replace(" ", "\n", 1)) if k != "ours" else "CVT-RAG"
                                                          for k in keys])
    ax.xaxis.tick_top(); ax.tick_params(axis="both", length=0)
    labels = []
    for i, q in enumerate(qs):
        labels.append(_short(q, 70))
    ax.set_yticks([len(qs) - 1 - i + 0.5 for i in range(len(qs))], labels, fontsize=5.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    prev = None
    for i, q in enumerate(qs):
        ev = EVENT_NAME[qmeta[q]["old_rec"]]
        if ev != prev:
            y = len(qs) - i
            ax.axhline(y, color="black", lw=0.5, xmin=0, xmax=1, clip_on=False)
            n_ev = sum(EVENT_NAME[qmeta[x]["old_rec"]] == ev for x in qs)
            ax.text(len(keys) + 0.08, y - n_ev / 2, ev, va="center", fontsize=5.9, fontweight="bold")
            prev = ev
    ax.axhline(0, color="black", lw=0.5, clip_on=False)
    from matplotlib.patches import Patch
    hs = [Patch(color=state_col["superseded"]), Patch(facecolor=state_col["superseded"], hatch="////",
                                                       edgecolor="#0072B2", lw=0),
          Patch(color=state_col["current"]), Patch(color=state_col["neither"])]
    ax.legend(hs, ["Used superseded guidance", "Superseded and current both stated", "Stated current guidance only",
                   "Neither detected"], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=5.8,
              handlelength=1.2)
    fig.subplots_adjust(left=0.36, right=0.8, top=0.9, bottom=0.1)
    save(fig, "fig_change_grid")


def fig_stability():
    runs = [("Run 1", ROOT / "runs/baselines_run1_unfiltered.json"),
            ("Run 2", ROOT / "runs/baselines_run2_cited_prompt.json"), ("Run 3", ROOT / "runs/baselines.json")]
    keys = [k for k, _ in SYS]
    data = {}
    for name, p in runs:
        rows = [r for r in json.loads(p.read_text())["rows"] if "error" not in r]
        for k in keys:
            R = [r for r in rows if r["system"] == k]
            C = [r for r in R if r["kind"] == "change"]
            n = sum(r["n"] for r in R) or 1
            data[(name, k)] = {"superseded": 100 * sum(bool(r["uses_superseded"]) for r in C) / len(C),
                               "current": 100 * sum(bool(r["states_current"]) for r in C) / len(C),
                               "detail": 100 * sum(r["detail_mismatch"] for r in R) / n,
                               "traceable": 100 * sum(r["traceable"] for r in R) / n}
    metrics = [("superseded", "Change questions answered with\nsuperseded guidance (%)", (-3, 100)),
               ("current", "Change questions stating the\ncurrent guidance (%)", (-3, 100)),
               ("detail", "Detail mismatch\n(per 100 sentences)", (-0.5, 15)),
               ("traceable", "Sentences traceable to a\nguideline passage (%)", (-3, 103))]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9), gridspec_kw={"wspace": 0.1})
    marks = {"Run 1": "^", "Run 2": "s", "Run 3": "o"}
    for i, (m, lab, lim) in enumerate(metrics):
        ax = axs[i]; Y = _sys_axis(ax, keys, labels=(i == 0))
        for k in keys:
            use = [(n, p_) for n, p_ in runs if not (m == "detail" and n == "Run 1")]
            if k == "graphrag" and m in ("detail", "traceable"):
                ax.text(lim[0] + 0.03 * (lim[1] - lim[0]), Y[k], "n/a (no guideline text)" if m == "detail" else
                        "0 (summaries, not passages)", va="center", fontsize=5.6, color="#404040")
                continue
            vals = [data[(n, k)][m] for n, _ in use]
            ax.plot([min(vals), max(vals)], [Y[k], Y[k]], color=SYS_COL[k], lw=1.0, alpha=0.6)
            dys = {"Run 1": 0.16, "Run 2": 0.0, "Run 3": -0.16}
            for (n, _), v in zip(use, vals):
                dy = dys[n]
                ax.plot([v], [Y[k] + dy], marks[n], ms=3.2, color=SYS_COL[k], mfc="white" if n != "Run 3" else SYS_COL[k],
                        mew=0.8)
        ax.set_xlim(*lim); ax.set_xlabel(lab, fontsize=6.2)
        panel(ax, "ABCD"[i], x=-0.03, y=1.03)
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([], [], marker=marks[n], ls="", color="#404040", mfc="white" if n != "Run 3" else "#404040",
                          ms=3.4, mew=0.8) for n, _ in runs],
                  ["Run 1", "Run 2", "Run 3 (reported)"], loc="lower right", fontsize=5.8, handletextpad=0.2)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.3)
    save(fig, "fig_stability")


def fig_mismatch_types():
    D = json.loads((ROOT / "runs/baselines.json").read_text())
    keys = [k for k in (k for k, _ in SYS) if k != "graphrag"]
    cats = [("quantity", "Number or unit not in source", "#E69F00"),
            ("strength", "Recommendation strength not in source", "#D55E00"),
            ("negation", "Negation dropped or added", "#CC79A7")]

    def kinds(detail):
        d = detail or ""
        out = set()
        if "quantity" in d:
            out.add("quantity")
        if "strong/must" in d or "conditional/suggest" in d:
            out.add("strength")
        if "negation" in d:
            out.add("negation")
        return out
    fig, ax = plt.subplots(figsize=(W1 + 1.2, 1.7))
    Y = _sys_axis(ax, keys)
    counts = {}
    for k in keys:
        S = [s for r in D["rows"] if r["system"] == k and "error" not in r for s in r.get("per_sentence", [])]
        n = len(S)
        left = 0.0
        for c, lab, col in cats:
            v = 100 * sum(c in kinds(s["detail"]) for s in S) / n
            counts[(k, c)] = sum(c in kinds(s["detail"]) for s in S)
            ax.barh(Y[k], v, 0.55, left=left, color=col, edgecolor="white", linewidth=0.5,
                    label=lab if k == keys[0] else None)
            left += v
        ax.text(left + 0.2, Y[k], f"n = {n} sentences", va="center", fontsize=5.8)
    ax.set_xlim(0, 14); ax.set_xlabel("Sentences with each kind of detail mismatch (per 100 shown)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.42), ncol=1, fontsize=5.8, handlelength=1.0)
    fig.subplots_adjust(left=0.3, right=0.97, top=0.97, bottom=0.5)
    save(fig, "fig_mismatch_types")
    (ROOT / "runs/mismatch_types.json").write_text(json.dumps({f"{k}:{c}": v for (k, c), v in counts.items()}, indent=1))


# ---- Supplement S9: guidance for the right patient -----------------------------------
POP_SHORT = {  # label only; titles checked against the manifest
    "f3e61e207cc2": ("WHO 2021, caesarean skin antisepsis", "cs"),
    "7c3781feaf22": ("WHO SSI 2018 (general; refined by 2021)", "general"),
    "a09c53c2bd03": ("NICE NG125, SSI prevention 2019", "ssi"),
    "1a12d3183519": ("CDC SSI 2017", "ssi"),
    "4a934c41413a": ("ERAS gyn/onc perioperative care 2019", "gyn"),
    "2552a848e495": ("ERAS gyn/onc pre/intra-op 2016", "gyn"),
    "a48112505623": ("SSI bundle, gyn-onc surgery 2024", "gyn"),
    "144cabf03653": ("RCOG GTG37a, VTE in pregnancy 2015", "other"),
}
POP_COL = {"cs": ("#D55E00", "Caesarean-only guidance (WHO 2021)"),
           "general": ("#0072B2", "General guidance the 2021 document refines (WHO 2018)"),
           "ssi": ("#56B4E9", "Other surgical-site-infection guidance"),
           "gyn": ("#009E73", "Gynecologic-surgery guidance"),
           "other": ("#bdbdbd", "Other")}


def fig_population():
    p = ROOT / "runs/population_example.json"
    if not p.exists():
        print("  runs/population_example.json missing (needs the live graph)"); return
    d = json.loads(p.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw={"wspace": 0.08})
    for ax, (key, title), letter in zip(axs, (("caesarean", "Caesarean question"),
                                              ("hysterectomy", "Hysterectomy question")), "AB"):
        q = d["questions"][key]
        ax.set_axis_off(); ax.set_xlim(0, 2); ax.set_ylim(-0.5, 7.4)
        ax.text(0.0, 7.25, f"\u201c{q['question']}\u201d", fontsize=6.0, style="italic", va="center")
        for c, (col_key, head) in enumerate((("search_only", "Search alone"), ("cvt_rag", "CVT-RAG retrieval"))):
            ax.text(c + 0.5, 6.55, head, ha="center", va="center", fontsize=6.6, fontweight="bold")
            for i, x in enumerate(q[col_key][:6]):
                name, cat = POP_SHORT.get(x["doc_id"], (x["doc_id"], "other"))
                y = 5.6 - i
                ax.add_patch(plt.Rectangle((c + 0.03, y - 0.42), 0.94, 0.84, color=POP_COL[cat][0], lw=0,
                                           alpha=0.9 if cat != "other" else 1))
                ax.text(c + 0.06, y, f"{i + 1}  {name}", va="center", fontsize=5.4,
                        color="white" if cat in ("cs", "general", "gyn") else "black")
        n_cs = [sum(x["doc_id"] == "f3e61e207cc2" for x in q[k][:6]) for k in ("search_only", "cvt_rag")]
        ax.text(0.5, -0.3, f"{n_cs[0]} of 6 caesarean-only", ha="center", fontsize=5.8)
        ax.text(1.5, -0.3, f"{n_cs[1]} of 6 caesarean-only", ha="center", fontsize=5.8)
        panel(ax, letter, x=-0.01, y=1.04)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=POP_COL[k][0]) for k in POP_COL], [POP_COL[k][1] for k in POP_COL], loc="lower center",
               ncol=3, fontsize=5.8, handlelength=1.0, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.17)
    save(fig, "fig_population")


# ---- Supplement S3: consistency while guidance is updated -----------------------------
REL_COL = {1: "#bdbdbd", 2: "#56B4E9", 3: "#0072B2"}


def fig_consistency():
    p = ROOT / "runs/consistency_replay.json"
    if not p.exists():
        print("  runs/consistency_replay.json missing"); return
    d = json.loads(p.read_text())
    log, ev = d["log"], {e["release"]: e for e in d["events"]}
    W = d["workers"]
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.25, 1], wspace=0.42, left=0.08, right=0.98, top=0.86, bottom=0.36)
    a, z, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    def lanes(ax, x0=None, x1=None, gap=0.0):
        for x in log:
            if x0 is not None and (x["end"] < x0 or x["start"] > x1):
                continue
            ax.plot([x["start"] + gap, x["end"] - gap], [x["worker"] + 1] * 2, color=REL_COL[x["snapshot"]],
                    lw=5, solid_capstyle="butt")
            if x["mixed"]:
                ax.plot([(x["start"] + x["end"]) / 2], [x["worker"] + 1], "x", color="#D55E00", ms=5)
        ax.set_yticks(range(1, W + 1), [f"worker {w}" for w in range(1, W + 1)])
        ax.tick_params(axis="y", length=0); ax.set_ylim(0.4, W + 0.9)

    lanes(a)
    for r in (2, 3):
        e = ev[r]
        a.axvline(e["start"], color="#D55E00", lw=0.6, ls=(0, (3, 2)))
        a.text(e["start"] - 0.15, W + 0.55, f"release {r}", fontsize=5.8, ha="right", color="#a63f00")
    a.set_xlim(min(x["start"] for x in log) - 0.2, max(x["end"] for x in log) + 0.2)
    a.set_xlabel("Time (s)")
    panel(a, "A", x=-0.2, y=1.03)
    e2 = ev[2]
    x0, x1 = e2["start"] - 0.25, e2["end"] + 0.35
    a.axvspan(x0, x1, color="black", alpha=0.07, lw=0)
    lanes(z, x0, x1, gap=0.0012)
    z.axvspan(e2["start"], e2["end"], color="#D55E00", alpha=0.18, lw=0)
    z.text((e2["start"] + e2["end"]) / 2, W + 0.55, "release 2 publish\n(one transaction)", fontsize=5.6,
           ha="center", va="center", color="#a63f00")
    z.set_xlim(x0, x1); z.set_yticklabels([]); z.set_xlabel("Time (s), zoomed")
    panel(z, "B", x=-0.04, y=1.03)
    from matplotlib.patches import Patch
    a.legend([Patch(color=REL_COL[r]) for r in (1, 2, 3)], ["search read release 1", "release 2", "release 3"],
             loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=3, fontsize=5.8, handlelength=1.0)
    ox = [x for x in log if x["query"] == 1]          # logged index 1 = the oxygen query
    cats = [("2016 edition only", "#bdbdbd", lambda x: x["old"] and not x["new"]),
            ("2018 edition only", "#0072B2", lambda x: x["new"] and not x["old"]),
            ("both (mixed)", "#D55E00", lambda x: x["new"] and x["old"]),
            ("neither", "#f0f0f0", lambda x: not x["new"] and not x["old"])]
    rels = sorted({x["snapshot"] for x in ox})
    for i, r in enumerate(rels):
        R = [x for x in ox if x["snapshot"] == r]
        left = 0
        for lab, col, f in cats:
            n = sum(f(x) for x in R)
            b.barh(i, n, 0.6, left=left, color=col, edgecolor="white", lw=0.4, label=lab if i == 0 else None)
            left += n
        b.text(left + 2, i, f"n={len(R)}", va="center", fontsize=5.8)
    b.set_yticks(range(len(rels)), [f"release {r}" for r in rels]); b.tick_params(axis="y", length=0)
    b.invert_yaxis()
    b.set_xlabel("Oxygen-question searches")
    b.set_xlim(0, max(sum(x["snapshot"] == r for x in ox) for r in rels) * 1.35)
    b.legend(loc="upper left", bbox_to_anchor=(-0.45, -0.3), ncol=2, fontsize=5.6, handlelength=1.0)
    b.text(1.0, 1.04, f"mixed: {d['mixed']} of {d['searches']}", transform=b.transAxes,
           ha="right", fontsize=6.0, fontweight="bold")
    panel(b, "C", x=-0.45, y=1.03)
    save(fig, "fig_consistency")


# ---- Reorganised set (2026-11-23): S5 "what the checks catch", Tables 2-3, S-tables ----
def fig_checks(bench):
    """S5. A: fate of CVT-RAG's claims; B: which check caught the first draft;
    C: the kinds of detail error in the baselines' displayed sentences."""
    qs = bench["questions"]
    origins = Counter(c["origin"] for q in qs for c in q["claims"])
    order = [("synthesized", "Passed on first check", "#0072B2"), ("repaired", "Repaired once", "#009E73"),
             ("verbatim", "Replaced by source text", "#E69F00"), ("withdrawn", "Withdrawn", "#7f7f7f")]
    total = sum(origins.values())
    fails = Counter(n for q in qs for c in q["claims"] for n in c["failed_checks"])
    fig = plt.figure(figsize=(W2, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 1.35], wspace=0.55, left=0.03, right=0.99, top=0.86, bottom=0.42)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    left = 0
    for key, lab, col in order:
        n = origins.get(key, 0)
        a.barh(0, n, 0.4, left=left, color=col, edgecolor="white", linewidth=0.5, label=f"{lab} ({n})")
        left += n
    a.set_xlim(0, total); a.set_ylim(-0.5, 0.5); a.set_yticks([]); a.spines["left"].set_visible(False)
    a.set_xlabel(f"CVT-RAG claims ({total} in {len(qs)} answers)")
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.5), ncol=1, handlelength=1.0, fontsize=5.8)
    panel(a, "A", x=-0.02, y=1.04)
    names = ["citation", "details", "support"]
    vals = [fails.get(n, 0) for n in names]
    b.bar(range(3), vals, 0.6, color="#0072B2")
    for i, v in enumerate(vals):
        b.text(i, v + max(vals + [1]) * 0.03, str(v), ha="center", fontsize=6.0)
    b.set_xticks(range(3), ["Citation\n(code)", "Details\n(code)", "Support\n(Jev)"], fontsize=6.0)
    b.set_ylabel("Claims failing\nfirst check", fontsize=6.4)
    from matplotlib.ticker import MaxNLocator
    b.yaxis.set_major_locator(MaxNLocator(integer=True))
    panel(b, "B", x=-0.55, y=1.04)
    D = json.loads((ROOT / "runs/baselines.json").read_text())
    keys = [k for k in (k for k, _ in SYS) if k != "graphrag"]
    cats = [("quantity", "Number or unit not in source", "#E69F00"),
            ("strength", "Strength word not in source", "#D55E00"),
            ("negation", "Negation dropped or added", "#CC79A7")]

    def kinds(detail):
        d = detail or ""
        return {k for k, hit in (("quantity", "quantity" in d),
                                 ("strength", "strong/must" in d or "conditional/suggest" in d),
                                 ("negation", "negation" in d)) if hit}
    Y = _sys_axis(c, keys)
    counts = {}
    for k in keys:
        S = [s for r in D["rows"] if r["system"] == k and "error" not in r for s in r.get("per_sentence", [])]
        left = 0.0
        for cat, lab, col in cats:
            n = sum(cat in kinds(s["detail"]) for s in S)
            counts[f"{k}:{cat}"] = n
            c.barh(Y[k], 100 * n / len(S), 0.55, left=left, color=col, edgecolor="white", linewidth=0.5,
                   label=lab if k == keys[0] else None)
            left += 100 * n / len(S)
        any_ = 100 * sum(bool(kinds(s["detail"])) for s in S) / len(S)
        c.text(left + 0.2, Y[k], f"{any_:.1f} with any", va="center", fontsize=5.8)
    c.set_xlim(0, 12); c.set_xlabel("Displayed sentences with a detail error (per 100)")
    c.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.5), ncol=1, fontsize=5.8, handlelength=1.0)
    panel(c, "C", x=-0.45, y=1.04)
    save(fig, "fig_checks")
    (ROOT / "runs/mismatch_types.json").write_text(json.dumps(counts, indent=1))


def change_rows():
    cdc = json.loads((ROOT / "runs/knowledge_cdc.json").read_text())
    pil = json.loads((ROOT / "runs/knowledge_pilot.json").read_text())
    n99, n_cdc = cdc["documents"]["99fccebb0aaa"]["graded"], len(cdc["pairs"])
    n_pairs, n_chg = pil["editions"]["pairs"], len(pil["editions"]["grade_changed"])
    return [
        (f"Partial update: CDC 1999 → 2017 ({n_cdc} of {n99} graded recommendations replaced; the rest stay in force)",
         [("Antibiotic prophylaxis, caesarean section", "After cord clamping (high-risk caesarean only)",
           "Before skin incision, every caesarean"),
          ("Perioperative glucose", "Control hyperglycaemia (diabetic patients)",
           "Target <200 mg/dL, with or without diabetes"),
          ("Skin preparation", "An appropriate antiseptic", "Alcohol-based antiseptic"),
          ("Preoperative bathing", "Antiseptic agent required", "Soap or antiseptic agent")]),
        (f"New edition: WHO 2016 → 2018 ({n_chg} of {n_pairs} paired recommendations changed)",
         [("80% inspired oxygen, intubated adults", "Strong (“the panel recommends”)",
           "Conditional (“the panel suggests”)")]),
        ("Narrower guidance: WHO 2018 → 2021 (adds a population; nothing retired)",
         [("Chlorhexidine–alcohol skin preparation", "All surgical patients (WHO 2018)",
           "Stated for elective or emergency caesarean section")]),
    ]


def tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    # Table 2: kinds of change and what changed at the bedside
    L = ["| Kind of change | Recommendation | Before | After |", "|---|---|---|---|"]
    for kind, items in change_rows():
        for i, (rec, before, after) in enumerate(items):
            L.append(f"| {'**' + kind + '**' if i == 0 else ''} | {rec} | {before} | {after} |")
    L.append("\nCounts read from `runs/knowledge_cdc.json` and `runs/knowledge_pilot.json`; the WHO grade change "
             "matches WHO's own change note (2018 edition, p.26); CDC 2017 states it does not re-evaluate most "
             "1999 recommendations.")
    (TABLES / "table2_changes.md").write_text("\n".join(L) + "\n")
    # Table 3: speed (from the timing report) + where CVT-RAG's answer time goes
    t = json.loads((ROOT / "runs/timing.json").read_text())
    L = ["| System | Take in one new edition (WHO SSI 2018, 508 passages) | Model calls to update | "
         "Time per answer, median (90th percentile) | Model calls per answer | Old edition still searchable? |",
         "|---|---|---|---|---|---|"]
    for k, label in SYS:
        x = t["systems"][k]
        ing = (f"{x['ingest_seconds'] / 60:.1f} min" if x["ingest_seconds"] >= 120 else f"{x['ingest_seconds']:.0f} s") \
            + (" (est.)" if x["ingest_extrapolated"] else "")
        mc = f"{x['ingest_llm_calls']} LLM" + (f" + {x['ingest_jev_calls']} Jev" if x["ingest_jev_calls"] else "")
        L.append(f"| {label} | {ing} | {mc} | {x['question_median_s']:.1f} s ({x['question_p90_s']:.1f} s) | "
                 f"{x['question_llm_calls_mean']:.1f} | {'no: retired at publication' if k == 'ours' else 'yes'} |")
    bench = load_bench()
    note = ("\n(est.) extrapolated from a timed sample at the pipeline's own parallelism. The writer model "
            "(gemini-3.7-flash, a reasoning model) dominates answer time for every system.")
    if bench:
        qs = bench["questions"]
        med = {k: statistics.median([q["timings"].get(k, 0) for q in qs])
               for k in ("search_s", "graph_s", "population_s", "synthesize_s", "verify_s", "repair_s")}
        note += (f" Where CVT-RAG's time goes, from a separate profiling run with a faster, non-reasoning "
                 f"writer ({bench['chat_model'].replace('-project', '')}; median, n={len(qs)}): writer {med['synthesize_s']:.2f} s, "
                 f"sentence checks {med['verify_s']:.2f} s, search {med['search_s']:.2f} s, graph lookups and "
                 f"population check <0.01 s; repair only when needed.")
    (TABLES / "table3_speed.md").write_text("\n".join(L) + "\n" + note + "\n")
    # Supplementary table: ablation A/B/C and stability across runs
    S = json.loads((ROOT / "runs/comparison.json").read_text())["summary"]
    L = ["**A. What each capability adds (ablation; same writer and questions)**", "",
         "| System | Change questions with superseded guidance | Stating current guidance | "
         "Detail mismatch (sentences) | Not supported (sentences) |", "|---|---|---|---|---|"]
    for m, lab in (("A", "A. Conventional RAG"), ("B", "B. + versioned releases"), ("C", "C. CVT-RAG (+ sentence checks)")):
        sh = S[f"change:{m}"]["sentences_shown"] + S[f"general:{m}"]["sentences_shown"]
        L.append(f"| {lab} | {S[f'change:{m}']['uses_superseded']}/{S[f'change:{m}']['answers']} | "
                 f"{S[f'change:{m}']['states_current']}/{S[f'change:{m}']['answers']} | "
                 f"{S[f'change:{m}']['detail_mismatch'] + S[f'general:{m}']['detail_mismatch']}/{sh} | "
                 f"{S[f'change:{m}']['verifier_unsupported'] + S[f'general:{m}']['verifier_unsupported']}/{sh} |")
    L += ["", "**B. Stability across three runs of the four-system comparison**", "",
          "| System | Superseded guidance, runs 1 / 2 / 3 | Stating current guidance, runs 1 / 2 / 3 |", "|---|---|---|"]
    runs = [ROOT / "runs/baselines_run1_unfiltered.json", ROOT / "runs/baselines_run2_cited_prompt.json",
            ROOT / "runs/baselines.json"]
    R = [[r for r in json.loads(p.read_text())["rows"] if "error" not in r and r["kind"] == "change"] for p in runs]
    for k, lab in SYS:
        sup = " / ".join(str(sum(bool(r["uses_superseded"]) for r in rr if r["system"] == k)) for rr in R)
        cur = " / ".join(str(sum(bool(r["states_current"]) for r in rr if r["system"] == k)) for rr in R)
        L.append(f"| {lab} | {sup} (of 15) | {cur} (of 15) |")
    (TABLES / "tableS1_ablation_stability.md").write_text("\n".join(L) + "\n")
    # Supplementary table: recommendation extraction per document
    man = json.loads((ROOT / "corpus/metadata/corpus_metadata.json").read_text())
    import hashlib
    title = {hashlib.sha1((x.get("sha256") or x.get("filename", "")).encode()).hexdigest()[:12]:
             (x["title"], x["publication_year"]) for x in man}
    pil = json.loads((ROOT / "runs/knowledge_pilot.json").read_text())["documents"]
    cdc = json.loads((ROOT / "runs/knowledge_cdc.json").read_text())["documents"]
    L = ["| Document | Passages | Candidate sentences | Recommendations | Graded |", "|---|---|---|---|---|"]
    for d, v in list(cdc.items()) + list(pil.items()):
        tt, yy = title.get(d, (d, ""))
        L.append(f"| {tt} ({yy}) | {v['chunks']} | {v['candidates']} | {v['recommendations']} | "
                 f"{v.get('graded', v.get('with_grade', ''))} |")
    (TABLES / "tableS2_extraction.md").write_text("\n".join(L) + "\n")
    # Table 4: retrieval accuracy (runs/retrieval_eval.json)
    rp = ROOT / "runs/retrieval_eval.json"
    if rp.exists():
        R = json.loads(rp.read_text())["summary"]
        L = ["| System | Recall@1 | Recall@3 | Recall@6 (95% CI) | MRR@6 |", "|---|---|---|---|---|"]
        for k, label in (("hybrid_rag", "Hybrid RAG"), ("neo4j_graphrag", "Neo4j GraphRAG"), ("ours", "CVT-RAG")):
            m, ci = R["all"][k], R["recall@6_ci"][k]
            L.append(f"| {label} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@6']:.3f} "
                     f"({ci[0]:.3f}-{ci[1]:.3f}) | {m['mrr@6']:.3f} |")
        L.append("| Microsoft GraphRAG | n/a | n/a | n/a | n/a |")
        L.append("| LLM alone (no retrieval) | n/a | n/a | n/a | n/a |")
        d = R["ours_vs_hybrid_k6"]
        L.append(f"\nProportion of the {R['questions']} source-linked test questions whose source passage (or a "
                 "duplicate of it) was among the first k passages retrieved; MRR, mean reciprocal rank within the "
                 f"first 6. Neo4j GraphRAG's first 6 passages are the hybrid search results, so its Recall@1-6 equal "
                 f"hybrid RAG's; its writer receives up to 12 passages, with Recall@12 {R['neo4j_graphrag_recall@12']:.3f}. "
                 f"CVT-RAG vs hybrid RAG at k=6: {d['hybrid_only']} questions found only by hybrid RAG, "
                 f"{d['ours_only']} only by CVT-RAG (exact McNemar P={d['mcnemar_exact_p']:.3f}); 14 of the "
                 f"{d['hybrid_only']} were lost to CVT-RAG's limit of 3 passages per document. Microsoft GraphRAG "
                 "retrieves community summaries, not passages; the LLM alone retrieves nothing.")
        (TABLES / "table4_retrieval.md").write_text("\n".join(L) + "\n")
    print("  wrote tables 2, 3, 4, S1, S2")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    corpus_table()
    fig_corpus(); fig_workflow(); fig_update(); fig_screen(); fig_baselines()
    timing_report(); tables()
    fig_procedure(); fig_population(); fig_consistency(); fig_change_grid()
    bench = load_bench()
    if bench:
        fig_checks(bench)
    fig_calibration()
