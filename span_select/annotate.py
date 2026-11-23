"""Generate the nurse annotation sheet for gold spans.

    python3 -m span_select.annotate              # 50 questions, batches of 10
    python3 -m span_select.annotate -n 100 --batch 20

Why this exists
---------------
The prototype proved the *guarantee* (every answer resolves to a verbatim span
at a named offset) and says nothing about the *quality* (whether the span
chosen was the right one). Measuring quality needs a human-chosen reference
span per question, which is what this sheet collects.

Design decisions that are part of the method, not cosmetics:

* The candidate pool shown to the annotator is produced by the SAME
  `enumerate_spans` call the selector sees, with the SAME dense indices. The
  annotator and the model are therefore answering the identical multiple-choice
  question. Nothing is scored against a free-text answer.
* Multiple candidates may be acceptable. Guideline chunks restate the same
  recommendation across a prose paragraph and a summary table row, so a single
  forced gold span would score correct selections as errors. The annotator
  marks a set of ACCEPTABLE spans and, within it, one BEST span. Both are
  scored (see span_select.score_spans).
* The LLM-written reference answer is HIDDEN by default. Revealed, the task
  degrades into "find the span the reference was generated from", which is
  circular: the reference came from this chunk. Hidden, the task is "find the
  span that answers this nurse's question", which is the thing the system is
  supposed to do. The sheet records whether the reference was revealed before a
  choice was made, so circular annotations can be excluded.
* "Not in this chunk" is a first-class verdict. A model that abstains on those
  is RIGHT, and without this option abstention cannot be scored at all. It also
  catches gold-label defects, of which this corpus has at least three (Italian
  translation chunks under English questions).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List

from .candidates import MAX_CHOICE_OPTIONS, enumerate_spans

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "eval" / "full_testset.jsonl"
CHUNKS = ROOT / "chunks" / "chunks.jsonl"
OUTDIR = ROOT / "eval" / "annotation"

FIELDS = [
    ("action", "the sentence stating what the nurse should do"),
    ("quantity", "the specific number and unit the question asks for"),
]


def load(n: int = 0, qids=None):
    """Load the test set. qid is the line index in the FULL file, always.

    Stable qids are what let an annotation sheet, a prediction dump and a
    second annotator's file be joined later without carrying a mapping around.
    """
    questions = [json.loads(line) for line in TESTSET.open()]
    for i, q in enumerate(questions):
        q["qid"] = "q%03d" % i
    if qids is not None:
        order = {qid: i for i, qid in enumerate(qids)}
        questions = sorted((q for q in questions if q["qid"] in order),
                           key=lambda q: order[q["qid"]])
    elif n:
        questions = questions[:n]
    wanted = {q["source_chunk_id"] for q in questions}
    chunks: Dict[str, dict] = {}
    for line in CHUNKS.open():
        row = json.loads(line)
        if row["chunk_id"] in wanted:
            chunks[row["chunk_id"]] = row
    return questions, chunks


def pools(chunk: dict):
    spans = enumerate_spans(chunk["chunk_id"], chunk["text"], max_options=0)
    spans = spans[:MAX_CHOICE_OPTIONS]
    return {
        "action": [s for s in spans if s.kind == "segment"],
        "quantity": [s for s in spans if s.kind == "quantity"],
    }


def payload(questions, chunks) -> List[dict]:
    """The sheet's data, identical in indexing to what the selector is given."""
    items = []
    for q in questions:
        chunk = chunks.get(q["source_chunk_id"])
        if not chunk:
            continue
        pool = pools(chunk)
        items.append({
            "qid": q["qid"],
            "question": q["question"],
            "question_type": q.get("question_type", ""),
            "reference": q.get("reference", ""),
            "chunk_id": chunk["chunk_id"],
            "source": "%s (%s, tier %s, p.%s)" % (
                chunk.get("title") or chunk.get("filename") or "?",
                chunk.get("publication_year") or "?",
                chunk.get("tier"), chunk.get("page")),
            "fields": {
                name: [{"i": i, "text": s.text, "start": s.start, "end": s.end}
                       for i, s in enumerate(pool[name])]
                for name, _ in FIELDS
            },
        })
    return items


def stratify(size: int, seed: int) -> List[str]:
    """Draw a sample balanced on the thing that limits the quantity field.

    Only 131 of 300 questions have any quantity candidate in their gold chunk,
    and they cluster in `dose` and `timing`. A uniform draw therefore leaves the
    quantity field with too few scorable decisions to report. This draws half
    the sample from quantity-bearing questions and half from the rest, spreading
    each half over the six question types as evenly as the pool allows.
    """
    import random

    questions, chunks = load()
    strata: Dict[tuple, List[str]] = {}
    for q in questions:
        chunk = chunks.get(q["source_chunk_id"])
        if not chunk:
            continue
        has_qty = bool(pools(chunk)["quantity"])
        strata.setdefault((has_qty, q.get("question_type", "?")), []).append(q["qid"])

    rng = random.Random(seed)
    for pool in strata.values():
        rng.shuffle(pool)

    chosen: List[str] = []
    for has_qty in (True, False):
        half = size // 2
        keys = sorted(k for k in strata if k[0] == has_qty)
        picked: List[str] = []
        # round-robin over question types, so a large type cannot crowd out a
        # small one, and a small one cannot block the quota being filled
        while len(picked) < half and any(strata[k] for k in keys):
            for k in keys:
                if len(picked) >= half:
                    break
                if strata[k]:
                    picked.append(strata[k].pop())
        chosen.extend(picked)
    rng.shuffle(chosen)
    return chosen


CSS = """
:root { --bg:#fff; --fg:#1b1b1b; --line:#d8d8d8; --mut:#666;
        --best:#0b6a2f; --bestbg:#e3f3e8; --ok:#1d4ed8; --okbg:#e6edfd; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
       color:var(--fg); background:var(--bg); }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
         padding:10px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header b { font-size:15px; }
#prog { color:var(--mut); font-size:13px; }
button { font:inherit; padding:6px 12px; border:1px solid var(--line);
         background:#f6f6f6; border-radius:5px; cursor:pointer; }
button.primary { background:var(--fg); color:#fff; border-color:var(--fg); }
main { padding:20px; max-width:1000px; }
.q { border:1px solid var(--line); border-radius:7px; margin-bottom:26px; padding:16px; }
.q.done { border-color:var(--best); }
.qh { font-size:17px; font-weight:600; margin:0 0 4px; }
.meta { color:var(--mut); font-size:13px; margin-bottom:12px; }
.verdict { margin:10px 0 14px; display:flex; gap:14px; flex-wrap:wrap; font-size:14px; }
.verdict label { cursor:pointer; }
h4 { margin:16px 0 2px; font-size:14px; text-transform:uppercase; letter-spacing:.04em; }
h4 span { text-transform:none; letter-spacing:0; font-weight:400; color:var(--mut); }
.opt { display:flex; gap:10px; padding:5px 8px; border-radius:5px; cursor:pointer;
       border:1px solid transparent; align-items:baseline; }
.opt:hover { background:#f4f4f4; }
.opt .n { color:var(--mut); font-variant-numeric:tabular-nums; min-width:2.4em;
          text-align:right; font-size:13px; }
.opt .t { white-space:pre-wrap; }
.opt[data-s="best"] { background:var(--bestbg); border-color:var(--best); }
.opt[data-s="ok"]   { background:var(--okbg);  border-color:var(--ok); }
.opt .m { min-width:1.3em; font-weight:700; }
.opt[data-s="best"] .m::before { content:"*"; color:var(--best); }
.opt[data-s="ok"] .m::before { content:"+"; color:var(--ok); }
.empty { color:var(--mut); font-size:14px; padding:4px 8px; }
.ref { margin-top:14px; font-size:14px; }
.ref summary { cursor:pointer; color:var(--mut); }
.ref .body { background:#fbf6e8; border-left:3px solid #d8b24a; padding:8px 10px;
             margin-top:6px; }
.note { margin-top:12px; }
.note input { width:100%; font:inherit; padding:6px 8px; border:1px solid var(--line);
              border-radius:5px; }
.instructions { padding:0 20px 30px; max-width:1000px; }
#out { width:100%; height:120px; font:12px/1.4 ui-monospace,monospace; display:none;
       margin-top:14px; }
"""

JS = r"""
const DATA = __DATA__;
const BATCH = "__BATCH__";

// Corpus text is injected as innerHTML, and 10 of 1089 candidates in the first
// 50 questions contain < > or &. In this corpus those are clinical content
// ("BMI > 35", "glucose < 200 mg/dL"), so dropping them silently would hand the
// annotator a different option than the one the selector sees.
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
const KEY = "goldspans:" + BATCH;
let A = {};
try { A = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { A = {}; }

function rec(qid, field) {
  A[qid] = A[qid] || { verdict: {}, marks: {}, note: "", revealed: false,
                       revealed_before_choice: {} };
  A[qid].marks[field] = A[qid].marks[field] || { best: null, ok: [] };
  return A[qid];
}
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(A)); } catch (e) {}
  const total = DATA.length * 2;
  let done = 0;
  for (const it of DATA) {
    const a = A[it.qid]; if (!a) continue;
    for (const f of ["action", "quantity"]) {
      const v = (a.verdict || {})[f];
      const m = (a.marks || {})[f] || {};
      if (v === "absent" || v === "unclear" || m.best !== null && m.best !== undefined) done++;
    }
  }
  document.getElementById("prog").textContent = done + " / " + total + " decisions recorded";
}

function cycle(qid, field, i, el) {
  const a = rec(qid, field), m = a.marks[field];
  if (!a.revealed) a.revealed_before_choice[field] = false;
  else if (a.revealed_before_choice[field] === undefined) a.revealed_before_choice[field] = true;
  const cur = el.dataset.s || "";
  if (cur === "") {
    if (m.best !== null && m.best !== undefined) {
      const p = document.querySelector('[data-q="'+qid+'"][data-f="'+field+'"][data-i="'+m.best+'"]');
      if (p) { p.dataset.s = "ok"; if (!m.ok.includes(m.best)) m.ok.push(m.best); }
    }
    m.best = i; el.dataset.s = "best"; m.ok = m.ok.filter(x => x !== i);
  } else if (cur === "best") {
    m.best = null; el.dataset.s = "ok"; if (!m.ok.includes(i)) m.ok.push(i);
  } else {
    el.dataset.s = ""; m.ok = m.ok.filter(x => x !== i);
  }
  a.verdict[field] = (m.best !== null || m.ok.length) ? "present" : (a.verdict[field] || "");
  const box = document.querySelector('[data-qbox="'+qid+'"]');
  box.querySelectorAll('input[name="v-'+qid+'-'+field+'"]').forEach(r => {
    r.checked = (r.value === a.verdict[field]);
  });
  save();
}

function verdict(qid, field, value) {
  const a = rec(qid, field);
  a.verdict[field] = value;
  if (value !== "present") {
    a.marks[field] = { best: null, ok: [] };
    document.querySelectorAll('[data-q="'+qid+'"][data-f="'+field+'"]')
      .forEach(el => el.dataset.s = "");
    if (a.revealed_before_choice[field] === undefined)
      a.revealed_before_choice[field] = a.revealed;
  }
  save();
}

function render() {
  const main = document.querySelector("main");
  main.innerHTML = DATA.map(it => {
    const blocks = ["action", "quantity"].map(f => {
      const opts = it.fields[f];
      const desc = f === "action"
        ? "the sentence that states what the nurse should do"
        : "the number and unit the question asks for";
      const rows = opts.length ? opts.map(o =>
        '<div class="opt" data-q="'+it.qid+'" data-f="'+f+'" data-i="'+o.i+'">' +
        '<span class="n">['+o.i+']</span><span class="m"></span>' +
        '<span class="t">'+esc(o.text)+'</span></div>').join("")
        : '<div class="empty">no candidate of this kind in the chunk &mdash; nothing to choose</div>';
      return '<h4>'+f+' <span>&mdash; '+desc+'</span></h4>' +
        '<div class="verdict">' +
        ['present|the chunk states it (mark it below)',
         'absent|not stated in this chunk',
         'unclear|cannot tell'].map(x => {
          const [v, lab] = x.split("|");
          return '<label><input type="radio" name="v-'+it.qid+'-'+f+'" value="'+v+
                 '" onchange="verdict(\''+it.qid+'\',\''+f+'\',\''+v+'\')"> '+lab+'</label>';
        }).join("") + '</div>' + rows;
    }).join("");
    return '<section class="q" data-qbox="'+it.qid+'">' +
      '<p class="qh">'+it.qid+'. '+esc(it.question)+'</p>' +
      '<p class="meta">'+esc(it.source)+' &middot; chunk '+esc(it.chunk_id)+
      ' &middot; '+esc(it.question_type)+'</p>' +
      blocks +
      '<details class="ref" ontoggle="if(this.open){rec(\''+it.qid+'\',\'action\').revealed=true;save();}">' +
      '<summary>show the written reference answer (open this only AFTER choosing)</summary>' +
      '<div class="body">'+esc(it.reference)+'</div></details>' +
      '<div class="note"><input placeholder="one-line note (optional): why none fit, which wording is wrong, a safety concern" ' +
      'oninput="rec(\''+it.qid+'\',\'action\').note=this.value;save();"></div>' +
      '</section>';
  }).join("");

  document.querySelectorAll(".opt").forEach(el => {
    el.onclick = () => cycle(el.dataset.q, el.dataset.f, +el.dataset.i, el);
  });
  for (const it of DATA) {
    const a = A[it.qid]; if (!a) continue;
    for (const f of ["action", "quantity"]) {
      const v = (a.verdict || {})[f];
      if (v) {
        const r = document.querySelector('input[name="v-'+it.qid+'-'+f+'"][value="'+v+'"]');
        if (r) r.checked = true;
      }
      const m = (a.marks || {})[f] || {};
      if (m.best !== null && m.best !== undefined) {
        const el = document.querySelector('[data-q="'+it.qid+'"][data-f="'+f+'"][data-i="'+m.best+'"]');
        if (el) el.dataset.s = "best";
      }
      (m.ok || []).forEach(i => {
        const el = document.querySelector('[data-q="'+it.qid+'"][data-f="'+f+'"][data-i="'+i+'"]');
        if (el) el.dataset.s = "ok";
      });
    }
    const n = document.querySelector('[data-qbox="'+it.qid+'"] .note input');
    if (n && a.note) n.value = a.note;
  }
  save();
}

function lines() {
  const out = [];
  for (const it of DATA) {
    const a = A[it.qid] || {};
    for (const f of ["action", "quantity"]) {
      const m = ((a.marks || {})[f]) || { best: null, ok: [] };
      out.push(JSON.stringify({
        qid: it.qid, chunk_id: it.chunk_id, field: f, batch: BATCH,
        annotator: document.getElementById("who").value.trim(),
        verdict: ((a.verdict || {})[f]) || "",
        best: (m.best === undefined ? null : m.best),
        acceptable: (m.ok || []).slice().sort((x, y) => x - y),
        n_candidates: it.fields[f].length,
        reference_revealed_first: !!((a.revealed_before_choice || {})[f]),
        note: a.note || ""
      }));
    }
  }
  return out.join("\n") + "\n";
}

function fail(e) {
  // A silently blank sheet is the failure an annotator cannot report usefully.
  // Say what broke, and name the fallback that needs no JavaScript.
  document.querySelector("main").innerHTML =
    '<div style="border:2px solid #a11;background:#fdecec;padding:14px;border-radius:7px">' +
    '<b>This sheet failed to load in your browser.</b><p>Please send this message to ' +
    'the study team, and use the spreadsheet fallback <code>gold_spans_all.csv</code> ' +
    'in the meantime.</p><pre style="white-space:pre-wrap;font-size:12px">' +
    esc(e && (e.stack || e.message || e)) + '</pre></div>';
}

function download() {
  const who = document.getElementById("who").value.trim() || "anon";
  const blob = new Blob([lines()], { type: "application/x-ndjson" });
  const u = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = u; a.download = "gold_spans_" + BATCH + "_" + who + ".jsonl";
  a.click(); URL.revokeObjectURL(u);
}
function show() {
  const t = document.getElementById("out");
  t.style.display = "block"; t.value = lines(); t.select();
}

try {
  render();
  if (!document.querySelectorAll("section.q").length)
    throw new Error("no questions rendered (expected " + DATA.length + ")");
} catch (e) { fail(e); }

// Autosave relies on localStorage, which some managed browsers refuse. Say so
// up front rather than losing an hour of work at the end.
(function () {
  let ok = false;
  try { localStorage.setItem(KEY + ":probe", "1");
        localStorage.removeItem(KEY + ":probe"); ok = true; } catch (e) {}
  if (!ok) document.getElementById("prog").innerHTML =
    '<b style="color:#a11">autosave is blocked in this browser &mdash; ' +
    'download your answers before closing the tab</b>';
})();
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold spans &mdash; batch %(batch)s</title><style>%(css)s</style></head>
<body>
<header>
  <b>Gold span annotation &mdash; batch %(batch)s</b>
  <label style="font-size:13px">your initials
    <input id="who" style="width:5em;font:inherit;padding:3px 5px" value="%(who)s"></label>
  <span id="prog"></span>
  <span style="flex:1"></span>
  <button onclick="show()">show JSON</button>
  <button class="primary" onclick="download()">download answers</button>
</header>
<noscript><div style="margin:20px;padding:14px;border:2px solid #a11;background:#fdecec">
This sheet needs JavaScript. If it cannot be enabled, use the spreadsheet
fallback <code>gold_spans_all.csv</code> instead.</div></noscript>
<main></main>
<div class="instructions"><textarea id="out" readonly></textarea>
<p style="color:#666;font-size:13px;max-width:62em">
Click a candidate once to mark it <b style="color:#0b6a2f">*</b> best, twice to
demote it to <b style="color:#1d4ed8">+</b> also acceptable, three times to clear it.
Mark every candidate that would be a correct thing to show a nurse, not only the
one you like most. Work is saved in this browser as you go; press
<b>download answers</b> when you finish and send the file back.</p></div>
<script>%(js)s</script>
</body></html>
"""


def write_batches(items, batch_size: int, who: str,
                  prefix: str = "gold_spans") -> List[Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    written = []
    for start in range(0, len(items), batch_size):
        part = items[start:start + batch_size]
        name = "%s%03d-%03d" % ("" if prefix == "gold_spans" else prefix + "-",
                                start + 1, start + len(part))
        js = JS.replace("__DATA__", json.dumps(part)).replace("__BATCH__", name)
        path = OUTDIR / ("%s_%03d-%03d.html" % (prefix, start + 1, start + len(part)))
        path.write_text(PAGE % {"batch": name, "css": CSS, "js": js,
                                "who": html.escape(who)}, encoding="utf-8")
        written.append(path)
    return written


def write_csv(items, path: Path) -> None:
    """Flat fallback for an annotator who would rather work in a spreadsheet."""
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "field", "candidate_index", "candidate_text",
                    "start", "end", "question", "chunk_id",
                    "mark (best / ok / blank)", "verdict (present/absent/unclear)",
                    "note"])
        for it in items:
            for field in ("action", "quantity"):
                opts = it["fields"][field]
                if not opts:
                    w.writerow([it["qid"], field, "", "(no candidate of this kind)",
                                "", "", it["question"], it["chunk_id"], "", "", ""])
                for o in opts:
                    w.writerow([it["qid"], field, o["i"], o["text"], o["start"],
                                o["end"], it["question"], it["chunk_id"], "", "", ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="span_select.annotate")
    ap.add_argument("-n", type=int, default=50, help="questions to include")
    ap.add_argument("--batch", type=int, default=10, help="questions per sheet")
    ap.add_argument("--who", default="", help="prefill annotator initials")
    ap.add_argument("--stratify", type=int, default=0, metavar="SIZE",
                    help="draw SIZE questions from all 300, half of them "
                         "quantity-bearing, instead of taking the first -n")
    ap.add_argument("--seed", type=int, default=20261118,
                    help="sampling seed, recorded in the manifest")
    ap.add_argument("--prefix", default="gold_spans",
                    help="output filename prefix")
    args = ap.parse_args(argv)

    if args.stratify:
        qids = stratify(args.stratify, args.seed)
        questions, chunks = load(qids=qids)
    else:
        qids = None
        questions, chunks = load(args.n)
    items = payload(questions, chunks)

    n_act = sum(len(i["fields"]["action"]) for i in items)
    n_qty = sum(len(i["fields"]["quantity"]) for i in items)
    no_qty = sum(1 for i in items if not i["fields"]["quantity"])

    paths = write_batches(items, args.batch, args.who, args.prefix)
    csv_path = OUTDIR / ("%s_all.csv" % args.prefix)
    write_csv(items, csv_path)
    manifest = OUTDIR / ("%s_manifest.json" % args.prefix)
    manifest.write_text(json.dumps(
        {"n_questions": len(items), "batch_size": args.batch,
         "sampling": ("stratified %d, seed %d" % (args.stratify, args.seed)
                      if args.stratify else "first %d" % args.n),
         "qids": [i["qid"] for i in items],
         "sheets": [p.name for p in paths],
         "questions": [{"qid": i["qid"], "chunk_id": i["chunk_id"],
                        "n_action": len(i["fields"]["action"]),
                        "n_quantity": len(i["fields"]["quantity"])} for i in items]},
        indent=2), encoding="utf-8")

    print("questions            %d" % len(items))
    print("action candidates    %d total, %.1f per question" % (n_act, n_act / len(items)))
    print("quantity candidates  %d total, %.1f per question" % (n_qty, n_qty / len(items)))
    print("questions with no quantity candidate: %d (nothing to annotate for that field)"
          % no_qty)
    print("decisions to make    %d  (2 fields x %d questions)" % (2 * len(items), len(items)))
    print()
    for p in paths:
        print("  sheet  %s" % p)
    print("  csv    %s" % csv_path)
    print("  index  %s" % manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
