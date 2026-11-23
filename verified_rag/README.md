# verified_rag

Perioperative nursing answers in which every sentence is checked against the
guideline text it cites. Design and measurements: see the top-level README and the accompanying paper
(under review).

## Layout

```
corpus_index/        data layer: chunks, embeddings, Neo4j, atomic releases (existing)
span_select/         candidate passages, read-back, forced choice (existing)
verified_rag/
  config.py          settings (.env via corpus_index.config)
  types.py           Source, Claim, Check, Signals, Response
  clients/           chat.py (OpenAI-compatible), jev.py (TypeSafe noul/choice)
  retrieval.py       release-pinned search + graph enrichment + population ranking
  answer/
    synthesize.py    chat model writes cited claims (JSON)
    details.py       check 2: quantities, strength words, negation (pure code)
    verify.py        checks 1-3 (citation, details, Jev support), in parallel
    repair.py        triage by failure type, one rewrite, verbatim fallback
    assemble.py      response + four separate signals
  knowledge/         recommendations, slots and links (pilot: 3 WHO documents)
  store.py           audit log, nurse feedback, calibration (SQLite)
  pipeline.py        the end-to-end answer path
  bench.py           benchmark over the study questions -> runs/bench.json
  tests/             offline tests with stubbed models and graph
```

## Commands

```bash
PYTHONPATH=. ./.venv/bin/python -m verified_rag ask "should my intubated patient get 80% oxygen?"
PYTHONPATH=. ./.venv/bin/python -m verified_rag ask "..." --as-of 1      # as answered at release 1
PYTHONPATH=. ./.venv/bin/python -m verified_rag feedback RESPONSE_ID 0 right
PYTHONPATH=. ./.venv/bin/python -m verified_rag.tests.test_offline     # no network
PYTHONPATH=. ./.venv/bin/python -m verified_rag.knowledge pilot        # ~680 Jev calls
PYTHONPATH=. ./.venv/bin/python -m verified_rag.bench                  # 58 questions, ~80 s
python3 -m span_select.mock_annotate --model claude-opus-5-project   # MOCK gold, not nurse data
PYTHONPATH=. ./.venv/bin/python figures/span_verified/make_figures.py
```

Needs Neo4j running with a published release (see `NEO4J_SETUP.md`),
`OPENAI_BASE_URL`/`OPENAI_API_KEY` and `TYPESAFE_API_KEY` in `.env`.

## Built vs. not yet

Built: release-pinned retrieval, population-aware ranking, cited synthesis,
three checks, repair ladder, four signals, audit/feedback/calibration store,
knowledge pilot (recommendations, grades, populations, edition and refinement
links) on three WHO documents.

Not yet: knowledge layer over the whole Tier 1–2 corpus; question hubs and
AGREES/CONFLICTS relations (consensus says "not yet assessed"); a nurse-facing
screen; Postgres event log (SQLite stands in). Calibration shows "not yet
calibrated" until 30 nurse verdicts exist per group.

## Data handling

`runs/verified_rag.sqlite` stores every question and answer. Real bedside
questions may contain patient details: treat the file as clinical data, and do
not send real questions to TypeSafe without de-identification (no HIPAA
agreement).
