# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Both servers at once (creates the venv and installs deps if missing):

```bash
./dev.sh
```

Backend alone — `backend/.venv` already exists:

```bash
cd backend && .venv/bin/uvicorn app.main:app --reload
```

Frontend alone (expects the backend on `:8000`, override with `NEXT_PUBLIC_API_BASE`):

```bash
cd frontend && npm run dev
```

Tests (42; `requirements-dev.txt` adds pytest and httpx):

```bash
cd backend && .venv/bin/python -m pytest tests/ -q
```

A single test, or a themed subset:

```bash
cd backend && .venv/bin/python -m pytest tests/test_agent.py::test_did_constructs_the_interaction_when_no_treatment_column_exists -v
cd backend && .venv/bin/python -m pytest tests/ -k "recovers or catches or flags" -v
```

Frontend checks — run all three before considering a change done; `npm run lint` is bare
`eslint` and needs no arguments:

```bash
cd frontend && npx tsc --noEmit && npm run lint && npm run build
```

There is no Python linter or formatter configured. Docker is not running on this machine,
so the backend image has never been built — the pins in `requirements.txt` were verified to
have cp312 manylinux wheels, but the image itself is unverified.

**This is not a git repository.** Initialise one before any commit-based workflow.

## Architecture

A four-stage agent pipeline. `POST /upload` → `POST /plan` → `POST /execute` → `POST /report`,
plus `GET /run` which does all four as a server-sent event stream. The SSE endpoint is what
the UI actually uses; the individual routes exist for scripting and tests.

### The load-bearing idea

Design selection is **deterministic**. `app/identify.py` decides what is identifiable from
the data's structure; the LLM in `app/planner.py` only chooses among options the engine
already marked feasible and writes prose. If the model names a blocked design, the planner
overrides it and records that in the reasoning trace. Everything works with no API key.

When changing planning behaviour, decide first whether the change belongs in the engine
(a structural fact about identifiability) or the planner (judgment about the question).
Putting structural logic in the prompt is how this system regresses.

### Where each concern lives

| File | Responsibility |
|---|---|
| `app/profiling.py` | CSV → semantic column types, role candidates, panel/cross-section structure |
| `app/identify.py` | five assessors; which designs are feasible, with what roles, and why not |
| `app/planner.py` | question → role hints; LLM call plus the guardrail; assembles the `Plan` |
| `app/methods/*.py` | one estimator per design, each self-contained |
| `app/reporter.py` | estimate + diagnostics → prose, with a deterministic fallback |
| `app/suggest.py` | feasible → grounded example questions; blocked → what is missing |
| `app/main.py` | routes, the SSE pipeline, session lookup |

`app/schemas.py` is the contract with the frontend. `frontend/lib/types.ts` mirrors it by
hand — **changing one requires changing the other**.

### Mechanisms worth knowing before editing

**Semantic types, not dtypes.** `_semantic_type` in `profiling.py` classifies columns as
binary / categorical / continuous / integer / datetime / identifier / text. Every downstream
decision keys off this, not off the pandas dtype.

**The variation test.** `identify._variation(df, col, unit, time)` returns `time_only`
(a period flag like `post`), `unit_only` (a group flag like `treated_group`), `both` (a real
treatment), or `constant`. This single primitive drives DiD treatment selection, the
period-flag guards, and the treatment veto in question matching.

**Agent-constructed columns.** When a dataset has a group flag and a period flag but no
interaction, the DiD and synthetic-control assessors emit a `Derivation`. Anything that runs
an estimator must call `identify.apply_derivations` on the dataframe first — `main._prepared_frame`
does this, and tests must do it manually.

**Role hints.** `planner.question_hints` matches question tokens to column names and claims
roles in priority order (outcome → running variable → instrument → treatment → time → unit),
each exclusively. Scoring is whole-token overlap with **no length normalisation** — normalising
made `hs_gpa` beat `college_gpa`. Ties fall back to the profiler's ranking on purpose.

**Scores encode credibility, not just feasibility.** PSM starts low and drops further on panel
data because unconfoundedness is a stronger assumption than a discontinuity needs. DiD is
penalised with a single treated unit. Do not "fix" these by flattening them.

**IV instruments must be nominated.** `assess_iv` only considers columns whose name reads as
an assignment mechanism (`INSTRUMENT_NAME_HINTS`) or that the question nominated. Discovering
an instrument from correlation is a bug, not a feature — a strong first stage is equally
consistent with a confounder.

### Adding a method

Create `app/methods/<id>.py` exporting a `SPEC: MethodSpec` and an
`estimate(df, roles, options) -> MethodOutput`, then add the module to `_MODULES` in
`app/methods/__init__.py` (`SPECS` and `ESTIMATORS` build themselves from it) and add an
assessor to `identify.ASSESSORS`. If it emits a new diagnostic `kind`, extend the `Literal`
in `schemas.Diagnostic`, the union in `frontend/lib/types.ts`, and the switch in
`components/diagnostics/DiagnosticCard.tsx`.

### Estimator conventions

Raise `IdentificationError` (from `methods/base.py`) for anything the data cannot support —
`main.py` turns it into a 422 with the message shown to the user, so write it for a person.
Diagnostics carry a `verdict` of pass/warn/fail/info plus a `data` payload the frontend charts
by `kind`. Run every numeric through `base.fnum`, which maps NaN and inf to null; raw NaN
breaks JSON serialisation and there is a test asserting this.

Several variance choices are deliberate and easy to "simplify" wrongly: PSM uses the
Abadie–Imbens matching variance because the bootstrap is invalid under matching with
replacement; IV computes second-stage variance from **structural-equation** residuals using
actual treatment, not fitted treatment.

### Frontend

`Diagnostic` in `lib/types.ts` is a discriminated union on `kind`, so the switch in
`DiagnosticCard` narrows `data` to one payload type per chart component. Recharts
`ResponsiveContainer` renders nothing at zero width, so charts appear blank in a hidden
browser pane — that is an artifact of the harness, not a bug.

Theme tokens live in `app/globals.css` and must be defined on bare `:root` before the
`prefers-color-scheme` and `[data-theme]` blocks redefine them. `NavBar.tsx` reads the theme
via `useSyncExternalStore` over the inline boot script in `layout.tsx`; `setState` in an
effect trips the lint rule. `<body>` carries `suppressHydrationWarning` because browser
extensions inject attributes there before React hydrates.

`frontend/CLAUDE.md` and `frontend/AGENTS.md` are generated by `next dev` and re-created if
deleted — leave them alone.

## Testing philosophy

The suite asserts that each estimator **recovers a treatment effect planted in synthetic
data** (`app/sample_data.py` holds the generators and their true effects) and that the
assumption checks actually fire: a divergent pre-trend must fail parallel trends, a hole
below an RDD cutoff must flag manipulation, a random instrument must be reported weak, and
unidentifiable data must be refused rather than estimated. A refactor that quietly breaks
identification fails here rather than passing silently.

`sample-data/` holds nine real datasets from the literature with a README of what the agent
does with each — four of the nine are correctly refused. Use them when changing profiling or
`identify.py`, since most of the heuristics in those files exist because of a specific
failure on one of them.

## Reference

`README.md` covers the API surface, deploy configs (Dockerfile plus Railway and Render), and
the design/assumption table. `frontend/app/about/page.tsx` and `/methods` render the same
material for end users, and `/methods` reads the catalogue from `GET /methods` so the UI and
the engine cannot disagree about what a design assumes.
