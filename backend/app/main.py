"""FastAPI app: upload -> plan -> execute -> report, plus an SSE stream of the whole run."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any, AsyncIterator

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import identify, llm, methods, planner, reporter, sample_data, suggest
from .config import settings
from .profiling import build_profile
from .schemas import (
    DataProfile,
    Suggestions,
    ExecuteRequest,
    ExecutionResult,
    PlanRequest,
    Report,
    ReportRequest,
)
from .store import Session, store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("causal.api")

app = FastAPI(
    title="Causal Inference Research Agent",
    description=(
        "Takes a dataset and a plain-English causal question, works out which design is "
        "identifiable, runs it, and reports the result with its diagnostics."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in settings.cors_origins else settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _require_session(session_id: str) -> Session:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Upload the dataset again.",
        )
    return session


def _read_csv(raw: bytes, filename: str) -> pd.DataFrame:
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.max_upload_bytes // (1024*1024)} MB limit.",
        )
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse '{filename}': {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse '{filename}': {exc}")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file has no rows or no columns.")
    if len(df) > settings.max_rows:
        raise HTTPException(
            status_code=413,
            detail=f"Dataset has {len(df):,} rows; the limit is {settings.max_rows:,}.",
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _prepared_frame(session: Session) -> pd.DataFrame:
    """The dataframe with any agent-constructed columns materialised."""
    derivations = (
        [d.to_dict() for d in session.assessment.derivations] if session.assessment else []
    )
    return identify.apply_derivations(session.df, derivations)


def _do_plan(session: Session, question: str, roles: dict | None) -> Any:
    if session.profile is None:
        session.profile = build_profile(session.id, session.filename, session.df)
    try:
        plan, assessment = planner.build_plan(
            session.id, question, session.df, session.profile, user_roles=roles
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    session.plan, session.assessment = plan, assessment
    return plan


def _do_execute(session: Session, method: str | None, roles: dict | None, options: dict) -> Any:
    if session.plan is None and method is None:
        raise HTTPException(
            status_code=409, detail="Run /plan first, or name a method explicitly."
        )
    chosen = method or session.plan.chosen_method

    if method and session.plan and method != session.plan.chosen_method:
        # The user overrode the agent's choice; take that method's own role proposal.
        alt = next(
            (a for a in identify.assess_all(session.df, session.profile) if a.method == method),
            None,
        )
        if alt is not None:
            session.assessment = alt

    effective_roles = dict(session.assessment.roles) if session.assessment else {}
    if roles:
        effective_roles.update({k: v for k, v in roles.items() if v})

    df = _prepared_frame(session)
    try:
        output = methods.run(chosen, df, effective_roles, options)
    except methods.IdentificationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Estimation failed")
        raise HTTPException(status_code=500, detail=f"Estimation failed: {exc}")

    spec = methods.SPECS[chosen]
    result = ExecutionResult(
        session_id=session.id,
        method=chosen,  # type: ignore[arg-type]
        method_label=spec.label,
        estimand=spec.estimand,
        estimate=output.estimate,
        diagnostics=output.diagnostics,
        specification={**output.specification, "roles": effective_roles},
        warnings=output.warnings,
    )
    session.result = result
    return result


def _do_report(session: Session) -> Report:
    if session.plan is None or session.result is None:
        raise HTTPException(status_code=409, detail="Run /plan and /execute first.")
    report = reporter.build_report(session.plan, session.result)
    session.report = report
    return report


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_configured": llm.available(),
        "llm_model": settings.anthropic_model if llm.available() else None,
        "methods": list(methods.SPECS),
        "active_sessions": store.count(),
    }


@app.get("/methods")
def list_methods() -> list[dict[str, Any]]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "estimand": spec.estimand,
            "required_roles": list(spec.required_roles),
            "optional_roles": list(spec.optional_roles),
            "assumptions": list(spec.assumptions),
            "plain_english": spec.plain_english,
        }
        for spec in (methods.SPECS[m] for m in methods.ORDER)
    ]


@app.get("/samples")
def list_samples() -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "question": s.question,
            "expected_method": s.expected_method,
            "expected_method_label": methods.SPECS[s.expected_method].label,
            "true_effect": s.true_effect,
        }
        for s in sample_data.SAMPLES.values()
    ]


@app.post("/samples/{sample_id}/load", response_model=DataProfile)
def load_sample(sample_id: str) -> DataProfile:
    """Load a built-in dataset with a known ground-truth effect, as if uploaded."""
    if sample_id not in sample_data.SAMPLES:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{sample_id}'.")
    df = sample_data.generate(sample_id)
    session = store.create(f"{sample_id}.csv", df)
    session.profile = build_profile(session.id, session.filename, df)
    return session.profile


@app.get("/samples/{sample_id}/download")
def download_sample(sample_id: str) -> StreamingResponse:
    if sample_id not in sample_data.SAMPLES:
        raise HTTPException(status_code=404, detail=f"Unknown sample '{sample_id}'.")
    csv = sample_data.generate(sample_id).to_csv(index=False)
    return StreamingResponse(
        io.StringIO(csv),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{sample_id}.csv"'},
    )


@app.post("/upload", response_model=DataProfile)
async def upload(file: UploadFile = File(...)) -> DataProfile:
    raw = await file.read()
    df = _read_csv(raw, file.filename or "upload.csv")
    session = store.create(file.filename or "upload.csv", df)
    session.profile = build_profile(session.id, session.filename, df)
    return session.profile


@app.get("/profile/{session_id}", response_model=DataProfile)
def get_profile(session_id: str) -> DataProfile:
    session = _require_session(session_id)
    if session.profile is None:
        session.profile = build_profile(session.id, session.filename, session.df)
    return session.profile


@app.get("/suggestions/{session_id}", response_model=Suggestions)
def suggestions(session_id: str) -> Suggestions:
    """Questions this dataset can answer — or, if none, what is missing and why."""
    session = _require_session(session_id)
    if session.profile is None:
        session.profile = build_profile(session.id, session.filename, session.df)
    return suggest.build(session.id, session.df, session.profile)


@app.post("/plan")
def create_plan(req: PlanRequest) -> Any:
    session = _require_session(req.session_id)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="A causal question is required.")
    return _do_plan(session, req.question.strip(), req.roles)


@app.post("/execute", response_model=ExecutionResult)
def execute(req: ExecuteRequest) -> ExecutionResult:
    session = _require_session(req.session_id)
    return _do_execute(session, req.method, req.roles, req.options)


@app.post("/report", response_model=Report)
def report(req: ReportRequest) -> Report:
    session = _require_session(req.session_id)
    return _do_report(session)


# ---------------------------------------------------------------------------
# streamed pipeline -- the reasoning trail the UI renders live
# ---------------------------------------------------------------------------


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _run_pipeline(
    session: Session, question: str, roles: dict | None, method: str | None
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()

    try:
        # 1. profile
        yield _sse("step_start", {"step": "profile", "title": "Profiling the dataset"})
        profile = await loop.run_in_executor(
            None, lambda: session.profile or build_profile(session.id, session.filename, session.df)
        )
        session.profile = profile
        yield _sse(
            "step_done",
            {
                "step": "profile",
                "title": "Profiling the dataset",
                "summary": (
                    f"{profile.n_rows:,} rows x {profile.n_cols} columns. "
                    + (profile.structure.notes[0] if profile.structure.notes else "")
                ),
                "payload": profile.model_dump(),
            },
        )
        await asyncio.sleep(0.15)

        # 2. plan
        yield _sse(
            "step_start",
            {
                "step": "plan",
                "title": "Deciding which causal design is identifiable",
                "detail": (
                    "Testing all five designs against the data's structure"
                    + (", then asking the model to choose" if llm.available() else "")
                ),
            },
        )
        plan = await loop.run_in_executor(None, lambda: _do_plan(session, question, roles))
        yield _sse(
            "step_done",
            {
                "step": "plan",
                "title": f"Chose {plan.chosen_label}",
                # Just the lead sentence: the full justification is rendered below the trail.
                "summary": plan.justification.split(". ")[0].rstrip(".") + ".",
                "payload": plan.model_dump(),
            },
        )
        await asyncio.sleep(0.15)

        # 3. execute
        yield _sse(
            "step_start",
            {
                "step": "execute",
                "title": f"Running {plan.chosen_label} and checking its assumptions",
            },
        )
        result = await loop.run_in_executor(
            None, lambda: _do_execute(session, method, roles, {})
        )
        est = result.estimate
        yield _sse(
            "step_done",
            {
                "step": "execute",
                "title": f"Estimated {result.method_label}",
                "summary": (
                    f"Effect = {est.point:,.4g}"
                    + (
                        f" (95% CI [{est.ci_low:,.4g}, {est.ci_high:,.4g}])"
                        if est.ci_low is not None and est.ci_high is not None
                        else ""
                    )
                    + f"; {len(result.diagnostics)} diagnostics run."
                ),
                "payload": result.model_dump(),
            },
        )
        await asyncio.sleep(0.15)

        # 4. report
        yield _sse("step_start", {"step": "report", "title": "Writing up the findings"})
        rep = await loop.run_in_executor(None, lambda: _do_report(session))
        yield _sse(
            "step_done",
            {
                "step": "report",
                "title": "Report ready",
                "summary": rep.headline,
                "payload": rep.model_dump(),
            },
        )
        yield _sse("complete", {"session_id": session.id})

    except HTTPException as exc:
        yield _sse("error", {"detail": exc.detail, "status": exc.status_code})
    except Exception as exc:  # pragma: no cover
        log.exception("Pipeline failed")
        yield _sse("error", {"detail": str(exc), "status": 500})


@app.post("/run")
async def run_pipeline(
    session_id: str = Form(...),
    question: str = Form(...),
    method: str | None = Form(default=None),
    roles: str | None = Form(default=None),
) -> StreamingResponse:
    """Run the whole agent and stream each step as it completes (SSE)."""
    session = _require_session(session_id)
    parsed_roles = json.loads(roles) if roles else None
    return StreamingResponse(
        _run_pipeline(session, question.strip(), parsed_roles, method or None),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/run")
async def run_pipeline_get(
    session_id: str, question: str, method: str | None = None, roles: str | None = None
) -> StreamingResponse:
    """GET variant so the browser's native EventSource can drive the stream."""
    session = _require_session(session_id)
    parsed_roles = json.loads(roles) if roles else None
    return StreamingResponse(
        _run_pipeline(session, question.strip(), parsed_roles, method or None),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
