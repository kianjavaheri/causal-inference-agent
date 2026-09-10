"""Turn the identifiability engine's verdicts into things a person can act on:
questions this dataset can actually answer, and — when it cannot answer any —
what is missing and why.
"""

from __future__ import annotations

import pandas as pd

from . import identify
from .identify import Assessment
from .methods import SPECS
from .schemas import DataProfile, DesignBlocker, Suggestions

# Keyword -> the column that would unlock the design. Matched against the engine's own
# blocking text so the advice stays in step with the checks that produced it.
_MISSING_HINTS: list[tuple[str, str]] = [
    ("time dimension", "a time column — the period each row was measured in"),
    ("two time periods", "at least two time periods per unit"),
    ("repeated units", "a unit column — which entity (person, store, state) each row belongs to"),
    ("identifies repeated units", "a unit column identifying which entity each row belongs to"),
    ("never-treated", "some units that are never treated, to serve as controls"),
    ("no variation", "variation in the treatment — some rows treated, some not"),
    ("treatment indicator", "a treatment column that is 1 for treated units in treated periods"),
    ("continuous outcome", "a numeric outcome column — the thing you want explained"),
    ("running variable", "a continuous score that determines treatment via a threshold"),
    ("not determined by a threshold", "a genuine cutoff rule assigning treatment"),
    ("covariates", "pre-treatment characteristics to adjust for"),
    ("instrument", "a variable that shifts treatment but affects the outcome no other way"),
    ("first stage", "an instrument that actually moves treatment (first-stage F above 10)"),
    ("single treated unit", "exactly one treated unit, plus untreated donors"),
    ("donor pool", "more untreated units to build a comparison from"),
    ("pre-treatment window", "more periods observed before treatment begins"),
    ("too few", "more observations"),
]


def _missing_for(reasons: list[str]) -> list[str]:
    out: list[str] = []
    joined = " ".join(reasons).lower()
    for needle, advice in _MISSING_HINTS:
        if needle in joined and advice not in out:
            out.append(advice)
    return out[:3]


def _question_for(a: Assessment) -> str | None:
    """A question phrased against this dataset's real columns."""
    r = a.roles
    outcome = r.get("outcome")
    treat = r.get("treatment")
    if not outcome:
        return None

    # An agent-constructed interaction has an internal name; show its inputs instead.
    if isinstance(treat, str) and treat.startswith("__treated_x_post__"):
        inputs = next((d.inputs for d in a.derivations if d.name == treat), None)
        treat = inputs[0] if inputs else "the treatment"

    if a.method == "did":
        return f"Did {treat} change {outcome} for the units that adopted it?"
    if a.method == "rdd":
        run = r.get("running_variable")
        return f"What is the effect of {treat} on {outcome} for units near the {run} cutoff?"
    if a.method == "psm":
        return f"Did {treat} change {outcome}, comparing similar units?"
    if a.method == "iv":
        z = r.get("instrument")
        return f"What is the effect of {treat} on {outcome}, using {z} as the source of variation?"
    if a.method == "synthetic_control":
        unit = a.evidence.get("treated_unit_name")
        who = f" for {unit}" if unit else ""
        return f"How did {treat} change {outcome}{who}?"
    return None


def build(session_id: str, df: pd.DataFrame, profile: DataProfile) -> Suggestions:
    assessments = identify.assess_all(df, profile)
    feasible = [a for a in assessments if a.feasible]
    blocked = [a for a in assessments if not a.feasible]

    questions: list[str] = []
    for a in feasible:
        q = _question_for(a)
        if q and q not in questions:
            questions.append(q)

    blockers = [
        DesignBlocker(
            method=a.method,  # type: ignore[arg-type]
            label=SPECS[a.method].label,
            reasons=a.blocking or [a.rationale],
            missing=_missing_for(a.blocking or [a.rationale]),
        )
        for a in blocked
    ]

    st = profile.structure
    if feasible:
        names = ", ".join(SPECS[a.method].label for a in feasible)
        guidance = (
            f"This dataset can support {names}. The questions below are phrased against "
            "columns that actually exist in it — edit one, or write your own naming the "
            "outcome and the treatment."
        )
    else:
        shape = (
            "a panel" if st.is_panel
            else "repeated cross-sections" if st.has_repeated_cross_sections
            else "a single cross-section"
        )
        wanted = sorted({m for b in blockers for m in b.missing})
        guidance = (
            f"No causal design is identifiable here. The data is {shape} of "
            f"{profile.n_rows:,} rows, but a causal comparison needs more than that: it "
            "needs something that separates treated from untreated units in a way the data "
            "records. "
            + (
                "The most useful additions would be " + "; ".join(wanted[:3]) + "."
                if wanted
                else "Check that the treatment and outcome are both present as columns."
            )
        )

    return Suggestions(
        session_id=session_id,
        questions=questions[:5],
        feasible_methods=[a.method for a in feasible],  # type: ignore[misc]
        blockers=blockers,
        guidance=guidance,
        can_run=bool(feasible),
    )
