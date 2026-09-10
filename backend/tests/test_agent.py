"""End-to-end checks.

The point of these tests is not that the code runs -- it is that each estimator
recovers a treatment effect we planted in the data, and that the planner picks the
design the data was built for. A refactor that breaks identification will fail here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import identify, methods, sample_data  # noqa: E402
from app.methods.base import IdentificationError  # noqa: E402
from app.planner import build_plan  # noqa: E402
from app.profiling import build_profile  # noqa: E402

SAMPLE_IDS = list(sample_data.SAMPLES)


def _plan_for(sample_id: str):
    sample = sample_data.SAMPLES[sample_id]
    df = sample_data.generate(sample_id)
    profile = build_profile("test", f"{sample_id}.csv", df)
    plan, assessment = build_plan("test", sample.question, df, profile, use_llm=False)
    return sample, df, profile, plan, assessment


# --- planning ---------------------------------------------------------------


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_planner_picks_the_intended_design(sample_id: str) -> None:
    sample, _, _, plan, _ = _plan_for(sample_id)
    assert plan.chosen_method == sample.expected_method, (
        f"{sample_id}: chose {plan.chosen_method}, expected {sample.expected_method}"
    )


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_planner_assigns_required_roles(sample_id: str) -> None:
    _, _, _, plan, _ = _plan_for(sample_id)
    spec = methods.SPECS[plan.chosen_method]
    for role in spec.required_roles:
        assert plan.roles.get(role), f"{sample_id}: role '{role}' unassigned"


# --- estimation -------------------------------------------------------------


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_estimate_recovers_the_planted_effect(sample_id: str) -> None:
    sample, df, _, plan, assessment = _plan_for(sample_id)
    prepared = identify.apply_derivations(df, [d.to_dict() for d in assessment.derivations])
    out = methods.run(plan.chosen_method, prepared, assessment.roles, {})
    est = out.estimate
    assert est.ci_low is not None and est.ci_high is not None
    assert est.ci_low <= sample.true_effect <= est.ci_high, (
        f"{sample_id}: 95% CI [{est.ci_low:.4g}, {est.ci_high:.4g}] excludes the true "
        f"effect {sample.true_effect}"
    )


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_diagnostics_are_produced_and_serialisable(sample_id: str) -> None:
    _, df, _, plan, assessment = _plan_for(sample_id)
    prepared = identify.apply_derivations(df, [d.to_dict() for d in assessment.derivations])
    out = methods.run(plan.chosen_method, prepared, assessment.roles, {})
    assert out.diagnostics, "every design should report at least one assumption check"
    for d in out.diagnostics:
        d.model_dump_json()  # NaN/inf would raise here


def test_did_constructs_the_interaction_when_no_treatment_column_exists() -> None:
    """The dataset has a group flag and a post flag but no treated x post column."""
    df = sample_data.make_did().drop(columns=["treated"])
    profile = build_profile("t", "did.csv", df)
    plan, assessment = build_plan(
        "t", sample_data.SAMPLES["did_minimum_wage"].question, df, profile, use_llm=False
    )
    assert plan.chosen_method == "did"
    assert assessment.derivations, "expected the agent to build the interaction term"

    prepared = identify.apply_derivations(df, [d.to_dict() for d in assessment.derivations])
    out = methods.run("did", prepared, assessment.roles, {})
    assert out.estimate.ci_low <= -2.5 <= out.estimate.ci_high


# --- assumption checks actually fire ---------------------------------------


def test_parallel_trends_check_catches_a_violated_pre_trend() -> None:
    """Give the treated group its own trend before treatment; the check must fail."""
    df = sample_data.make_did()
    treated_units = set(df.loc[df["treated_group"] == 1, "county_id"])
    bump = df["county_id"].isin(treated_units) * df["quarter"] * 1.5
    df = df.assign(employment=df["employment"] + bump)

    out = methods.run(
        "did",
        df,
        {"outcome": "employment", "treatment": "treated", "time": "quarter", "unit": "county_id"},
        {},
    )
    pretrend = next(d for d in out.diagnostics if d.id == "parallel_trends")
    assert pretrend.verdict == "fail", "a divergent pre-trend should be caught"


def test_rdd_density_check_catches_manipulation() -> None:
    """Drop most students just below the cutoff, as if they had nudged themselves over."""
    df = sample_data.make_rdd()
    just_below = (df["exam_score"] >= 68) & (df["exam_score"] < 70)
    keep = ~just_below | (df.groupby(just_below).cumcount() % 6 == 0)
    manipulated = df[keep].reset_index(drop=True)

    out = methods.run(
        "rdd",
        manipulated,
        {"outcome": "college_gpa", "running_variable": "exam_score", "treatment": "scholarship"},
        {},
    )
    density = next(d for d in out.diagnostics if d.id == "density_test")
    assert density.verdict in ("warn", "fail"), "a hole below the cutoff should be flagged"


def test_iv_flags_a_weak_instrument() -> None:
    """A near-random instrument must be reported as weak, not silently used."""
    df = sample_data.make_iv()
    rng = pd.Series(range(len(df)))
    df = df.assign(assigned_noise=(rng % 2).to_numpy())

    out = methods.run(
        "iv",
        df,
        {
            "outcome": "spend_90d",
            "treatment": "completed_onboarding",
            "instrument": "assigned_noise",
        },
        {},
    )
    first_stage = next(d for d in out.diagnostics if d.id == "first_stage")
    assert first_stage.verdict in ("warn", "fail")
    assert any("weak instrument" in w.lower() for w in out.warnings)


# --- guardrails -------------------------------------------------------------


def test_unidentifiable_data_is_refused_rather_than_estimated() -> None:
    """Pure noise with no design should not yield a confident causal claim."""
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"row_id": [f"R{i}" for i in range(200)], "noise": rng.normal(size=200)})
    profile = build_profile("t", "noise.csv", df)
    with pytest.raises(ValueError, match="No causal design is identifiable"):
        build_plan("t", "Did the thing cause the other thing?", df, profile, use_llm=False)


def test_synthetic_control_rejects_a_period_flag_as_treatment() -> None:
    df = sample_data.make_synthetic_control()
    with pytest.raises(IdentificationError, match="marks the post period"):
        methods.run(
            "synthetic_control",
            df,
            {"outcome": "cigarette_sales", "treatment": "post_tax", "time": "year", "unit": "state"},
            {},
        )


def test_psm_requires_covariates() -> None:
    df = sample_data.make_psm()
    with pytest.raises(IdentificationError, match="covariate"):
        methods.run(
            "psm", df, {"outcome": "earnings_after", "treatment": "enrolled_training"}, {}
        )


def test_missing_values_are_dropped_not_silently_imputed() -> None:
    df = sample_data.make_did()
    df.loc[:19, "employment"] = None
    out = methods.run(
        "did",
        df,
        {"outcome": "employment", "treatment": "treated", "time": "quarter", "unit": "county_id"},
        {},
    )
    assert out.estimate.n_obs == len(df) - 20
    assert any("missing" in w.lower() for w in out.warnings)


def test_every_assessor_survives_a_dataset_with_no_binary_columns() -> None:
    """A dataset of pure measurements must produce clean blocking reasons, not crashes."""
    df = sample_data.make_synthetic_control()[["state", "year", "cigarette_sales"]]
    profile = build_profile("t", "no_binaries.csv", df)
    for a in identify.assess_all(df, profile):
        assert not a.feasible
        joined = " ".join(a.blocking or [a.rationale]).lower()
        assert "cannot access local variable" not in joined
        assert "traceback" not in joined
