"""Instrumental Variables (two-stage least squares).

Identification: find a variable that shifts treatment but affects the outcome only
through treatment. The instrument supplies the exogenous variation that the treatment
variable itself lacks.

Estimator: 2SLS. With one binary instrument and one binary treatment this reduces to
the Wald ratio (reduced form / first stage), which is the LATE for compliers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..schemas import Diagnostic, Estimate
from .base import (
    IdentificationError,
    MethodOutput,
    MethodSpec,
    clean_frame,
    coef_inference,
    dummies,
    fnum,
    numeric,
    ols,
    require,
    robust_vcov,
)

SPEC = MethodSpec(
    id="iv",
    label="Instrumental Variables (2SLS)",
    estimand="LATE — local average treatment effect for compliers",
    required_roles=("outcome", "treatment", "instrument"),
    optional_roles=("covariates", "unit"),
    assumptions=(
        "Relevance: the instrument genuinely moves treatment. Testable — the first-stage "
        "F statistic should comfortably exceed 10.",
        "Exclusion restriction: the instrument affects the outcome ONLY through treatment. "
        "Untestable, and the assumption that usually fails.",
        "Independence: the instrument is as good as randomly assigned with respect to "
        "potential outcomes and unmeasured confounders.",
        "Monotonicity: nobody is a defier — the instrument never pushes someone out of "
        "treatment while pushing others in.",
    ),
    plain_english=(
        "When treatment is tangled up with things you cannot see, find a nudge that "
        "changes treatment for reasons unrelated to the outcome. Scale the nudge's effect "
        "on the outcome by its effect on treatment, and you recover the causal effect for "
        "the people the nudge actually moved."
    ),
)


def _cov_block(frame: pd.DataFrame, covs: list[str]) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for c in covs:
        col = frame[c]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 2:
            blocks.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)[:, None])
            names.append(c)
        elif col.nunique() == 2:
            vals = sorted(pd.unique(col.dropna()), key=str)
            blocks.append((col.to_numpy() == vals[-1]).astype(float)[:, None])
            names.append(f"{c}={vals[-1]}")
        else:
            block, levels = dummies(col.to_numpy())
            if block.size:
                blocks.append(block)
                names.extend(f"{c}={lvl}" for lvl in levels)
    if not blocks:
        return np.zeros((len(frame), 0)), []
    return np.column_stack(blocks), names


def estimate(df: pd.DataFrame, roles: dict[str, str | None], options: dict) -> MethodOutput:
    y_col = require(roles, "outcome", "IV")
    d_col = require(roles, "treatment", "IV")
    z_col = require(roles, "instrument", "IV")
    if z_col == d_col:
        raise IdentificationError("The instrument and the treatment cannot be the same column.")
    covs = [c for c in (roles.get("covariates") or []) if c and c in df.columns
            and c not in (y_col, d_col, z_col)] if isinstance(roles.get("covariates"), list) else []
    cluster_col = roles.get("unit")

    frame, dropped = clean_frame(df, [y_col, d_col, z_col, *covs, cluster_col])
    if len(frame) < 50:
        raise IdentificationError("IV needs at least 50 complete rows; fewer remain after cleaning.")

    y = numeric(frame, y_col)
    d = numeric(frame, d_col)
    z = numeric(frame, z_col)
    if np.std(z) == 0:
        raise IdentificationError(f"The instrument '{z_col}' has no variation.")
    if np.std(d) == 0:
        raise IdentificationError(f"The treatment '{d_col}' has no variation.")

    W, w_names = _cov_block(frame, covs)
    n = len(y)
    cluster = frame[cluster_col].to_numpy() if cluster_col and cluster_col in frame.columns else None
    n_clusters = len(pd.unique(cluster)) if cluster is not None else n
    dof = (n_clusters - 1) if cluster is not None else (n - 2 - W.shape[1])

    warnings: list[str] = []
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing values in the modelled columns.")

    exog = np.column_stack([np.ones(n), W]) if W.size else np.ones((n, 1))

    # --- first stage:  d ~ z + W --------------------------------------------
    X1 = np.column_stack([exog, z])
    b1, r1 = ols(X1, d)
    v1 = robust_vcov(X1, r1, cluster=cluster)
    z_idx = X1.shape[1] - 1
    fs_coef, fs_se, fs_lo, fs_hi, fs_p = coef_inference(b1, v1, z_idx, dof)
    f_stat = float((fs_coef / fs_se) ** 2) if fs_se and np.isfinite(fs_se) and fs_se > 0 else 0.0
    d_hat = X1 @ b1

    if f_stat < 10:
        warnings.append(
            f"Weak instrument: first-stage F = {f_stat:.1f}, below the conventional threshold of "
            "10. 2SLS is badly biased toward OLS and the confidence interval understates "
            "uncertainty when the first stage is this weak."
        )

    # --- second stage: y ~ d_hat + W, with correct 2SLS residuals ------------
    X2 = np.column_stack([exog, d_hat])
    b2, _ = ols(X2, y)
    d_idx = X2.shape[1] - 1
    # Variance must use residuals from the *structural* equation (actual d, not d_hat).
    X_struct = np.column_stack([exog, d])
    resid2 = y - X_struct @ b2
    v2 = robust_vcov(X2, resid2, cluster=cluster)
    point, se, lo, hi, p = coef_inference(b2, v2, d_idx, dof)

    # --- reduced form: y ~ z + W --------------------------------------------
    Xr = np.column_stack([exog, z])
    br, rr = ols(Xr, y)
    vr = robust_vcov(Xr, rr, cluster=cluster)
    rf_coef, rf_se, rf_lo, rf_hi, rf_p = coef_inference(br, vr, z_idx, dof)

    # --- OLS for comparison --------------------------------------------------
    Xo = np.column_stack([exog, d])
    bo, ro = ols(Xo, y)
    vo = robust_vcov(Xo, ro, cluster=cluster)
    ols_coef, ols_se, _, _, ols_p = coef_inference(bo, vo, Xo.shape[1] - 1, dof)

    diagnostics: list[Diagnostic] = []

    if f_stat >= 100:
        fs_verdict, fs_note = "pass", "Very strong first stage."
    elif f_stat >= 10:
        fs_verdict, fs_note = "pass", "First stage clears the conventional F > 10 threshold."
    elif f_stat >= 5:
        fs_verdict, fs_note = "warn", (
            "Borderline-weak instrument. 2SLS is biased toward OLS here and coverage is poor."
        )
    else:
        fs_verdict, fs_note = "fail", (
            "The instrument barely moves treatment. The IV estimate is not trustworthy."
        )
    diagnostics.append(
        Diagnostic(
            id="first_stage",
            title="Instrument strength (first stage)",
            kind="table",
            verdict=fs_verdict,  # type: ignore[arg-type]
            summary=(
                f"A one-unit change in {z_col} shifts {d_col} by {fs_coef:.4g} "
                f"(SE {fs_se:.4g}). F = {f_stat:.1f}. {fs_note}"
            ),
            detail=(
                "This is the one IV assumption the data can verify. Below F ≈ 10, weak-instrument "
                "bias pulls 2SLS back toward the confounded OLS estimate and the reported "
                "confidence interval is too narrow."
            ),
            data={
                "rows": [
                    {"label": f"First-stage coefficient on {z_col}", "value": fnum(fs_coef),
                     "se": fnum(fs_se), "p_value": fnum(fs_p)},
                    {"label": "First-stage F statistic", "value": fnum(f_stat)},
                    {"label": "Conventional threshold", "value": 10.0},
                ],
                "columns": ["label", "value", "se", "p_value"],
                "f_stat": fnum(f_stat),
            },
        )
    )

    diagnostics.append(
        Diagnostic(
            id="reduced_form",
            title="Reduced form & the Wald ratio",
            kind="table",
            verdict="info" if abs(rf_coef / rf_se) > 1.96 else "warn",  # type: ignore[arg-type]
            summary=(
                f"The instrument moves the outcome by {rf_coef:.4g} (p = {rf_p:.3g}) and moves "
                f"treatment by {fs_coef:.4g}. Their ratio, {rf_coef/fs_coef:.4g}, is the IV estimate."
                + ("" if abs(rf_coef / rf_se) > 1.96 else " The reduced form is not itself "
                   "significant, so the IV result rests on a weak signal.")
            ),
            detail=(
                "IV is just the instrument's effect on the outcome, rescaled by how much it "
                "shifted treatment. If the reduced form is a null, no amount of rescaling "
                "creates a real effect."
            ),
            data={
                "rows": [
                    {"label": f"Reduced form: {y_col} on {z_col}", "value": fnum(rf_coef),
                     "se": fnum(rf_se), "p_value": fnum(rf_p)},
                    {"label": f"First stage: {d_col} on {z_col}", "value": fnum(fs_coef),
                     "se": fnum(fs_se), "p_value": fnum(fs_p)},
                    {"label": "Wald ratio (reduced form / first stage)",
                     "value": fnum(rf_coef / fs_coef if fs_coef else None)},
                ],
                "columns": ["label", "value", "se", "p_value"],
            },
        )
    )

    gap = abs(point - ols_coef)
    diagnostics.append(
        Diagnostic(
            id="ols_vs_iv",
            title="OLS vs IV",
            kind="table",
            verdict="info",
            summary=(
                f"OLS gives {ols_coef:.4g}; IV gives {point:.4g}. "
                + (
                    "The gap is large, which is what you would expect if treatment was "
                    "confounded — and is also a reminder that IV answers a narrower question."
                    if se and np.isfinite(se) and gap > 2 * se
                    else "The two are close, suggesting either little confounding or a weak "
                    "instrument pulling IV back toward OLS."
                )
            ),
            detail=(
                "OLS estimates the effect for everyone under no-confounding. IV estimates it "
                "only for compliers — the units whose treatment status the instrument actually "
                "changed. A difference can reflect bias, effect heterogeneity, or both."
            ),
            data={
                "rows": [
                    {"label": "OLS", "value": fnum(ols_coef), "se": fnum(ols_se),
                     "p_value": fnum(ols_p)},
                    {"label": "IV (2SLS)", "value": fnum(point), "se": fnum(se), "p_value": fnum(p)},
                ],
                "columns": ["label", "value", "se", "p_value"],
            },
        )
    )

    # --- complier share, when both instrument and treatment are binary -------
    z_binary = set(np.unique(z)).issubset({0.0, 1.0})
    d_binary = set(np.unique(d)).issubset({0.0, 1.0})
    if z_binary and d_binary:
        p1, p0 = float(d[z == 1].mean()), float(d[z == 0].mean())
        compliers = p1 - p0
        always, never = p0, 1 - p1
        diagnostics.append(
            Diagnostic(
                id="complier_share",
                title="Who does this estimate apply to?",
                kind="composition",
                verdict="info" if compliers > 0.10 else "warn",  # type: ignore[arg-type]
                summary=(
                    f"Compliers are {compliers:.1%} of the sample. The estimate describes them "
                    "only — not always-takers, never-takers, or the population average."
                    + ("" if compliers > 0.10 else " With so few compliers, the LATE generalises "
                       "to a very narrow group.")
                ),
                detail=(
                    "Under monotonicity, the sample splits into always-takers (treated regardless), "
                    "never-takers (untreated regardless), and compliers (moved by the instrument). "
                    "IV can only speak to the compliers."
                ),
                data={
                    "segments": [
                        {
                            "label": "Always-takers",
                            "share": fnum(always),
                            "note": "treated regardless of the instrument",
                            "estimated": False,
                        },
                        {
                            "label": "Compliers",
                            "share": fnum(compliers),
                            "note": "moved into treatment by the instrument",
                            "estimated": True,
                        },
                        {
                            "label": "Never-takers",
                            "share": fnum(never),
                            "note": "untreated regardless of the instrument",
                            "estimated": False,
                        },
                    ],
                    "rows": [
                        {"label": f"Take-up when {z_col} = 1", "value": fnum(p1)},
                        {"label": f"Take-up when {z_col} = 0", "value": fnum(p0)},
                        {"label": "Difference (the first stage)", "value": fnum(compliers)},
                    ],
                },
            )
        )

    diagnostics.append(
        Diagnostic(
            id="exclusion_restriction",
            title="Exclusion restriction cannot be tested",
            kind="text",
            verdict="warn",
            summary=(
                f"Everything above tests whether '{z_col}' moves treatment. Nothing tests whether "
                f"'{z_col}' affects '{y_col}' through any other channel — and with a single "
                "instrument, nothing can."
            ),
            detail=(
                f"Argue it substantively: is there any route from {z_col} to {y_col} that does "
                f"not pass through {d_col}? Direct effects, correlated omitted variables, or "
                "behavioural responses to the instrument itself all break IV. If a plausible "
                "alternative channel exists, the estimate is biased, and no diagnostic here "
                "will reveal it."
            ),
            data={},
        )
    )

    return MethodOutput(
        estimate=Estimate(
            point=point,
            se=fnum(se),
            ci_low=fnum(lo),
            ci_high=fnum(hi),
            p_value=fnum(p),
            n_obs=n,
            n_treated=int((d > 0).sum()),
            n_control=int((d == 0).sum()),
            units=y_col,
        ),
        diagnostics=diagnostics,
        specification={
            "first_stage": f"{d_col} = pi0 + pi1*{z_col}"
                           + (f" + {' + '.join(w_names)}" if w_names else ""),
            "second_stage": f"{y_col} = b0 + beta*{d_col}_hat"
                            + (f" + {' + '.join(w_names)}" if w_names else ""),
            "instrument": z_col,
            "first_stage_f": fnum(f_stat),
            "se_type": (f"cluster-robust on {cluster_col}" if cluster is not None
                        else "heteroskedasticity-robust (HC1)"),
            "covariates": covs,
        },
        warnings=warnings,
    )
