"""Difference-in-Differences.

Identification: the treated group's outcome would have moved in parallel with the
control group's, absent treatment. We cannot test that counterfactual directly, so
we test its observable implication -- parallel *pre*-treatment trends -- via an
event-study specification.

Estimator: two-way fixed effects,  y_it = a_i + g_t + tau * D_it + e_it,
with standard errors clustered on the unit (Bertrand-Duflo-Mullainathan).
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
    as_binary,
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
    id="did",
    label="Difference-in-Differences",
    estimand="ATT — average treatment effect on the treated",
    required_roles=("outcome", "treatment", "time", "unit"),
    optional_roles=("covariates",),
    assumptions=(
        "Parallel trends: absent treatment, treated and control outcomes would have "
        "followed the same path over time.",
        "No anticipation: units do not change behaviour before treatment begins.",
        "Stable composition: the same units are observed before and after (or entry "
        "and exit are unrelated to treatment).",
        "No spillovers (SUTVA): control units are unaffected by others' treatment.",
    ),
    plain_english=(
        "Compare the before-to-after change in the treated group against the "
        "before-to-after change in an untreated comparison group. The second "
        "difference nets out anything that shifted both groups equally."
    ),
)


def _treatment_timing(df: pd.DataFrame, unit: str, time: str, d: np.ndarray) -> tuple[dict, list]:
    """First treated period per unit; None for never-treated units."""
    work = pd.DataFrame({"unit": df[unit].to_numpy(), "time": df[time].to_numpy(), "d": d})
    periods = sorted(pd.unique(work["time"]).tolist(), key=lambda v: (v is None, v))
    order = {p: i for i, p in enumerate(periods)}
    first: dict = {}
    for u, grp in work.groupby("unit", observed=True):
        treated_rows = grp[grp["d"] > 0]
        first[u] = min(treated_rows["time"], key=lambda p: order[p]) if len(treated_rows) else None
    return first, periods


def estimate(df: pd.DataFrame, roles: dict[str, str | None], options: dict) -> MethodOutput:
    y_col = require(roles, "outcome", "DiD")
    d_col = require(roles, "treatment", "DiD")
    t_col = require(roles, "time", "DiD")
    u_col = require(roles, "unit", "DiD")
    covs = [c for c in (roles.get("covariates") or []) if c and c in df.columns] \
        if isinstance(roles.get("covariates"), list) else []

    frame, dropped = clean_frame(df, [y_col, d_col, t_col, u_col, *covs])
    if len(frame) < 20:
        raise IdentificationError("Fewer than 20 complete rows remain after dropping missing values.")

    y = numeric(frame, y_col)
    d = as_binary(frame, d_col)
    units = frame[u_col].to_numpy()
    times = frame[t_col].to_numpy()

    if d.sum() == 0 or d.sum() == len(d):
        raise IdentificationError("The treatment indicator has no variation across rows.")

    first_treated, periods = _treatment_timing(frame, u_col, t_col, d)
    order = {p: i for i, p in enumerate(periods)}
    n_periods = len(periods)
    if n_periods < 2:
        raise IdentificationError("DiD needs at least two time periods; the data has one.")

    ever = {u: (ft is not None) for u, ft in first_treated.items()}
    n_treated_units = sum(ever.values())
    n_control_units = len(ever) - n_treated_units
    if n_treated_units == 0 or n_control_units == 0:
        raise IdentificationError(
            "DiD needs both ever-treated and never-treated units; the data has only one kind."
        )

    warnings: list[str] = []
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing values in the modelled columns.")

    adoption_periods = {ft for ft in first_treated.values() if ft is not None}
    staggered = len(adoption_periods) > 1
    if staggered:
        warnings.append(
            f"Treatment is staggered across {len(adoption_periods)} adoption dates. The two-way "
            "fixed-effects estimate can be biased when effects vary over time "
            "(Goodman-Bacon 2021); treat it as a weighted average, not a clean ATT."
        )

    # --- main TWFE regression ------------------------------------------------
    unit_d, _ = dummies(units)
    time_d, _ = dummies(times)
    cov_blocks = []
    for c in covs:
        col = frame[c]
        if pd.api.types.is_numeric_dtype(col):
            cov_blocks.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)[:, None])
        else:
            block, _ = dummies(col.to_numpy())
            if block.size:
                cov_blocks.append(block)

    X = np.column_stack([np.ones(len(y)), d, unit_d, time_d, *cov_blocks])
    beta, resid = ols(X, y)
    vcov = robust_vcov(X, resid, cluster=units)
    n_clusters = len(pd.unique(units))
    point, se, lo, hi, p = coef_inference(beta, vcov, 1, dof=n_clusters - 1)

    if n_clusters < 30:
        warnings.append(
            f"Only {n_clusters} clusters. Cluster-robust standard errors are unreliable "
            "below roughly 30-40 clusters; treat the confidence interval as optimistic."
        )

    diagnostics: list[Diagnostic] = []

    # --- event study: leads and lags relative to the last untreated period ----
    rel = np.full(len(frame), np.nan)
    for i, (u, t) in enumerate(zip(units, times)):
        ft = first_treated.get(u)
        if ft is not None:
            rel[i] = order[t] - order[ft]
    has_rel = ~np.isnan(rel)

    event_data: dict = {}
    pretrend_verdict = "info"
    pretrend_summary = "Not enough pre-periods to test parallel trends."
    pretrend_detail = None

    rel_values = sorted({int(r) for r in rel[has_rel]})
    leads = [r for r in rel_values if r < -1]
    if leads and n_periods >= 3:
        # Reference period is r = -1 (the period just before treatment).
        event_cols, labels = [], []
        for r in rel_values:
            if r == -1:
                continue
            col = np.where(has_rel & (rel == r), 1.0, 0.0)
            if col.sum() >= 3:
                event_cols.append(col)
                labels.append(r)
        if event_cols:
            Xe = np.column_stack([np.ones(len(y)), np.column_stack(event_cols), unit_d, time_d])
            beta_e, resid_e = ols(Xe, y)
            vcov_e = robust_vcov(Xe, resid_e, cluster=units)
            points = []
            for j, r in enumerate(labels):
                pt, se_e, lo_e, hi_e, p_e = coef_inference(beta_e, vcov_e, 1 + j, n_clusters - 1)
                points.append(
                    {
                        "period": r,
                        "coef": fnum(pt),
                        "se": fnum(se_e),
                        "ci_low": fnum(lo_e),
                        "ci_high": fnum(hi_e),
                        "is_pre": r < 0,
                    }
                )
            points.append(
                {"period": -1, "coef": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                 "is_pre": True, "is_reference": True}
            )
            points.sort(key=lambda pt: pt["period"])
            event_data = {"points": points, "reference_period": -1}

            # Joint F-test that all pre-treatment leads are zero.
            lead_idx = [1 + j for j, r in enumerate(labels) if r < -1]
            if lead_idx:
                R = np.zeros((len(lead_idx), Xe.shape[1]))
                for row, ci in enumerate(lead_idx):
                    R[row, ci] = 1.0
                Rb = R @ beta_e
                mid = R @ vcov_e @ R.T
                try:
                    wald = float(Rb.T @ np.linalg.pinv(mid) @ Rb)
                    q = len(lead_idx)
                    f_stat = wald / q
                    p_pre = float(stats.f.sf(f_stat, q, max(n_clusters - 1, 1)))
                    biggest = max(abs(pt["coef"] or 0) for pt in points if pt["is_pre"])
                    rel_size = biggest / abs(point) if point else float("inf")
                    if p_pre < 0.05:
                        pretrend_verdict = "fail"
                        pretrend_summary = (
                            f"Pre-treatment trends diverge (joint F-test p = {p_pre:.3f}). "
                            "Parallel trends is rejected."
                        )
                    elif rel_size > 0.5:
                        pretrend_verdict = "warn"
                        pretrend_summary = (
                            f"Pre-trends are not statistically significant (p = {p_pre:.3f}), but the "
                            f"largest pre-period coefficient is {rel_size:.0%} of the estimated effect."
                        )
                    else:
                        pretrend_verdict = "pass"
                        pretrend_summary = (
                            f"Pre-treatment coefficients are jointly indistinguishable from zero "
                            f"(F = {f_stat:.2f}, p = {p_pre:.3f}). Parallel trends is supported."
                        )
                    pretrend_detail = (
                        "Each point is the treated-vs-control gap in that period relative to the "
                        "period just before treatment. Points to the left of 0 should sit on zero "
                        "if the two groups were moving together beforehand."
                    )
                    event_data["f_stat"] = fnum(f_stat)
                    event_data["p_value"] = fnum(p_pre)
                except np.linalg.LinAlgError:
                    pass

    diagnostics.append(
        Diagnostic(
            id="parallel_trends",
            title="Parallel pre-trends (event study)",
            kind="event_study",
            verdict=pretrend_verdict,  # type: ignore[arg-type]
            summary=pretrend_summary,
            detail=pretrend_detail,
            data=event_data,
        )
    )

    # --- raw group means over time (the classic two-line DiD picture) ---------
    ever_arr = np.array([1.0 if ever[u] else 0.0 for u in units])
    means = (
        pd.DataFrame({"time": times, "group": ever_arr, "y": y})
        .groupby(["time", "group"], observed=True)["y"]
        .mean()
        .reset_index()
    )
    series = []
    for grp, label in ((1.0, "Treated"), (0.0, "Control")):
        sub = means[means["group"] == grp].sort_values("time", key=lambda s: s.map(order))
        series.append(
            {
                "name": label,
                "points": [
                    {"time": str(t), "value": fnum(v)}
                    for t, v in zip(sub["time"], sub["y"])
                ],
            }
        )
    treat_start = min(adoption_periods, key=lambda p: order[p]) if adoption_periods else None
    diagnostics.append(
        Diagnostic(
            id="group_means",
            title="Outcome paths by group",
            kind="path",
            verdict="info",
            summary=(
                "Raw group averages over time. The vertical line marks when treatment begins."
            ),
            detail=(
                "DiD reads the vertical gap between these two lines after treatment, minus "
                "the gap before it."
            ),
            data={
                "series": series,
                "treatment_time": str(treat_start) if treat_start is not None else None,
                "y_label": y_col,
                "x_label": t_col,
            },
        )
    )

    # --- placebo: fake treatment one period before the real one --------------
    if treat_start is not None and order[treat_start] >= 2:
        fake_start = order[treat_start] - 1
        pre_mask = np.array([order[t] < order[treat_start] for t in times])
        if pre_mask.sum() >= 20:
            d_fake = np.array(
                [
                    1.0 if (ever[u] and order[t] >= fake_start) else 0.0
                    for u, t in zip(units[pre_mask], times[pre_mask])
                ]
            )
            if 0 < d_fake.sum() < len(d_fake):
                ud, _ = dummies(units[pre_mask])
                td, _ = dummies(times[pre_mask])
                Xp = np.column_stack([np.ones(int(pre_mask.sum())), d_fake, ud, td])
                bp, rp = ols(Xp, y[pre_mask])
                vp = robust_vcov(Xp, rp, cluster=units[pre_mask])
                ppt, pse, plo, phi, ppv = coef_inference(
                    bp, vp, 1, len(pd.unique(units[pre_mask])) - 1
                )
                placebo_ok = (ppv is not None and np.isfinite(ppv) and ppv > 0.10)
                diagnostics.append(
                    Diagnostic(
                        id="placebo_timing",
                        title="Placebo: fake treatment date",
                        kind="table",
                        verdict="pass" if placebo_ok else "warn",  # type: ignore[arg-type]
                        summary=(
                            f"Re-running DiD on pre-treatment periods only, pretending treatment "
                            f"started one period early, gives {ppt:.3f} (p = {ppv:.3f})."
                            + ("" if placebo_ok else " A significant placebo effect suggests the "
                               "groups were already diverging.")
                        ),
                        detail=(
                            "A well-identified design should find nothing here: there was no "
                            "treatment yet, so the estimate should be near zero."
                        ),
                        data={
                            "rows": [
                                {"label": "Placebo estimate", "value": fnum(ppt)},
                                {"label": "Std. error", "value": fnum(pse)},
                                {"label": "p-value", "value": fnum(ppv)},
                                {"label": "Real estimate", "value": fnum(point)},
                            ]
                        },
                    )
                )

    # --- balance of the panel ------------------------------------------------
    counts = pd.Series(units).value_counts()
    balanced = counts.nunique() == 1
    diagnostics.append(
        Diagnostic(
            id="panel_balance",
            title="Panel balance & composition",
            kind="table",
            verdict="pass" if balanced else "warn",  # type: ignore[arg-type]
            summary=(
                f"Balanced panel: all {len(counts)} units appear in all {n_periods} periods."
                if balanced
                else f"Unbalanced panel: units are observed between {counts.min()} and "
                f"{counts.max()} times. Composition changes can masquerade as treatment effects."
            ),
            data={
                "rows": [
                    {"label": "Units", "value": int(len(counts))},
                    {"label": "Treated units", "value": int(n_treated_units)},
                    {"label": "Control units", "value": int(n_control_units)},
                    {"label": "Periods", "value": int(n_periods)},
                    {"label": "Observations", "value": int(len(frame))},
                ]
            },
        )
    )

    estimate_obj = Estimate(
        point=point,
        se=fnum(se),
        ci_low=fnum(lo),
        ci_high=fnum(hi),
        p_value=fnum(p),
        n_obs=int(len(frame)),
        n_treated=int(d.sum()),
        n_control=int(len(d) - d.sum()),
        units=y_col,
    )

    return MethodOutput(
        estimate=estimate_obj,
        diagnostics=diagnostics,
        specification={
            "equation": f"{y_col}_it = alpha_i + gamma_t + tau * {d_col}_it + e_it",
            "fixed_effects": [u_col, t_col],
            "se_type": f"cluster-robust on {u_col}",
            "n_clusters": int(n_clusters),
            "covariates": covs,
            "staggered_adoption": bool(staggered),
            "periods": [str(p) for p in periods],
        },
        warnings=warnings,
    )
