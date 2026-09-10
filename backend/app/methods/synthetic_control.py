"""Synthetic Control.

Identification: when one unit is treated and no single control is a good match,
build a weighted average of untreated donors that reproduces the treated unit's
pre-treatment outcome path. If the fit is good over a long pre-period, the synthetic
unit is a credible counterfactual afterwards.

Estimator: donor weights on the simplex (non-negative, summing to one) chosen to
minimise pre-period RMSE. Inference is by placebo-in-space: re-run the whole
procedure pretending each donor was treated, and see where the real gap ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ..schemas import Diagnostic, Estimate
from .base import (
    IdentificationError,
    MethodOutput,
    MethodSpec,
    as_binary,
    clean_frame,
    fnum,
    numeric,
    require,
)

SPEC = MethodSpec(
    id="synthetic_control",
    label="Synthetic Control",
    estimand="Effect on the treated unit — treated outcome minus its synthetic counterfactual",
    required_roles=("outcome", "treatment", "time", "unit"),
    assumptions=(
        "Good pre-treatment fit: the synthetic unit tracks the treated unit closely "
        "before treatment. A poor pre-fit invalidates the counterfactual outright.",
        "No interference: donor units are unaffected by the treated unit's treatment, "
        "or the counterfactual is contaminated.",
        "Convex hull: the treated unit's characteristics lie inside the range spanned "
        "by the donors — no extrapolation beyond what the donor pool can express.",
        "No other shocks: nothing else hits the treated unit at the same moment as "
        "treatment, or the two are indistinguishable.",
        "Enough pre-periods: a long pre-treatment window is what makes a good fit "
        "informative rather than coincidental.",
    ),
    plain_english=(
        "When only one unit is treated, no other single unit is a fair comparison. So "
        "build one: a weighted blend of untreated units that mimics the treated unit's "
        "history. After treatment, the gap between the real unit and its synthetic twin "
        "is the estimated effect."
    ),
)


def _fit_weights(Y0_pre: np.ndarray, y1_pre: np.ndarray) -> np.ndarray:
    """Weights on the simplex minimising ||y1_pre - Y0_pre @ w||, via a penalised NNLS.

    Appending a heavily-weighted row of ones drives sum(w) -> 1 while nnls enforces w >= 0.
    """
    penalty = 1e6 * (np.linalg.norm(Y0_pre) / max(np.sqrt(Y0_pre.shape[0]), 1) + 1.0)
    A = np.vstack([Y0_pre, penalty * np.ones((1, Y0_pre.shape[1]))])
    b = np.concatenate([y1_pre, [penalty]])
    w, _ = nnls(A, b)
    total = w.sum()
    return w / total if total > 0 else np.full(Y0_pre.shape[1], 1 / Y0_pre.shape[1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def estimate(df: pd.DataFrame, roles: dict[str, str | None], options: dict) -> MethodOutput:
    y_col = require(roles, "outcome", "Synthetic control")
    d_col = require(roles, "treatment", "Synthetic control")
    t_col = require(roles, "time", "Synthetic control")
    u_col = require(roles, "unit", "Synthetic control")

    frame, dropped = clean_frame(df, [y_col, d_col, t_col, u_col])
    y_all = numeric(frame, y_col)
    d_all = as_binary(frame, d_col)
    units = frame[u_col].to_numpy()
    times = frame[t_col].to_numpy()

    periods = sorted(pd.unique(times).tolist(), key=lambda v: (v is None, v))
    order = {p: i for i, p in enumerate(periods)}
    if len(periods) < 6:
        raise IdentificationError(
            f"Synthetic control needs a meaningful pre-treatment window; only {len(periods)} "
            "periods are present."
        )

    # Treated unit = the one that is ever treated. Its first treated period is the event date.
    ever = pd.DataFrame({"u": units, "d": d_all}).groupby("u", observed=True)["d"].max()
    treated_units = [u for u, v in ever.items() if v > 0]
    if not treated_units:
        raise IdentificationError("No unit is ever treated.")
    if len(treated_units) > 1:
        # A column that switches on for *every* unit at once is a period flag, not a treatment.
        by_period = (
            pd.DataFrame({"t": times, "d": d_all})
            .groupby("t", observed=True)["d"]
            .agg(["min", "max"])
        )
        if bool((by_period["min"] == by_period["max"]).all()):
            raise IdentificationError(
                f"'{d_col}' switches on for every unit at the same time, so it marks the post "
                "period rather than who was treated. Point the treatment role at a column that "
                "is 1 only for the treated unit during the post period."
            )
        raise IdentificationError(
            f"Synthetic control handles one treated unit; {len(treated_units)} are treated here. "
            "Use difference-in-differences, or run synthetic control per treated unit."
        )
    treated_unit = treated_units[0]

    treat_rows = (units == treated_unit) & (d_all > 0)
    treat_start = min(times[treat_rows], key=lambda p: order[p])
    t0 = order[treat_start]

    pre_periods = periods[:t0]
    post_periods = periods[t0:]
    if len(pre_periods) < 3:
        raise IdentificationError(
            f"Only {len(pre_periods)} pre-treatment periods. Synthetic control needs a longer "
            "history to build a credible counterfactual."
        )

    wide = (
        pd.DataFrame({"u": units, "t": times, "y": y_all})
        .pivot_table(index="t", columns="u", values="y", aggfunc="mean")
        .reindex(periods)
    )
    donors = [c for c in wide.columns if c != treated_unit]
    if len(donors) < 2:
        raise IdentificationError(
            f"Only {len(donors)} donor unit(s) available. Synthetic control needs a donor pool."
        )

    warnings: list[str] = []
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing values in the modelled columns.")

    # Drop donors with gaps -- an incomplete series cannot carry a weight.
    complete = [c for c in donors if wide[c].notna().all()]
    if len(complete) < len(donors):
        warnings.append(
            f"Excluded {len(donors) - len(complete)} donor unit(s) with incomplete series."
        )
    donors = complete
    if len(donors) < 2 or wide[treated_unit].isna().any():
        raise IdentificationError("The treated unit or the donor pool has gaps in its outcome series.")

    Y0 = wide[donors].to_numpy(dtype=float)
    y1 = wide[treated_unit].to_numpy(dtype=float)
    Y0_pre, y1_pre = Y0[:t0], y1[:t0]

    w = _fit_weights(Y0_pre, y1_pre)
    synth = Y0 @ w
    gap = y1 - synth
    pre_rmse = _rmse(y1_pre, synth[:t0])
    post_gaps = gap[t0:]
    att = float(np.mean(post_gaps))
    post_rmse = _rmse(y1[t0:], synth[t0:])

    # --- placebo-in-space inference -----------------------------------------
    placebo_ratios, placebo_gaps = [], []
    for j, donor in enumerate(donors):
        others = [k for k in range(len(donors)) if k != j]
        if len(others) < 2:
            continue
        Yp = Y0[:, others]
        yp = Y0[:, j]
        wp = _fit_weights(Yp[:t0], yp[:t0])
        sp = Yp @ wp
        gp = yp - sp
        pre_r = _rmse(yp[:t0], sp[:t0])
        if pre_r <= 0:
            continue
        post_r = _rmse(yp[t0:], sp[t0:])
        placebo_ratios.append({"unit": str(donor), "ratio": post_r / pre_r,
                               "att": float(np.mean(gp[t0:])), "pre_rmse": pre_r})
        placebo_gaps.append({"unit": str(donor), "pre_rmse": pre_r,
                             "gaps": [fnum(v) for v in gp]})

    real_ratio = post_rmse / pre_rmse if pre_rmse > 0 else float("inf")
    p_value = None
    if placebo_ratios:
        n_worse = sum(1 for pr in placebo_ratios if pr["ratio"] >= real_ratio)
        p_value = (n_worse + 1) / (len(placebo_ratios) + 1)

    # Placebo spread gives an interval; there is no standard error here.
    placebo_atts = np.array([pr["att"] for pr in placebo_ratios]) if placebo_ratios else np.array([])
    ci_low = ci_high = pseudo_se = None
    if len(placebo_atts) >= 5:
        pseudo_se = float(np.std(placebo_atts, ddof=1))
        ci_low = float(att - 1.96 * pseudo_se)
        ci_high = float(att + 1.96 * pseudo_se)

    diagnostics: list[Diagnostic] = []

    # --- pre-treatment fit quality ------------------------------------------
    # Judge the fit against two benchmarks: the naive equal-weighted donor average
    # (what you would get without optimising), and the outcome's own scale (RMSPE).
    # Normalising by the pre-period SD would unfairly punish a flat series.
    naive_rmse = _rmse(y1_pre, Y0_pre.mean(axis=1))
    improvement = 1 - (pre_rmse / naive_rmse) if naive_rmse > 0 else 0.0
    level = abs(float(np.mean(y1_pre)))
    rmspe = pre_rmse / level if level > 0 else float("inf")
    fit_ratio = pre_rmse / float(np.std(y1_pre)) if float(np.std(y1_pre)) > 0 else float("inf")

    if improvement >= 0.5 or rmspe < 0.02:
        fit_verdict = "pass"
        fit_msg = (
            f"Good pre-treatment fit: RMSE {pre_rmse:.4g} ({rmspe:.1%} of the outcome's average "
            f"level), {improvement:.0%} better than a simple average of all donors."
        )
    elif improvement > 0 or rmspe < 0.05:
        fit_verdict = "warn"
        fit_msg = (
            f"Mediocre pre-treatment fit: RMSE {pre_rmse:.4g} ({rmspe:.1%} of the outcome's "
            f"average level), only {improvement:.0%} better than a simple donor average. A loose "
            "fit before treatment weakens the counterfactual after it."
        )
    else:
        fit_verdict = "fail"
        fit_msg = (
            f"Poor pre-treatment fit: RMSE {pre_rmse:.4g} is no better than simply averaging all "
            "donors. The synthetic unit does not reproduce the treated unit's history, so the "
            "post-period gap cannot be read as a treatment effect."
        )

    diagnostics.append(
        Diagnostic(
            id="path_plot",
            title="Treated unit vs synthetic control",
            kind="path",
            verdict=fit_verdict,  # type: ignore[arg-type]
            summary=fit_msg,
            detail=(
                "Before the vertical line the two series should be nearly indistinguishable — "
                "that is what the weights were chosen to achieve. After it, the divergence is "
                "the estimated effect."
            ),
            data={
                "series": [
                    {
                        "name": f"{treated_unit} (actual)",
                        "points": [{"time": str(p), "value": fnum(v)} for p, v in zip(periods, y1)],
                    },
                    {
                        "name": "Synthetic control",
                        "points": [{"time": str(p), "value": fnum(v)} for p, v in zip(periods, synth)],
                    },
                ],
                "treatment_time": str(treat_start),
                "y_label": y_col,
                "x_label": t_col,
                "pre_rmse": fnum(pre_rmse),
                "naive_rmse": fnum(naive_rmse),
                "rmspe": fnum(rmspe),
                "improvement_over_naive": fnum(improvement),
            },
        )
    )

    diagnostics.append(
        Diagnostic(
            id="gap_plot",
            title="Gap: actual minus synthetic",
            kind="gap",
            verdict="info",
            summary=(
                f"The gap averages {np.mean(gap[:t0]):.3g} before treatment and {att:.3g} after."
            ),
            detail=(
                "The same information as the path plot, differenced. A flat line at zero before "
                "the intervention and a clear step afterwards is the signature of a real effect."
            ),
            data={
                "points": [
                    {"time": str(p), "value": fnum(v), "is_post": order[p] >= t0}
                    for p, v in zip(periods, gap)
                ],
                "treatment_time": str(treat_start),
                "y_label": f"{y_col} gap",
                "x_label": t_col,
            },
        )
    )

    # --- placebo distribution ------------------------------------------------
    if placebo_gaps:
        # Convention: drop placebos that fit their own pre-period far worse than the treated unit.
        cutoff = 5 * pre_rmse
        kept = [pg for pg in placebo_gaps if pg["pre_rmse"] <= cutoff]
        excluded = len(placebo_gaps) - len(kept)
        rank = 1 + sum(1 for pr in placebo_ratios if pr["ratio"] > real_ratio)
        total_units = len(placebo_ratios) + 1
        sig = p_value is not None and p_value <= 0.10
        diagnostics.append(
            Diagnostic(
                id="placebo_in_space",
                title="Placebo test (every donor treated in turn)",
                kind="placebo_distribution",
                verdict="pass" if sig else "warn",  # type: ignore[arg-type]
                summary=(
                    f"The treated unit ranks {rank} of {total_units} on post/pre RMSE ratio "
                    f"(exact p = {p_value:.3f})."
                    + (
                        " Its post-treatment divergence is unusual against the placebo distribution."
                        if sig
                        else " Several untreated units show gaps as large, so the effect is not "
                        "clearly distinguishable from noise."
                    )
                ),
                detail=(
                    "With one treated unit there is no sampling distribution to appeal to. Instead "
                    "we pretend each donor was treated and re-run everything. If the real unit's "
                    "post-treatment gap is unremarkable among those placebos, the result is not "
                    f"credible. {excluded} placebo(s) with poor pre-fit are shown faded."
                ),
                data={
                    "treated_name": str(treated_unit),
                    "treated_gaps": [
                        {"time": str(p), "value": fnum(v)} for p, v in zip(periods, gap)
                    ],
                    "placebos": [
                        {
                            "unit": pg["unit"],
                            "poor_fit": pg["pre_rmse"] > cutoff,
                            "points": [
                                {"time": str(p), "value": v} for p, v in zip(periods, pg["gaps"])
                            ],
                        }
                        for pg in placebo_gaps
                    ],
                    "treatment_time": str(treat_start),
                    "p_value": fnum(p_value),
                    "rank": rank,
                    "n_units": total_units,
                    "x_label": t_col,
                },
            )
        )

    # --- donor weights -------------------------------------------------------
    weight_rows = sorted(
        ({"label": str(u), "value": fnum(wi)} for u, wi in zip(donors, w) if wi > 1e-4),
        key=lambda r: -(r["value"] or 0),
    )
    top_share = sum(r["value"] or 0 for r in weight_rows[:3])
    concentrated = top_share > 0.9 and len(weight_rows) <= 3
    diagnostics.append(
        Diagnostic(
            id="donor_weights",
            title="Donor weights",
            kind="table",
            verdict="warn" if concentrated else "pass",  # type: ignore[arg-type]
            summary=(
                f"{len(weight_rows)} of {len(donors)} donors receive positive weight; the top "
                f"three carry {top_share:.0%} of it."
                + (" With the counterfactual resting on so few donors, it is fragile to any one "
                   "of them being idiosyncratic." if concentrated else "")
            ),
            detail=(
                "Weights are non-negative and sum to one, so the synthetic unit is an "
                "interpolation of real units — never an extrapolation beyond them."
            ),
            data={"rows": weight_rows, "columns": ["label", "value"]},
        )
    )

    if len(pre_periods) < 8:
        warnings.append(
            f"Only {len(pre_periods)} pre-treatment periods. A short pre-window makes a good fit "
            "easy to achieve by chance."
        )
    if len(donors) < 10:
        warnings.append(
            f"Donor pool has {len(donors)} units. Placebo inference is coarse — the smallest "
            f"attainable p-value is {1/(len(donors)+1):.3f}."
        )

    return MethodOutput(
        estimate=Estimate(
            point=att,
            se=fnum(pseudo_se),
            ci_low=fnum(ci_low),
            ci_high=fnum(ci_high),
            p_value=fnum(p_value),
            n_obs=int(len(frame)),
            n_treated=1,
            n_control=len(donors),
            units=y_col,
        ),
        diagnostics=diagnostics,
        specification={
            "treated_unit": str(treated_unit),
            "treatment_period": str(treat_start),
            "n_donors": len(donors),
            "pre_periods": len(pre_periods),
            "post_periods": len(post_periods),
            "pre_treatment_rmse": fnum(pre_rmse),
            "pre_treatment_rmspe": fnum(rmspe),
            "pre_rmse_vs_naive_donor_average": fnum(naive_rmse),
            "weights_objective": "minimise pre-treatment outcome RMSE, weights on the simplex",
            "inference": "placebo-in-space (exact permutation p-value on post/pre RMSE ratio)",
            "se_note": "The reported SE is the spread of placebo effects, not a sampling SE.",
        },
        warnings=warnings,
    )
