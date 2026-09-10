"""Sharp Regression Discontinuity.

Identification: treatment is assigned by whether a running variable crosses a
cutoff. Units just below the cutoff are a credible counterfactual for units just
above, provided nothing else jumps there and units cannot precisely sort across it.

Estimator: local linear regression on each side of the cutoff with a triangular
kernel, using an Imbens-Kalyanaraman optimal bandwidth. The estimate is the
difference in intercepts at the cutoff.
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
    fnum,
    numeric,
    ols,
    require,
    robust_vcov,
)

SPEC = MethodSpec(
    id="rdd",
    label="Regression Discontinuity",
    estimand="LATE at the cutoff — local average treatment effect for units at the threshold",
    required_roles=("outcome", "running_variable"),
    optional_roles=("treatment", "covariates"),
    assumptions=(
        "Continuity: all other determinants of the outcome vary smoothly through the "
        "cutoff, so the only thing that jumps is treatment.",
        "No precise manipulation: units cannot finely control which side of the cutoff "
        "they land on (testable via the density of the running variable).",
        "Correct functional form: the relationship on each side is well approximated "
        "locally, so the jump is not an artifact of curvature.",
    ),
    plain_english=(
        "When a rule hands out treatment based on a score crossing a threshold, people "
        "just above and just below the line are near-identical by luck. Any sudden jump "
        "in outcomes exactly at the line is the treatment effect."
    ),
)


def _triangular(u: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(u))


def _local_linear(
    x: np.ndarray, y: np.ndarray, cutoff: float, h: float
) -> tuple[float, float, float, int, int]:
    """Weighted local-linear fit on both sides. Returns (tau, se, dof, n_left, n_right)."""
    d = (x >= cutoff).astype(float)
    xc = x - cutoff
    w = _triangular(xc / h)
    keep = w > 0
    if keep.sum() < 10:
        raise IdentificationError(
            f"Only {int(keep.sum())} observations fall inside the bandwidth; too few to fit."
        )
    xk, yk, dk, wk = xc[keep], y[keep], d[keep], w[keep]
    n_left, n_right = int((dk == 0).sum()), int((dk == 1).sum())
    if n_left < 5 or n_right < 5:
        raise IdentificationError(
            f"Need observations on both sides of the cutoff (found {n_left} below, {n_right} above)."
        )
    # y = a + tau*D + b1*x + b2*(D*x): tau is the jump at x = 0.
    X = np.column_stack([np.ones(len(xk)), dk, xk, dk * xk])
    sw = np.sqrt(wk)
    beta, _ = ols(X * sw[:, None], yk * sw)
    resid = yk - X @ beta
    Xw = X * sw[:, None]
    vcov = robust_vcov(Xw, resid * sw, cluster=None)
    tau = float(beta[1])
    se = float(np.sqrt(max(vcov[1, 1], 0)))
    return tau, se, len(xk) - 4, n_left, n_right


def _ik_bandwidth(x: np.ndarray, y: np.ndarray, cutoff: float) -> float:
    """Imbens-Kalyanaraman (2012) optimal bandwidth for a triangular kernel."""
    n = len(x)
    xc = x - cutoff
    sd = float(np.std(xc))
    # Step 1: pilot bandwidth (Silverman) and density/variance at the cutoff.
    h1 = 1.84 * sd * n ** (-1 / 5)
    near = np.abs(xc) <= h1
    if near.sum() < 20:
        return max(h1, sd * 0.5)
    f_hat = float(near.sum()) / (2 * h1 * n)
    left, right = near & (xc < 0), near & (xc >= 0)
    if left.sum() < 5 or right.sum() < 5:
        return max(h1, sd * 0.5)
    sigma2 = 0.5 * (float(np.var(y[left])) + float(np.var(y[right])))

    # Step 2: second derivatives from cubic fits on each side.
    def curv(mask: np.ndarray) -> float:
        xm, ym = xc[mask], y[mask]
        if len(xm) < 10:
            return 0.0
        X = np.column_stack([np.ones(len(xm)), xm, xm**2, xm**3])
        b, _ = ols(X, ym)
        return float(2 * b[2])

    h2 = 3.56 * sd * n ** (-1 / 7)
    m2_r = curv((xc >= 0) & (xc <= h2))
    m2_l = curv((xc < 0) & (xc >= -h2))

    # Regularisation terms guard against a near-zero curvature difference.
    n_r, n_l = max(int(((xc >= 0) & (xc <= h2)).sum()), 1), max(int(((xc < 0) & (xc >= -h2)).sum()), 1)
    r_r = 2160 * sigma2 / (n_r * h2**4)
    r_l = 2160 * sigma2 / (n_l * h2**4)

    denom = (m2_r - m2_l) ** 2 + r_r + r_l
    if denom <= 0 or f_hat <= 0:
        return max(h1, sd * 0.5)
    h = 3.4375 * ((2 * sigma2) / (f_hat * denom)) ** (1 / 5) * n ** (-1 / 5)
    # Keep the bandwidth in a sane range relative to the data's spread.
    return float(np.clip(h, 0.05 * sd, 2.0 * sd))


def _infer_cutoff(x: np.ndarray, treat: np.ndarray | None) -> tuple[float, int]:
    """Return the cutoff and which side is treated: +1 at-or-above, -1 below.

    Plenty of real assignment rules run downward -- a benefit for income below a line, a
    remediation class for scores below a pass mark -- so both directions are considered.
    """
    if treat is None:
        raise IdentificationError(
            "RDD needs a cutoff. Provide a treatment column, or set options.cutoff explicitly."
        )
    above = x[treat > 0]
    below = x[treat == 0]
    if len(above) == 0 or len(below) == 0:
        raise IdentificationError("The treatment indicator does not split the running variable.")

    # Sharp designs: the cutoff sits between the two groups' nearest values.
    if above.min() > below.max():
        return float((above.min() + below.max()) / 2), 1
    if below.min() > above.max():
        return float((below.min() + above.max()) / 2), -1

    # Fuzzy: fall back to the threshold that best reproduces the assignment.
    from ..identify import best_threshold

    cutoff, _acc, side = best_threshold(x, treat)
    return float(cutoff), side


def estimate(df: pd.DataFrame, roles: dict[str, str | None], options: dict) -> MethodOutput:
    y_col = require(roles, "outcome", "RDD")
    x_col = require(roles, "running_variable", "RDD")
    d_col = roles.get("treatment")
    covs = [c for c in (roles.get("covariates") or []) if c and c in df.columns] \
        if isinstance(roles.get("covariates"), list) else []

    frame, dropped = clean_frame(df, [y_col, x_col, d_col, *covs])
    if len(frame) < 50:
        raise IdentificationError("RDD needs at least 50 complete rows; fewer remain after cleaning.")

    y = numeric(frame, y_col)
    x = numeric(frame, x_col)
    treat = None
    if d_col and d_col in frame.columns:
        t_raw = pd.to_numeric(frame[d_col], errors="coerce")
        if t_raw.notna().all() and t_raw.nunique() == 2:
            treat = (t_raw.to_numpy() == t_raw.max()).astype(float)

    if options.get("cutoff") is not None:
        cutoff = float(options["cutoff"])
        side = int(options.get("treated_side", 1)) or 1
    else:
        cutoff, side = _infer_cutoff(x, treat)
    treated_above = side >= 0

    warnings: list[str] = []
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing values in the modelled columns.")

    # Fuzzy check: if treatment is not perfectly determined by the cutoff, say so.
    if treat is not None:
        implied = ((x >= cutoff) if treated_above else (x <= cutoff)).astype(float)
        compliance = float((implied == treat).mean())
        if compliance < 0.99:
            warnings.append(
                f"The cutoff rule predicts treatment for only {compliance:.1%} of rows. This is a "
                "fuzzy discontinuity; the sharp estimate below understates the effect on compliers. "
                "Consider instrumental variables using the cutoff as the instrument."
            )

    h = float(options["bandwidth"]) if options.get("bandwidth") else _ik_bandwidth(x, y, cutoff)
    raw_tau, se, dof, n_left, n_right = _local_linear(x, y, cutoff, h)
    # _local_linear always reports (above - below). When the rule treats the units *below*
    # the cutoff, the treated-minus-control effect is the negative of that.
    flip = 1.0 if treated_above else -1.0
    tau = flip * raw_tau
    crit = float(stats.t.ppf(0.975, max(dof, 1)))
    p = float(2 * stats.t.sf(abs(tau / se), max(dof, 1))) if se > 0 else float("nan")
    if not treated_above:
        warnings.append(
            f"Treatment is assigned to units *below* {x_col} = {cutoff:g}. The estimate is "
            "reported as treated minus untreated, so its sign already accounts for that."
        )

    diagnostics: list[Diagnostic] = []

    # --- the jump plot: binned means + fitted lines on each side -------------
    window = min(3.0 * h, float(np.max(np.abs(x - cutoff))))
    in_window = np.abs(x - cutoff) <= window
    n_bins = 40
    edges = np.linspace(cutoff - window, cutoff + window, n_bins + 1)
    # Force a bin boundary exactly at the cutoff so no bin straddles it.
    edges = np.unique(np.concatenate([edges, [cutoff]]))
    bins = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = in_window & (x >= lo_e) & (x < hi_e)
        if m.sum() >= 3:
            bins.append(
                {
                    "x": fnum((lo_e + hi_e) / 2),
                    "y": fnum(float(np.mean(y[m]))),
                    "n": int(m.sum()),
                    "side": "right" if (lo_e + hi_e) / 2 >= cutoff else "left",
                }
            )

    # Fitted lines from the same weighted local-linear model used for the estimate.
    def _fit_line(side: str) -> list[dict]:
        mask = (x >= cutoff) if side == "right" else (x < cutoff)
        mask = mask & (np.abs(x - cutoff) <= h)
        if mask.sum() < 5:
            return []
        xm, ym = x[mask] - cutoff, y[mask]
        w = _triangular(xm / h)
        X = np.column_stack([np.ones(len(xm)), xm])
        sw = np.sqrt(w)
        b, _ = ols(X * sw[:, None], ym * sw)
        grid = np.linspace(0, h, 25) if side == "right" else np.linspace(-h, 0, 25)
        return [{"x": fnum(cutoff + g), "y": fnum(float(b[0] + b[1] * g))} for g in grid]

    left_line, right_line = _fit_line("left"), _fit_line("right")
    intercept_gap = None
    if left_line and right_line:
        intercept_gap = fnum(right_line[0]["y"] - left_line[-1]["y"])

    diagnostics.append(
        Diagnostic(
            id="jump_plot",
            title="Discontinuity at the cutoff",
            kind="scatter_fit",
            verdict="info",
            summary=(
                f"Binned outcome averages either side of {x_col} = {cutoff:g}, with the local "
                f"linear fits used for the estimate (bandwidth {h:.3g})."
            ),
            detail=(
                "The estimate is the vertical gap between the two fitted lines exactly at the "
                "cutoff. Everything outside the shaded bandwidth is shown for context but "
                "receives little or no weight."
            ),
            data={
                "bins": bins,
                "left_fit": left_line,
                "right_fit": right_line,
                "cutoff": fnum(cutoff),
                "bandwidth": fnum(h),
                "jump": intercept_gap,
                "treated_side": "right" if treated_above else "left",
                "x_label": x_col,
                "y_label": y_col,
            },
        )
    )

    # --- manipulation: does the density of the running variable jump? --------
    bw_d = max(window / 20, 1e-9)
    left_counts = int(((x >= cutoff - 5 * bw_d) & (x < cutoff)).sum())
    right_counts = int(((x >= cutoff) & (x < cutoff + 5 * bw_d)).sum())
    total = left_counts + right_counts
    density_verdict, density_summary = "info", "Too few observations near the cutoff to test."
    density_p = None
    if total >= 30:
        # Binomial test: with no manipulation, points near the cutoff split ~50/50.
        density_p = float(stats.binomtest(right_counts, total, 0.5).pvalue)
        if density_p < 0.01:
            density_verdict = "fail"
            density_summary = (
                f"Density jumps at the cutoff ({left_counts} below vs {right_counts} above, "
                f"p = {density_p:.3f}). This is the signature of units sorting across the "
                "threshold, which breaks RDD."
            )
        elif density_p < 0.10:
            density_verdict = "warn"
            density_summary = (
                f"Mild density imbalance at the cutoff ({left_counts} vs {right_counts}, "
                f"p = {density_p:.3f}). Worth a closer look at whether units can game the score."
            )
        else:
            density_verdict = "pass"
            density_summary = (
                f"No density jump at the cutoff ({left_counts} below vs {right_counts} above, "
                f"p = {density_p:.3f}). No evidence of manipulation."
            )

    hist_edges = np.linspace(cutoff - window, cutoff + window, 31)
    hist_edges = np.unique(np.concatenate([hist_edges, [cutoff]]))
    hist = [
        {
            "x": fnum((a + b) / 2),
            "count": int(((x >= a) & (x < b)).sum()),
            "side": "right" if (a + b) / 2 >= cutoff else "left",
        }
        for a, b in zip(hist_edges[:-1], hist_edges[1:])
    ]
    diagnostics.append(
        Diagnostic(
            id="density_test",
            title="Manipulation test (density of the running variable)",
            kind="overlap",
            verdict=density_verdict,  # type: ignore[arg-type]
            summary=density_summary,
            detail=(
                "If units could precisely control their score, we would see a pile-up on the "
                "side that gets treated. The histogram should pass smoothly through the cutoff."
            ),
            data={"histogram": hist, "cutoff": fnum(cutoff), "p_value": fnum(density_p),
                  "x_label": x_col},
        )
    )

    # --- covariate continuity: predetermined variables should not jump -------
    if covs:
        rows, worst_p = [], 1.0
        for c in covs:
            if c in (y_col, x_col, d_col):
                continue
            col = pd.to_numeric(frame[c], errors="coerce")
            if col.isna().any() or col.nunique() < 3:
                continue
            try:
                t_c, se_c, dof_c, _, _ = _local_linear(x, col.to_numpy(dtype=float), cutoff, h)
            except IdentificationError:
                continue
            p_c = float(2 * stats.t.sf(abs(t_c / se_c), max(dof_c, 1))) if se_c > 0 else 1.0
            worst_p = min(worst_p, p_c)
            rows.append({"label": c, "value": fnum(flip * t_c), "se": fnum(se_c),
                         "p_value": fnum(p_c)})
        if rows:
            ok = worst_p > 0.05
            diagnostics.append(
                Diagnostic(
                    id="covariate_continuity",
                    title="Covariate continuity at the cutoff",
                    kind="table",
                    verdict="pass" if ok else "warn",  # type: ignore[arg-type]
                    summary=(
                        "No predetermined covariate jumps at the cutoff — consistent with "
                        "as-good-as-random assignment locally."
                        if ok
                        else f"At least one covariate jumps at the cutoff (smallest p = {worst_p:.3f}). "
                        "Something other than treatment changes there, which threatens continuity."
                    ),
                    detail=(
                        "Running the same discontinuity estimate on variables determined *before* "
                        "treatment. All of these should be flat through the cutoff."
                    ),
                    data={"rows": rows, "columns": ["label", "value", "se", "p_value"]},
                )
            )

    # --- bandwidth sensitivity ----------------------------------------------
    sens = []
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        try:
            t_b, se_b, dof_b, nl, nr = _local_linear(x, y, cutoff, h * mult)
        except IdentificationError:
            continue
        crit_b = float(stats.t.ppf(0.975, max(dof_b, 1)))
        sens.append(
            {
                "bandwidth": fnum(h * mult),
                "multiplier": mult,
                "estimate": fnum(t_b),
                "ci_low": fnum(t_b - crit_b * se_b),
                "ci_high": fnum(t_b + crit_b * se_b),
                "n": nl + nr,
            }
        )
    if sens:
        ests = [s["estimate"] for s in sens if s["estimate"] is not None]
        spread = (max(ests) - min(ests)) / abs(tau) if tau else float("inf")
        stable = spread < 0.5
        diagnostics.append(
            Diagnostic(
                id="bandwidth_sensitivity",
                title="Bandwidth sensitivity",
                kind="table",
                verdict="pass" if stable else "warn",  # type: ignore[arg-type]
                summary=(
                    f"The estimate ranges from {min(ests):.4g} to {max(ests):.4g} across bandwidths "
                    f"from {h*0.5:.3g} to {h*2:.3g}."
                    + ("" if stable else " That is a wide swing; the result depends on how much "
                       "data you include.")
                ),
                detail=(
                    "RDD trades bias against variance through the bandwidth. A credible result "
                    "should not move much as the window widens or narrows."
                ),
                data={
                    "rows": sens,
                    "columns": ["bandwidth", "estimate", "ci_low", "ci_high", "n"],
                    "optimal_bandwidth": fnum(h),
                },
            )
        )

    # --- placebo cutoffs -----------------------------------------------------
    placebos = []
    for q in (0.25, 0.35, 0.65, 0.75):
        fake = float(np.quantile(x, q))
        if abs(fake - cutoff) < h:
            continue
        side = x < cutoff if fake < cutoff else x >= cutoff
        try:
            t_f, se_f, dof_f, _, _ = _local_linear(x[side], y[side], fake, h)
        except IdentificationError:
            continue
        p_f = float(2 * stats.t.sf(abs(t_f / se_f), max(dof_f, 1))) if se_f > 0 else 1.0
        placebos.append(
            {"label": f"{x_col} = {fake:.3g}", "value": fnum(flip * t_f), "p_value": fnum(p_f)}
        )
    if placebos:
        n_sig = sum(1 for pl in placebos if (pl["p_value"] or 1) < 0.05)
        diagnostics.append(
            Diagnostic(
                id="placebo_cutoffs",
                title="Placebo cutoffs",
                kind="table",
                verdict="pass" if n_sig == 0 else "warn",  # type: ignore[arg-type]
                summary=(
                    f"No significant jump at any of the {len(placebos)} fake cutoffs tested."
                    if n_sig == 0
                    else f"{n_sig} of {len(placebos)} fake cutoffs show a significant jump, which "
                    "suggests the method finds discontinuities where none should exist."
                ),
                detail=(
                    "Re-running the estimate at thresholds where nothing happened. Finding jumps "
                    "there would mean the functional form, not the policy, is driving the result."
                ),
                data={"rows": placebos, "columns": ["label", "value", "p_value"]},
            )
        )

    return MethodOutput(
        estimate=Estimate(
            point=tau,
            se=fnum(se),
            ci_low=fnum(tau - crit * se),
            ci_high=fnum(tau + crit * se),
            p_value=fnum(p),
            n_obs=int(n_left + n_right),
            n_treated=int(n_right),
            n_control=int(n_left),
            units=y_col,
        ),
        diagnostics=diagnostics,
        specification={
            "equation": (
                f"{y_col} = a + tau*1[{x_col} {'>=' if treated_above else '<='} {cutoff:g}] "
                f"+ b1*({x_col}-c) + b2*1[...]*({x_col}-c), triangular kernel"
            ),
            "cutoff": fnum(cutoff),
            "treated_side": "at or above the cutoff" if treated_above else "below the cutoff",
            "bandwidth": fnum(h),
            "bandwidth_selector": "user-specified" if options.get("bandwidth") else "Imbens-Kalyanaraman",
            "kernel": "triangular",
            "polynomial_order": 1,
            "se_type": "heteroskedasticity-robust (HC1)",
        },
        warnings=warnings,
    )
