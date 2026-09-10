"""Propensity Score Matching.

Identification: conditional on the observed covariates, treatment is as good as
random (selection on observables / conditional independence). Matching does not
create this assumption -- it only makes it easier to satisfy by comparing units
with similar propensity to be treated.

Estimator: logit propensity model, 1:M nearest-neighbour matching on the linear
propensity index with a caliper and replacement, ATT with Abadie-Imbens (2006)
matching standard errors, which account for controls being reused.
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
    dummies,
    fnum,
    numeric,
    ols,
    require,
)

SPEC = MethodSpec(
    id="psm",
    label="Propensity Score Matching",
    estimand="ATT — average treatment effect on the treated",
    required_roles=("outcome", "treatment", "covariates"),
    assumptions=(
        "Conditional independence (unconfoundedness): once you condition on the "
        "covariates supplied, treatment assignment is independent of potential outcomes. "
        "This is untestable -- it fails if anything unobserved drives both selection "
        "and the outcome.",
        "Common support / overlap: every treated unit has comparable untreated units, "
        "so propensity scores stay away from 0 and 1.",
        "Covariates are pre-treatment: nothing conditioned on is itself affected by "
        "treatment, which would block a causal path rather than close a backdoor.",
        "SUTVA: one unit's treatment does not affect another's outcome.",
    ),
    plain_english=(
        "Model each unit's chance of being treated from its observed characteristics, "
        "then pair every treated unit with untreated units that looked equally likely "
        "to be treated. Comparing outcomes within those pairs strips out the differences "
        "you can see -- but nothing you cannot."
    ),
)


def _design_matrix(frame: pd.DataFrame, covs: list[str]) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
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
        raise IdentificationError("No usable covariates were supplied for the propensity model.")
    return np.column_stack(blocks), names


def _fit_logit(X: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Newton-Raphson logit with ridge damping. Returns the linear index for each row."""
    Xc = np.column_stack([np.ones(len(d)), X])
    # Standardise so the optimiser is well conditioned; undo it implicitly via the index.
    mu, sd = Xc.mean(axis=0), Xc.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = Xc.copy()
    Xs[:, 1:] = (Xc[:, 1:] - mu[1:]) / sd[1:]

    beta = np.zeros(Xs.shape[1])
    for _ in range(60):
        eta = np.clip(Xs @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-8, None)
        grad = Xs.T @ (d - p)
        H = (Xs * W[:, None]).T @ Xs + 1e-6 * np.eye(Xs.shape[1])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return np.clip(Xs @ beta, -35, 35)


def _std_diff(x: np.ndarray, d: np.ndarray, w: np.ndarray | None = None) -> float:
    """Standardised mean difference, pooled-SD denominator (Rosenbaum-Rubin)."""
    t, c = d == 1, d == 0
    if t.sum() == 0 or c.sum() == 0:
        return float("nan")
    if w is None:
        mt, mc = x[t].mean(), x[c].mean()
    else:
        wt, wc = w[t], w[c]
        mt = np.average(x[t], weights=wt) if wt.sum() > 0 else np.nan
        mc = np.average(x[c], weights=wc) if wc.sum() > 0 else np.nan
    denom = np.sqrt((np.var(x[t]) + np.var(x[c])) / 2)
    return float((mt - mc) / denom) if denom > 0 else 0.0


def _within_group_sigma2(index: np.ndarray, y: np.ndarray, d: np.ndarray, J: int = 2) -> np.ndarray:
    """Abadie-Imbens conditional variance: match each unit within its own arm."""
    sigma2 = np.zeros(len(y))
    for arm in (0.0, 1.0):
        idx = np.flatnonzero(d == arm)
        if len(idx) < 2:
            continue
        xi = index[idx]
        order = np.argsort(xi)
        xs, ys = xi[order], y[idx][order]
        m = len(xs)
        j = min(J, m - 1)
        for pos in range(m):
            lo, hi = max(0, pos - j), min(m, pos + j + 1)
            window = [k for k in range(lo, hi) if k != pos]
            dists = sorted(window, key=lambda k: abs(xs[k] - xs[pos]))[:j]
            if not dists:
                continue
            neighbour_mean = float(np.mean([ys[k] for k in dists]))
            sigma2[idx[order[pos]]] = (j / (j + 1)) * (ys[pos] - neighbour_mean) ** 2
    return sigma2


def estimate(df: pd.DataFrame, roles: dict[str, str | None], options: dict) -> MethodOutput:
    y_col = require(roles, "outcome", "Propensity score matching")
    d_col = require(roles, "treatment", "Propensity score matching")
    covs_raw = roles.get("covariates") or []
    covs = [c for c in covs_raw if c and c in df.columns and c not in (y_col, d_col)]
    if len(covs) < 1:
        raise IdentificationError(
            "Propensity score matching needs at least one pre-treatment covariate to match on."
        )

    frame, dropped = clean_frame(df, [y_col, d_col, *covs])
    if len(frame) < 50:
        raise IdentificationError("Fewer than 50 complete rows remain after dropping missing values.")

    y = numeric(frame, y_col)
    d = as_binary(frame, d_col)
    n_t, n_c = int(d.sum()), int((1 - d).sum())
    if n_t < 10 or n_c < 10:
        raise IdentificationError(
            f"Need at least 10 units in each arm (found {n_t} treated, {n_c} control)."
        )

    X, cov_names = _design_matrix(frame, covs)
    index = _fit_logit(X, d)
    pscore = 1 / (1 + np.exp(-index))

    # Complete (or near-complete) separation: the covariates predict treatment perfectly,
    # so there is no region where treated and untreated units coexist. This is almost
    # always a sign that a covariate encodes the treatment rule rather than a confounder.
    separated = float(np.mean((d == 1) == (pscore >= 0.5)))
    if separated >= 0.999:
        culprits = ", ".join(f"'{c}'" for c in covs[:4])
        raise IdentificationError(
            f"The covariates predict '{d_col}' perfectly ({separated:.1%} of rows classified "
            f"correctly), so no treated unit has a comparable untreated one. Usually this means "
            f"one of the covariates ({culprits}) encodes the assignment rule itself rather than "
            "a pre-treatment characteristic. Drop it, or use a design that exploits the "
            "assignment rule directly."
        )

    warnings: list[str] = []
    if dropped:
        warnings.append(f"Dropped {dropped} rows with missing values in the modelled columns.")

    M = int(options.get("n_neighbors", 1))
    caliper_sd = float(options.get("caliper", 0.2))
    caliper = caliper_sd * float(np.std(index))

    treated_idx = np.flatnonzero(d == 1)
    control_idx = np.flatnonzero(d == 0)
    c_index = index[control_idx]
    c_order = np.argsort(c_index)
    c_sorted_idx = control_idx[c_order]
    c_sorted_val = c_index[c_order]

    matches: dict[int, list[int]] = {}
    unmatched: list[int] = []
    for i in treated_idx:
        pos = np.searchsorted(c_sorted_val, index[i])
        # Expand outward from the insertion point until M candidates are collected.
        lo, hi = pos - 1, pos
        picked: list[int] = []
        while len(picked) < M and (lo >= 0 or hi < len(c_sorted_val)):
            d_lo = abs(index[i] - c_sorted_val[lo]) if lo >= 0 else np.inf
            d_hi = abs(c_sorted_val[hi] - index[i]) if hi < len(c_sorted_val) else np.inf
            if d_lo <= d_hi:
                if d_lo > caliper:
                    break
                picked.append(int(c_sorted_idx[lo]))
                lo -= 1
            else:
                if d_hi > caliper:
                    break
                picked.append(int(c_sorted_idx[hi]))
                hi += 1
        if picked:
            matches[int(i)] = picked
        else:
            unmatched.append(int(i))

    if not matches:
        raise IdentificationError(
            "No treated unit found a control within the caliper. The two groups do not overlap "
            "on the propensity score, so no comparison is possible."
        )

    n_matched = len(matches)
    if unmatched:
        pct = len(unmatched) / n_t
        msg = (
            f"{len(unmatched)} of {n_t} treated units ({pct:.1%}) had no control inside the "
            f"caliper and were dropped. The estimate applies to the {n_matched} matched treated "
            "units, not the full treated population."
        )
        warnings.append(msg)

    # --- ATT and Abadie-Imbens variance --------------------------------------
    diffs = np.array([y[i] - np.mean(y[js]) for i, js in matches.items()])
    tau = float(np.mean(diffs))

    # K_i: how many times control i is used as a match (weighted by 1/M).
    K = np.zeros(len(y))
    for js in matches.values():
        for j in js:
            K[j] += 1.0 / len(js)

    sigma2 = _within_group_sigma2(index, y, d)
    weight = np.zeros(len(y))
    for i in matches:
        weight[i] = 1.0
    weight[control_idx] = -K[control_idx]
    var_cond = float(np.sum((weight**2) * sigma2)) / (n_matched**2)
    # Treatment-effect heterogeneity across matched treated units.
    var_het = float(np.var(diffs, ddof=1)) / n_matched if n_matched > 1 else 0.0
    se = float(np.sqrt(max(var_cond + var_het, 0.0)))
    dof = max(n_matched - 1, 1)
    crit = float(stats.t.ppf(0.975, dof))
    p = float(2 * stats.t.sf(abs(tau / se), dof)) if se > 0 else float("nan")

    diagnostics: list[Diagnostic] = []

    # --- covariate balance, before and after matching ------------------------
    match_w = np.zeros(len(y))
    match_w[list(matches.keys())] = 1.0
    match_w[control_idx] = K[control_idx]

    balance_rows = []
    worst_after = 0.0
    for j, name in enumerate(cov_names):
        col = X[:, j]
        before = _std_diff(col, d)
        after = _std_diff(col, d, w=match_w)
        if np.isfinite(after):
            worst_after = max(worst_after, abs(after))
        balance_rows.append(
            {"label": name, "before": fnum(before), "after": fnum(after)}
        )
    ps_before = _std_diff(index, d)
    ps_after = _std_diff(index, d, w=match_w)
    balance_rows.append(
        {"label": "propensity index", "before": fnum(ps_before), "after": fnum(ps_after)}
    )

    if worst_after < 0.10:
        bal_verdict, bal_msg = "pass", (
            f"All covariates are balanced after matching (largest standardised difference "
            f"{worst_after:.3f}, well under the 0.10 rule of thumb)."
        )
    elif worst_after < 0.25:
        bal_verdict, bal_msg = "warn", (
            f"Balance is imperfect: the largest standardised difference after matching is "
            f"{worst_after:.3f}, above the 0.10 target. Residual differences on observables remain."
        )
    else:
        bal_verdict, bal_msg = "fail", (
            f"Matching failed to balance the covariates (largest standardised difference "
            f"{worst_after:.3f}). The matched groups are still visibly different."
        )
    diagnostics.append(
        Diagnostic(
            id="covariate_balance",
            title="Covariate balance (love plot)",
            kind="balance",
            verdict=bal_verdict,  # type: ignore[arg-type]
            summary=bal_msg,
            detail=(
                "Standardised mean differences between treated and control units, before and "
                "after matching. This is the only assumption matching can actually verify: it "
                "shows the observables now line up. It says nothing about unobservables."
            ),
            data={"rows": balance_rows, "threshold": 0.1},
        )
    )

    # --- overlap / common support -------------------------------------------
    edges = np.linspace(0, 1, 31)
    hist = []
    for a, b in zip(edges[:-1], edges[1:]):
        hist.append(
            {
                "x": fnum((a + b) / 2),
                "treated": int(((pscore >= a) & (pscore < b) & (d == 1)).sum()),
                "control": int(((pscore >= a) & (pscore < b) & (d == 0)).sum()),
            }
        )
    extreme = float(((pscore > 0.95) | (pscore < 0.05)).mean())
    t_range = (float(pscore[d == 1].min()), float(pscore[d == 1].max()))
    c_range = (float(pscore[d == 0].min()), float(pscore[d == 0].max()))
    overlap_ok = extreme < 0.10 and len(unmatched) / n_t < 0.10
    diagnostics.append(
        Diagnostic(
            id="overlap",
            title="Common support (propensity score overlap)",
            kind="overlap",
            verdict="pass" if overlap_ok else "warn",  # type: ignore[arg-type]
            summary=(
                f"Treated scores span {t_range[0]:.2f}-{t_range[1]:.2f} and controls span "
                f"{c_range[0]:.2f}-{c_range[1]:.2f}; {extreme:.1%} of units sit in the extreme "
                "tails."
                + ("" if overlap_ok else " Thin overlap means some treated units have no real "
                   "comparison group, and the estimate leans on extrapolation.")
            ),
            detail=(
                "Where the two distributions overlap, matching is comparing like with like. "
                "Where they do not, there is simply no counterfactual in the data."
            ),
            data={"histogram": hist, "x_label": "estimated propensity score"},
        )
    )

    # --- naive vs matched ----------------------------------------------------
    naive = float(y[d == 1].mean() - y[d == 0].mean())
    Xr = np.column_stack([np.ones(len(y)), d, X])
    beta_r, _ = ols(Xr, y)
    reg_adj = float(beta_r[1])
    diagnostics.append(
        Diagnostic(
            id="estimator_comparison",
            title="How much did adjustment change the answer?",
            kind="table",
            verdict="info",
            summary=(
                f"Raw difference in means is {naive:.4g}; OLS with the same covariates gives "
                f"{reg_adj:.4g}; matching gives {tau:.4g}."
            ),
            detail=(
                "Large gaps between these tell you selection on observables was severe. Close "
                "agreement between matching and regression suggests the result is not an "
                "artifact of one particular adjustment strategy."
            ),
            data={
                "rows": [
                    {"label": "Raw difference in means", "value": fnum(naive)},
                    {"label": "OLS with covariates", "value": fnum(reg_adj)},
                    {"label": "Matched ATT", "value": fnum(tau)},
                ]
            },
        )
    )

    # --- caliper sensitivity -------------------------------------------------
    sens = []
    for mult in (0.5, 1.0, 2.0):
        cal = caliper_sd * mult * float(np.std(index))
        picked_diffs = []
        for i in treated_idx:
            pos = np.searchsorted(c_sorted_val, index[i])
            best, best_d = None, np.inf
            for k in (pos - 1, pos):
                if 0 <= k < len(c_sorted_val):
                    dd = abs(index[i] - c_sorted_val[k])
                    if dd < best_d:
                        best, best_d = int(c_sorted_idx[k]), dd
            if best is not None and best_d <= cal:
                picked_diffs.append(y[i] - y[best])
        if picked_diffs:
            sens.append(
                {
                    "label": f"caliper = {caliper_sd*mult:.2f} SD",
                    "value": fnum(float(np.mean(picked_diffs))),
                    "matched": len(picked_diffs),
                }
            )
    if sens:
        vals = [s["value"] for s in sens if s["value"] is not None]
        spread = (max(vals) - min(vals)) / abs(tau) if tau else float("inf")
        diagnostics.append(
            Diagnostic(
                id="caliper_sensitivity",
                title="Caliper sensitivity",
                kind="table",
                verdict="pass" if spread < 0.25 else "warn",  # type: ignore[arg-type]
                summary=(
                    f"Tightening and loosening the caliper moves the estimate between "
                    f"{min(vals):.4g} and {max(vals):.4g}."
                ),
                detail=(
                    "A tighter caliper means closer matches but fewer of them. If the estimate "
                    "swings sharply, the result depends on match quality rather than the data."
                ),
                data={"rows": sens, "columns": ["label", "value", "matched"]},
            )
        )

    diagnostics.append(
        Diagnostic(
            id="unconfoundedness",
            title="Unconfoundedness is assumed, not tested",
            kind="text",
            verdict="warn",
            summary=(
                "Every check above concerns *observed* covariates. Matching cannot detect an "
                "unobserved variable that drives both selection into treatment and the outcome, "
                "and no diagnostic in this report rules one out."
            ),
            detail=(
                f"Matched on: {', '.join(covs)}. Ask directly whether anything omitted — "
                "motivation, health, private information, an unrecorded eligibility rule — "
                "would push both treatment and the outcome in the same direction. If so, this "
                "estimate is biased in that direction, and a design with an external source of "
                "variation (an experiment, a discontinuity, an instrument) would be stronger."
            ),
            data={},
        )
    )

    return MethodOutput(
        estimate=Estimate(
            point=tau,
            se=fnum(se),
            ci_low=fnum(tau - crit * se),
            ci_high=fnum(tau + crit * se),
            p_value=fnum(p),
            n_obs=int(len(frame)),
            n_treated=n_matched,
            n_control=int((K > 0).sum()),
            units=y_col,
        ),
        diagnostics=diagnostics,
        specification={
            "propensity_model": f"logit({d_col} ~ {' + '.join(covs)})",
            "matching": f"{M}-nearest-neighbour on the linear propensity index, with replacement",
            "caliper": f"{caliper_sd} SD of the propensity index ({caliper:.4g})",
            "se_type": "Abadie-Imbens matching variance (accounts for reuse of controls)",
            "n_treated_matched": n_matched,
            "n_treated_dropped": len(unmatched),
            "n_controls_used": int((K > 0).sum()),
            "covariates": covs,
        },
        warnings=warnings,
    )
