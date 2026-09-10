"""Shared machinery for the causal estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..schemas import Diagnostic, Estimate


class IdentificationError(Exception):
    """Raised when the data cannot support the requested method."""


@dataclass
class MethodSpec:
    """Static description of a method: what it needs and what it assumes."""

    id: str
    label: str
    estimand: str
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    plain_english: str = ""


@dataclass
class MethodOutput:
    estimate: Estimate
    diagnostics: list[Diagnostic] = field(default_factory=list)
    specification: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def require(roles: dict[str, str | None], key: str, method: str) -> str:
    value = roles.get(key)
    if not value:
        raise IdentificationError(f"{method} requires a '{key}' column, but none was assigned.")
    return value


def numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    """Coerce a column to float, raising a clear error if it isn't numeric."""
    series = pd.to_numeric(df[col], errors="coerce")
    if series.isna().all():
        raise IdentificationError(f"Column '{col}' could not be read as numeric.")
    return series.to_numpy(dtype=float)


def as_binary(df: pd.DataFrame, col: str) -> np.ndarray:
    """Map a two-valued column to {0, 1}, treating the larger/'true' value as 1."""
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy().astype(float)
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().all():
        uniq = np.unique(numeric_series.to_numpy())
        if len(uniq) > 2:
            raise IdentificationError(
                f"Column '{col}' takes {len(uniq)} distinct values; a binary treatment is required."
            )
        if len(uniq) == 1:
            raise IdentificationError(f"Column '{col}' has no variation -- every row is {uniq[0]}.")
        return (numeric_series.to_numpy() == uniq.max()).astype(float)
    uniq = pd.unique(series.dropna())
    if len(uniq) != 2:
        raise IdentificationError(
            f"Column '{col}' takes {len(uniq)} distinct values; a binary treatment is required."
        )
    positive = sorted(str(u) for u in uniq)[-1]
    return (series.astype(str) == positive).to_numpy().astype(float)


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares via pinv (tolerates collinear dummy blocks). Returns beta, residuals."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def robust_vcov(X: np.ndarray, resid: np.ndarray, cluster: np.ndarray | None = None) -> np.ndarray:
    """HC1 heteroskedasticity-robust, or CR1 cluster-robust when `cluster` is given."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    if cluster is None:
        scale = n / max(n - k, 1)
        meat = (X * (resid**2)[:, None]).T @ X
        return scale * XtX_inv @ meat @ XtX_inv

    groups = pd.unique(cluster)
    g = len(groups)
    meat = np.zeros((k, k))
    index = pd.Series(np.arange(n)).groupby(pd.Series(cluster), observed=True).indices
    for idx in index.values():
        Xg = X[idx]
        ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return scale * XtX_inv @ meat @ XtX_inv


def coef_inference(
    beta: np.ndarray, vcov: np.ndarray, idx: int, dof: int
) -> tuple[float, float, float, float, float]:
    """Return (point, se, ci_low, ci_high, p_value) for one coefficient."""
    point = float(beta[idx])
    var = float(vcov[idx, idx])
    se = float(np.sqrt(var)) if var > 0 else float("nan")
    if not np.isfinite(se) or se == 0:
        return point, float("nan"), float("nan"), float("nan"), float("nan")
    dof = max(dof, 1)
    crit = float(stats.t.ppf(0.975, dof))
    t_stat = point / se
    p = float(2 * stats.t.sf(abs(t_stat), dof))
    return point, se, point - crit * se, point + crit * se, p


def dummies(values: np.ndarray, drop_first: bool = True) -> tuple[np.ndarray, list[Any]]:
    """One-hot encode, dropping the first level to avoid collinearity with an intercept."""
    levels = list(pd.unique(pd.Series(values).dropna()))
    levels.sort(key=lambda v: str(v))
    use = levels[1:] if drop_first else levels
    if not use:
        return np.zeros((len(values), 0)), []
    mat = np.column_stack([(values == lvl).astype(float) for lvl in use])
    return mat, use


def clean_frame(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop rows with missing values in the columns the estimator actually uses."""
    used = [c for c in dict.fromkeys(cols) if c]
    before = len(df)
    out = df.dropna(subset=used).reset_index(drop=True)
    return out, before - len(out)


def fnum(x: Any) -> float | None:
    """JSON-safe float: NaN/inf become null rather than breaking serialisation."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
