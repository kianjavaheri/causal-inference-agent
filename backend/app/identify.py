"""Identifiability engine.

Decides, from the data's structure alone, which causal designs are even *possible*
and what column plays each role. The LLM chooses among what this module says is
feasible -- it never gets to invent feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .methods import SPECS
from .schemas import ColumnProfile, DataProfile, MethodAssessment


@dataclass
class Derivation:
    """A column the agent constructs rather than finds, e.g. treated x post."""

    name: str
    kind: str
    inputs: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": self.inputs,
            "explanation": self.explanation,
        }


@dataclass
class Assessment:
    method: str
    feasible: bool
    score: float
    rationale: str
    blocking: list[str] = field(default_factory=list)
    roles: dict[str, Any] = field(default_factory=dict)
    derivations: list[Derivation] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def _hint_score_unit(name: str) -> bool:
    """True when a column name reads as an entity identifier rather than a measurement."""
    from .profiling import UNIT_HINTS, _hint_score

    return _hint_score(name, UNIT_HINTS) >= 0.8


def _binary_cols(cols: list[ColumnProfile]) -> list[str]:
    return [c.name for c in cols if c.semantic_type == "binary"]


def _to01(series: pd.Series) -> np.ndarray | None:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy().astype(float)
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().all():
        vals = np.unique(num.to_numpy())
        if len(vals) == 2:
            return (num.to_numpy() == vals.max()).astype(float)
        return None
    vals = pd.unique(series.dropna())
    if len(vals) == 2:
        top = sorted((str(v) for v in vals))[-1]
        return (series.astype(str) == top).to_numpy().astype(float)
    return None


def _variation(df: pd.DataFrame, col: str, unit: str | None, time: str | None) -> str:
    """Where a column's variation lives: across units, across time, both, or neither."""
    vals = _to01(df[col])
    if vals is None:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    series = pd.Series(vals)
    if series.nunique(dropna=True) <= 1:
        return "constant"
    varies_within_unit = (
        bool((series.groupby(df[unit], observed=True).nunique() > 1).any()) if unit else False
    )
    varies_within_time = (
        bool((series.groupby(df[time], observed=True).nunique() > 1).any()) if time else False
    )
    if varies_within_unit and varies_within_time:
        return "both"
    if varies_within_unit:
        return "time_only"      # constant across units within a period -> a period flag
    if varies_within_time:
        return "unit_only"      # constant over time within a unit -> a group flag
    return "constant"


def _threshold_above(x: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Best cutoff c such that 1[x >= c] matches d, and the resulting accuracy."""
    order = np.argsort(x)
    xs, ds = x[order], d[order]
    n = len(xs)
    # Accuracy of predicting 1 for every position at or after i.
    right_ones = np.concatenate([np.cumsum(ds[::-1])[::-1], [0.0]])
    left_zeros = np.concatenate([[0.0], np.cumsum(1 - ds)])
    correct = right_ones + left_zeros
    best_i = int(np.argmax(correct))
    acc = float(correct[best_i] / n)
    if best_i == 0:
        cutoff = float(xs[0])
    elif best_i >= n:
        cutoff = float(xs[-1]) + 1e-9
    else:
        cutoff = float((xs[best_i - 1] + xs[best_i]) / 2)
    return cutoff, acc


def best_threshold(x: np.ndarray, d: np.ndarray) -> tuple[float, float, int]:
    """Best cutoff, its accuracy, and which side is treated (+1 above, -1 below).

    Plenty of real rules run the other way -- benefits for income *below* a line,
    remediation for scores *below* a pass mark -- so both directions are tested.
    """
    c_up, acc_up = _threshold_above(x, d)
    c_dn, acc_dn = _threshold_above(x, 1.0 - d)
    if acc_dn > acc_up:
        return c_dn, acc_dn, -1
    return c_up, acc_up, 1


def _covariate_pool(
    profile: DataProfile, exclude: set[str], max_n: int = 12
) -> list[str]:
    """Plausible pre-treatment controls: measured, low-missingness, not an id."""
    out = []
    for c in profile.columns:
        if c.name in exclude or c.semantic_type in ("identifier", "text", "datetime"):
            continue
        if c.missing_pct > 40 or c.n_unique <= 1:
            continue
        if c.semantic_type == "categorical" and c.n_unique > 15:
            continue
        out.append(c.name)
    return out[:max_n]


# ---------------------------------------------------------------------------
# Per-method assessments
# ---------------------------------------------------------------------------


def _pick(hints: dict, role: str, ranked: list[str], exclude: set = frozenset()) -> str | None:
    """Honour a hinted column when it is a legitimate candidate; else take the top rank."""
    hinted = hints.get(role)
    if hinted and hinted not in exclude and hinted in ranked:
        return hinted
    for c in ranked:
        if c not in exclude:
            return c
    return None


def assess_did(df: pd.DataFrame, profile: DataProfile, hints: dict) -> Assessment:
    cand, struct = profile.candidates, profile.structure
    blocking: list[str] = []
    time = _pick(hints, "time", cand.time)
    unit = _pick(hints, "unit", cand.unit, exclude={time} if time else set())
    outcome = _pick(hints, "outcome", cand.outcome, exclude={time, unit} - {None})

    if not time or struct.n_periods < 2:
        blocking.append("No time dimension with at least two periods.")
    if not unit:
        blocking.append("No column identifies repeated units over time.")
    if not outcome:
        blocking.append("No continuous outcome column found.")

    treatment, derivations, evidence = None, [], {}
    if unit and time:
        binaries = _binary_cols(profile.columns)
        variation = {c: _variation(df, c, unit, time) for c in binaries}
        evidence["variation"] = variation

        # 1. A column that already varies both across units and over time is the treatment.
        both = [c for c, v in variation.items() if v == "both" and c not in (unit, time)]
        if both:
            # Rank by the question/profiler's own treatment ordering first, then by name.
            # "post" is a legitimate DiD treatment here precisely *because* this column
            # varies across units too -- a pure period flag never reaches this list.
            prefer = [hints.get("treatment")] + list(cand.treatment)
            order = {c: i for i, c in enumerate(x for x in prefer if x)}
            treatment = sorted(
                both,
                key=lambda c: (
                    -(0.8 if any(h in c.lower() for h in
                                 ("treat", "policy", "program", "active", "adopt",
                                  "exposed", "post", "law", "enacted")) else 0.0),
                    order.get(c, len(order)),
                ),
            )[0]
        else:
            # 2. Otherwise construct treated x post from a group flag and a period flag.
            groups = [c for c, v in variation.items() if v == "unit_only"]
            posts = [c for c, v in variation.items() if v == "time_only"]
            if groups and posts:
                g, p = groups[0], posts[0]
                treatment = f"__treated_x_post__{g}__{p}"
                derivations.append(
                    Derivation(
                        name=treatment,
                        kind="interaction",
                        inputs=[g, p],
                        explanation=(
                            f"No single column marks who is treated *when*: '{g}' identifies the "
                            f"treated group and '{p}' marks the post period. The DiD treatment "
                            f"indicator is their product, so it equals 1 only for treated units "
                            f"after the intervention."
                        ),
                    )
                )
        if treatment is None:
            blocking.append(
                "No treatment indicator that turns on for some units at some point in time "
                "(and no group flag + post flag pair to build one from)."
            )
        elif not derivations:
            d01 = _to01(df[treatment])
            if d01 is not None:
                ever = pd.Series(d01).groupby(df[unit], observed=True).max()
                n_ever = int((ever > 0).sum())
                evidence["ever_treated_units"] = n_ever
                evidence["never_treated_units"] = int(len(ever) - n_ever)
                if n_ever == len(ever):
                    blocking.append(
                        "Every unit is eventually treated, so there is no never-treated "
                        "comparison group."
                    )

    if blocking:
        return Assessment(
            method="did",
            feasible=False,
            score=0.0,
            rationale="; ".join(blocking),
            blocking=blocking,
            evidence=evidence,
        )

    score = 0.65
    if struct.is_panel:
        score += 0.2
    if struct.has_never_treated_units:
        score += 0.1
    if struct.n_periods >= 4:
        score += 0.05
    if derivations:
        score -= 0.05

    # With a single treated unit there is nothing to average over: cluster-robust
    # inference is meaningless and synthetic control is the purpose-built design.
    n_treated_units = evidence.get("ever_treated_units")
    if n_treated_units is None and unit and treatment:
        probe = apply_derivations(df, [d.to_dict() for d in derivations]) if derivations else df
        d01 = _to01(probe[treatment]) if treatment in probe.columns else None
        if d01 is not None:
            ever = pd.Series(d01).groupby(probe[unit], observed=True).max()
            n_treated_units = int((ever > 0).sum())
            evidence["ever_treated_units"] = n_treated_units
            evidence["never_treated_units"] = int(len(ever) - n_treated_units)
    single_treated = n_treated_units == 1
    if single_treated:
        score -= 0.45

    rationale = (
        f"The data is {'a balanced panel' if struct.is_panel else 'repeated observations'} of "
        f"{struct.n_units} units over {struct.n_periods} periods, with treatment switching on "
        f"partway through and untreated units available throughout. That is exactly the "
        f"structure difference-in-differences needs: the untreated units supply the "
        f"counterfactual trend."
    )
    if single_treated:
        rationale += (
            " Only one unit is ever treated, though, so DiD would weight every control unit "
            "equally and its clustered standard errors would rest on a single treated cluster. "
            "Synthetic control is built for exactly this case."
        )
    exclude = {outcome, treatment, time, unit} | {
        i for d in derivations for i in d.inputs
    }
    return Assessment(
        method="did",
        feasible=True,
        score=min(score, 1.0),
        rationale=rationale,
        roles={
            "outcome": outcome,
            "treatment": treatment,
            "time": time,
            "unit": unit,
            "covariates": [],
        },
        derivations=derivations,
        evidence=evidence,
    )


def assess_rdd(df: pd.DataFrame, profile: DataProfile, hints: dict) -> Assessment:
    cand = profile.candidates
    blocking: list[str] = []

    # Find the assignment rule FIRST; the outcome is whatever is left over. Doing it the
    # other way round lets a running variable be mistaken for the outcome.
    # (accuracy, treatment, running variable, cutoff, treated side)
    best: tuple[float, str, str, float, int] | None = None
    evidence: dict[str, Any] = {"candidates": []}
    time_cols = set(cand.time[:2])
    binaries = [
        c for c in _binary_cols(profile.columns)
        if not (cand.unit and cand.time
                and _variation(df, c, cand.unit[0], cand.time[0]) == "time_only")
    ]
    # Unit identifiers must be excluded: with enough binary columns, a threshold search
    # will happily "discover" a cutoff in a state id, which is meaningless.
    id_cols = set(cand.unit[:3]) | {
        c.name for c in profile.columns if _hint_score_unit(c.name)
    }
    runners = [
        c.name
        for c in profile.columns
        if c.semantic_type == "continuous"
        and c.n_unique >= 20
        and c.name not in time_cols          # a calendar axis is not a running variable
        and c.name not in id_cols
        and c.semantic_type != "datetime"
    ]
    hinted_running = hints.get("running_variable")
    if hinted_running in runners:
        runners = [hinted_running] + [r for r in runners if r != hinted_running]
    for t in binaries:
        d = _to01(df[t])
        if d is None or d.sum() in (0, len(d)):
            continue
        for r in runners:
            x = pd.to_numeric(df[r], errors="coerce")
            mask = x.notna() & pd.Series(d).notna()
            if mask.sum() < 50:
                continue
            cutoff, acc, side = best_threshold(
                x[mask].to_numpy(dtype=float), d[mask.to_numpy()]
            )
            evidence["candidates"].append(
                {"treatment": t, "running_variable": r, "cutoff": round(cutoff, 6),
                 "assignment_accuracy": round(acc, 4),
                 "treated_side": "below" if side < 0 else "at or above"}
            )
            if best is None or acc > best[0]:
                best = (acc, t, r, cutoff, side)

    evidence["candidates"] = sorted(
        evidence["candidates"], key=lambda c: -c["assignment_accuracy"]
    )[:5]

    if best is None:
        blocking.append(
            "No pairing of a binary treatment with a continuous running variable was found."
        )
    elif best[0] < 0.90:
        blocking.append(
            f"The best cutoff rule ({best[1]} at {best[3]:.4g} of {best[2]}) explains only "
            f"{best[0]:.1%} of treatment assignment. Treatment is not determined by a threshold."
        )

    if blocking:
        return Assessment(
            method="rdd", feasible=False, score=0.0, rationale="; ".join(blocking),
            blocking=blocking, evidence=evidence,
        )

    acc, treat, running, cutoff, side = best  # type: ignore[misc]
    outcome = _pick(hints, "outcome", cand.outcome, exclude={running, treat} | time_cols)
    if not outcome:
        return Assessment(
            method="rdd", feasible=False, score=0.0,
            rationale="No outcome column remains once the running variable is set aside.",
            blocking=["No continuous outcome column distinct from the running variable."],
            evidence=evidence,
        )
    sharp = acc >= 0.995
    score = 0.9 if sharp else 0.6
    where = "falls below" if side < 0 else "reaches"
    rationale = (
        f"Treatment '{treat}' is {'exactly' if sharp else 'almost entirely'} determined by "
        f"whether '{running}' {where} {cutoff:.4g} ({acc:.1%} of rows follow the rule). That is a "
        f"{'sharp' if sharp else 'fuzzy'} discontinuity: units just either side of the threshold "
        "are comparable by luck, which identifies the effect locally."
    )
    return Assessment(
        method="rdd",
        feasible=True,
        score=score,
        rationale=rationale,
        roles={
            "outcome": outcome,
            "treatment": treat,
            "running_variable": running,
            "covariates": _covariate_pool(profile, {outcome, treat, running}, 8),
        },
        evidence={**evidence, "cutoff": cutoff, "assignment_accuracy": acc,
                  "sharp": sharp, "treated_side": side},
    )


def assess_psm(df: pd.DataFrame, profile: DataProfile, hints: dict) -> Assessment:
    cand = profile.candidates
    blocking: list[str] = []
    time = _pick(hints, "time", cand.time)
    unit = _pick(hints, "unit", cand.unit, exclude={time} if time else set())
    outcome = _pick(hints, "outcome", cand.outcome, exclude={time, unit} - {None})
    if not outcome:
        blocking.append("No continuous outcome column found.")

    treatment: str | None = None
    binaries = [c for c in _binary_cols(profile.columns) if c != outcome]
    ranked = [c for c in cand.treatment if c in binaries] or binaries
    if hints.get("treatment") in binaries:
        ranked = [hints["treatment"]] + [c for c in ranked if c != hints["treatment"]]
    # A pure period flag is not a treatment.
    for c in ranked:
        if unit and time and _variation(df, c, unit, time) == "time_only":
            continue
        treatment = c
        break
    if treatment is None:
        blocking.append("No binary treatment indicator found.")

    covs = _covariate_pool(profile, {outcome, treatment, unit, time} - {None})  # type: ignore[operator]
    if len(covs) < 1:
        blocking.append("No pre-treatment covariates available to match on.")

    if blocking:
        return Assessment(
            method="psm", feasible=False, score=0.0, rationale="; ".join(blocking),
            blocking=blocking,
        )

    d = _to01(df[treatment])
    n_t = int(d.sum()) if d is not None else 0
    n_c = int(len(d) - n_t) if d is not None else 0
    if min(n_t, n_c) < 10:
        return Assessment(
            method="psm", feasible=False, score=0.0,
            rationale=f"Only {n_t} treated and {n_c} control units -- too few to match.",
            blocking=[f"Only {n_t} treated and {n_c} control units."],
        )

    # PSM is always *runnable*; it scores low because its assumption is the weakest.
    score = 0.45
    if len(covs) >= 4:
        score += 0.1
    if min(n_t, n_c) >= 100:
        score += 0.05
    if profile.structure.is_panel:
        # With a panel in hand, cross-sectional matching throws away the time variation
        # that lets a design difference out fixed unit characteristics.
        score -= 0.2
    rationale = (
        f"There is a binary treatment ('{treatment}') with {n_t} treated and {n_c} untreated "
        f"units, and {len(covs)} covariates to adjust on. Matching is always computable here, but "
        "it identifies a causal effect only if those covariates capture everything that drives "
        "both selection and the outcome -- a strictly stronger assumption than the designs above."
    )
    return Assessment(
        method="psm",
        feasible=True,
        score=score,
        rationale=rationale,
        roles={"outcome": outcome, "treatment": treatment, "covariates": covs},
        evidence={"n_treated": n_t, "n_control": n_c, "n_covariates": len(covs)},
    )


INSTRUMENT_NAME_HINTS = (
    "instrument", "assign", "lottery", "random", "encourag", "offer", "invit",
    "eligib", "draw", "_z", "iv_", "email", "voucher", "nudge", "reminder",
    "quarter_of_birth", "rainfall", "distance_to",
)


def _looks_like_instrument(name: str) -> bool:
    return any(h in name.lower() for h in INSTRUMENT_NAME_HINTS)


def assess_iv(df: pd.DataFrame, profile: DataProfile, hints: dict) -> Assessment:
    """Instrument and treatment are chosen together: an instrument only makes sense
    relative to the endogenous variable it is meant to shift."""
    from .methods.base import ols, robust_vcov

    cand = profile.candidates
    outcome = _pick(hints, "outcome", cand.outcome)
    if not outcome:
        return Assessment(
            method="iv", feasible=False, score=0.0,
            rationale="No continuous outcome column found.",
            blocking=["No continuous outcome column found."],
        )

    structural = set(cand.unit[:1]) | set(cand.time[:1]) | {outcome}
    hinted_z = hints.get("instrument")
    hinted_d = hints.get("treatment")

    # Candidate instruments must be *nominated* -- by the user's question or by a name that
    # denotes an assignment mechanism. Exogeneity is untestable, so strong correlation with
    # treatment is not evidence of an instrument; it usually signals a confounder or a
    # mechanical function of treatment.
    z_pool = [
        c.name
        for c in profile.columns
        if c.name not in structural
        and c.semantic_type in ("binary", "continuous", "integer")
        and (c.name == hinted_z or _looks_like_instrument(c.name))
        and c.name != hinted_d
    ]
    d_pool = [
        c.name
        for c in profile.columns
        if c.name not in structural and c.semantic_type in ("binary", "integer", "continuous")
    ]
    # Prefer whatever the profiler and the question flagged as the treatment.
    ranked_d = ([hinted_d] if hinted_d in d_pool else []) + \
        [c for c in cand.treatment if c in d_pool] + \
        [c for c in d_pool if c not in cand.treatment]

    evidence: dict[str, Any] = {"candidates": []}
    best: tuple[float, str, str] | None = None
    for z_col in z_pool:
        z = pd.to_numeric(df[z_col], errors="coerce")
        for d_col in dict.fromkeys(ranked_d):
            if d_col == z_col or _looks_like_instrument(d_col) and d_col != hinted_d:
                continue
            d = pd.to_numeric(df[d_col], errors="coerce")
            mask = z.notna() & d.notna()
            if mask.sum() < 50 or z[mask].nunique() < 2 or d[mask].nunique() < 2:
                continue
            zv = z[mask].to_numpy(dtype=float)
            dv = d[mask].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(zv)), zv])
            beta, resid = ols(X, dv)
            vc = robust_vcov(X, resid)
            se = float(np.sqrt(max(vc[1, 1], 0)))
            f = float((beta[1] / se) ** 2) if se > 0 else 0.0
            evidence["candidates"].append(
                {"instrument": z_col, "treatment": d_col, "first_stage_f": round(f, 2)}
            )
            bonus = (0.5 if d_col == hinted_d else 0.0) + (0.5 if z_col == hinted_z else 0.0)
            if f >= 10 and (best is None or f * (1 + bonus) > best[0]):
                best = (f * (1 + bonus), z_col, d_col)

    evidence["candidates"] = sorted(
        evidence["candidates"], key=lambda c: -c["first_stage_f"]
    )[:6]

    if best is None:
        if z_pool:
            blocking = [
                f"Candidate instrument(s) {', '.join(z_pool[:3])} do not move any treatment "
                "variable strongly enough (first-stage F below 10)."
            ]
        else:
            blocking = [
                "No column is identifiable as an instrument. An instrument cannot be discovered "
                "from correlations -- a variable that predicts treatment is just as likely a "
                "confounder. It has to be nominated on substantive grounds as something that "
                "shifts treatment and reaches the outcome no other way."
            ]
        return Assessment(
            method="iv", feasible=False, score=0.0, rationale=blocking[0],
            blocking=blocking, evidence=evidence,
        )

    _, z_col, d_col = best
    raw_f = next(
        c["first_stage_f"] for c in evidence["candidates"]
        if c["instrument"] == z_col and c["treatment"] == d_col
    )
    score = 0.72 + (0.08 if z_col == hinted_z else 0.0) + (0.1 if raw_f >= 50 else 0.0)
    rationale = (
        f"'{z_col}' reads as an assignment mechanism and strongly predicts '{d_col}' "
        f"(first-stage F = {raw_f:.0f}), so it can supply exogenous variation in a treatment "
        f"that people otherwise select into. IV is valid *if* '{z_col}' reaches '{outcome}' only "
        f"through '{d_col}' -- a substantive claim about the world, not something the data "
        "can settle."
    )
    return Assessment(
        method="iv",
        feasible=True,
        score=min(score, 1.0),
        rationale=rationale,
        roles={
            "outcome": outcome,
            "treatment": d_col,
            "instrument": z_col,
            "covariates": _covariate_pool(profile, {outcome, d_col, z_col} | structural, 8),
        },
        evidence={**evidence, "chosen_first_stage_f": raw_f},
    )


def assess_synthetic_control(df: pd.DataFrame, profile: DataProfile, hints: dict) -> Assessment:
    cand, struct = profile.candidates, profile.structure
    time = _pick(hints, "time", cand.time)
    unit = _pick(hints, "unit", cand.unit, exclude={time} if time else set())
    outcome = _pick(hints, "outcome", cand.outcome, exclude={time, unit} - {None})
    blocking: list[str] = []
    evidence: dict[str, Any] = {}

    if not (unit and time and outcome):
        blocking.append("Synthetic control needs unit, time and outcome columns.")
    elif struct.n_periods < 6:
        blocking.append(f"Only {struct.n_periods} time periods; a long pre-window is required.")
    else:
        binaries = _binary_cols(profile.columns)
        treatment, derivations = None, []
        variation = {c: _variation(df, c, unit, time) for c in binaries}
        both = [c for c, v in variation.items() if v == "both" and c not in (unit, time)]
        n_treated_units = None
        for c in both:
            d = _to01(df[c])
            if d is None:
                continue
            ever = pd.Series(d).groupby(df[unit], observed=True).max()
            n = int((ever > 0).sum())
            if n == 1:
                treatment, n_treated_units = c, n
                break
            if n_treated_units is None:
                n_treated_units = n
        if treatment is None:
            groups = [c for c, v in variation.items() if v == "unit_only"]
            posts = [c for c, v in variation.items() if v == "time_only"]
            for g in groups:
                gd = _to01(df[g])
                if gd is None:
                    continue
                ever = pd.Series(gd).groupby(df[unit], observed=True).max()
                if int((ever > 0).sum()) == 1 and posts:
                    treatment = f"__treated_x_post__{g}__{posts[0]}"
                    derivations = [
                        Derivation(
                            name=treatment,
                            kind="interaction",
                            inputs=[g, posts[0]],
                            explanation=(
                                f"'{g}' marks the single treated unit and '{posts[0]}' marks the "
                                "post period; their product is the treatment indicator."
                            ),
                        )
                    ]
                    n_treated_units = 1
                    break
        evidence["n_treated_units"] = n_treated_units
        evidence["n_donors"] = max(struct.n_units - (n_treated_units or 0), 0)

        if treatment is None:
            blocking.append("No treatment indicator identifying a single treated unit.")
        elif n_treated_units != 1:
            blocking.append(
                f"{n_treated_units} units are treated. Synthetic control is designed for a single "
                "treated unit; with several, difference-in-differences is the natural choice."
            )
        elif struct.n_units < 4:
            blocking.append(f"Only {struct.n_units} units -- the donor pool is too small.")

    if blocking:
        return Assessment(
            method="synthetic_control", feasible=False, score=0.0,
            rationale="; ".join(blocking), blocking=blocking, evidence=evidence,
        )

    score = 0.85
    if struct.n_units >= 15:
        score += 0.05
    if struct.n_periods >= 20:
        score += 0.05
    rationale = (
        f"Exactly one of {struct.n_units} units is treated, observed over {struct.n_periods} "
        "periods with a long pre-treatment window. With a single treated unit, no other unit is a "
        "fair comparison on its own -- but a weighted blend of the "
        f"{evidence.get('n_donors')} donors can reproduce its history and serve as the "
        "counterfactual."
    )
    return Assessment(
        method="synthetic_control",
        feasible=True,
        score=min(score, 1.0),
        rationale=rationale,
        roles={"outcome": outcome, "treatment": treatment, "time": time, "unit": unit},
        derivations=derivations,
        evidence=evidence,
    )


ASSESSORS = {
    "did": assess_did,
    "rdd": assess_rdd,
    "iv": assess_iv,
    "synthetic_control": assess_synthetic_control,
    "psm": assess_psm,
}


def assess_all(
    df: pd.DataFrame, profile: DataProfile, hints: dict | None = None
) -> list[Assessment]:
    hints = {k: v for k, v in (hints or {}).items() if v}
    out: list[Assessment] = []
    for method, fn in ASSESSORS.items():
        try:
            out.append(fn(df, profile, hints))
        except Exception as exc:  # a broken assessor must not sink the whole plan
            out.append(
                Assessment(
                    method=method,
                    feasible=False,
                    score=0.0,
                    rationale=f"Could not assess: {exc}",
                    blocking=[str(exc)],
                )
            )
    out.sort(key=lambda a: (-a.score, a.method))
    return out


def to_schema(a: Assessment) -> MethodAssessment:
    spec = SPECS[a.method]
    return MethodAssessment(
        method=a.method,  # type: ignore[arg-type]
        label=spec.label,
        feasible=a.feasible,
        score=round(a.score, 3),
        rationale=a.rationale,
        blocking_reasons=a.blocking,
        required_roles={k: (v if isinstance(v, str) else None) for k, v in a.roles.items()},
        key_assumptions=list(spec.assumptions),
    )


def apply_derivations(df: pd.DataFrame, derivations: list[dict[str, Any]]) -> pd.DataFrame:
    """Materialise agent-constructed columns (currently: interactions) onto a copy."""
    if not derivations:
        return df
    out = df.copy()
    for d in derivations:
        if d.get("kind") != "interaction":
            continue
        inputs = d.get("inputs") or []
        if not all(c in out.columns for c in inputs):
            continue
        product = np.ones(len(out))
        for c in inputs:
            vals = _to01(out[c])
            if vals is None:
                vals = pd.to_numeric(out[c], errors="coerce").fillna(0).to_numpy(dtype=float)
            product = product * vals
        out[d["name"]] = product
    return out
