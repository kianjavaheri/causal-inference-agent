"""Data profiling: turn a raw CSV into the structural facts the planner reasons over."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .schemas import ColumnProfile, DataProfile, RoleCandidates, StructureFlags

# Name hints. These only *rank* candidates -- structure decides feasibility.
OUTCOME_HINTS = ("outcome", "y", "revenue", "sales", "spend", "conversion", "value",
                 "amount", "result", "gmv", "retention", "churn", "engagement",
                 "clicks", "views", "earnings", "wage", "income", "price", "profit",
                 "quantity", "target", "metric", "employment", "attendance",
                 "enrollment", "consumption", "output", "usage", "gpa", "mortality",
                 "crime", "yield", "productivity", "after", "post_outcome")
# Names that usually mark a control variable rather than the thing being explained.
COVARIATE_HINTS = ("population", "pop", "age", "size", "area", "density", "baseline",
                   "prior", "pre", "lag", "gender", "sex", "race", "married",
                   "education", "tenure", "channel", "device", "segment")
TREATMENT_HINTS = ("treat", "treated", "treatment", "exposed", "intervention", "policy",
                   "program", "enrolled", "adopt", "variant", "group", "arm", "d",
                   "participate", "received", "eligible_treated")
TIME_HINTS = ("date", "time", "period", "month", "year", "week", "day", "quarter",
              "t", "timestamp", "wave", "post", "epoch")
UNIT_HINTS = ("id", "unit", "user", "customer", "store", "state", "region", "county",
              "firm", "school", "market", "city", "country", "entity", "group_id",
              "cohort", "individual", "person", "hospital", "district")
RUNNING_HINTS = ("score", "rating", "index", "margin", "percentile", "rank",
                 "threshold", "cutoff", "running", "vote_share", "test", "exam",
                 "eligibility", "points")
INSTRUMENT_HINTS = ("instrument", "iv", "z", "lottery", "assigned", "random", "draw",
                    "encouragement", "offer", "distance", "rainfall", "quarter_of_birth",
                    "assignment", "invited")

DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    re.compile(r"^\d{2}/\d{2}/\d{4}"),
    re.compile(r"^\d{4}/\d{2}/\d{2}"),
    re.compile(r"^\d{4}-\d{2}$"),
    re.compile(r"^\d{4}Q[1-4]$", re.IGNORECASE),
)


def _hint_score(name: str, hints: tuple[str, ...]) -> float:
    """How strongly a column name suggests a role. Exact match beats substring."""
    lowered = name.lower().strip()
    tokens = set(re.split(r"[^a-z0-9]+", lowered)) - {""}
    if lowered in hints:
        return 1.0
    if tokens & set(hints):
        return 0.8
    for hint in hints:
        if len(hint) > 2 and hint in lowered:
            return 0.5
    return 0.0


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty or not pd.api.types.is_object_dtype(non_null) and not pd.api.types.is_string_dtype(non_null):
        return False
    sample = non_null.astype(str).head(50)
    hits = sum(any(p.match(v) for p in DATE_PATTERNS) for v in sample)
    return hits >= max(3, int(0.8 * len(sample)))


def _semantic_type(name: str, series: pd.Series, n_rows: int) -> str:
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    if _looks_like_datetime(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        uniques = set(np.unique(non_null.to_numpy())) if n_unique <= 2 else set()
        if n_unique <= 2 and uniques <= {0, 1, 0.0, 1.0, True, False}:
            return "binary"
        if pd.api.types.is_integer_dtype(series):
            # A near-unique integer column is an identifier, not a measurement.
            if n_unique > 0.95 * max(n_rows, 1) and n_rows > 20:
                return "identifier"
            if n_unique <= 20:
                return "integer"
            return "continuous"
        return "continuous"
    # Non-numeric.
    if n_unique <= 2:
        return "binary"
    if n_unique > 0.95 * max(n_rows, 1) and n_rows > 20:
        return "identifier"
    # Values that recur are labels (unit ids, groups), not free text -- however many there are.
    repeats = n_rows / max(n_unique, 1)
    if repeats >= 1.5 and n_unique <= 5000:
        return "categorical"
    if n_unique <= max(50, 0.05 * n_rows):
        return "categorical"
    return "text"


def _to_jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return value


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    n_rows = len(df)
    profiles: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        non_null = series.dropna()
        sem = _semantic_type(str(name), series, n_rows)
        prof = ColumnProfile(
            name=str(name),
            dtype=str(series.dtype),
            semantic_type=sem,  # type: ignore[arg-type]
            missing_count=int(series.isna().sum()),
            missing_pct=round(float(series.isna().mean() * 100), 2),
            n_unique=int(non_null.nunique()),
            sample_values=[_to_jsonable(v) for v in non_null.head(5).tolist()],
        )
        if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
            prof.mean = _to_jsonable(float(non_null.mean()))
            prof.std = _to_jsonable(float(non_null.std())) if len(non_null) > 1 else None
            prof.min = _to_jsonable(float(non_null.min()))
            prof.max = _to_jsonable(float(non_null.max()))
        profiles.append(prof)
    return profiles


def _rank(names_scores: list[tuple[str, float]]) -> list[str]:
    """Best-scoring first, de-duplicated: a column can be scored by several rules."""
    best: dict[str, float] = {}
    for name, score in names_scores:
        if score > best.get(name, float("-inf")):
            best[name] = score
    return [n for n, s in sorted(best.items(), key=lambda kv: -kv[1]) if s > 0]


_STEM = re.compile(r"^(.*?)(\d+)$")


def _prefer_latest_in_series(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Among columns that share a stem and differ only by a trailing number
    (re74, re75, re78 / spend_2019, spend_2020), the highest number is the most recent
    and therefore the likely outcome; the earlier ones are pre-treatment measurements.
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for name, _ in scores:
        m = _STEM.match(name)
        if m and m.group(1):
            groups.setdefault(m.group(1), []).append((name, int(m.group(2))))

    adjust: dict[str, float] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        latest = max(members, key=lambda kv: kv[1])[0]
        for name, _ in members:
            adjust[name] = 0.6 if name == latest else -0.6
    return [(n, s + adjust.get(n, 0.0)) for n, s in scores]


def _dummy_block_penalty(scores: list[tuple[str, float]], min_family: int = 5) -> list[tuple[str, float]]:
    """Down-rank members of a large numbered family (r20001, r20002, ... d1990, d1991).

    Datasets shipped for regression work often carry dozens of pre-built fixed-effect
    dummies. They are binary and they vary the same way a treatment does, so without this
    they crowd out the real treatment indicator purely by being numerous.
    """
    families: dict[str, int] = {}
    for name, _ in scores:
        m = _STEM.match(name)
        if m and m.group(1):
            families[m.group(1)] = families.get(m.group(1), 0) + 1
    out = []
    for name, score in scores:
        m = _STEM.match(name)
        stem = m.group(1) if m else None
        penalty = 0.35 if stem and families.get(stem, 0) >= min_family else 0.0
        out.append((name, score - penalty))
    return out


def detect_candidates(df: pd.DataFrame, cols: list[ColumnProfile]) -> RoleCandidates:
    by_name = {c.name: c for c in cols}
    n_rows = len(df)

    outcome, treatment, time_c, unit, running, instrument, covariates = ([] for _ in range(7))

    for c in cols:
        name = c.name
        sem = c.semantic_type

        # Outcome: a measured quantity. Penalise names that read as time, id, or control.
        if sem in ("continuous", "integer"):
            score = 0.3 + _hint_score(name, OUTCOME_HINTS)
            score += 0.35 if sem == "continuous" else 0.0
            score -= 0.9 * _hint_score(name, TIME_HINTS)
            score -= 0.9 * _hint_score(name, UNIT_HINTS)
            score -= 0.5 * _hint_score(name, RUNNING_HINTS)
            score -= 0.45 * _hint_score(name, COVARIATE_HINTS)
            score -= 0.4 * _hint_score(name, INSTRUMENT_HINTS)
            if sem == "integer" and c.n_unique < 10:
                score -= 0.3
            # Constant or two-valued columns are not outcomes.
            if c.n_unique <= 2:
                score = 0.0
            outcome.append((name, max(score, 0.0)))

        # Treatment: binary columns, strongly name-driven.
        if sem == "binary":
            score = 0.4 + _hint_score(name, TREATMENT_HINTS)
            treatment.append((name, score))
        elif sem in ("categorical",) and c.n_unique <= 5:
            score = _hint_score(name, TREATMENT_HINTS) * 0.6
            if score > 0:
                treatment.append((name, score))

        # Time: datetime columns, or low-cardinality ordered integers.
        if sem == "datetime":
            time_c.append((name, 1.0 + _hint_score(name, TIME_HINTS)))
        elif sem in ("integer", "continuous") and 2 <= c.n_unique <= max(60, 0.2 * n_rows):
            hint = _hint_score(name, TIME_HINTS)
            if hint >= 0.8:  # exact or whole-token match only
                time_c.append((name, 0.5 + hint))
        elif sem in ("categorical",) and _hint_score(name, TIME_HINTS) > 0:
            time_c.append((name, 0.4 + _hint_score(name, TIME_HINTS)))

        # Unit: repeated identifiers -- more than one row per value, but not unique.
        if sem in ("identifier", "categorical", "integer"):
            if 1 < c.n_unique < n_rows:
                repeats = n_rows / max(c.n_unique, 1)
                score = _hint_score(name, UNIT_HINTS)
                if repeats >= 2:
                    score += 0.5
                # A time-looking column is the period axis, not the unit axis.
                score -= 1.2 * _hint_score(name, TIME_HINTS)
                if sem == "identifier":
                    score += 0.3
                if score > 0:
                    unit.append((name, score))
        elif sem == "identifier":
            unit.append((name, _hint_score(name, UNIT_HINTS)))

        # Running variable: continuous with enough support to fit either side of a cutoff.
        if sem == "continuous" and c.n_unique >= 20:
            running.append((name, 0.3 + _hint_score(name, RUNNING_HINTS)))

        # Instrument: binary or continuous, name-driven (structure can't reveal exclusion).
        hint = _hint_score(name, INSTRUMENT_HINTS)
        if hint > 0 and sem in ("binary", "continuous", "integer"):
            instrument.append((name, hint))

        # Covariates: anything usable as a control.
        if sem in ("continuous", "integer", "binary", "categorical"):
            covariates.append((name, 1.0))

    # A unit column is one where (column, time) nearly uniquely keys the rows -- that is
    # what a panel *is*. Testing it structurally catches numeric ids like `fcode` or `sid`
    # that name hints and dtype heuristics both miss.
    ranked_time = _rank(time_c)
    if ranked_time and n_rows > 4:
        t = ranked_time[0]
        for c in cols:
            if c.name == t or c.n_unique <= 1 or c.n_unique >= n_rows:
                continue
            if n_rows / c.n_unique < 1.5:      # too few repeats to be a unit
                continue
            try:
                pair_unique = int((~df.duplicated(subset=[c.name, t])).sum())
            except (TypeError, ValueError):
                continue
            if pair_unique >= 0.98 * n_rows:
                unit.append((c.name, 1.5))     # outranks any name-based guess

    return RoleCandidates(
        outcome=_rank(_prefer_latest_in_series(outcome))[:10],
        treatment=_rank(_dummy_block_penalty(treatment))[:10],
        time=_rank(time_c)[:10],
        unit=_rank(unit)[:10],
        running_variable=_rank(running)[:10],
        instrument=_rank(instrument)[:10],
        covariates=_rank(covariates)[:40],
    )


def detect_structure(
    df: pd.DataFrame, cols: list[ColumnProfile], cand: RoleCandidates
) -> StructureFlags:
    notes: list[str] = []
    by_name = {c.name: c for c in cols}

    time_col = cand.time[0] if cand.time else None
    unit_col = cand.unit[0] if cand.unit else None
    treat_col = cand.treatment[0] if cand.treatment else None

    n_periods = int(df[time_col].nunique()) if time_col else 0
    n_units = int(df[unit_col].nunique()) if unit_col else 0

    is_panel = False
    has_repeated_cross_sections = False
    if time_col and unit_col and n_periods >= 2 and n_units >= 2:
        rows_per_cell = len(df) / max(n_periods * n_units, 1)
        # A balanced panel has ~1 row per (unit, period) cell.
        dup = df.duplicated(subset=[unit_col, time_col]).mean()
        if rows_per_cell <= 1.5 and dup < 0.1:
            is_panel = True
            notes.append(
                f"Panel structure detected: {n_units} units observed over {n_periods} periods."
            )
        else:
            has_repeated_cross_sections = True
            notes.append(
                f"Repeated cross-sections: multiple rows per ({unit_col}, {time_col}) cell."
            )
    elif time_col and n_periods >= 2:
        has_repeated_cross_sections = True
        notes.append(f"Time variation present ({n_periods} periods) but no clear unit id.")

    has_binary_treatment = bool(treat_col and by_name[treat_col].semantic_type == "binary")

    has_pre_post = False
    has_never_treated = False
    if treat_col and time_col and has_binary_treatment:
        treat_num = pd.to_numeric(df[treat_col], errors="coerce")
        if treat_num.notna().any():
            by_period = treat_num.groupby(df[time_col], observed=True).mean()
            has_pre_post = bool((by_period == 0).any() and (by_period > 0).any())
            if unit_col:
                ever = treat_num.groupby(df[unit_col], observed=True).max()
                has_never_treated = bool((ever == 0).any() and (ever > 0).any())
                if has_never_treated:
                    n_ever = int((ever > 0).sum())
                    notes.append(
                        f"{n_ever} of {len(ever)} units are ever-treated; "
                        f"{len(ever) - n_ever} are never-treated controls."
                    )
            if has_pre_post:
                notes.append(
                    "Treatment switches on over time -- pre/post variation available for DiD."
                )

    if not notes:
        notes.append("Cross-sectional data: no usable time dimension detected.")

    return StructureFlags(
        is_panel=is_panel,
        n_periods=n_periods,
        n_units=n_units,
        has_pre_post_variation=has_pre_post,
        has_binary_treatment=has_binary_treatment,
        has_never_treated_units=has_never_treated,
        has_repeated_cross_sections=has_repeated_cross_sections,
        notes=notes,
    )


def build_profile(session_id: str, filename: str, df: pd.DataFrame) -> DataProfile:
    cols = profile_columns(df)
    cand = detect_candidates(df, cols)
    structure = detect_structure(df, cols, cand)
    preview = [
        {str(k): _to_jsonable(v) for k, v in row.items()}
        for row in df.head(8).to_dict(orient="records")
    ]
    return DataProfile(
        session_id=session_id,
        filename=filename,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=cols,
        candidates=cand,
        structure=structure,
        preview=preview,
    )
