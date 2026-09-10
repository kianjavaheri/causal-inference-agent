"""The /plan step: choose a method and justify it.

Two layers. The deterministic layer decides what is *identifiable* from the data's
structure. The LLM layer chooses among those options in light of what the user
actually asked, and writes the justification. The LLM can never make an
unidentifiable design feasible -- if it names one, we override it and say so.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from . import identify, llm
from .identify import Assessment
from .methods import SPECS
from .schemas import DataProfile, Plan

STOPWORDS = {
    "the", "a", "an", "of", "on", "in", "to", "for", "and", "or", "is", "was", "were",
    "did", "do", "does", "what", "how", "why", "effect", "effects", "impact", "affect",
    "affected", "cause", "caused", "causal", "change", "changed", "much", "many",
    "their", "its", "that", "this", "it", "by", "with", "from", "at", "as", "be",
    "been", "have", "has", "had", "our", "we", "us", "my", "any", "all", "more",
    "less", "than", "after", "before", "when", "who", "which", "there",
}


def _tokens(text: str) -> set[str]:
    raw = re.split(r"[^a-z0-9]+", text.lower())
    out: set[str] = set()
    for t in raw:
        if len(t) < 3 or t in STOPWORDS:
            continue
        out.add(t)
        # Cheap stemming so "earnings" matches "earning" and "sales" matches "sale".
        if t.endswith("s") and len(t) > 4:
            out.add(t[:-1])
        if t.endswith("ing") and len(t) > 6:
            out.add(t[:-3])
        if t.endswith("ed") and len(t) > 5:
            out.add(t[:-2])
    return out


def _column_match(question: str, columns: list[str]) -> dict[str, float]:
    """How strongly each column name is echoed by the question.

    Scored by whole-token overlap only. Any normalisation by name length rewards terse
    names -- 'hs_gpa' would beat 'college_gpa' on a question about GPA purely for being
    shorter. Ties are left for the caller to break on the profiler's own ranking, which
    already encodes which columns look like outcomes rather than controls.
    """
    q = _tokens(question)
    scores: dict[str, float] = {}
    for col in columns:
        c = _tokens(col)
        if not c:
            continue
        overlap = len(q & c)
        if overlap:
            scores[col] = float(overlap)
    return scores


def question_hints(
    question: str, profile: DataProfile, df: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Read role hints straight out of the question by matching column names.

    Deterministic and cheap. It is what lets the agent honour "effect on GPA" rather
    than picking whichever numeric column happened to rank first. Roles are claimed in
    priority order so one column never fills two roles, and the profiler's own ranking
    breaks ties between equally-good textual matches.
    """
    names = [c.name for c in profile.columns]
    matched = _column_match(question, names)
    if not matched:
        return {}

    cand = profile.candidates
    hints: dict[str, Any] = {}
    claimed: set[str] = set()

    def claim(role: str, pool: list[str], veto=None, min_score: float = 1.0) -> None:
        best, best_key = None, None
        for rank, name in enumerate(pool):
            if name in claimed:
                continue
            score = matched.get(name, 0.0)
            if score < min_score or (veto and veto(name)):
                continue
            # Whole-token overlap dominates; the profiler's own ranking breaks ties, so a
            # merely-equal textual match never overrides the structural ordering.
            key = (-score, rank)
            if best_key is None or key < best_key:
                best, best_key = name, key
        if best:
            hints[role] = best
            claimed.add(best)

    numericish = {
        c.name
        for c in profile.columns
        if c.semantic_type in ("binary", "continuous", "integer")
    }
    instrument_pool = [
        n for n in names if n in numericish and identify._looks_like_instrument(n)
    ]

    # Order matters: the outcome is the thing being explained, so it is claimed first.
    # The instrument is claimed before the treatment, because a question that names both
    # ("effect of onboarding, given only the email was randomised") mentions the instrument
    # just as prominently -- and a column that reads as an assignment mechanism is the
    # instrument, not the treatment it shifts.
    unit0 = cand.unit[0] if cand.unit else None
    time0 = cand.time[0] if cand.time else None

    def not_a_treatment(name: str) -> bool:
        """Veto instruments, and any flag that just marks the post period."""
        if identify._looks_like_instrument(name):
            return True
        if df is not None and unit0 and time0 and name not in (unit0, time0):
            try:
                return identify._variation(df, name, unit0, time0) == "time_only"
            except Exception:
                return False
        return False

    claim("outcome", cand.outcome)
    claim("running_variable", cand.running_variable)
    claim("instrument", instrument_pool)
    claim("treatment", cand.treatment, veto=not_a_treatment)
    claim("time", cand.time)
    claim("unit", cand.unit)

    hints["_matched"] = {
        k: round(v, 3) for k, v in sorted(matched.items(), key=lambda kv: -kv[1])[:8]
    }
    return hints


PLANNER_SYSTEM = """You are a causal inference methodologist advising an analyst.

A deterministic identifiability engine has already examined the dataset's structure and \
determined which designs are FEASIBLE and which are BLOCKED, along with the exact columns \
that would fill each role. That analysis is authoritative about the data. Your job is \
judgment, not recomputation:

1. Choose the design that best answers the user's actual question among the FEASIBLE ones.
2. Confirm or correct the column-to-role assignment using the question's wording.
3. Explain the choice in plain English an intelligent non-specialist can follow.

Rules you must not break:
- Choose ONLY from the feasible methods listed. If none is feasible, say so in the \
justification and set chosen_method to null.
- Assign roles ONLY from the column names given. Never invent a column.
- Prefer designs whose identifying assumption is more credible, not merely those that \
produce a number. Propensity score matching assumes no unobserved confounding, which is \
strictly stronger than what a discontinuity, an instrument, or a parallel-trends design needs; \
choose it only when nothing better is available or the question calls for it specifically.
- Be candid about what the design cannot deliver. Do not oversell.

Return JSON with exactly these keys:
{
  "chosen_method": "<method id or null>",
  "roles": {"outcome": "...", "treatment": "...", ...},
  "justification": "2-4 sentences: why this design, why not the runners-up, in plain English",
  "estimand": "one sentence stating precisely what quantity is being estimated and for whom",
  "reasoning_trace": ["short step", "short step", "..."]
}
reasoning_trace should be 3-6 terse steps showing how you moved from the question and the \
data's structure to the design -- the kind of notes a careful analyst would jot down."""


def _profile_digest(profile: DataProfile) -> str:
    lines = [
        f"Dataset: {profile.filename} — {profile.n_rows} rows x {profile.n_cols} columns",
        "",
        "Columns:",
    ]
    for c in profile.columns:
        bits = [f"  - {c.name} ({c.semantic_type}, {c.n_unique} distinct"]
        if c.missing_pct > 0:
            bits.append(f", {c.missing_pct}% missing")
        bits.append(")")
        if c.mean is not None:
            bits.append(f" mean={c.mean:.4g} range=[{c.min:.4g}, {c.max:.4g}]")
        else:
            sample = ", ".join(str(v) for v in c.sample_values[:3])
            bits.append(f" e.g. {sample}")
        lines.append("".join(bits))
    s = profile.structure
    lines += [
        "",
        "Structure:",
        f"  panel: {s.is_panel}; periods: {s.n_periods}; units: {s.n_units}",
        f"  pre/post variation: {s.has_pre_post_variation}; "
        f"never-treated units present: {s.has_never_treated_units}",
    ]
    lines += [f"  note: {n}" for n in s.notes]
    return "\n".join(lines)


def _assessment_digest(assessments: list[Assessment]) -> str:
    feasible = [a for a in assessments if a.feasible]
    blocked = [a for a in assessments if not a.feasible]
    lines = ["FEASIBLE designs (choose one of these):"]
    if not feasible:
        lines.append("  (none)")
    for a in feasible:
        spec = SPECS[a.method]
        lines.append(f"  * {a.method} — {spec.label} (structural fit score {a.score:.2f})")
        lines.append(f"      estimand: {spec.estimand}")
        lines.append(f"      engine says: {a.rationale}")
        lines.append(f"      proposed roles: {a.roles}")
        if a.derivations:
            for d in a.derivations:
                lines.append(f"      constructed column '{d.name}': {d.explanation}")
    lines.append("")
    lines.append("BLOCKED designs (must not be chosen):")
    for a in blocked:
        lines.append(f"  * {a.method}: {'; '.join(a.blocking) or a.rationale}")
    return "\n".join(lines)


def _fallback_justification(chosen: Assessment, others: list[Assessment]) -> str:
    """Why this design, in two sentences.

    Deliberately short. Each ruled-out design already carries its own blocking reason in
    the assessment list, and the method's plain-English explanation has its own section in
    the report -- piling all three into one paragraph makes it unreadable.
    """
    parts = [chosen.rationale]
    runners = [a for a in others if a.feasible and a.method != chosen.method][:2]
    if runners:
        names = " and ".join(SPECS[a.method].label for a in runners)
        parts.append(
            f"{names} {'are' if len(runners) > 1 else 'is'} also computable here, but "
            f"{'they rest' if len(runners) > 1 else 'it rests'} on assumptions this data "
            "supports less well."
        )
    n_blocked = sum(1 for a in others if not a.feasible)
    if n_blocked:
        parts.append(
            f"The other {n_blocked} design{'s' if n_blocked > 1 else ''} "
            f"{'are' if n_blocked > 1 else 'is'} not identifiable from this data at all."
        )
    return " ".join(parts)


def build_plan(
    session_id: str,
    question: str,
    df: pd.DataFrame,
    profile: DataProfile,
    user_roles: dict[str, Any] | None = None,
    use_llm: bool = True,
) -> tuple[Plan, Assessment]:
    hints = question_hints(question, profile, df)
    matched = hints.pop("_matched", {})
    if user_roles:
        hints.update({k: v for k, v in user_roles.items() if v})

    assessments = identify.assess_all(df, profile, hints)
    feasible = [a for a in assessments if a.feasible]

    trace: list[str] = [
        f"Profiled {profile.n_rows} rows across {profile.n_cols} columns.",
        profile.structure.notes[0] if profile.structure.notes else "Examined data structure.",
    ]
    if matched:
        top = ", ".join(f"'{k}'" for k in list(matched)[:3])
        trace.append(f"Matched the question's wording to columns: {top}.")
    trace.append(
        f"Tested all five designs for identifiability: "
        f"{len(feasible)} feasible, {len(assessments) - len(feasible)} blocked."
    )

    if not feasible:
        reasons = "; ".join(
            f"{SPECS[a.method].label}: {(a.blocking or [a.rationale])[0]}" for a in assessments
        )
        raise ValueError(
            "No causal design is identifiable from this dataset as structured. " + reasons
        )

    chosen = feasible[0]
    justification = _fallback_justification(chosen, assessments)
    estimand = SPECS[chosen.method].estimand
    roles = dict(chosen.roles)
    llm_used = False

    if use_llm and llm.available():
        user_msg = (
            f"USER'S QUESTION:\n{question}\n\n"
            f"{_profile_digest(profile)}\n\n"
            f"{_assessment_digest(assessments)}\n"
        )
        parsed = llm.complete_json(PLANNER_SYSTEM, user_msg, max_tokens=1800)
        if parsed:
            picked = parsed.get("chosen_method")
            match = next((a for a in feasible if a.method == picked), None)
            if match:
                chosen = match
                roles = dict(chosen.roles)
                llm_used = True
                # Accept role overrides only for columns that actually exist.
                valid = {c.name for c in profile.columns}
                for role, value in (parsed.get("roles") or {}).items():
                    if role not in roles:
                        continue
                    if isinstance(value, str) and value in valid:
                        roles[role] = value
                    elif isinstance(value, list):
                        roles[role] = [v for v in value if v in valid]
                if isinstance(parsed.get("justification"), str):
                    justification = parsed["justification"].strip()
                if isinstance(parsed.get("estimand"), str):
                    estimand = parsed["estimand"].strip()
                steps = parsed.get("reasoning_trace")
                if isinstance(steps, list):
                    trace += [str(s) for s in steps if isinstance(s, (str, int, float))][:6]
            elif picked:
                trace.append(
                    f"The language model proposed '{picked}', which the identifiability engine "
                    "ruled out on this data. Overriding it and keeping the highest-scoring "
                    "feasible design."
                )
                trace.append(
                    f"Selected {SPECS[chosen.method].label} on structural grounds instead."
                )
    if not llm_used:
        trace.append(
            f"Selected {SPECS[chosen.method].label} — the highest-scoring identifiable design "
            f"(structural fit {chosen.score:.2f})."
        )

    for d in chosen.derivations:
        trace.append(f"Constructed '{d.name}': {d.explanation}")

    plan = Plan(
        session_id=session_id,
        question=question,
        chosen_method=chosen.method,  # type: ignore[arg-type]
        chosen_label=SPECS[chosen.method].label,
        justification=justification,
        estimand=estimand,
        roles={k: v for k, v in roles.items()},
        assumptions=list(SPECS[chosen.method].assumptions),
        assessments=[identify.to_schema(a) for a in assessments],
        llm_used=llm_used,
        reasoning_trace=trace,
    )
    chosen.roles = roles
    return plan, chosen
