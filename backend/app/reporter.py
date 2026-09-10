"""The /report step: turn an estimate plus diagnostics into plain English."""

from __future__ import annotations

import numpy as np

from . import llm
from .config import settings
from .methods import SPECS
from .schemas import ExecutionResult, Plan, Report

REPORTER_SYSTEM = """You are writing the results section of a causal analysis for an \
intelligent reader who is not a statistician -- a product manager or an executive.

Write with precision and restraint:
- Lead with the number and what it means in the units of the problem.
- Say plainly who the estimate applies to. A discontinuity speaks only about units near \
the cutoff; an instrument only about compliers; matching only about the treated.
- Report the uncertainty honestly. Never call a result that crosses zero "an effect".
- Take the diagnostics seriously. A failed assumption check is the headline, not a footnote.
- Never claim an assumption is verified when it is untestable. Say which assumptions rest \
on judgment rather than evidence.
- No hedging boilerplate, no "it is important to note", no restating the question back.

Return JSON with exactly these keys:
{
  "headline": "one sentence, the finding and its size, in the outcome's own units",
  "interpretation": "2-4 sentences on what the number means in practical terms and for whom",
  "method_explanation": "2-3 sentences explaining how this design works and why it was right here",
  "assumptions_discussion": "2-4 sentences on what had to be true, what the diagnostics showed, and what remains an act of faith",
  "caveats": ["specific, concrete caveat", "..."],
  "confidence": "high" | "moderate" | "low"
}
Base `confidence` on the diagnostics and the precision of the estimate, not on how \
interesting the result is."""


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:,.{digits}g}"


def _evidence_block(plan: Plan, result: ExecutionResult) -> str:
    e = result.estimate
    lines = [
        f"QUESTION: {plan.question}",
        f"DESIGN: {result.method_label}",
        f"ESTIMAND: {result.estimand}",
        f"WHY THIS DESIGN: {plan.justification}",
        "",
        "RESULT:",
        f"  point estimate: {_fmt(e.point)} ({e.units})",
        f"  95% CI: [{_fmt(e.ci_low)}, {_fmt(e.ci_high)}]",
        f"  standard error: {_fmt(e.se)}   p-value: {_fmt(e.p_value)}",
        f"  observations: {e.n_obs}"
        + (f"  (treated {e.n_treated}, control {e.n_control})" if e.n_treated is not None else ""),
        "",
        "SPECIFICATION:",
    ]
    lines += [f"  {k}: {v}" for k, v in result.specification.items()]
    lines += ["", "DIAGNOSTICS:"]
    for d in result.diagnostics:
        lines.append(f"  [{d.verdict.upper()}] {d.title}: {d.summary}")
    if result.warnings:
        lines += ["", "WARNINGS RAISED DURING ESTIMATION:"]
        lines += [f"  - {w}" for w in result.warnings]
    lines += ["", "ASSUMPTIONS THIS DESIGN REQUIRES:"]
    lines += [f"  - {a}" for a in plan.assumptions]
    return "\n".join(lines)


def _confidence(result: ExecutionResult) -> str:
    verdicts = [d.verdict for d in result.diagnostics]
    e = result.estimate
    crosses_zero = (
        e.ci_low is not None and e.ci_high is not None and e.ci_low <= 0 <= e.ci_high
    )
    if "fail" in verdicts or crosses_zero:
        return "low"
    if verdicts.count("warn") >= 2 or result.warnings:
        return "moderate"
    return "high" if verdicts.count("warn") == 0 else "moderate"


def _template_report(plan: Plan, result: ExecutionResult) -> Report:
    """Deterministic report used when no LLM is configured, or when the call fails."""
    e = result.estimate
    spec = SPECS[result.method]
    direction = "increase" if e.point > 0 else "decrease"
    crosses_zero = (
        e.ci_low is not None and e.ci_high is not None and e.ci_low <= 0 <= e.ci_high
    )

    if crosses_zero:
        headline = (
            f"No statistically detectable effect: the estimate is {_fmt(e.point)} {e.units}, "
            f"but the 95% interval [{_fmt(e.ci_low)}, {_fmt(e.ci_high)}] includes zero."
        )
    else:
        headline = (
            f"Estimated {direction} of {_fmt(abs(e.point))} in {e.units} "
            f"(95% CI [{_fmt(e.ci_low)}, {_fmt(e.ci_high)}])."
        )

    interpretation = (
        f"{result.method_label} gives an estimate of {_fmt(e.point)} {e.units}. "
        f"This is the {result.estimand.lower()}, computed on {e.n_obs:,} observations"
        + (f" ({e.n_treated:,} treated, {e.n_control:,} control)."
           if e.n_treated is not None and e.n_control is not None else ".")
        + (
            " Because the interval spans zero, the data are consistent with no effect at all; "
            "treat the point estimate as a best guess, not a finding."
            if crosses_zero
            else f" The interval excludes zero, so the direction of the effect is well "
            f"determined even though its exact size is not."
        )
    )

    failed = [d for d in result.diagnostics if d.verdict == "fail"]
    warned = [d for d in result.diagnostics if d.verdict == "warn"]
    passed = [d for d in result.diagnostics if d.verdict == "pass"]
    parts = []
    if passed:
        parts.append(
            f"{len(passed)} assumption check(s) passed: "
            + "; ".join(d.title.lower() for d in passed) + "."
        )
    if warned:
        parts.append("Flagged for attention: " + "; ".join(d.summary for d in warned))
    if failed:
        parts.append("FAILED: " + "; ".join(d.summary for d in failed))
    if not parts:
        parts.append("No automated assumption checks were available for this design.")
    assumptions_discussion = " ".join(parts)

    caveats = [w for w in result.warnings]
    caveats += [d.summary for d in failed + warned]
    caveats.append(
        f"{result.method_label} identifies a causal effect only under: "
        + spec.assumptions[0]
    )
    if result.method in ("rdd", "iv"):
        caveats.append(
            "This estimate is local. It describes a specific subgroup — units at the cutoff, "
            "or compliers — and need not generalise to everyone."
        )

    return Report(
        session_id=result.session_id,
        headline=headline,
        interpretation=interpretation,
        method_explanation=spec.plain_english,
        assumptions_discussion=assumptions_discussion,
        caveats=caveats[:6],
        confidence=_confidence(result),  # type: ignore[arg-type]
        llm_used=False,
    )


def build_report(plan: Plan, result: ExecutionResult, use_llm: bool = True) -> Report:
    if not (use_llm and llm.available()):
        return _template_report(plan, result)

    parsed = llm.complete_json(
        REPORTER_SYSTEM,
        _evidence_block(plan, result),
        model=settings.anthropic_report_model,
        max_tokens=1800,
        temperature=0.3,
    )
    if not parsed or not isinstance(parsed.get("headline"), str):
        return _template_report(plan, result)

    fallback = _template_report(plan, result)
    caveats = parsed.get("caveats")
    if not isinstance(caveats, list) or not caveats:
        caveats = fallback.caveats
    confidence = parsed.get("confidence")
    if confidence not in ("high", "moderate", "low"):
        confidence = fallback.confidence

    def text(key: str, default: str) -> str:
        v = parsed.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else default

    return Report(
        session_id=result.session_id,
        headline=parsed["headline"].strip(),
        interpretation=text("interpretation", fallback.interpretation),
        method_explanation=text("method_explanation", fallback.method_explanation),
        assumptions_discussion=text("assumptions_discussion", fallback.assumptions_discussion),
        caveats=[str(c) for c in caveats][:8],
        confidence=confidence,  # type: ignore[arg-type]
        llm_used=True,
    )
