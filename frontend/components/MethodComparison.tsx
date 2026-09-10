"use client";

import { useState } from "react";
import type { MethodAssessment, MethodId, Plan } from "@/lib/types";

/** The "why this method" panel: every design considered, and why the others were ruled out. */
export function MethodComparison({
  plan,
  onRerun,
  running,
}: {
  plan: Plan;
  onRerun?: (method: MethodId) => void;
  running?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const chosen = plan.chosen_method;
  const sorted = [...plan.assessments].sort(
    (a, b) =>
      Number(b.method === chosen) - Number(a.method === chosen) ||
      Number(b.feasible) - Number(a.feasible) ||
      b.score - a.score
  );

  return (
    <div className="rounded-xl border border-border-base bg-surface">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full cursor-pointer items-center justify-between gap-4 px-5 py-4 text-left"
        aria-expanded={open}
      >
        <div className="min-w-0">
          <h3 className="text-[14.5px] font-semibold tracking-tight text-ink">
            Why {plan.chosen_label}?
          </h3>
          <p className="mt-0.5 text-[12.5px] text-ink-3">
            {plan.assessments.filter((a) => a.feasible).length} of {plan.assessments.length}{" "}
            designs were identifiable from this data
          </p>
        </div>
        <svg
          className={`h-4 w-4 shrink-0 text-ink-3 transition-transform ${open ? "rotate-180" : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.75}
        >
          <path d="m4 6 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      <div className="border-t border-border-base px-5 py-4">
        <p className="max-w-3xl text-[14px] leading-relaxed text-ink-2">{plan.justification}</p>
        <p className="mt-3 text-[12.5px] text-ink-3">
          <span className="font-medium text-ink-2">Estimand:</span> {plan.estimand}
        </p>
      </div>

      {open ? (
        <div className="space-y-3 border-t border-border-base px-5 py-4">
          {sorted.map((a) => (
            <AssessmentRow
              key={a.method}
              a={a}
              isChosen={a.method === chosen}
              onRerun={onRerun}
              running={running}
            />
          ))}

          <div className="rounded-lg bg-surface-2 p-4">
            <h5 className="text-[11px] font-medium uppercase tracking-wider text-ink-3">
              What {plan.chosen_label} requires
            </h5>
            <ul className="mt-2 space-y-1.5">
              {plan.assumptions.map((x, i) => (
                <li key={i} className="flex gap-2 text-[13px] leading-relaxed text-ink-2">
                  <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
                  {x}
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function AssessmentRow({
  a,
  isChosen,
  onRerun,
  running,
}: {
  a: MethodAssessment;
  isChosen: boolean;
  onRerun?: (m: MethodId) => void;
  running?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border p-4 ${
        isChosen ? "border-accent bg-accent-soft" : "border-border-base bg-surface"
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13.5px] font-semibold text-ink">{a.label}</span>
        {isChosen ? (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
            style={{ background: "var(--accent)", color: "white" }}
          >
            selected
          </span>
        ) : a.feasible ? (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide"
            style={{ color: "var(--pass)", background: "var(--pass-soft)" }}
          >
            identifiable
          </span>
        ) : (
          <span
            className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide"
            style={{ color: "var(--ink-3)", background: "var(--surface-2)" }}
          >
            ruled out
          </span>
        )}

        {a.feasible ? (
          <span className="tnum ml-auto text-[11px] text-ink-3">
            structural fit {a.score.toFixed(2)}
          </span>
        ) : null}

        {a.feasible && !isChosen && onRerun ? (
          <button
            onClick={() => onRerun(a.method)}
            disabled={running}
            className="cursor-pointer rounded-md border border-border-strong px-2 py-1 text-[11px] font-medium text-ink-2 transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
          >
            run this instead
          </button>
        ) : null}
      </div>

      <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-2">
        {a.feasible ? a.rationale : a.blocking_reasons[0] || a.rationale}
      </p>
    </div>
  );
}
