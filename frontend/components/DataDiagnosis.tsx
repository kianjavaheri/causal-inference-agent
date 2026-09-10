"use client";

import type { Suggestions } from "@/lib/types";

/**
 * Shown when no design is identifiable. The point is to be specific: which design was
 * considered, exactly why it does not apply, and what column would change that. A bare
 * "cannot analyse this" tells the user nothing about how to move forward.
 */
export function DataDiagnosis({ suggestions }: { suggestions: Suggestions }) {
  return (
    <div
      className="rounded-xl border p-5 sm:p-6"
      style={{ borderColor: "var(--warn)", background: "var(--warn-soft)" }}
    >
      <h3 className="text-[15px] font-semibold tracking-tight text-ink">
        This dataset cannot support a causal design yet
      </h3>
      <p className="mt-2 max-w-2xl text-[13.5px] leading-relaxed text-ink-2">
        {suggestions.guidance}
      </p>

      <div className="mt-5 space-y-2.5">
        {suggestions.blockers.map((b) => (
          <div key={b.method} className="rounded-lg border border-border-base bg-surface p-4">
            <h4 className="text-[13.5px] font-semibold text-ink">{b.label}</h4>
            {b.reasons.slice(0, 1).map((r, i) => (
              <p key={i} className="mt-1 text-[12.5px] leading-relaxed text-ink-2">
                {r}
              </p>
            ))}
            {b.missing.length ? (
              <p className="mt-2 text-[12.5px] leading-relaxed text-ink-3">
                <span className="font-medium text-ink-2">Would need:</span>{" "}
                {b.missing.join("; ")}.
              </p>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}
