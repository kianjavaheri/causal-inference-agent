"use client";

import type { CompositionData } from "@/lib/types";


/**
 * The IV complier decomposition. Under monotonicity the sample splits three ways, and
 * instrumental variables can only speak to the middle group -- so the point of this
 * chart is to show how much of the population the estimate silently excludes.
 *
 * Drawn with plain elements rather than a chart library: it is one bar, and it needs to
 * render legibly at any width.
 */
export function CompositionBar({ data }: { data: CompositionData }) {
  const segments = data.segments ?? [];
  if (!segments.length) return null;

  const total = segments.reduce((a, s) => a + (s.share ?? 0), 0) || 1;
  const rows = data.rows ?? [];

  return (
    <div>
      <div className="flex h-11 w-full overflow-hidden rounded-lg">
        {segments.map((s) => {
          const pct = ((s.share ?? 0) / total) * 100;
          if (pct <= 0) return null;
          return (
            <div
              key={s.label}
              title={`${s.label}: ${pct.toFixed(1)}%`}
              className="flex items-center justify-center overflow-hidden transition-all"
              style={{
                width: `${pct}%`,
                background: s.estimated ? "var(--accent)" : "var(--surface-2)",
                color: s.estimated ? "white" : "var(--ink-3)",
                borderRight: "1px solid var(--paper)",
              }}
            >
              {pct > 11 ? (
                <span className="tnum text-[12px] font-medium">{pct.toFixed(0)}%</span>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        {segments.map((s) => (
          <div key={s.label} className="flex gap-2">
            <span
              className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{
                background: s.estimated ? "var(--accent)" : "var(--surface-2)",
                outline: s.estimated ? "none" : "1px solid var(--border-strong)",
              }}
            />
            <span className="text-[12px] leading-snug">
              <span
                className="font-medium"
                style={{ color: s.estimated ? "var(--accent)" : "var(--ink-2)" }}
              >
                {s.label}
              </span>
              <span className="block text-ink-3">{s.note}</span>
            </span>
          </div>
        ))}
      </div>

      {rows.length ? (
        <dl className="mt-4 space-y-1.5 border-t border-border-base pt-3">
          {rows.map((r) => (
            <div key={r.label} className="flex justify-between gap-4 text-[12.5px]">
              <dt className="text-ink-3">{r.label}</dt>
              <dd className="tnum text-ink-2">
                {r.value === null || r.value === undefined
                  ? "—"
                  : `${(r.value * 100).toFixed(1)}%`}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  );
}
