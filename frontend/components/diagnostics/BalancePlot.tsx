"use client";

import { sig } from "@/lib/format";
import type { BalanceData } from "@/lib/types";

/**
 * Love plot. Standardised mean differences before and after matching, with the
 * conventional 0.1 threshold marked. Drawn by hand rather than with a chart library:
 * a dumbbell across a symmetric axis is clearer than any bar chart of the same numbers.
 */
export function BalancePlot({ data }: { data: BalanceData }) {
  const rows = data.rows ?? [];
  if (!rows.length) return null;

  const threshold = data.threshold ?? 0.1;
  const maxAbs = Math.max(
    0.15,
    ...rows.flatMap((r) => [Math.abs(r.before ?? 0), Math.abs(r.after ?? 0)])
  );
  const domain = Math.ceil(maxAbs * 10) / 10;
  const pos = (v: number | null) =>
    v === null || !Number.isFinite(v) ? 50 : ((v + domain) / (2 * domain)) * 100;

  return (
    <div>
      <div className="mb-3 flex items-center gap-5 text-[11px] text-ink-3">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full ring-1 ring-ink-3" /> before matching
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-accent" /> after matching
        </span>
        <span className="ml-auto">|SMD| &lt; {threshold} is balanced</span>
      </div>

      <div className="space-y-1.5">
        {rows.map((r) => {
          const bad = Math.abs(r.after ?? 0) >= threshold;
          return (
            <div key={r.label} className="group grid grid-cols-[minmax(0,9rem)_1fr_4.5rem] items-center gap-3">
              <span className="truncate text-[12px] text-ink-2" title={r.label}>
                {r.label}
              </span>
              <div className="relative h-6 rounded bg-surface-2">
                {/* Balanced band */}
                <div
                  className="absolute inset-y-0 bg-pass-soft"
                  style={{
                    left: `${pos(-threshold)}%`,
                    width: `${pos(threshold) - pos(-threshold)}%`,
                  }}
                />
                <div className="absolute inset-y-1 left-1/2 w-px bg-border-strong" />
                {/* Connector */}
                <div
                  className="absolute top-1/2 h-px -translate-y-1/2 bg-border-strong"
                  style={{
                    left: `${Math.min(pos(r.before), pos(r.after))}%`,
                    width: `${Math.abs(pos(r.after) - pos(r.before))}%`,
                  }}
                />
                <span
                  className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-paper ring-1 ring-ink-3"
                  style={{ left: `${pos(r.before)}%` }}
                  title={`before: ${sig(r.before, 3)}`}
                />
                <span
                  className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
                  style={{
                    left: `${pos(r.after)}%`,
                    background: bad ? "var(--warn)" : "var(--accent)",
                  }}
                  title={`after: ${sig(r.after, 3)}`}
                />
              </div>
              <span
                className="tnum text-right text-[11px]"
                style={{ color: bad ? "var(--warn)" : "var(--ink-3)" }}
              >
                {sig(r.after, 3)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex justify-between text-[10px] text-ink-3 tnum">
        <span>−{domain.toFixed(1)}</span>
        <span>0</span>
        <span>+{domain.toFixed(1)}</span>
      </div>
    </div>
  );
}
