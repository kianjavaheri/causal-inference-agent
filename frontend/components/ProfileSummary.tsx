"use client";

import { useState } from "react";
import type { DataProfile } from "@/lib/types";
import { count } from "@/lib/format";

const TYPE_COLOR: Record<string, string> = {
  binary: "var(--accent)",
  continuous: "var(--pass)",
  integer: "var(--pass)",
  categorical: "var(--warn)",
  datetime: "var(--info)",
  identifier: "var(--ink-3)",
  text: "var(--ink-3)",
};

export function ProfileSummary({ profile }: { profile: DataProfile }) {
  const [open, setOpen] = useState(false);
  const s = profile.structure;

  const facts = [
    { label: "Rows", value: count(profile.n_rows) },
    { label: "Columns", value: count(profile.n_cols) },
    ...(s.n_units ? [{ label: "Units", value: count(s.n_units) }] : []),
    ...(s.n_periods ? [{ label: "Periods", value: count(s.n_periods) }] : []),
    {
      label: "Shape",
      value: s.is_panel ? "Panel" : s.has_repeated_cross_sections ? "Repeated cross-sections" : "Cross-section",
    },
  ];

  return (
    <div className="rounded-xl border border-border-base bg-surface">
      <div className="flex flex-wrap items-center gap-x-8 gap-y-3 px-5 py-4">
        {facts.map((f) => (
          <div key={f.label}>
            <div className="text-[10.5px] font-medium uppercase tracking-wider text-ink-3">
              {f.label}
            </div>
            <div className="tnum mt-0.5 text-[15px] font-semibold text-ink">{f.value}</div>
          </div>
        ))}
        <button
          onClick={() => setOpen((o) => !o)}
          className="ml-auto cursor-pointer text-[12px] font-medium text-accent transition-opacity hover:opacity-70"
        >
          {open ? "Hide columns" : `Inspect ${profile.n_cols} columns`}
        </button>
      </div>

      {open ? (
        <div className="scroll-x border-t border-border-base px-5 py-4">
          <table className="w-full min-w-[38rem] border-collapse text-[12.5px]">
            <thead>
              <tr className="border-b border-border-base text-[10.5px] uppercase tracking-wider text-ink-3">
                <th className="pb-2 pr-4 text-left font-medium">Column</th>
                <th className="pb-2 pr-4 text-left font-medium">Type</th>
                <th className="pb-2 pr-4 text-right font-medium">Distinct</th>
                <th className="pb-2 pr-4 text-right font-medium">Missing</th>
                <th className="pb-2 text-left font-medium">Sample values</th>
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr key={c.name} className="border-b border-border-base/50 last:border-0">
                  <td className="py-2 pr-4 font-medium text-ink">{c.name}</td>
                  <td className="py-2 pr-4">
                    <span
                      className="rounded px-1.5 py-0.5 text-[10.5px]"
                      style={{
                        color: TYPE_COLOR[c.semantic_type],
                        background: "var(--surface-2)",
                      }}
                    >
                      {c.semantic_type}
                    </span>
                  </td>
                  <td className="tnum py-2 pr-4 text-right text-ink-2">{count(c.n_unique)}</td>
                  <td
                    className="tnum py-2 pr-4 text-right"
                    style={{ color: c.missing_pct > 0 ? "var(--warn)" : "var(--ink-3)" }}
                  >
                    {c.missing_pct > 0 ? `${c.missing_pct}%` : "—"}
                  </td>
                  <td className="py-2 text-ink-3">
                    <span className="line-clamp-1">
                      {c.sample_values.slice(0, 3).map(String).join(", ")}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {s.notes.length ? (
        <div className="border-t border-border-base px-5 py-3">
          {s.notes.map((n, i) => (
            <p key={i} className="text-[12.5px] leading-relaxed text-ink-2">
              {n}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
