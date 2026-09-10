"use client";

import { pval, sig, titleCase } from "@/lib/format";
import type { TableCell, TableData } from "@/lib/types";

const NUMERIC_KEYS = new Set([
  "value", "se", "p_value", "estimate", "ci_low", "ci_high", "bandwidth",
  "before", "after", "n", "matched", "multiplier",
]);

/** Generic diagnostic table. Columns come from the payload; numbers are right-aligned. */
export function DataTable({ data }: { data: TableData }) {
  const rows = data.rows ?? [];
  if (!rows.length) return null;

  const columns: string[] =
    data.columns ??
    Array.from(new Set(rows.flatMap((r) => Object.keys(r)))).filter((c) => c !== "side");

  const fmt = (key: string, v: TableCell) => {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return key === "p_value" ? pval(v) : sig(v, 4);
    return String(v);
  };

  return (
    <div className="scroll-x -mx-1 px-1">
      <table className="w-full min-w-[22rem] border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-border-base">
            {columns.map((c) => (
              <th
                key={c}
                className={`pb-2 pr-4 text-[11px] font-medium uppercase tracking-wider text-ink-3 ${
                  NUMERIC_KEYS.has(c) ? "text-right" : "text-left"
                }`}
              >
                {c === "label" ? "" : titleCase(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-border-base/60 last:border-0">
              {columns.map((c) => (
                <td
                  key={c}
                  className={`py-2 pr-4 ${
                    NUMERIC_KEYS.has(c)
                      ? "tnum text-right text-ink"
                      : "text-left text-ink-2"
                  }`}
                >
                  {fmt(c, r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
