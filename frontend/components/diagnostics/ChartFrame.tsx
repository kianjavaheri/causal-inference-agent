"use client";

import { ReactNode } from "react";

/** Shared chart chrome: fixed height, responsive width, a caption slot underneath. */
export function ChartFrame({
  height = 280,
  children,
  caption,
}: {
  height?: number;
  children: ReactNode;
  caption?: ReactNode;
}) {
  return (
    <div>
      <div style={{ height }} className="w-full">
        {children}
      </div>
      {caption ? (
        <p className="mt-3 text-[13px] leading-relaxed text-ink-3">{caption}</p>
      ) : null}
    </div>
  );
}

export const axisStyle = {
  fontSize: 11,
  fill: "var(--ink-3)",
  fontFamily: "var(--font-mono)",
};

export const gridStyle = { stroke: "var(--grid)", strokeDasharray: "2 4" };

export function tooltipStyle() {
  return {
    contentStyle: {
      background: "var(--surface)",
      border: "1px solid var(--border-strong)",
      borderRadius: 8,
      fontSize: 12,
      boxShadow: "0 8px 24px rgb(0 0 0 / 0.10)",
      color: "var(--ink)",
    },
    labelStyle: { color: "var(--ink-2)", fontWeight: 600, marginBottom: 4 },
    itemStyle: { color: "var(--ink)" },
  };
}
