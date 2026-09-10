"use client";

import {
  CartesianGrid, ComposedChart, ErrorBar, Line, ReferenceLine, ResponsiveContainer,
  Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";
import { sig } from "@/lib/format";
import type { EventStudyData, EventStudyPoint } from "@/lib/types";

/**
 * The parallel-trends test. Each point is the treated-control gap in that period,
 * relative to the period just before treatment. Pre-period points should sit on zero.
 */
export function EventStudyChart({ data }: { data: EventStudyData }) {
  const points: EventStudyPoint[] = data.points ?? [];
  if (!points.length) return <Empty />;

  const rows = points.map((p) => ({
    ...p,
    coef: p.coef ?? 0,
    // Recharts ErrorBar wants [downwards, upwards] offsets from the value.
    err: [
      Math.max(0, (p.coef ?? 0) - (p.ci_low ?? p.coef ?? 0)),
      Math.max(0, (p.ci_high ?? p.coef ?? 0) - (p.coef ?? 0)),
    ] as [number, number],
  }));

  return (
    <ChartFrame
      caption={
        data.p_value !== undefined && data.p_value !== null ? (
          <>
            Joint test that every pre-treatment coefficient is zero:{" "}
            <span className="tnum">F = {sig(data.f_stat, 3)}</span>, p ={" "}
            <span className="tnum">{sig(data.p_value, 3)}</span>. Period −1 is the reference
            and is fixed at zero by construction.
          </>
        ) : null
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis
            dataKey="period"
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            label={{
              value: "periods relative to treatment",
              position: "insideBottom",
              offset: -14,
              style: { ...axisStyle, fontSize: 11 },
            }}
          />
          <YAxis
            tick={axisStyle}
            tickLine={false}
            axisLine={false}
            width={54}
            tickFormatter={(v) => sig(v, 2)}
          />
          <ReferenceLine y={0} stroke="var(--border-strong)" strokeWidth={1} />
          <ReferenceLine
            x={-0.5}
            stroke="var(--accent)"
            strokeDasharray="4 4"
            label={{
              value: "treatment",
              position: "top",
              style: { fontSize: 10, fill: "var(--accent)" },
            }}
          />
          <Tooltip
            {...tooltipStyle()}
            formatter={(v: unknown, name: unknown) =>
              name === "coef" ? [sig(v as number, 4), "estimate"] : null
            }
            labelFormatter={(l) => `period ${l}`}
          />
          <Line
            type="linear"
            dataKey="coef"
            stroke="var(--ink-3)"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />
          <Scatter dataKey="coef" isAnimationActive={false}>
            <ErrorBar dataKey="err" width={4} strokeWidth={1.5} stroke="var(--accent)" />
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

function Empty() {
  return (
    <p className="rounded-lg border border-dashed border-border-base bg-surface-2 p-4 text-[13px] text-ink-3">
      Not enough pre-treatment periods to run an event study.
    </p>
  );
}
