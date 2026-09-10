"use client";

import {
  Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";
import { sig } from "@/lib/format";
import type { GapData } from "@/lib/types";

/** Treated minus synthetic. Flat at zero before treatment, stepping away after it. */
export function GapChart({ data }: { data: GapData }) {
  const points = data.points ?? [];
  if (!points.length) return null;

  return (
    <ChartFrame height={240}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={points} margin={{ top: 8, right: 12, bottom: 28, left: 4 }}>
          <defs>
            <linearGradient id="gapFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.28} />
              <stop offset="100%" stopColor="var(--accent)" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis
            dataKey="time"
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            minTickGap={24}
            label={{
              value: data.x_label,
              position: "insideBottom",
              offset: -16,
              style: { ...axisStyle, fontSize: 11 },
            }}
          />
          <YAxis
            tick={axisStyle}
            tickLine={false}
            axisLine={false}
            width={58}
            tickFormatter={(v) => sig(v, 3)}
          />
          <ReferenceLine y={0} stroke="var(--border-strong)" />
          {data.treatment_time ? (
            <ReferenceLine
              x={data.treatment_time}
              stroke="var(--accent)"
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
          ) : null}
          <Tooltip
            {...tooltipStyle()}
            formatter={(v: unknown) => [sig(v as number, 4), data.y_label ?? "gap"]}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke="var(--accent)"
            strokeWidth={2}
            fill="url(#gapFill)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
