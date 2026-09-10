"use client";

import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";
import { sig } from "@/lib/format";
import type { PathData } from "@/lib/types";

/** Two outcome paths over time: treated vs control (DiD) or treated vs synthetic (SC). */
export function PathChart({ data }: { data: PathData }) {
  const series = data.series ?? [];
  if (!series.length) return null;

  const times = series[0].points.map((p) => p.time);
  const rows = times.map((t, i) => {
    const row: Record<string, string | number | null> = { time: t };
    for (const s of series) row[s.name] = s.points[i]?.value ?? null;
    return row;
  });

  const colors = ["var(--accent)", "var(--ink-3)"];

  return (
    <ChartFrame
      height={300}
      caption={
        data.pre_rmse !== undefined && data.pre_rmse !== null ? (
          <>
            Pre-treatment fit error: <span className="tnum">{sig(data.pre_rmse, 4)}</span> RMSE.
            The two lines are matched on the pre-period only — everything after the dashed line
            is out of sample.
          </>
        ) : null
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 14, right: 12, bottom: 28, left: 4 }}>
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
            domain={["auto", "auto"]}
            tickFormatter={(v) => sig(v, 3)}
          />
          {data.treatment_time ? (
            <ReferenceLine
              x={data.treatment_time}
              stroke="var(--accent)"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: "treatment",
                position: "insideTopLeft",
                offset: 8,
                style: { fontSize: 10, fill: "var(--accent)" },
              }}
            />
          ) : null}
          <Tooltip {...tooltipStyle()} formatter={(v: unknown) => sig(v as number, 4)} />
          <Legend
            verticalAlign="top"
            align="right"
            height={34}
            iconType="plainline"
            wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }}
          />
          {series.map((s, i) => (
            <Line
              key={s.name}
              type="monotone"
              dataKey={s.name}
              stroke={colors[i % colors.length]}
              strokeWidth={2.25}
              strokeDasharray={i === 1 ? "5 3" : undefined}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
