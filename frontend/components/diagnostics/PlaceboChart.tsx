"use client";

import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";
import { sig } from "@/lib/format";
import type { PlaceboData } from "@/lib/types";

/**
 * Placebo-in-space: the treated unit's gap drawn against the gap you get by pretending
 * each donor was treated. The real line should stand apart from the grey ones.
 */
export function PlaceboChart({ data }: { data: PlaceboData }) {
  const treated = data.treated_gaps ?? [];
  const placebos = data.placebos ?? [];
  if (!treated.length) return null;

  const rows = treated.map((p, i) => {
    const row: Record<string, string | number | null> = { time: p.time, __treated: p.value };
    placebos.forEach((pl) => {
      row[pl.unit] = pl.points[i]?.value ?? null;
    });
    return row;
  });

  return (
    <ChartFrame
      height={300}
      caption={
        <>
          <span className="tnum">{placebos.length}</span> placebo runs in grey, the treated unit
          in colour. p ={" "}
          <span className="tnum">{sig(data.p_value, 3)}</span> is the share of units whose
          post-treatment divergence is at least as extreme, relative to how well each was
          fitted beforehand.
        </>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 28, left: 4 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis
            dataKey="time"
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            minTickGap={28}
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
            <ReferenceLine x={data.treatment_time} stroke="var(--accent)" strokeDasharray="4 4" />
          ) : null}
          <Tooltip
            {...tooltipStyle()}
            formatter={(v: unknown, n: unknown) => [
              sig(v as number, 4),
              n === "__treated" ? data.treated_name : String(n),
            ]}
          />
          {placebos.map((pl) => (
            <Line
              key={pl.unit}
              type="monotone"
              dataKey={pl.unit}
              stroke="var(--ink-3)"
              strokeOpacity={pl.poor_fit ? 0.12 : 0.3}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
            />
          ))}
          <Line
            type="monotone"
            dataKey="__treated"
            stroke="var(--accent)"
            strokeWidth={2.75}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
