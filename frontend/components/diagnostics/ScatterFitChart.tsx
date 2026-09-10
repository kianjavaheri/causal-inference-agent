"use client";

import {
  CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";
import { sig } from "@/lib/format";
import type { ScatterFitData } from "@/lib/types";

/** The RDD jump plot: binned means plus the local-linear fits either side of the cutoff. */
export function ScatterFitChart({ data }: { data: ScatterFitData }) {
  const bins = data.bins ?? [];
  const leftFit = data.left_fit ?? [];
  const rightFit = data.right_fit ?? [];
  const { cutoff, bandwidth: bw } = data;
  if (!bins.length) return null;

  const left = bins.filter((b) => b.side === "left");
  const right = bins.filter((b) => b.side === "right");
  // The treated side is not always the right one: benefits below an income line, say.
  const treatedRight = (data.treated_side ?? "right") === "right";
  const rightColor = treatedRight ? "var(--accent)" : "var(--ink-3)";
  const leftColor = treatedRight ? "var(--ink-3)" : "var(--accent)";

  return (
    <ChartFrame
      height={300}
      caption={
        <>
          Cutoff at <span className="tnum">{sig(cutoff, 4)}</span>; the shaded band is the
          estimation bandwidth (±<span className="tnum">{sig(bw, 3)}</span>). Each dot averages
          the outcome within a narrow slice of {data.x_label}.
        </>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis
            type="number"
            dataKey="x"
            domain={["dataMin", "dataMax"]}
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            tickFormatter={(v) => sig(v, 3)}
            label={{
              value: data.x_label,
              position: "insideBottom",
              offset: -14,
              style: { ...axisStyle, fontSize: 11 },
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={["auto", "auto"]}
            tick={axisStyle}
            tickLine={false}
            axisLine={false}
            width={54}
            tickFormatter={(v) => sig(v, 3)}
          />
          <ZAxis range={[26, 26]} />
          <ReferenceArea
            x1={cutoff - bw}
            x2={cutoff + bw}
            fill="var(--accent)"
            fillOpacity={0.06}
          />
          <ReferenceLine
            x={cutoff}
            stroke="var(--accent)"
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />
          <Tooltip
            {...tooltipStyle()}
            formatter={(v: unknown, n: unknown) =>
              [sig(v as number, 4), n === "y" ? data.y_label : String(n)]
            }
            labelFormatter={() => ""}
          />
          <Scatter
            name={treatedRight ? "below cutoff" : "treated (below cutoff)"}
            data={left}
            fill={leftColor}
            fillOpacity={0.85}
            isAnimationActive={false}
          />
          <Scatter
            name={treatedRight ? "treated (at or above cutoff)" : "at or above cutoff"}
            data={right}
            fill={rightColor}
            fillOpacity={0.85}
            isAnimationActive={false}
          />
          <Line
            data={leftFit}
            dataKey="y"
            type="linear"
            stroke={leftColor}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            data={rightFit}
            dataKey="y"
            type="linear"
            stroke={rightColor}
            strokeWidth={2.5}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
