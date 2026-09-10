"use client";

import {
  Bar, BarChart, Cell, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { ChartFrame, axisStyle, gridStyle, tooltipStyle } from "./ChartFrame";
import { sig } from "@/lib/format";
import type { OverlapData } from "@/lib/types";

/**
 * Two shapes share this component:
 *  - propensity-score overlap, drawn as a mirrored histogram (treated above the axis,
 *    control below) so the two distributions can be compared shape-for-shape rather
 *    than stacked on top of each other;
 *  - the RDD manipulation test, a single histogram of the running variable, with bars
 *    either side of the cutoff coloured differently.
 *
 * Note both use a *category* x-axis, because Recharts' BarChart does. A ReferenceLine
 * therefore has to name an actual category value, not an arbitrary number -- hence
 * `cutoffCategory` below rather than passing the raw cutoff.
 */
export function OverlapChart({ data }: { data: OverlapData }) {
  const raw = data.histogram ?? [];
  if (!raw.length) return null;
  const twoGroup = raw[0].treated !== undefined;

  if (twoGroup) {
    const rows = raw.map((r) => ({ ...r, controlNeg: -(r.control ?? 0) }));
    return (
      <ChartFrame
        height={260}
        caption="Treated units above the axis, control units below. Wherever one side has mass and the other does not, there is no comparison group to match against."
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 26, left: 4 }} barGap={0}>
            <CartesianGrid {...gridStyle} vertical={false} />
            <XAxis
              dataKey="x"
              tick={axisStyle}
              tickLine={false}
              axisLine={{ stroke: "var(--border-strong)" }}
              tickFormatter={(v) => sig(v, 2)}
              minTickGap={20}
              label={{
                value: data.x_label,
                position: "insideBottom",
                offset: -14,
                style: { ...axisStyle, fontSize: 11 },
              }}
            />
            <YAxis
              tick={axisStyle}
              tickLine={false}
              axisLine={false}
              width={44}
              tickFormatter={(v) => String(Math.abs(v as number))}
            />
            <ReferenceLine y={0} stroke="var(--border-strong)" />
            <Tooltip
              {...tooltipStyle()}
              formatter={(v: unknown, n: unknown) => [Math.abs(v as number), String(n)]}
              labelFormatter={(l) => `score ≈ ${sig(l as number, 3)}`}
            />
            <Legend
              verticalAlign="top"
              align="right"
              height={26}
              wrapperStyle={{ fontSize: 12, color: "var(--ink-2)" }}
            />
            <Bar dataKey="treated" name="treated" fill="var(--accent)" fillOpacity={0.85} />
            <Bar dataKey="controlNeg" name="control" fill="var(--ink-3)" fillOpacity={0.6} />
          </BarChart>
        </ResponsiveContainer>
      </ChartFrame>
    );
  }

  // Single histogram. The bars are coloured by side rather than marked with a
  // ReferenceLine: on a band axis a reference line snaps to a bin's centre, which would
  // sit half a bin off the true cutoff -- a misleading thing to draw on a test that is
  // entirely about what happens exactly at the threshold. The backend forces a bin edge
  // at the cutoff, so the colour change falls precisely on it.
  const cutoff: number | undefined = data.cutoff ?? undefined;

  return (
    <ChartFrame
      height={220}
      caption={
        <>
          Bars left of the colour change fall below the cutoff
          {cutoff !== undefined && cutoff !== null ? (
            <> (<span className="tnum">{sig(cutoff, 4)}</span>)</>
          ) : null}
          ; bars right of it are at or above it. A pile-up on one side would mean units are
          steering themselves across the threshold. The counts should pass through smoothly.
        </>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={raw} margin={{ top: 8, right: 12, bottom: 26, left: 4 }}>
          <CartesianGrid {...gridStyle} vertical={false} />
          <XAxis
            dataKey="x"
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            tickFormatter={(v) => sig(v, 3)}
            minTickGap={20}
            label={{
              value: data.x_label,
              position: "insideBottom",
              offset: -14,
              style: { ...axisStyle, fontSize: 11 },
            }}
          />
          <YAxis tick={axisStyle} tickLine={false} axisLine={false} width={44} />
          <Tooltip
            {...tooltipStyle()}
            labelFormatter={(l) => sig(l as number, 4)}
            formatter={(v: unknown) => [v as number, "observations"]}
          />
          <Bar dataKey="count" name="observations">
            {raw.map((r, i) => (
              <Cell
                key={i}
                fill={r.side === "right" ? "var(--accent)" : "var(--ink-3)"}
                fillOpacity={r.side === "right" ? 0.8 : 0.55}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
