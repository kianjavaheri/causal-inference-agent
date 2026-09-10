"use client";

import type { ExecutionResult, Report } from "@/lib/types";
import { count, pval, sig, titleCase } from "@/lib/format";
import { DiagnosticCard } from "./diagnostics/DiagnosticCard";

const CONFIDENCE = {
  high: { fg: "var(--pass)", bg: "var(--pass-soft)", label: "High confidence" },
  moderate: { fg: "var(--warn)", bg: "var(--warn-soft)", label: "Moderate confidence" },
  low: { fg: "var(--fail)", bg: "var(--fail-soft)", label: "Low confidence" },
} as const;

/** A confidence interval drawn to scale, so its width and its distance from zero are visible. */
function IntervalBar({ lo, hi, point }: { lo: number; hi: number; point: number }) {
  const span = Math.max(Math.abs(lo), Math.abs(hi), Math.abs(point)) * 1.25 || 1;
  const pos = (v: number) => ((v + span) / (2 * span)) * 100;
  const crossesZero = lo <= 0 && 0 <= hi;

  return (
    <div className="mt-5">
      <div className="relative h-9">
        <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-border-base" />
        <div
          className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full"
          style={{
            left: `${pos(lo)}%`,
            width: `${pos(hi) - pos(lo)}%`,
            background: crossesZero ? "var(--warn)" : "var(--accent)",
            opacity: 0.35,
          }}
        />
        <div
          className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-paper"
          style={{ left: `${pos(point)}%`, background: crossesZero ? "var(--warn)" : "var(--accent)" }}
        />
        <div
          className="absolute inset-y-1 w-px"
          style={{ left: `${pos(0)}%`, background: "var(--ink-3)" }}
        />
        <span
          className="absolute top-0 -translate-x-1/2 text-[10px] text-ink-3"
          style={{ left: `${pos(0)}%` }}
        >
          0
        </span>
      </div>
      <div className="tnum flex justify-between text-[11px] text-ink-3">
        <span>{sig(lo, 4)}</span>
        <span className={crossesZero ? "" : "font-medium"} style={{ color: crossesZero ? "var(--warn)" : undefined }}>
          {crossesZero ? "interval includes zero" : "interval excludes zero"}
        </span>
        <span>{sig(hi, 4)}</span>
      </div>
    </div>
  );
}

export function Results({
  result,
  report,
}: {
  result: ExecutionResult;
  report: Report | null;
}) {
  const e = result.estimate;
  const conf = report ? CONFIDENCE[report.confidence] : null;
  const roles = (result.specification.roles ?? {}) as Record<string, unknown>;

  return (
    <div className="space-y-6">
      {/* --- headline estimate --- */}
      <section className="animate-rise rounded-xl border border-border-base bg-surface p-6 sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wider text-ink-3">
              {result.estimand}
            </p>
            <div className="tnum mt-2 text-[44px] font-semibold leading-none tracking-tight text-ink sm:text-[56px]">
              {e.point > 0 ? "+" : ""}
              {sig(e.point, 4)}
            </div>
            <p className="mt-2 text-[13.5px] text-ink-2">
              in <span className="font-medium">{e.units}</span>, via {result.method_label}
            </p>
          </div>
          {conf ? (
            <span
              className="rounded-full px-3 py-1.5 text-[12px] font-medium"
              style={{ color: conf.fg, background: conf.bg }}
            >
              {conf.label}
            </span>
          ) : null}
        </div>

        {e.ci_low !== null && e.ci_high !== null && e.ci_low !== undefined && e.ci_high !== undefined ? (
          <IntervalBar lo={e.ci_low} hi={e.ci_high} point={e.point} />
        ) : null}

        <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 border-t border-border-base pt-5 sm:grid-cols-4">
          {[
            { k: "95% CI", v: `[${sig(e.ci_low, 4)}, ${sig(e.ci_high, 4)}]` },
            { k: "Std. error", v: sig(e.se, 4) },
            { k: "p-value", v: pval(e.p_value) },
            {
              k: "Observations",
              v:
                count(e.n_obs) +
                (e.n_treated != null ? ` · ${count(e.n_treated)} treated` : ""),
            },
          ].map((f) => (
            <div key={f.k}>
              <dt className="text-[10.5px] font-medium uppercase tracking-wider text-ink-3">
                {f.k}
              </dt>
              <dd className="tnum mt-1 text-[14px] text-ink">{f.v}</dd>
            </div>
          ))}
        </dl>

        {result.warnings.length ? (
          <ul className="mt-5 space-y-2 border-t border-border-base pt-4">
            {result.warnings.map((w, i) => (
              <li
                key={i}
                className="flex gap-2 text-[13px] leading-relaxed"
                style={{ color: "var(--warn)" }}
              >
                <span aria-hidden>⚠</span>
                <span>{w}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      {/* --- written report --- */}
      {report ? (
        <section className="animate-rise rounded-xl border border-border-base bg-surface p-6 sm:p-8">
          <h3 className="text-[19px] font-semibold leading-snug tracking-tight text-ink">
            {report.headline}
          </h3>
          <div className="mt-4 space-y-4 text-[14.5px] leading-[1.7] text-ink-2">
            <p>{report.interpretation}</p>
            <div>
              <h4 className="mb-1 text-[11px] font-medium uppercase tracking-wider text-ink-3">
                How this design works
              </h4>
              <p>{report.method_explanation}</p>
            </div>
            <div>
              <h4 className="mb-1 text-[11px] font-medium uppercase tracking-wider text-ink-3">
                What had to be true
              </h4>
              <p>{report.assumptions_discussion}</p>
            </div>
          </div>

          {report.caveats.length ? (
            <div className="mt-6 rounded-lg bg-surface-2 p-4">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-ink-3">
                Caveats
              </h4>
              <ul className="mt-2 space-y-2">
                {report.caveats.map((c, i) => (
                  <li key={i} className="flex gap-2 text-[13.5px] leading-relaxed text-ink-2">
                    <span className="mt-[8px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <p className="mt-4 text-[11.5px] text-ink-3">
            {report.llm_used
              ? "Interpretation written by Claude from the estimate and its diagnostics."
              : "Interpretation generated from the estimate and its diagnostics (no model configured)."}
          </p>
        </section>
      ) : null}

      {/* --- diagnostics --- */}
      <section>
        <h3 className="mb-3 text-[13px] font-medium uppercase tracking-wider text-ink-3">
          Assumption checks
        </h3>
        <div className="space-y-4">
          {result.diagnostics.map((d) => (
            <DiagnosticCard key={d.id} diagnostic={d} />
          ))}
        </div>
      </section>

      {/* --- specification --- */}
      <section className="rounded-xl border border-border-base bg-surface p-5">
        <h3 className="mb-3 text-[13px] font-medium uppercase tracking-wider text-ink-3">
          Specification
        </h3>
        <dl className="grid gap-x-8 gap-y-2.5 sm:grid-cols-2">
          {Object.entries(result.specification)
            .filter(([k]) => k !== "roles")
            .map(([k, v]) => (
              <div key={k} className="flex flex-wrap gap-x-2 text-[12.5px]">
                <dt className="text-ink-3">{titleCase(k)}:</dt>
                <dd className="min-w-0 break-words font-mono text-[11.5px] text-ink-2">
                  {Array.isArray(v) ? (v.length ? v.join(", ") : "none") : String(v)}
                </dd>
              </div>
            ))}
        </dl>
        {Object.keys(roles).length ? (
          <div className="mt-4 border-t border-border-base pt-3">
            <dt className="mb-2 text-[10.5px] font-medium uppercase tracking-wider text-ink-3">
              Column roles
            </dt>
            <div className="flex flex-wrap gap-2">
              {Object.entries(roles)
                .filter(([, v]) => v && (!Array.isArray(v) || v.length))
                .map(([k, v]) => (
                  <span
                    key={k}
                    className="rounded-md bg-surface-2 px-2 py-1 text-[11.5px] text-ink-2"
                  >
                    <span className="text-ink-3">{titleCase(k)}</span>{" "}
                    <span className="font-mono">{Array.isArray(v) ? v.join(", ") : String(v)}</span>
                  </span>
                ))}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}
