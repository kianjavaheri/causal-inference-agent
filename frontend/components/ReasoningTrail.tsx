"use client";

import type { Plan, TrailStep } from "@/lib/types";

function Dot({ status }: { status: TrailStep["status"] }) {
  if (status === "running") {
    return (
      <span className="relative flex h-3 w-3">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-3 w-3 rounded-full bg-accent" />
      </span>
    );
  }
  if (status === "error") {
    return (
      <span
        className="flex h-3 w-3 items-center justify-center rounded-full"
        style={{ background: "var(--fail)" }}
      />
    );
  }
  return (
    <span className="flex h-3 w-3 items-center justify-center rounded-full bg-accent">
      <svg viewBox="0 0 12 12" className="h-2 w-2" fill="none" stroke="white" strokeWidth={2.5}>
        <path d="M2.5 6.2 4.8 8.5 9.5 3.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

/**
 * The agent's working notes, streamed in as each backend step completes.
 * The plan step additionally shows the model's own reasoning trace.
 */
export function ReasoningTrail({
  steps,
  plan,
  error,
}: {
  steps: TrailStep[];
  plan: Plan | null;
  error: string | null;
}) {
  if (!steps.length) return null;

  return (
    <div className="rounded-xl border border-border-base bg-surface p-5 sm:p-6">
      <div className="mb-5 flex items-baseline justify-between">
        <h3 className="text-[13px] font-medium uppercase tracking-wider text-ink-3">
          Reasoning trail
        </h3>
        {plan ? (
          <span className="text-[11.5px] text-ink-3">
            {plan.llm_used ? "planned with Claude" : "planned by the identifiability engine"}
          </span>
        ) : null}
      </div>

      <ol className="relative space-y-6 pl-7">
        <span
          className="absolute left-[5px] top-2 w-px bg-border-base"
          style={{ height: "calc(100% - 1.5rem)" }}
          aria-hidden
        />
        {steps.map((step, i) => (
          <li key={`${step.step}-${i}`} className="animate-rise relative">
            <span className="absolute -left-7 top-1">
              <Dot status={step.status} />
            </span>

            <div className="flex flex-wrap items-baseline gap-x-3">
              <h4 className="text-[14.5px] font-semibold tracking-tight text-ink">
                {step.title}
              </h4>
              {step.finishedAt ? (
                <span className="tnum text-[11px] text-ink-3">
                  {step.finishedAt - step.startedAt < 100
                    ? "<0.1s"
                    : `${((step.finishedAt - step.startedAt) / 1000).toFixed(1)}s`}
                </span>
              ) : (
                <span className="animate-pulse-soft text-[11px] text-ink-3">working…</span>
              )}
            </div>

            {step.detail && step.status === "running" ? (
              <p className="mt-1 text-[13px] text-ink-3">{step.detail}</p>
            ) : null}

            {step.summary ? (
              <p className="mt-1.5 max-w-3xl text-[13.5px] leading-relaxed text-ink-2">
                {step.summary}
              </p>
            ) : null}

            {step.step === "plan" && step.status === "done" && plan ? (
              <ul className="mt-3 space-y-1.5 border-l-2 border-accent-soft pl-4">
                {plan.reasoning_trace.map((t, j) => (
                  <li key={j} className="text-[12.5px] leading-relaxed text-ink-3">
                    {t}
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ol>

      {error ? (
        <div
          className="mt-5 rounded-lg px-4 py-3 text-[13px] leading-relaxed"
          style={{ color: "var(--fail)", background: "var(--fail-soft)" }}
        >
          {error}
        </div>
      ) : null}
    </div>
  );
}
