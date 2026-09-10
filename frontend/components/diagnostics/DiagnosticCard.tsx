"use client";

import { useState } from "react";
import type { Diagnostic } from "@/lib/types";
import { verdictColor } from "@/lib/format";
import { EventStudyChart } from "./EventStudyChart";
import { ScatterFitChart } from "./ScatterFitChart";
import { PathChart } from "./PathChart";
import { GapChart } from "./GapChart";
import { PlaceboChart } from "./PlaceboChart";
import { BalancePlot } from "./BalancePlot";
import { OverlapChart } from "./OverlapChart";
import { CompositionBar } from "./CompositionBar";
import { DataTable } from "./DataTable";

/** Dispatch on `kind`, which narrows `d.data` to that variant's payload type. */
function Body({ d }: { d: Diagnostic }) {
  switch (d.kind) {
    case "event_study": return <EventStudyChart data={d.data} />;
    case "scatter_fit": return <ScatterFitChart data={d.data} />;
    case "path": return <PathChart data={d.data} />;
    case "gap": return <GapChart data={d.data} />;
    case "placebo_distribution": return <PlaceboChart data={d.data} />;
    case "balance": return <BalancePlot data={d.data} />;
    case "overlap": return <OverlapChart data={d.data} />;
    case "composition": return <CompositionBar data={d.data} />;
    case "table": return <DataTable data={d.data} />;
    // Text diagnostics carry no payload; the summary and detail say everything.
    case "text": return null;
  }
}

export function DiagnosticCard({ diagnostic }: { diagnostic: Diagnostic }) {
  const [showDetail, setShowDetail] = useState(false);
  const v = verdictColor[diagnostic.verdict];

  return (
    <section className="animate-rise rounded-xl border border-border-base bg-surface p-5">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h4 className="text-[15px] font-semibold tracking-tight text-ink">
            {diagnostic.title}
          </h4>
          <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-ink-2">
            {diagnostic.summary}
          </p>
        </div>
        <span
          className="shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium"
          style={{ color: v.fg, background: v.bg }}
        >
          {v.label}
        </span>
      </header>

      <Body d={diagnostic} />

      {diagnostic.detail ? (
        <div className="mt-4 border-t border-border-base pt-3">
          <button
            onClick={() => setShowDetail((s) => !s)}
            className="cursor-pointer text-[12px] font-medium text-accent transition-opacity hover:opacity-70"
          >
            {showDetail ? "Hide" : "What am I looking at?"}
          </button>
          {showDetail ? (
            <p className="mt-2 max-w-2xl text-[13px] leading-relaxed text-ink-2">
              {diagnostic.detail}
            </p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
