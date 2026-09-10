"use client";

import type { Suggestions } from "@/lib/types";

/**
 * The question box, with suggestions generated from the dataset's own columns. The
 * suggestions matter: a user who does not know which designs their data can support
 * cannot phrase a question it can answer, and guessing wastes a run.
 */
export function QuestionForm({
  question,
  onChange,
  onRun,
  running,
  suggestions,
}: {
  question: string;
  onChange: (q: string) => void;
  onRun: () => void;
  running: boolean;
  suggestions: Suggestions | null;
}) {
  const chips = suggestions?.questions ?? [];

  return (
    <div className="rounded-xl border border-border-base bg-surface p-4">
      <textarea
        value={question}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onRun();
        }}
        rows={3}
        disabled={running}
        placeholder="e.g. Did the price change reduce customer churn?"
        className="w-full resize-none bg-transparent text-[15px] leading-relaxed text-ink outline-none placeholder:text-ink-3"
      />

      {chips.length ? (
        <div className="mt-3 border-t border-border-base pt-3">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-ink-3">
            Questions this data can answer
          </p>
          <div className="flex flex-wrap gap-2">
            {chips.map((q) => (
              <button
                key={q}
                onClick={() => onChange(q)}
                disabled={running}
                className="cursor-pointer rounded-full border border-border-base bg-surface-2 px-3 py-1.5 text-left text-[12.5px] text-ink-2 transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-border-base pt-3">
        <p className="text-[12px] text-ink-3">
          Name the outcome and the treatment — the agent matches your words to columns.
        </p>
        <button
          onClick={onRun}
          disabled={running || !question.trim()}
          className="cursor-pointer rounded-lg px-4 py-2 text-[13.5px] font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          style={{ background: "var(--accent)" }}
        >
          {running ? "Running…" : "Run the agent"}
        </button>
      </div>
    </div>
  );
}
