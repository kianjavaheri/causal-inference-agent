"use client";

import { useCallback, useRef, useState } from "react";
import type { DataProfile, SampleDataset } from "@/lib/types";
import { API_BASE, loadSample, uploadCsv } from "@/lib/api";

export function UploadPanel({
  samples,
  onLoaded,
  disabled,
}: {
  samples: SampleDataset[];
  onLoaded: (profile: DataProfile, question?: string) => void;
  disabled?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith(".csv")) {
        setError("Please choose a .csv file.");
        return;
      }
      setError(null);
      setBusy("upload");
      try {
        onLoaded(await uploadCsv(file));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setBusy(null);
      }
    },
    [onLoaded]
  );

  const handleSample = async (s: SampleDataset) => {
    setError(null);
    setBusy(s.id);
    try {
      onLoaded(await loadSample(s.id), s.question);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the sample.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (disabled) return;
          const file = e.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        aria-label="Upload a CSV file"
        className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
          dragging
            ? "border-accent bg-accent-soft"
            : "border-border-strong bg-surface hover:border-accent hover:bg-surface-2"
        } ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = "";
          }}
        />
        <svg
          className="mx-auto mb-3 h-8 w-8 text-ink-3"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L8 8m4-4 4 4" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
        <p className="text-[15px] font-medium text-ink">
          {busy === "upload" ? "Reading your file…" : "Drop a CSV here, or click to browse"}
        </p>
        <p className="mt-1 text-[13px] text-ink-3">
          One row per observation. Column names matter — the agent reads them.
        </p>
      </div>

      {error ? (
        <p
          className="rounded-lg px-3 py-2 text-[13px]"
          style={{ color: "var(--fail)", background: "var(--fail-soft)" }}
        >
          {error}
        </p>
      ) : null}

      <div>
        <div className="mb-3 flex items-baseline justify-between">
          <h3 className="text-[13px] font-medium uppercase tracking-wider text-ink-3">
            Or try a dataset with a known answer
          </h3>
          <span className="text-[12px] text-ink-3">
            each has a treatment effect planted in it
          </span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {samples.map((s) => (
            <button
              key={s.id}
              onClick={() => handleSample(s)}
              disabled={disabled || busy !== null}
              className="group cursor-pointer rounded-xl border border-border-base bg-surface p-4 text-left transition-all hover:border-accent hover:shadow-[0_2px_12px_rgb(0_0_0/0.05)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-[14px] font-semibold text-ink">{s.name}</span>
                <span
                  className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide"
                  style={{ color: "var(--accent-ink)", background: "var(--accent-soft)" }}
                >
                  {s.expected_method_label}
                </span>
              </div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-2">{s.description}</p>
              <p className="mt-2 text-[11.5px] text-ink-3">
                True effect:{" "}
                <span className="tnum">
                  {s.true_effect > 0 ? "+" : ""}
                  {s.true_effect}
                </span>
                {busy === s.id ? <span className="ml-2 animate-pulse-soft">loading…</span> : null}
                <a
                  href={`${API_BASE}/samples/${s.id}/download`}
                  onClick={(e) => e.stopPropagation()}
                  className="ml-2 cursor-pointer text-accent underline-offset-2 hover:underline"
                >
                  download csv
                </a>
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
