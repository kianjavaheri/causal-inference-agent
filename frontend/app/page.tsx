"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { UploadPanel } from "@/components/UploadPanel";
import { ProfileSummary } from "@/components/ProfileSummary";
import { ReasoningTrail } from "@/components/ReasoningTrail";
import { MethodComparison } from "@/components/MethodComparison";
import { Results } from "@/components/Results";
import { QuestionForm } from "@/components/QuestionForm";
import { DataDiagnosis } from "@/components/DataDiagnosis";
import {
  buildReport, executeMethod, fetchHealth, fetchSamples, fetchSuggestions, runPipeline,
} from "@/lib/api";
import type {
  DataProfile, ExecutionResult, MethodId, Plan, Report, SampleDataset, StepName,
  Suggestions, TrailStep,
} from "@/lib/types";

export default function Home() {
  const [samples, setSamples] = useState<SampleDataset[]>([]);
  const [llmOn, setLlmOn] = useState<boolean | null>(null);
  const [apiDown, setApiDown] = useState(false);

  const [profile, setProfile] = useState<DataProfile | null>(null);
  const [question, setQuestion] = useState("");
  const [steps, setSteps] = useState<TrailStep[]>([]);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestions | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const [health, s] = await Promise.all([fetchHealth(), fetchSamples()]);
        setLlmOn(health.llm_configured);
        setSamples(s);
      } catch {
        setApiDown(true);
      }
    })();
  }, []);

  const resetRun = () => {
    abortRef.current?.abort();
    setSteps([]);
    setPlan(null);
    setResult(null);
    setReport(null);
    setError(null);
  };

  const handleLoaded = (p: DataProfile, presetQuestion?: string) => {
    resetRun();
    setProfile(p);
    setSuggestions(null);
    if (presetQuestion) setQuestion(presetQuestion);
    // Work out what this dataset can answer before the user commits to a question.
    fetchSuggestions(p.session_id).then(setSuggestions).catch(() => setSuggestions(null));
  };

  /** Clear everything and return to the upload screen. */
  const startOver = () => {
    resetRun();
    setProfile(null);
    setQuestion("");
    setSuggestions(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const startStep = useCallback((step: StepName, title: string, detail?: string) => {
    setSteps((prev) => [
      ...prev.filter((s) => s.step !== step),
      { step, title, detail, status: "running", startedAt: Date.now() },
    ]);
  }, []);

  const finishStep = useCallback((step: StepName, title: string, summary: string) => {
    setSteps((prev) =>
      prev.map((s) =>
        s.step === step ? { ...s, title, summary, status: "done", finishedAt: Date.now() } : s
      )
    );
  }, []);

  const run = async () => {
    if (!profile || !question.trim() || running) return;
    resetRun();
    setRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await runPipeline(
        profile.session_id,
        question.trim(),
        {
          onStepStart: startStep,
          onProfile: (p, summary) => {
            setProfile(p);
            finishStep("profile", "Profiled the dataset", summary);
          },
          onPlan: (p, summary) => {
            setPlan(p);
            finishStep("plan", `Chose ${p.chosen_label}`, summary);
          },
          onExecute: (r, summary) => {
            setResult(r);
            finishStep("execute", `Estimated ${r.method_label}`, summary);
          },
          onReport: (r, summary) => {
            setReport(r);
            finishStep("report", "Wrote up the findings", summary);
          },
          onError: (message) => {
            setError(message);
            if (profile) {
              fetchSuggestions(profile.session_id).then(setSuggestions).catch(() => {});
            }
            setSteps((prev) =>
              prev.map((s) => (s.status === "running" ? { ...s, status: "error" } : s))
            );
          },
          onComplete: () => {
            setTimeout(
              () => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
              200
            );
          },
        },
        controller.signal
      );
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "The run failed.");
      }
    } finally {
      setRunning(false);
    }
  };

  /** Re-estimate with a design the user picked over the agent's choice. */
  const rerunWith = async (method: MethodId) => {
    if (!profile || running) return;
    setRunning(true);
    setError(null);
    try {
      setResult(await executeMethod(profile.session_id, method));
      setReport(await buildReport(profile.session_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not run that design.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="mx-auto max-w-4xl px-5 py-10 sm:px-8 sm:py-16">
      <header className="mb-10">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-[28px] font-semibold leading-tight tracking-tight text-ink sm:text-[34px]">
            Causal Inference Agent
          </h1>
          {llmOn !== null ? (
            <span
              className="rounded-full px-2.5 py-1 text-[11px] font-medium"
              style={{
                color: llmOn ? "var(--pass)" : "var(--ink-3)",
                background: llmOn ? "var(--pass-soft)" : "var(--surface-2)",
              }}
              title={
                llmOn
                  ? "Claude is planning the design and writing the report."
                  : "No API key configured — the rule-based engine handles planning and reporting."
              }
            >
              {llmOn ? "Claude connected" : "rule-based mode"}
            </span>
          ) : null}
        </div>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-ink-2">
          Upload a dataset and ask a causal question in plain English. The agent works out
          which design the data can actually support, says why the others were ruled out,
          runs the estimate, and checks the assumptions it depends on.
        </p>
      </header>

      {apiDown ? (
        <div
          className="mb-8 rounded-xl px-5 py-4 text-[13.5px] leading-relaxed"
          style={{ color: "var(--fail)", background: "var(--fail-soft)" }}
        >
          <p className="font-medium">Cannot reach the backend.</p>
          <p className="mt-1">
            Start it with{" "}
            <code className="font-mono text-[12.5px]">
              cd backend &amp;&amp; uvicorn app.main:app --reload
            </code>
            , or set <code className="font-mono text-[12.5px]">NEXT_PUBLIC_API_BASE</code> to
            point at a deployed instance.
          </p>
        </div>
      ) : null}

      {/* Step 1 — data */}
      <section className="mb-8">
        <StepHeading n={1} title="Bring a dataset" />
        {profile ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-[14px] text-ink">
                <span className="font-medium">{profile.filename}</span>
              </p>
              <button
                onClick={startOver}
                disabled={running}
                className="cursor-pointer text-[12.5px] font-medium text-accent transition-opacity hover:opacity-70 disabled:cursor-not-allowed disabled:opacity-40"
              >
                use a different dataset
              </button>
            </div>
            <ProfileSummary profile={profile} />
          </div>
        ) : (
          <UploadPanel samples={samples} onLoaded={handleLoaded} disabled={running} />
        )}
      </section>

      {/* No design is possible — say why here, before asking for a question. */}
      {profile && suggestions && !suggestions.can_run ? (
        <section className="mb-8">
          <DataDiagnosis suggestions={suggestions} />
        </section>
      ) : null}

      {/* Step 2 — question */}
      {profile && suggestions?.can_run !== false ? (
        <section className="mb-8">
          <StepHeading n={2} title="Ask a causal question" />
          <QuestionForm
            question={question}
            onChange={setQuestion}
            onRun={run}
            running={running}
            suggestions={suggestions}
          />
        </section>
      ) : null}

      {/* Step 3 — the trail */}
      {steps.length ? (
        <section className="mb-8">
          <StepHeading n={3} title="What the agent did" />
          <ReasoningTrail steps={steps} plan={plan} error={error} />
        </section>
      ) : null}

      {plan ? (
        <section className="mb-8">
          <MethodComparison plan={plan} onRerun={rerunWith} running={running} />
        </section>
      ) : null}

      {/* Step 4 — results */}
      {result ? (
        <section ref={resultsRef} className="scroll-mt-8">
          <StepHeading n={4} title="Results" />
          <Results result={result} report={report} />
        </section>
      ) : null}

      {result ? (
        <div className="mt-10 flex flex-wrap gap-3 border-t border-border-base pt-6">
          <button
            onClick={startOver}
            className="cursor-pointer rounded-lg px-4 py-2 text-[13.5px] font-medium text-white transition-opacity hover:opacity-90"
            style={{ background: "var(--accent)" }}
          >
            Analyse another dataset
          </button>
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="cursor-pointer rounded-lg border border-border-strong px-4 py-2 text-[13.5px] font-medium text-ink-2 transition-colors hover:border-accent hover:text-accent"
          >
            Back to top
          </button>
        </div>
      ) : null}

      <footer className="mt-16 border-t border-border-base pt-6 text-[12px] leading-relaxed text-ink-3">
        <p>
          A causal estimate is only as good as the assumption behind it. Read the assumption
          checks before quoting the number —{" "}
          <Link href="/about" className="cursor-pointer text-accent hover:underline">
            more about the project
          </Link>
          .
        </p>
      </footer>
    </main>
  );
}

function StepHeading({ n, title }: { n: number; title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2.5">
      <span className="tnum flex h-5 w-5 items-center justify-center rounded-full bg-surface-2 text-[11px] font-semibold text-ink-3">
        {n}
      </span>
      <h2 className="text-[13px] font-medium uppercase tracking-wider text-ink-3">{title}</h2>
    </div>
  );
}
