import Link from "next/link";
import { API_BASE } from "@/lib/api";
import type { MethodSpec } from "@/lib/types";
import { BackendUnavailable } from "@/components/BackendUnavailable";

export const metadata = {
  title: "Methods · Causal Inference Agent",
  description: "The five causal designs supported, what each identifies, and what each assumes.",
};

// The catalogue lives in the backend so the UI and the engine can never disagree about
// what a design assumes.
async function getMethods(): Promise<MethodSpec[] | null> {
  try {
    const res = await fetch(`${API_BASE}/methods`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function Methods() {
  const methods = await getMethods();

  return (
    <main className="mx-auto max-w-3xl px-5 py-12 sm:px-8 sm:py-16">
      <h1 className="text-[30px] font-semibold leading-tight tracking-tight text-ink sm:text-[36px]">
        The five designs
      </h1>
      <p className="mt-4 text-[16.5px] leading-relaxed text-ink-2">
        Each one identifies a causal effect only under assumptions the data cannot fully
        verify. What separates them is which assumption you have to defend.
      </p>

      {methods === null ? (
        <BackendUnavailable className="mt-10" />
      ) : (
        <div className="mt-10 space-y-5">
          {methods.map((m) => (
            <article key={m.id} className="rounded-xl border border-border-base bg-surface p-6">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-[17px] font-semibold tracking-tight text-ink">{m.label}</h2>
                <span className="text-[12px] text-ink-3">{m.estimand}</span>
              </div>

              <p className="mt-3 text-[14.5px] leading-relaxed text-ink-2">{m.plain_english}</p>

              <div className="mt-4 flex flex-wrap gap-1.5">
                {m.required_roles.map((r) => (
                  <span
                    key={r}
                    className="rounded-md bg-surface-2 px-2 py-1 font-mono text-[11px] text-ink-3"
                  >
                    needs {r}
                  </span>
                ))}
              </div>

              <details className="group mt-4 border-t border-border-base pt-3">
                <summary className="cursor-pointer list-none text-[12.5px] font-medium text-accent transition-opacity hover:opacity-70">
                  What it assumes ({m.assumptions.length})
                </summary>
                <ul className="mt-3 space-y-2">
                  {m.assumptions.map((a, i) => (
                    <li key={i} className="flex gap-2 text-[13.5px] leading-relaxed text-ink-2">
                      <span className="mt-[8px] h-1 w-1 shrink-0 rounded-full bg-ink-3" />
                      {a}
                    </li>
                  ))}
                </ul>
              </details>
            </article>
          ))}
        </div>
      )}

      <div className="mt-8 border-t border-border-base pt-8">
        <Link
          href="/"
          className="cursor-pointer rounded-lg px-4 py-2 text-[13.5px] font-medium text-white transition-opacity hover:opacity-90"
          style={{ background: "var(--accent)" }}
        >
          Run one on your data
        </Link>
      </div>
    </main>
  );
}
