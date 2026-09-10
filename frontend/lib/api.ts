import type {
  DataProfile, ExecutionResult, MethodSpec, Plan, Report, SampleDataset, StepName,
  Suggestions,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://127.0.0.1:8000";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : detail;
    } catch { /* body was not JSON; keep the status line */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth() {
  return json<{ status: string; llm_configured: boolean; llm_model: string | null }>(
    await fetch(`${API_BASE}/health`, { cache: "no-store" })
  );
}

export async function fetchSamples(): Promise<SampleDataset[]> {
  return json(await fetch(`${API_BASE}/samples`, { cache: "no-store" }));
}

export async function fetchMethods(): Promise<MethodSpec[]> {
  return json(await fetch(`${API_BASE}/methods`, { cache: "no-store" }));
}

export async function loadSample(id: string): Promise<DataProfile> {
  return json(await fetch(`${API_BASE}/samples/${id}/load`, { method: "POST" }));
}

export async function uploadCsv(file: File): Promise<DataProfile> {
  const form = new FormData();
  form.append("file", file);
  return json(await fetch(`${API_BASE}/upload`, { method: "POST", body: form }));
}

export async function fetchSuggestions(sessionId: string): Promise<Suggestions> {
  return json(await fetch(`${API_BASE}/suggestions/${sessionId}`, { cache: "no-store" }));
}

export async function executeMethod(
  sessionId: string,
  method?: string,
  roles?: Record<string, unknown>
): Promise<ExecutionResult> {
  return json(
    await fetch(`${API_BASE}/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, method, roles, options: {} }),
    })
  );
}

export async function buildReport(sessionId: string): Promise<Report> {
  return json(
    await fetch(`${API_BASE}/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    })
  );
}

/** One server-sent event from /run. Fields are per-event, hence all optional. */
interface SseFrame {
  step?: StepName;
  title?: string;
  detail?: string;
  summary?: string;
  payload?: unknown;
  status?: number;
  session_id?: string;
}

export interface RunHandlers {
  onStepStart: (step: StepName, title: string, detail?: string) => void;
  onProfile: (profile: DataProfile, summary: string) => void;
  onPlan: (plan: Plan, summary: string) => void;
  onExecute: (result: ExecutionResult, summary: string) => void;
  onReport: (report: Report, summary: string) => void;
  onError: (message: string) => void;
  onComplete: () => void;
}

/**
 * Drive the streamed pipeline. Uses fetch + a ReadableStream rather than EventSource
 * so the request can be aborted when the user starts a new run, and so errors surface
 * as real HTTP failures instead of a silent reconnect loop.
 */
export async function runPipeline(
  sessionId: string,
  question: string,
  handlers: RunHandlers,
  signal?: AbortSignal
): Promise<void> {
  const params = new URLSearchParams({ session_id: sessionId, question });
  const res = await fetch(`${API_BASE}/run?${params}`, {
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (!res.ok || !res.body) {
    handlers.onError(`Could not start the run (${res.status} ${res.statusText}).`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (event: string, raw: string) => {
    let frame: SseFrame;
    try {
      frame = JSON.parse(raw) as SseFrame;
    } catch {
      return;
    }
    if (event === "step_start" && frame.step) {
      handlers.onStepStart(frame.step, frame.title ?? "", frame.detail);
    } else if (event === "step_done") {
      const summary = frame.summary ?? "";
      switch (frame.step) {
        case "profile": handlers.onProfile(frame.payload as DataProfile, summary); break;
        case "plan": handlers.onPlan(frame.payload as Plan, summary); break;
        case "execute": handlers.onExecute(frame.payload as ExecutionResult, summary); break;
        case "report": handlers.onReport(frame.payload as Report, summary); break;
      }
    } else if (event === "error") {
      handlers.onError(frame.detail ?? "The run failed.");
    } else if (event === "complete") {
      handlers.onComplete();
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      }
      if (dataLines.length) dispatch(event, dataLines.join("\n"));
    }
  }
}
