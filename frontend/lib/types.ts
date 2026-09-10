/** Mirrors of the backend Pydantic schemas. */

export type MethodId = "did" | "rdd" | "psm" | "iv" | "synthetic_control";
export type Verdict = "pass" | "warn" | "fail" | "info";

export interface ColumnProfile {
  name: string;
  dtype: string;
  semantic_type:
    | "binary" | "categorical" | "continuous" | "integer"
    | "datetime" | "identifier" | "text";
  missing_count: number;
  missing_pct: number;
  n_unique: number;
  sample_values: unknown[];
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
}

export interface RoleCandidates {
  outcome: string[];
  treatment: string[];
  time: string[];
  unit: string[];
  running_variable: string[];
  instrument: string[];
  covariates: string[];
}

export interface StructureFlags {
  is_panel: boolean;
  n_periods: number;
  n_units: number;
  has_pre_post_variation: boolean;
  has_binary_treatment: boolean;
  has_never_treated_units: boolean;
  has_repeated_cross_sections: boolean;
  notes: string[];
}

export interface DataProfile {
  session_id: string;
  filename: string;
  n_rows: number;
  n_cols: number;
  columns: ColumnProfile[];
  candidates: RoleCandidates;
  structure: StructureFlags;
  preview: Record<string, unknown>[];
}

export interface MethodAssessment {
  method: MethodId;
  label: string;
  feasible: boolean;
  score: number;
  rationale: string;
  blocking_reasons: string[];
  required_roles: Record<string, unknown>;
  key_assumptions: string[];
}

export interface Plan {
  session_id: string;
  question: string;
  chosen_method: MethodId;
  chosen_label: string;
  justification: string;
  estimand: string;
  roles: Record<string, unknown>;
  assumptions: string[];
  assessments: MethodAssessment[];
  llm_used: boolean;
  reasoning_trace: string[];
}

export interface Estimate {
  point: number;
  se?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  p_value?: number | null;
  n_obs: number;
  n_treated?: number | null;
  n_control?: number | null;
  units: string;
}

// --- diagnostics -----------------------------------------------------------
//
// Each diagnostic carries a payload whose shape depends on its `kind`, so Diagnostic is
// a discriminated union: switching on `kind` narrows `data` to exactly one of these.

export type DiagnosticKind = Diagnostic["kind"];

export interface EventStudyPoint {
  period: number;
  coef: number | null;
  se?: number | null;
  ci_low: number | null;
  ci_high: number | null;
  is_pre: boolean;
  is_reference?: boolean;
}

export interface EventStudyData {
  points: EventStudyPoint[];
  reference_period?: number;
  f_stat?: number | null;
  p_value?: number | null;
}

export interface ScatterFitData {
  bins: { x: number; y: number; n: number; side: "left" | "right" }[];
  left_fit: { x: number; y: number }[];
  right_fit: { x: number; y: number }[];
  cutoff: number;
  bandwidth: number;
  jump: number | null;
  /** Which side of the cutoff receives treatment. */
  treated_side?: "left" | "right";
  x_label: string;
  y_label: string;
}

export interface PathData {
  series: { name: string; points: { time: string; value: number | null }[] }[];
  treatment_time: string | null;
  x_label: string;
  y_label: string;
  pre_rmse?: number | null;
  naive_rmse?: number | null;
  rmspe?: number | null;
  improvement_over_naive?: number | null;
}

export interface GapData {
  points: { time: string; value: number | null; is_post: boolean }[];
  treatment_time: string | null;
  x_label: string;
  y_label: string;
}

export interface PlaceboData {
  treated_name: string;
  treated_gaps: { time: string; value: number | null }[];
  placebos: {
    unit: string;
    poor_fit: boolean;
    points: { time: string; value: number | null }[];
  }[];
  treatment_time: string;
  p_value: number | null;
  rank: number;
  n_units: number;
  x_label: string;
}

export interface BalanceData {
  rows: { label: string; before: number | null; after: number | null }[];
  threshold: number;
}

/** One histogram bin. `count`/`side` for a single distribution; `treated`/`control` for two. */
export interface HistogramBin {
  x: number;
  count?: number;
  side?: "left" | "right";
  treated?: number;
  control?: number;
}

export interface OverlapData {
  histogram: HistogramBin[];
  cutoff?: number | null;
  p_value?: number | null;
  x_label: string;
}

export interface CompositionData {
  segments: { label: string; share: number; note: string; estimated: boolean }[];
  rows: { label: string; value: number | null }[];
}

export type TableCell = string | number | boolean | null | undefined;

export interface TableData {
  rows: Record<string, TableCell>[];
  columns?: string[];
  optimal_bandwidth?: number | null;
}

interface DiagnosticBase {
  id: string;
  title: string;
  verdict: Verdict;
  summary: string;
  detail?: string | null;
}

export type Diagnostic =
  | (DiagnosticBase & { kind: "event_study"; data: EventStudyData })
  | (DiagnosticBase & { kind: "scatter_fit"; data: ScatterFitData })
  | (DiagnosticBase & { kind: "path"; data: PathData })
  | (DiagnosticBase & { kind: "gap"; data: GapData })
  | (DiagnosticBase & { kind: "placebo_distribution"; data: PlaceboData })
  | (DiagnosticBase & { kind: "balance"; data: BalanceData })
  | (DiagnosticBase & { kind: "overlap"; data: OverlapData })
  | (DiagnosticBase & { kind: "composition"; data: CompositionData })
  | (DiagnosticBase & { kind: "table"; data: TableData })
  | (DiagnosticBase & { kind: "text"; data: Record<string, never> });

export interface ExecutionResult {
  session_id: string;
  method: MethodId;
  method_label: string;
  estimand: string;
  estimate: Estimate;
  diagnostics: Diagnostic[];
  specification: Record<string, unknown>;
  warnings: string[];
}

export interface Report {
  session_id: string;
  headline: string;
  interpretation: string;
  method_explanation: string;
  assumptions_discussion: string;
  caveats: string[];
  confidence: "high" | "moderate" | "low";
  llm_used: boolean;
}

export interface SampleDataset {
  id: string;
  name: string;
  description: string;
  question: string;
  expected_method: MethodId;
  expected_method_label: string;
  true_effect: number;
}

export interface MethodSpec {
  id: MethodId;
  label: string;
  estimand: string;
  required_roles: string[];
  optional_roles: string[];
  assumptions: string[];
  plain_english: string;
}

export interface DesignBlocker {
  method: MethodId;
  label: string;
  reasons: string[];
  missing: string[];
}

export interface Suggestions {
  session_id: string;
  questions: string[];
  feasible_methods: MethodId[];
  blockers: DesignBlocker[];
  guidance: string;
  can_run: boolean;
}

export type StepName = "profile" | "plan" | "execute" | "report";

export interface TrailStep {
  step: StepName;
  title: string;
  detail?: string;
  summary?: string;
  status: "running" | "done" | "error";
  payload?: unknown;
  startedAt: number;
  finishedAt?: number;
}
