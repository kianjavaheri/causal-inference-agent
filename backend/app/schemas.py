"""Pydantic schemas shared by the API and the frontend."""

from typing import Any, Literal

from pydantic import BaseModel, Field

MethodId = Literal[
    "did",
    "rdd",
    "psm",
    "iv",
    "synthetic_control",
]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic_type: Literal[
        "binary", "categorical", "continuous", "integer", "datetime", "identifier", "text"
    ]
    missing_count: int
    missing_pct: float
    n_unique: int
    sample_values: list[Any]
    # Populated for numeric columns only.
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


class RoleCandidates(BaseModel):
    """Columns that plausibly fill each causal-inference role, best guess first."""

    outcome: list[str] = Field(default_factory=list)
    treatment: list[str] = Field(default_factory=list)
    time: list[str] = Field(default_factory=list)
    unit: list[str] = Field(default_factory=list)
    running_variable: list[str] = Field(default_factory=list)
    instrument: list[str] = Field(default_factory=list)
    covariates: list[str] = Field(default_factory=list)


class StructureFlags(BaseModel):
    """Structural facts about the dataset that drive method identifiability."""

    is_panel: bool
    n_periods: int
    n_units: int
    has_pre_post_variation: bool
    has_binary_treatment: bool
    has_never_treated_units: bool
    has_repeated_cross_sections: bool
    notes: list[str] = Field(default_factory=list)


class DataProfile(BaseModel):
    session_id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    candidates: RoleCandidates
    structure: StructureFlags
    preview: list[dict[str, Any]]


class MethodAssessment(BaseModel):
    method: MethodId
    label: str
    feasible: bool
    score: float = Field(ge=0, le=1)
    rationale: str
    blocking_reasons: list[str] = Field(default_factory=list)
    required_roles: dict[str, Any] = Field(default_factory=dict)
    key_assumptions: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    session_id: str
    question: str
    chosen_method: MethodId
    chosen_label: str
    justification: str
    estimand: str
    roles: dict[str, Any]
    assumptions: list[str]
    assessments: list[MethodAssessment]
    llm_used: bool
    reasoning_trace: list[str] = Field(default_factory=list)


class Estimate(BaseModel):
    point: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    n_obs: int
    n_treated: int | None = None
    n_control: int | None = None
    units: str = "outcome units"


class Diagnostic(BaseModel):
    """A single assumption check, with data the frontend can plot."""

    id: str
    title: str
    kind: Literal[
        "event_study",
        "scatter_fit",
        "balance",
        "overlap",
        "path",
        "gap",
        "placebo_distribution",
        "composition",
        "table",
        "text",
    ]
    verdict: Literal["pass", "warn", "fail", "info"]
    summary: str
    detail: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    session_id: str
    method: MethodId
    method_label: str
    estimand: str
    estimate: Estimate
    diagnostics: list[Diagnostic]
    specification: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class Report(BaseModel):
    session_id: str
    headline: str
    interpretation: str
    method_explanation: str
    assumptions_discussion: str
    caveats: list[str]
    confidence: Literal["high", "moderate", "low"]
    llm_used: bool


# ---- request bodies -------------------------------------------------------


class PlanRequest(BaseModel):
    session_id: str
    question: str
    # Optional user overrides for role assignment.
    roles: dict[str, Any] | None = None


class ExecuteRequest(BaseModel):
    session_id: str
    method: MethodId | None = None
    roles: dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    session_id: str


# ---- data diagnosis & question suggestions --------------------------------


class DesignBlocker(BaseModel):
    """Why one design cannot be used here, and what would change that."""

    method: MethodId
    label: str
    reasons: list[str]
    missing: list[str] = Field(default_factory=list)


class Suggestions(BaseModel):
    session_id: str
    # Questions phrased against columns that actually exist in this dataset.
    questions: list[str]
    feasible_methods: list[MethodId]
    blockers: list[DesignBlocker]
    guidance: str
    can_run: bool
