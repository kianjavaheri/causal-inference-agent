"""Synthetic demo datasets with known ground-truth treatment effects.

Each generator plants a specific true effect so the estimators can be validated
against it, and so the demo has a defensible "correct answer".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleDataset:
    id: str
    name: str
    description: str
    question: str
    true_effect: float
    expected_method: str


SAMPLES: dict[str, SampleDataset] = {
    "did_minimum_wage": SampleDataset(
        id="did_minimum_wage",
        name="Minimum wage & restaurant employment",
        description=(
            "Balanced panel of 60 counties over 12 quarters. Half the counties raise "
            "their minimum wage in Q7. Outcome is average restaurant employment."
        ),
        question=(
            "Did the minimum wage increase change restaurant employment in the "
            "counties that adopted it?"
        ),
        true_effect=-2.5,
        expected_method="did",
    ),
    "rdd_scholarship": SampleDataset(
        id="rdd_scholarship",
        name="Merit scholarship & college GPA",
        description=(
            "2,000 students. A scholarship is awarded to anyone scoring at or above 70 "
            "on an entrance exam. Outcome is first-year college GPA."
        ),
        question="What is the effect of receiving the merit scholarship on first-year GPA?",
        true_effect=0.35,
        expected_method="rdd",
    ),
    "psm_training": SampleDataset(
        id="psm_training",
        name="Job training program & earnings",
        description=(
            "3,000 workers who self-selected into a training program, with rich "
            "pre-treatment covariates. Selection is on observables."
        ),
        question="Did the job training program raise participants' earnings?",
        true_effect=1800.0,
        expected_method="psm",
    ),
    "iv_encouragement": SampleDataset(
        id="iv_encouragement",
        name="App onboarding encouragement",
        description=(
            "4,000 users randomly assigned an onboarding email (the instrument). "
            "Only some comply and complete onboarding. Outcome is 90-day spend."
        ),
        question=(
            "What is the effect of completing onboarding on 90-day spend, given that "
            "only the email was randomised?"
        ),
        true_effect=42.0,
        expected_method="iv",
    ),
    "sc_tobacco": SampleDataset(
        id="sc_tobacco",
        name="State tobacco tax & consumption",
        description=(
            "One treated state plus 24 donor states over 30 years. The treated state "
            "passes a tobacco tax in year 20. Outcome is per-capita consumption."
        ),
        question="How much did the tobacco tax reduce per-capita cigarette consumption?",
        true_effect=-18.0,
        expected_method="synthetic_control",
    ),
}


def make_did(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_units, n_periods, treat_period = 60, 12, 7
    unit_fe = rng.normal(100, 12, n_units)
    treated_units = set(rng.choice(n_units, size=n_units // 2, replace=False).tolist())

    rows = []
    for u in range(n_units):
        is_treated = u in treated_units
        for t in range(1, n_periods + 1):
            time_fe = 1.2 * t                      # common trend, shared by both groups
            post = int(t >= treat_period)
            effect = -2.5 if (is_treated and post) else 0.0
            y = unit_fe[u] + time_fe + effect + rng.normal(0, 2.0)
            rows.append(
                {
                    "county_id": f"C{u:03d}",
                    "quarter": t,
                    "treated_group": int(is_treated),
                    "post": post,
                    "treated": int(is_treated and post),
                    "min_wage": 7.25 + (2.0 if (is_treated and post) else 0.0),
                    "employment": round(float(y), 3),
                    "population_k": round(float(50 + unit_fe[u] * 0.8), 1),
                }
            )
    return pd.DataFrame(rows)


def make_rdd(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2000
    exam = np.clip(rng.normal(68, 12, n), 20, 100)
    above = (exam >= 70).astype(int)
    # Smooth function of the running variable + a genuine jump at the cutoff.
    gpa = (
        1.6
        + 0.022 * (exam - 70)
        - 0.00015 * (exam - 70) ** 2
        + 0.35 * above
        + rng.normal(0, 0.35, n)
    )
    return pd.DataFrame(
        {
            "student_id": [f"S{i:05d}" for i in range(n)],
            "exam_score": np.round(exam, 2),
            "scholarship": above,
            "college_gpa": np.round(np.clip(gpa, 0, 4.0), 3),
            "family_income_k": np.round(rng.normal(62, 20, n), 1),
            "hs_gpa": np.round(np.clip(rng.normal(3.1, 0.4, n), 0, 4), 2),
        }
    )


def make_psm(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 3000
    age = rng.integers(20, 60, n)
    education = np.clip(rng.normal(13, 2.5, n), 8, 20)
    prior_earnings = np.clip(rng.normal(28000, 9000, n), 5000, 80000)
    married = rng.binomial(1, 0.55, n)

    # Selection on observables: low earners and more-educated workers enrol more.
    index = (
        -1.0
        + 0.09 * (education - 13)
        - 0.00006 * (prior_earnings - 28000)
        - 0.02 * (age - 40)
        + 0.25 * married
    )
    p = 1 / (1 + np.exp(-index))
    enrolled = rng.binomial(1, p)

    earnings = (
        18000
        + 1.05 * prior_earnings
        + 900 * (education - 13)
        + 120 * (age - 40)
        + 1500 * married
        + 1800 * enrolled                       # true ATT
        + rng.normal(0, 4200, n)
    )
    return pd.DataFrame(
        {
            "worker_id": [f"W{i:05d}" for i in range(n)],
            "enrolled_training": enrolled,
            "age": age,
            "education_years": np.round(education, 1),
            "prior_earnings": np.round(prior_earnings, 0),
            "married": married,
            "earnings_after": np.round(earnings, 0),
        }
    )


def make_iv(seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 4000
    email = rng.binomial(1, 0.5, n)                     # randomised instrument
    ability = rng.normal(0, 1, n)                       # unobserved confounder

    # Compliance: the email lifts onboarding, but motivated users onboard anyway.
    p_onboard = 1 / (1 + np.exp(-(-0.8 + 1.6 * email + 0.9 * ability)))
    onboarded = rng.binomial(1, p_onboard)

    spend = 120 + 42 * onboarded + 25 * ability + rng.normal(0, 30, n)
    return pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n)],
            "onboarding_email": email,
            "completed_onboarding": onboarded,
            "spend_90d": np.round(spend, 2),
            "signup_channel": rng.choice(["organic", "paid", "referral"], n),
            "device_ios": rng.binomial(1, 0.45, n),
        }
    )


def make_synthetic_control(seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_donors, n_years, treat_year = 24, 30, 20
    # Two latent factors drive every state's path; the treated state is a blend of donors.
    f1 = np.cumsum(rng.normal(0, 1.0, n_years)) + np.linspace(0, -8, n_years)
    f2 = np.cumsum(rng.normal(0, 0.8, n_years))

    rows = []
    donor_loadings = rng.uniform(0.3, 1.6, (n_donors, 2))
    donor_levels = rng.uniform(90, 140, n_donors)
    for d in range(n_donors):
        for t in range(n_years):
            y = (
                donor_levels[d]
                + donor_loadings[d, 0] * f1[t]
                + donor_loadings[d, 1] * f2[t]
                + rng.normal(0, 2.0)
            )
            rows.append(
                {
                    "state": f"Donor{d:02d}",
                    "year": 1970 + t,
                    "treated_state": 0,
                    "post_tax": int(t >= treat_year),
                    "tax_active": 0,
                    "cigarette_sales": round(float(y), 2),
                }
            )

    treated_loadings = donor_loadings.mean(axis=0)
    treated_level = donor_levels.mean()
    for t in range(n_years):
        effect = -18.0 if t >= treat_year else 0.0
        y = (
            treated_level
            + treated_loadings[0] * f1[t]
            + treated_loadings[1] * f2[t]
            + effect
            + rng.normal(0, 2.0)
        )
        rows.append(
            {
                "state": "California",
                "year": 1970 + t,
                "treated_state": 1,
                "post_tax": int(t >= treat_year),
                "tax_active": int(t >= treat_year),
                "cigarette_sales": round(float(y), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["state", "year"]).reset_index(drop=True)


GENERATORS = {
    "did_minimum_wage": make_did,
    "rdd_scholarship": make_rdd,
    "psm_training": make_psm,
    "iv_encouragement": make_iv,
    "sc_tobacco": make_synthetic_control,
}


def generate(sample_id: str) -> pd.DataFrame:
    if sample_id not in GENERATORS:
        raise KeyError(f"Unknown sample dataset: {sample_id}")
    return GENERATORS[sample_id]()
