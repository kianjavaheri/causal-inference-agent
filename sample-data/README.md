# Real datasets

Canonical causal inference datasets, for testing the agent against data it was not
designed around. All are public and widely used for teaching; each is credited below.
Download links and licences belong to the original sources — these copies are here for
convenience.

Upload any of them on the home page.

## What the agent does with each

| File | Rows × cols | Agent picks | Notes |
|---|---|---|---|
| `lalonde.csv` | 614 × 10 | Propensity score matching | **Balance FAILS**, confidence low. Correct: this is the textbook case where matching does *not* recover the experimental benchmark (≈ $1,794). |
| `Guns.csv` | 1,173 × 14 | Difference-in-differences | **Parallel trends REJECTED** (p = 0.02) — the central methodological critique of the "more guns, less crime" literature. |
| `castle.csv` | 550 × 139 | Difference-in-differences | Finds `post` among 139 columns (60 are region dummies). Pre-trends rejected. |
| `gov_transfers.csv` | 1,948 × 5 | Regression discontinuity | Treatment is assigned *below* the income cutoff. **Density test FAILS** — sorting at the threshold. |
| `jtrain.csv` | 471 × 31 | Difference-in-differences | Michigan training grants; unit is the numeric firm code `fcode`. |
| `close_college.csv` | 3,010 × 8 | Propensity score matching | *Should* be IV (`nearc4` instruments `educ`). The agent will not guess an instrument from correlation — nominate it via the roles override. |
| `organ_donation.csv` | 162 × 3 | — refuses | No treatment column exists. The agent says what is missing instead of estimating. |
| `texas.csv` | 816 × 12 | — refuses | Same: no column marks who was treated when. |
| `CigarettesSW.csv` | 96 × 10 | — refuses | Continuous-treatment IV, which this build does not support. |

The refusals are the point as much as the estimates: four of nine datasets genuinely lack
an identifiable design as shipped, and the agent says so with reasons.

## Sources

- `lalonde.csv`, `CigarettesSW.csv`, `Guns.csv`, `jtrain.csv` — via
  [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) (from the R packages
  `MatchIt`, `AER`, and `wooldridge`).
  - LaLonde, R. (1986); Dehejia & Wahba (1999).
  - Card & Krueger; Stock & Watson, *Introduction to Econometrics*.
  - Ayres & Donohue / Lott & Mustard — `Guns`.
  - Wooldridge, *Introductory Econometrics* — `jtrain`.
- `castle.csv`, `gov_transfers.csv`, `close_college.csv`, `organ_donation.csv`,
  `texas.csv` — from [causaldata](https://github.com/NickCH-K/causaldata) (Nick
  Huntington-Klein, *The Effect*). Stata `.dta` originals converted to CSV with pandas.
  - Cheng & Hoekstra (2013) — castle doctrine.
  - Card (1995) — proximity to college.
  - Kessler & Roth (2014) — organ donation.
