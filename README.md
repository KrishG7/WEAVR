<div align="center">

# WEAVR

### Hybrid AI–NWP multi-model rainfall forecast blending for India

*Weigh every forecast model by how much it has actually earned, region by region and lead by lead, and be honest about what the evidence does and does not show.*

[![CI](https://github.com/ATMOSYNC/WEAVR/actions/workflows/ci.yml/badge.svg)](https://github.com/ATMOSYNC/WEAVR/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2ea44f)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![SIH](https://img.shields.io/badge/Smart%20India%20Hackathon-PS%2026081-blueviolet)

[Why WEAVR](#why-weavr) · [What it does](#what-it-does) · [The dashboard](#the-dashboard) · [Quick start](#quick-start) · [Evidence](#evidence-and-honesty) · [Documentation](#documentation)

</div>

---

## Why WEAVR

The monsoon is forecast today by a growing crowd of models: physics-based
numerical weather prediction (NWP) such as ECMWF's HRES and IFS ensemble, and
a new generation of AI models such as Google's GraphCast. Each is good in
some places, at some lead times, for some kinds of rain, and poor in others.
No single model is best everywhere, and simply averaging them throws away
what is known about each one's strengths.

**WEAVR blends them properly.** It learns, from history, how much to trust
each model in each region of India at each lead time, produces a single
blended forecast and calibrated exceedance probabilities at the thresholds
India's weather service (IMD) warns on, and verifies every claim against IMD
rain-gauge analyses using a leave-out protocol that never lets the test data
leak into training.

> **A research prototype, not an official product.** WEAVR is not an IMD or
> NCMRWF product. Any warnings it shows are demonstrations.

## What it does

<table>
<tr>
<td width="50%" valign="top">

### Blend
- Combines **GraphCast, HRES and the IFS ensemble** for rainfall (Pangu for
  temperature) on a common **0.25° India grid**, 6.5–38.5°N, 66.5–100.0°E
- **Lead times of 24 to 96 h** for the June–September monsoon
- Regional, lead-specific weights instead of one global average

</td>
<td width="50%" valign="top">

### Calibrate
- A ladder of methods, each benchmarked against the last: equal-weight mean,
  regional weights, hierarchical EMOS and BMA, and a regime-conditioned
  variant
- **Probabilities at IMD's rain thresholds**, not just a single number
- Drift detection and weight renormalisation when a source drops out

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Verify
- Leave-one-year-out and seasonal-block splits only; **never a random split**
- RMSE, bias, ACC, CRPS, Brier, SEEPS, FSS and POD/FAR/CSI/ETS
- Paired block bootstrap and Diebold–Mariano tests, with claims
  **pre-registered** before they are tested

</td>
<td width="50%" valign="top">

### Explore
- An interactive dashboard: weight map, skill trends, blended map and
  extreme-probability map in IMD's colour scheme
- An optional **offline OpenStreetMap basemap** under the forecast grid
- A daily pipeline for live forecasts and verification

</td>
</tr>
</table>

## How it works

```mermaid
flowchart LR
    subgraph S["Sources"]
        G["GraphCast"]
        H["HRES"]
        E["IFS ensemble"]
    end
    O[("IMD gauge<br/>analysis")]
    S --> R["Common 0.25° grid<br/>and time convention"]
    R --> T["Tier ladder<br/>equal weights → regional weights<br/>→ EMOS / BMA"]
    O --> T
    T --> B["Blended forecast<br/>and exceedance probabilities"]
    B --> V["Verification<br/>leave-out, CIs, pre-registered claims"]
    B --> D["Dashboard and API"]
```

Every stage is a small, tested module in `src/weavr/`: `grid`, `ensemble`,
`weighting`, `emos`, `bma`, `regions`, `verify`, `significance`,
`climatology`, `independence`, `drift`, `renormalize`, `stacking`,
`quantile_mapping` and `tail`, with the runnable pipelines in `scripts/`.

### Tail repair: Quantile mapping (`weavr.quantile_mapping`)

To repair under-forecast extreme rainfall before blending or parametric fitting, WEAVR provides empirical quantile mapping calibrated against historical IMD gauge analyses (`fit_quantile_map`, `apply_quantile_mapping`). It zeroes dry/drizzle values below a data-driven wet-day threshold (0.1 mm), performs linear quantile interpolation across wet ranks, and preserves forecast extreme anomalies above historical training maximums via an additive upper-tail extrapolation rule.

Evaluated over both daily JJAS seasons (LOYO, Tier 1 blend, paired moving-block bootstrap CIs), QM **fails** the pre-registered H7 test, on its non-inferiority guard: SEDI at 115.6 mm improves at 5 of 5 leads with the CI excluding zero, but Brier at 7.5 mm is significantly worse at 24 and 48 h for both reference variants, which H7 does not allow. Threshold-weighted CRPS at 64.5 mm is also significantly *worse* at every lead, and RMSE rises by roughly 3.6–6.2 mm depending on reference variant. Until 31 October this read as a pass, on Brier scores that counted unobserved sea cells as correct dry forecasts. QM trades false alarms for detection: probability of detection at 115.6 mm rises by 0.2–0.3 for GraphCast, the IFS-ENS mean and the blend, and false-alarm ratio rises with it. Applied to the ensemble members before EMOS-CSG it makes CRPS worse on GraphCast at every lead and helps IFS-ENS only slightly at long leads, and every arm fails the same Brier guard. See [tail repair results](docs/tail-repair-results.md) for the tables and the per-variant trade-off.

### Tail repair: Extreme-value tail (`weavr.tail`)

For rain bins where extreme events are too scarce to fit parametric EMOS-CSG (such as $\ge 204.5\text{ mm}$), WEAVR anchors probabilities on a pooled Generalized Pareto distribution (GPD) fitted to the training season's IMD extremes, the test year held out (`weavr.tail`). The GPD shares a single shape parameter $\xi$ nationwide while allowing region-specific scale parameters $\sigma_r$. The tail is continuously spliced at threshold $u$ ($64.5\text{ mm}$) with the nearest fittable EMOS-CSG bin: $P(Y > y) = P_{\text{CSGD}}(Y > u) \cdot (1 + \xi (y - u)/\sigma_r)^{-1/\xi}$, eliminating false-zero fallback grey cells and giving every cell a real probability for $115.6\text{ mm}$ and $204.5\text{ mm}$ extremes.

Measured over both daily JJAS seasons with the tail fitted on the 15-year IMD archive (test year excluded), the spliced tail eliminates every false zero-probability cell (193 → 0 at both 115.6 mm and 204.5 mm) and is calibrated in total (it forecasts 0.96× the observed count of 115.6 mm days and 1.15× at 204.5 mm, where the point-mass control forecast about half). It does **not** improve the scores H7 names: tw-CRPS and Brier are level with the control and SEDI is worse at all 5 leads, so **H7's EVT part fails**. Applying step 11's QM before EMOS-CSG changes little. See [tail repair results](docs/tail-repair-results.md).

### Tier 2b: Combiner meta-blend (`weavr.stacking`)

WEAVR meta-blends distributional combiners (EMOS-CSG and BMA) through three stacking strategies (`weavr.stacking`): per-bin selection (applying each method where its training density is well-supported), linear density pooling with weights $\alpha \cdot f_{\text{EMOS}} + (1-\alpha) \cdot f_{\text{BMA}}$, and quantile averaging across the predictive distributions with exact root-finding quantile inversion for the censored shifted gamma distribution (CSGD).

The pre-registered H10 test **passes at 5 of 5 leads**: the arm nominated on training data beats both EMOS-CSG and BMA on CRPS at every lead, every 95% interval excluding zero. Quantile averaging is the recommended probabilistic combiner (the only arm that beats both parents at all five leads); per-bin selection beats both at only 2 of 5, so the "strongest combiner per bin" claim is not supported. The dashboard can draw its extreme-rain map from the quantile average with `--combiner quantile_avg`; the published map stays EMOS-CSG: its measured reliability is no better (equal Brier, lower SEDI at every lead, over-forecasts 1.14x / 1.39x in total at 115.6 / 204.5 mm; Tier 2b results §5b). See [Tier 2b results](docs/tier2b-combined-results.md).



## The dashboard

A FastAPI service (`dashboard/api.py`) serves a dependency-light front end
(`dashboard-web/`, plain HTML, CSS and JavaScript, no build step).

| View | What it shows |
|---|---|
| **Weight map** | Which model earns the most trust in each region, by lead |
| **Skill trends** | RMSE and CRPS by lead, blend against every raw source |
| **Blended map** | The blended rainfall forecast on the grid, in IMD's rain bins |
| **Extreme-probability map** | The probability of exceeding a warning threshold, with fallback cells marked |

The two maps have a **Show geography** switch that draws the forecast grid
over a basemap built from **OpenStreetMap** data and served from a local
file, so it works with no internet connection. The basemap deliberately draws
**no national or disputed boundaries**; an official outline you supply is
drawn on top. It falls back to the plain grid if the tiles or WebGL are
missing. Details: [docs/basemap-scope.md](docs/basemap-scope.md).

## Quick start

```bash
git clone https://github.com/ATMOSYNC/WEAVR.git
cd WEAVR
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard-api]"

uvicorn dashboard.api:app --port 8000
```

Then open <http://127.0.0.1:8000/>. The dashboard runs out of the box on the
committed example grids in `dashboard/data/`.

**Optional: the offline basemap**

```bash
brew install pmtiles                 # once (see docs/basemap-scope.md for other platforms)
python scripts/build_basemap.py      # about 40 s, 148 MB, gitignored
```

**Rebuilding the results** needs the large forecast and observation stores,
which are not in the repository (`data/` is gitignored). The order, sources
and licences are in [docs/data-sources.md](docs/data-sources.md) and
[docs/baseline-store.md](docs/baseline-store.md).

For the second monsoon season, run the baseline and lagged builders with
`--year 2018`, then validate with `scripts/validate_daily_stores.py --year 2018
--no-require-ifs-ens`. The year-specific archive map and
`weavr.stores.open_multi_season` join matching 2018 and 2020 groups for the
next leave-one-year-out evaluation. See the
[2018 store notes](docs/baseline-store.md#second-season-2018) for the data
and full-member IFS-ENS scope.

## Evidence and honesty

WEAVR is built around reporting what the data shows, including the parts that
do not flatter it. Measured on two seasons (2018 + 2020) against IMD gauges,
daily cadence, leave-one-year-out:

- **Tier 1 beats Tier 0 at every lead** on both RMSE and ACC. Regional weights
  add value over equal weighting.
- **GraphCast is the best single member** at every lead (RMSE 13.05–13.53 mm).
  HRES is the worst (RMSE 16.15–19.69 mm). IFS-ENS mean is intermediate.
- **The three sources are worth about 1.1 independent models.** Their errors
  correlate at 0.72–0.87 on the daily two-season base, which is why plain
  averaging fails and why adding a genuinely different source matters more
  than re-weighting.
- **The blend beats a single ensemble member, and the best single-source
  calibration only at short leads.** H1 passes twice: the blend beats the best
  single member on RMSE and on CRPS at 5 of 5 leads each, every interval
  excluding zero. H2 is unproven: the multi-source combiner beats EMOS-IFS on
  CRPS at every lead, significantly, but the best single-source EMOS
  (GraphCast) only at 24–72 h, significantly only at 24 h (by 0.035 mm), and
  is level with it at 96–120 h. Calibrating one good source captures most,
  not all, of what the mixture adds.
- **Combining the two calibrations does help.** Averaging the EMOS-CSG and BMA
  quantile functions beats both at every lead (H10 passes, 5 of 5), by
  0.01–0.05 mm of daily CRPS.
- **Extremes are the open problem.** GraphCast detects 6% of 115.6 mm events
  at 24 h and under 1% by 96 h, and only HRES detects any 204.5 mm event past
  day 1 (1–4%) -- while being the worst source on RMSE. RMSE and warning skill
  point in opposite directions.
- **A regime-conditioned tier was tested and rejected** against
  pre-declared criteria, and the no-go is documented.
- **Two seasons, 234–242 test days per lead** (about 120 per fold), so
  comparisons now have meaningful confidence intervals. All numbers are on the
  corrected forecast-observation day pairing (#106).

Numbers, method and caveats:
[v2 evidence base](docs/v2-evidence-base-results.md),
[single-source and independence](docs/single-source-and-independence-results.md),
[scorecard and significance](docs/scorecard-and-significance.md),
[pre-registered claims](docs/preregistration.md).

## Repository layout

```
src/weavr/        the library: grid, blending, calibration, verification
scripts/          data-store builders, tier runners, scorecard, daily pipeline
dashboard/        FastAPI service and example data
dashboard-web/    the front end (vanilla JS, vendored MapLibre and pmtiles)
results/          committed result tables (CSV)
docs/             scope notes, result write-ups, the development log
tests/            unit and API tests
Improvements/     the forward plan: steps, order and execution prompts
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest --cov=weavr
```

CI runs lint, type-check and tests on Python 3.11 and 3.12 for pull requests
that touch code or committed results (documentation-only and front-end-only
changes skip it). Split data with `weavr.splits` only, never a random split.

## Documentation

| | |
|---|---|
| Plan and history | [Phase plan](docs/phase-plan.md) · [Development log](docs/development-log.md) · [Improvement plan](Improvements/WEAVR-SIH-improvement-plan.md) · [Execution plan](Improvements/EXECUTION-PLAN.md) |
| Data | [Data sources](docs/data-sources.md) · [Grid and time convention](docs/grid-and-time-convention.md) · [Baseline store](docs/baseline-store.md) |
| Methods and results | [Tier 0](docs/tier0-baseline-results.md) · [Tier 1](docs/tier1-regional-weights-results.md) · [Tier 2](docs/tier2-hierarchical-baseline-results.md) · [Tier 2b combined](docs/tier2b-combined-results.md) · [Tail repair](docs/tail-repair-results.md) · [Regime conditioning](docs/phase5-regime-conditioned-results.md) |
| Dashboard | [Dashboard scope](docs/phase7-dashboard-scope.md) · [Basemap](docs/basemap-scope.md) |

## Acknowledgements and data

Forecast data: ECMWF open data (CC BY 4.0), WeatherBench 2 and Google
DeepMind's GraphCast. Observations: India Meteorological Department gridded
rainfall. Basemap: © OpenStreetMap contributors (ODbL), served via
Protomaps, with Natural Earth and MapLibre GL JS. Each source's licence and
attribution is recorded in [docs/data-sources.md](docs/data-sources.md) and
[docs/basemap-scope.md](docs/basemap-scope.md).

## License

MIT, as declared in `pyproject.toml`.

<div align="center"><sub>Built for Smart India Hackathon, problem statement 26081.</sub></div>
