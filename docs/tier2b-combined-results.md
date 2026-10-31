# Tier 2b Combined Combiners (EMOS-CSG + BMA Stacking)

## 1. Overview and Motivation

Phase 4 evaluated two advanced probabilistic combiners:
- **EMOS-CSG** (`src/weavr/emos.py`): Parametric Censored Shifted Gamma Distribution regression per rain-intensity bin. Excels at discrimination and sharp threshold forecasts.
- **Hierarchical BMA** (`src/weavr/bma.py`): Mixture modeling with point-mass at zero and power-transformed Gamma components. Excels at calibration and reliability across spread regimes.

As demonstrated by **Javanshiri (2021)** and **Ji (2025)**, parametric EMOS and mixture BMA exhibit complementary trade-offs: EMOS provides superior discrimination for heavy events, whereas BMA produces better-calibrated probabilities. Slide 2 previously asserted that WEAVR *"uses the strongest combiner per bin"*, but no such combined selector existed in the codebase. Step 13 closes this gap with two principled combination methods implemented in `src/weavr/stacking.py`.

---

## 2. Combination Methods

### A. Per-Bin x Lead Selection
Selects the combiner with the lowest mean CRPS on the training fold for each `(rain_bin, lead)` stratum:
$$\text{Combiner}^*(b, l) = \arg\min_{c \in \{\text{EMOS-CSG}, \text{BMA}\}} \overline{\text{CRPS}}_{\text{train}}(b, l, c)$$

- **Tie-break rule**: If CRPS values match within $10^{-9}$, prefers `emos_csg` to favor parametric efficiency.
- **Sparse bin fallback**: If a bin has insufficient training cases (e.g. extreme precipitation), falls back to the domain-wide pooled-best combiner.

### B. Quantile Averaging (Vincentization)
Rather than averaging probabilities (linear pooling), which artificially flattens sharp distributions and leads to over-dispersion, **Lichtendahl, Grushka-Cockayne & Winkler (2013)** demonstrated that **quantile averaging (Vincentization)** preserves distribution shape, unimodality, and sharpness:
$$q_{\text{avg}}(\tau) = (1 - w) \cdot q_{\text{EMOS}}(\tau) + w \cdot q_{\text{BMA}}(\tau), \quad \tau \in (0, 1)$$

- **Monotonicity**: Because $q_{\text{EMOS}}(\tau)$ and $q_{\text{BMA}}(\tau)$ are monotone non-decreasing in $\tau$ and $w \in [0, 1]$, $q_{\text{avg}}(\tau)$ is monotone non-decreasing by construction.
- **CSGD Quantiles**: Inverted analytically from the CSGD survival function, accounting for the point mass at zero:
  $$p_0 = F_X(-\delta / \theta; \kappa), \quad q_{\text{EMOS}}(\tau) = \begin{cases} 0 & \text{if } \tau \le p_0 \\ \max(0, \delta + \theta \cdot F_X^{-1}(\tau; \kappa)) & \text{if } \tau > p_0 \end{cases}$$
- **BMA Quantiles**: Exact, by inverting the closed-form mixture CDF (`weavr.bma.bma_predictive_quantiles`): interpolated on a value grid reaching 1e5 mm, and solved by bisection for any level the grid does not reach. An earlier version read them from 500 Monte Carlo draws, which understated the upper quantiles by enough to matter at H10's margins.

### C. Linear Pool (Reference Baseline)
Standard mixture of predictive probabilities:
$$P_{\text{pool}}(Y > t) = (1 - w) P_{\text{EMOS}}(Y > t) + w P_{\text{BMA}}(Y > t)$$

---

## 3. Quantile-Based CRPS Approximation

To evaluate quantile-averaged predictive distributions without closed forms, `crps_from_quantiles` integrates the pinball loss across 99 quantile levels:
$$\text{CRPS}(F, y) = 2 \int_0^1 (y - q(\tau)) \cdot (\tau - \mathbb{I}(y < q(\tau))) \, d\tau$$
Verified against closed-form `csgd_crps` with $< 1.5\%$ relative error across all test conditions in `tests/test_stacking.py`.

**What the 99 levels leave out, measured (31 Oct).** The runner uses 99 levels,
0.01 to 0.99 (`DEFAULT_QUANTILE_LEVELS`); the 1.5% check above uses 199 levels
from 0.005, so it did not cover the runner's own grid. A second test now does:
on the runner's grid the estimate is never above the closed form and at most
2% below it (1.72% at worst on the test's cases). Leaving out the two 1% tails
makes every arm read low. On the same days and cells, against the exact CRPS
that Tier 2 computes for the same fits, the 99-level CRPS is lower by 0.0084–
0.0090 mm for EMOS-CSG and 0.0090–0.0099 mm for BMA, depending on lead; the two
differ by at most 0.0010 mm at any lead. Can the omission flatter a
combination over a parent? Per cell, by at most the gap between the two
parents' omissions: the pinball loss is convex, so a quantile average omits no
more than the weighted mean of what its parents omit. Pooled, that gap is at
most 0.0010 mm, about a tenth of H10's smallest margin (0.0094 mm, 24 h vs
`bma`). That is reassurance, not a bound, because cell-level gaps can be larger
than their mean. The bound needs an exact CRPS for the combinations, which is
not implemented.

---

## 3a. Measured Results

Two-season daily LOYO (2018 JJAS / 2020 JJAS), leads 24–120 h, Tier 1 blend
scope, against the IMD observed grid, on the corrected forecast-observation
pairing (#106; 234–242 test days per lead). Pooled scores are cell-weighted over both
folds and all five leads. Reproduce with
`scripts/run_tier2b_combined.py --fit-weight` (`--fit-weight` adds the
train-fitted `quantile_avg` arm alongside the fixed w=0.5 one; without it both
arms use 0.5). Results: `results/tier2b_combined.csv`, `_paired.csv`,
`_train.csv` and `results/per_day/tier2b_*`.

| arm | pooled CRPS (mm) | PIT mean | PIT MAD from 0.5 | central-90% coverage | role |
|---|---|---|---|---|---|
| `quantile_avg_fixed` (w=0.5) | **4.2023** | 0.494 | 0.252 | 0.899 | combination |
| `linear_pool` | **4.2024** | 0.494 | 0.251 | 0.900 | comparison only |
| `quantile_avg` (w fitted on train) | 4.2039 | 0.495 | 0.253 | 0.900 | combination |
| `per_bin` | 4.2134 | 0.501 | 0.252 | 0.905 | combination |
| `bma` | 4.2193 | 0.500 | 0.251 | 0.904 | parent |
| `emos_csg` | 4.2288 | 0.499 | 0.250 | 0.900 | parent |

A uniform PIT has mean 0.5, mean absolute deviation from 0.5 of 0.25, and 90%
of observations inside the forecast's own central 90% interval. All six arms sit
on those values: MAD 0.250–0.253 and coverage 0.899–0.905 pooled, and 0.896–0.909
in every lead x fold row (`results/tier2b_combined.csv`; per day in
`results/per_day/tier2b_*`). So the combined distributions are calibrated in
the sense of the whole predictive distribution, not only on average; this says
nothing about the tail above 64.5 mm (§5a). Dry observations enter as the
expectation of the randomised PIT (`pit_cell_statistics`).

Brier score by IMD threshold (cell-weighted, lower is better):

| arm | 7.5 mm | 35.5 mm | 64.5 mm | 115.6 mm | 204.5 mm |
|---|---|---|---|---|---|
| `quantile_avg_fixed` | 0.12978 | **0.03660** | 0.01191 | 0.00255 | **0.00027** |
| `linear_pool` | 0.12976 | 0.03663 | **0.01190** | 0.00254 | 0.00027 |
| `quantile_avg` | **0.12975** | 0.03664 | 0.01192 | 0.00255 | 0.00027 |
| `per_bin` | 0.13010 | 0.03672 | 0.01191 | **0.00254** | 0.00027 |
| `bma` | 0.13021 | 0.03678 | 0.01195 | 0.00255 | 0.00027 |
| `emos_csg` | 0.13033 | 0.03693 | 0.01200 | 0.00257 | 0.00027 |

The event is `obs >= t` (IMD's categories are "at least"), the same as every
other runner and `score_io.exceedance_event`; Tier 2b used `obs > t` until
this follow-up. The data has 6 observations exactly on 7.5 mm and 1 on 35.5 mm
(2020) and none on 64.5, 115.6 or 204.5, so only the 7.5 and 35.5 columns moved,
by at most 1e-6 pooled (0.0002 on one day); the table below is unchanged at
five decimals and no verdict moved.

The Brier spread across arms is small -- the fifth decimal at 204.5 mm, the
third at 7.5 mm -- so Brier does not separate these methods. CRPS does, which
says the trade-off is about the *shape* of the predictive distribution rather
than threshold decisions.

**Every arm is calibrated on average: PIT means 0.49–0.50.** An earlier version
of this file reported 0.61–0.63 for every arm and read it as a shared
over-forecast upstream of the combination rule. It was the PIT itself: a dry
observation was scored as `F(0) = p0`, the top of the point mass at zero, when
every level in `[0, p0]` is equally consistent with it. A forecast that is
calibrated by construction scored 0.638 under that rule. `pit_values` now
scores the point mass as the randomised PIT's expectation, and a regression
test pins it (`tests/test_run_tier2b_combined.py`). Only `pit_mean` changed:
every CRPS, Brier and paired difference is byte-identical between the two
codes.

**What changed against the previous runs.** On the corrected pairing (#106),
every arm's pooled CRPS fell by about 0.34 mm (BMA 4.5645 → 4.2194, EMOS-CSG
4.5657 → 4.2288), and the ordering held: every combination beats both parents
on pooled CRPS, and BMA stays ahead of EMOS-CSG, now by 0.009 mm instead of
0.001. Before that, both parents and every combination moved because BMA did:
the run before used BMA quantiles from a library with two defects, fixed on
#102 (a variance floor compounded by an independent `theta` clip, and Monte
Carlo quantiles), which put BMA at 4.5918 mm.

**31 October: BMA's quantiles are now inverted in cube root.** The mixture CDF
rises steeply just above 0 mm, and interpolating it linearly in mm on a 1 mm
grid misplaced the low quantiles. The same grid, with a double count, made
Tier 2's exact BMA CRPS read 0.175–0.182 mm high
(`docs/v2-evidence-base-results.md` §7, F5). The grid is now
even in the cube root of rainfall. BMA's pooled CRPS moves by −0.00013 mm,
every combination by less than 0.0001 mm, `emos_csg` not at all, and H10 is
unchanged at 5 of 5. The tables above are from that run.

---

## 4. Pre-registered Hypothesis H10

- **Null Hypothesis**: The combined combiner (quantile averaging or per-bin selection) does not improve upon both individual constituent combiners.
- **Pass Rule**: Significantly lower CRPS than both standalone EMOS-CSG and BMA at $\ge 3$ of 5 leads, with the 95% paired block-bootstrap CI strictly excluding 0, for the **primary arm nominated on train-fold CRPS**.

### 4.1 H10 verdict: **PASS** (5 of 5 leads)

The primary arm is nominated on **train**-fold CRPS, pooled over the two
folds, never on test. On the corrected pairing (#106) and with the per-bin
train CRPS fixed (§7), it is `per_bin` at 24 h and `quantile_avg` at 48–120 h.
Paired daily CRPS differences, negative meaning the combination is better, 95%
paired moving-block bootstrap over both seasons (234–242 test days per lead):

| lead | nominated | vs `emos_csg` | vs `bma` | both? |
|---|---|---|---|---|
| 24 h | `per_bin` | −0.0454 CI[−0.0655, −0.0263] | −0.0094 CI[−0.0168, −0.0034] | **pass** |
| 48 h | `quantile_avg` | −0.0266 CI[−0.0393, −0.0141] | −0.0140 CI[−0.0218, −0.0069] | **pass** |
| 72 h | `quantile_avg` | −0.0193 CI[−0.0311, −0.0086] | −0.0173 CI[−0.0261, −0.0094] | **pass** |
| 96 h | `quantile_avg` | −0.0141 CI[−0.0293, −0.0012] | −0.0181 CI[−0.0290, −0.0085] | **pass** |
| 120 h | `quantile_avg` | −0.0167 CI[−0.0339, −0.0007] | −0.0161 CI[−0.0256, −0.0080] | **pass** |

**5 of 5 leads beat both parents, above the pre-registered 3. H10 passes as
registered.** The nomination margins on train are small: `per_bin` leads
`quantile_avg` by 0.0016 mm at 24 h, and trails it by 0.0024–0.0057 mm at
48–120 h.

Every arm is reported separately, so the verdict on the nominated arm cannot
hide how the others did:

| arm | leads where it beat **both** parents |
|---|---|
| `quantile_avg` (weight fitted on train, 0.65–0.80 towards BMA) | **5 / 5** |
| `quantile_avg_fixed` (w=0.5, the pre-registered weight) | **4 / 5** (not 24 h vs `bma`: −0.0105, CI[−0.0232, +0.0022]) |
| `linear_pool` | **4 / 5** (comparison only; not 24 h vs `bma`) |
| `per_bin` | **2 / 5** (24 and 48 h; at 72–120 h it never separates from `emos_csg`) |

**The pass does not depend on the weight.** These results use `--fit-weight`,
as the committed Tier 2b results have since 2048060. With the pre-registered
fixed weight of 0.5 the quantile-averaging arm still beats both parents at 4
of 5 leads, so H10 passes either way.

**Why this differs from the previous verdict (FAIL, 2 of 5).** Two things
changed, and both are recorded rather than averaged away:

1. **The pairing (#106).** Every arm and both parents moved by about 0.34 mm
   (§3a); the margins between them changed too.
2. **The nomination was decided on the wrong numbers.** `per_bin`'s train CRPS
   averaged only the bins where BMA had been picked (§7, last row), so at the
   leads where BMA won only small heavy-rain bins it looked better than it
   was. The previous run nominated `per_bin` at 24, 48 and 72 h by margins of
   0.001–0.007 mm, which is the size of that bias, so the previous FAIL may
   itself have rested on it. It is not re-decided here on the old data; the
   verdict above is the one the corrected pairing and the corrected nomination
   support.

The verdict before that (PASS, 3 of 5) rested on BMA quantiles from the
defective library (#102) and was already withdrawn.

---

## 5. What the verdict licenses

**It licenses H10:** a combination of EMOS-CSG and BMA beats both on CRPS at 5
of 5 leads, every interval excluding 0, nominated on train before test was
seen.

**The recommended probabilistic combiner is quantile averaging**
(Vincentization) of EMOS-CSG and BMA. It is the nominated arm at four of five
leads and the only arm that beats both parents at all five, by 0.014–0.027 mm
of daily CRPS against EMOS-CSG and 0.014–0.018 mm against BMA (48–120 h).

**It still does not license Slide 2's "uses the strongest combiner per
bin".** That claim is `per_bin`, which beats both parents at only 2 of 5
leads. What the evidence supports is: *WEAVR combines EMOS-CSG and BMA by
averaging their quantile functions, which beats both at every lead (95%
intervals excluding 0).*

**Three qualifications:**

1. **The margin is small.** 0.01–0.05 mm of daily CRPS against a ~4.2 mm
   total: real and significant under the paired test, not the size of a
   method change.
2. **The weight is fitted.** 0.65–0.80 towards BMA by lead and fold, read from
   `results/tier2b_combined_train.csv`; the pre-registered 0.5 gives one lead
   fewer, as above.
3. **CRPS is the verdict's metric, not tail reliability.** H10 says nothing
   about how well-calibrated the combined distribution is above 64.5 mm; see
   §5a for what that means for the extreme-rain map.

### 5a. The dashboard flag (#75)

`scripts/export_dashboard_example_grids.py --combiner quantile_avg` (with the
`--fit-*` stores) draws the extreme-probability map from the recommended
combiner, configured exactly as evaluated: the EMOS source and weight come
from the Tier 2b fold whose test season is the displayed one (2020, trained on
2018), and EMOS-CSG and BMA quantiles are averaged per (bin, region) as Tier
2b scores them. The GPD tail needs only P(Y > 64.5 mm) from the distribution
below it, so it splices onto the averaged distribution unchanged; that
probability is read off the averaged quantile function on levels reaching
1 − 1e-10 (`weavr.stacking.exceedance_from_quantiles`), because the 99 levels
used for CRPS cannot resolve anything below 1%. With weight 0 the path
reproduces the EMOS map to 1.3e-5 relative error in every cell
(`test_dashboard_quantile_avg_at_weight_zero_is_the_emos_map`).

**The default stays `emos_csg`, and the flag is experimental.** On the real
2020 display day the quantile-averaged map's probabilities are bounded,
ordered (P(>204.5) ≤ P(>115.6) everywhere) and never fall back, but they are
1.9 times the EMOS map's on average at 24 h and 1.4 times at 120 h, with a
median per-cell ratio of 10 and 4.9 (the highest cell is about the same:
0.021 against 0.023 at 115.6 mm, 24 h). BMA's upper tail is heavier than
EMOS-CSG's, and at a weight of 0.8 towards BMA that lifts the anchor of the
splice. §5b measures whether that is better or worse calibrated.

### 5b. Reliability of the quantile-averaged extreme probabilities (#75 follow-up)

`scripts/run_quantile_avg_reliability.py` scores the quantile-averaged
distribution with the pooled GPD tail spliced above u = 64.5 mm exactly as
step 12 scored the EMOS tail (`docs/tail-repair-results.md`): the same
leave-one-year-out folds and test cells, the tail fitted on the IMD archive
minus the test year, the same reliability bins, Brier on cells IMD observes
(`obs >= t`), climatology without the test year, expected-count SEDI. The EMOS
source and weight are Tier 2b's for each lead and fold, read from
`results/tier2b_combined_train.csv`. Outputs: `results/quantile_avg_extreme.csv`,
`results/quantile_avg_reliability.csv`. The runner was cross-checked first
with weight 0 (EMOS only) against step 12's `csgd_gpd_tail`: Brier within 0.02%,
SEDI within 0.001, identical cells, events and climatology.

| threshold | lead | BSS quantile-avg | BSS EMOS tail | SEDI quantile-avg | SEDI EMOS tail | forecast / observed, quantile-avg | forecast / observed, EMOS tail |
|---|---|---|---|---|---|---|---|
| 115.6 mm | 24 h | 0.0414 | 0.0423 | 0.298 | 0.311 | 1.16x | 0.98x |
| 115.6 mm | 48 h | 0.0406 | 0.0410 | 0.296 | 0.307 | 1.13x | 0.97x |
| 115.6 mm | 72 h | 0.0387 | 0.0386 | 0.290 | 0.301 | 1.13x | 0.97x |
| 115.6 mm | 96 h | 0.0292 | 0.0302 | 0.265 | 0.277 | 1.13x | 0.95x |
| 115.6 mm | 120 h | 0.0260 | 0.0256 | 0.255 | 0.263 | 1.15x | 0.94x |
| 204.5 mm | 24 h | 0.0042 | 0.0062 | 0.222 | 0.241 | 1.42x | 1.18x |
| 204.5 mm | 48 h | 0.0042 | 0.0058 | 0.221 | 0.238 | 1.38x | 1.17x |
| 204.5 mm | 72 h | 0.0041 | 0.0051 | 0.219 | 0.232 | 1.37x | 1.16x |
| 204.5 mm | 96 h | 0.0014 | 0.0033 | 0.193 | 0.215 | 1.37x | 1.13x |
| 204.5 mm | 120 h | 0.0009 | 0.0024 | 0.186 | 0.204 | 1.39x | 1.12x |

**It is no better than the EMOS tail, and it over-forecasts more.**

- **Brier** is equal to five decimals at every lead and both thresholds, so Brier
  does not separate the two maps.
- **Brier skill** is slightly lower at 8 of 10 lead x threshold pairs (higher
  at 2), and **SEDI is lower at all 10**, by 0.008–0.022.
- **Calibration in total:** the EMOS tail forecasts 0.94–0.98x the observed
  frequency at 115.6 mm and 1.12–1.18x at 204.5 mm; the quantile average
  forecasts 1.13–1.16x and 1.37–1.42x. The "about twice" in the earlier
  version of this section was the single display day; over every test cell it
  is 1.14x and 1.39x in total.
- **Reliability** (pooled over leads and folds, 115.6 mm): both maps
  over-forecast the low-probability bins and under-forecast the high ones.
  Above about 2% the observed frequency exceeds the forecast for both (e.g. the
  0.05-0.1 bin: forecast 0.069, observed 0.106 for the quantile average and
  0.068 against 0.125 for the EMOS tail). In the low bins the quantile average
  over-forecasts more: in the 1e-4..1e-3 bin it forecasts 0.00039 against 0.00012
  observed (3.2x), the EMOS tail 0.00033 against 0.00020 (1.7x); in the
  1e-3..5e-3 bin 0.0029 against 0.0008 (3.5x) and 0.0025 against 0.0011 (2.4x).
- **Fallback cells** (neither parent fitted for the cell's bin and region, probability
  0): 78 observed cells per threshold, 14 of which are 115.6 mm events.

These are point estimates: the runner writes no per-day files, so there is no
paired block-bootstrap CI on the differences, and the Brier tie, the SEDI gap
and the Brier-skill gap should be read as consistent in sign, not as tested.
They do not support promoting this map. The H10 result stands: it is a CRPS
verdict, and this run only shows that the CRPS gain does not carry over to
the 115.6 and 204.5 mm exceedance probabilities.

**Decision: the dashboard default stays `emos_csg`.** `--combiner quantile_avg`
remains available and experimental.

---

## 6. Open problems this step did not fix

- **The CRPS is a 99-level quantile approximation** for all six arms,
  deliberately uniform so that no arm is advantaged by a better estimator.
- **The linear pool's inverse is interpolated**, not closed-form (§2 C). It is
  kept comparison-only for that reason, and a test pins it inside the parents'
  CRPS range so an inversion failure cannot pass as a pooling result.
- **The quantile-averaged extreme map is measured (§5b) and not better** than the
  EMOS tail: equal Brier, lower SEDI at every lead, 1.14x / 1.39x over-forecast
  in total. No paired CI was computed on those differences (§5b).

**Resolved since the first version of this file:**

- The Tier 2 `heavy` x NE1/SI "variance blow-up" was a defect in the BMA parent
  -- a negative-slope variance regression floored at `_TINY`, compounded by an
  independent clip of `theta` that moved the component mean to ~1850 mm. It is
  fixed on #102 (merged into this branch), and BMA quantiles are now exact
  (§2 B) rather than 500-draw Monte Carlo.
- The "universal PIT bias" was the dry-observation PIT bug (§3a).

---

## 7. Bugs found by reading the numbers, not the tests

This step needed five runs before this one. Every run exited 0, passed the test suite, and wrote
finite numbers; each was wrong in a way only inspection caught. They are
recorded because the pattern is the finding, not the individual slips.

| Bug | Symptom | Why tests missed it |
|---|---|---|
| `ci_low > 0` used as the improvement test | A combination winning by `d=-0.047`, CI `[-0.074,-0.022]`, was recorded as a loss | The helper returned a plausible boolean |
| Train table keyed `lead=0` while the test lookup asked for the real lead | `per_bin` silently collapsed to one parent everywhere, showing `0.000000` differences | A collapse looks exactly like a clean result |
| `sum(means) / sum(counts)` used as a mean | `per_bin` train CRPS read 7e-5 mm against `quantile_avg`'s 4.8 mm -- 56,000x -- so `<=` always picked `per_bin`, and H10 was decided on the weakest arm at 4 of 5 leads | Both values were finite and plausible-looking |
| Raw rainfall passed where a binary indicator belonged | Every `brier_*` column read ~281 mm^2 instead of a value in [0,1] | Nothing bounded a Brier score |
| `per_bin`'s train CRPS looked up its choices in the raw train table, where EMOS is labelled `emos_graphcast`, instead of the selection table, where it is `emos_csg` | Every bin that picked EMOS silently dropped out: `per_bin` train CRPS read 8.1 mm against 4.5 mm at 72 h on the 2018 fold, and at other leads was biased by the size of the nomination margins | Within 10x of the other arm, so the scale guard passed; the value was finite and, at most leads, plausible |
| `quantile_avg` was nominated on its train CRPS at the grid-minimum weight even when it ran at another | Inert in every run so far (every fold fitted an interior weight), but a fold that did not would have been nominated on one weight and scored on another | No run had exercised it |

Each is now covered by a test built from the numbers that exposed it, and the
third has a runtime guard (`comparable_train_crps`) that aborts the run rather
than nominating an arm from incomparable quantities. The fifth is now caught by
an invariant checked after every regeneration: picking the better parent per
bin can never be worse on train than either parent, and the broken value broke
it by 3.6 mm. The general lesson: the
suite checks that the code runs and that outputs are well-formed, and none of
that catches a computation that is well-formed and means the wrong thing.
Sanity bounds on internal quantities -- a Brier score in [0,1], two CRPS values
on a common scale -- are cheap and would have caught two of these immediately.
