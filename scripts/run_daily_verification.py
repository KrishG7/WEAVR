#!/usr/bin/env python3
"""Rolling verification and drift detection -- issue #8's last checkbox.

Reuses `weavr.drift`'s trailing-window rescoring and drift-flagging on top
of `weavr.verify`'s existing metrics (RMSE/bias, the same functions every
`run_tierN_*.py` script already calls), rather than a one-shot full-season
score.

## What this script actually runs against today, stated plainly

`docs/phase6-operational-scope.md` decided the real daily pipeline (step 4)
fetches AIFS/IFS/HRES fresh each day via `ecmwf_open_data.py`, not the
frozen `data/baseline_2020_jjas.zarr` store -- but step 4 hasn't been built
yet, so there is no real accumulated daily history to roll a window over
yet. This script demonstrates the real mechanism (`weavr.drift`) against
the only real accumulated history that exists today: the baseline store's
**18 real weekly JJAS-2020 samples** (`graphcast`/`hres`/`ifs_ens_mean`
precipitation vs. `imd_observed`, the same sources/alignment
`scripts/run_tier0_baseline.py` uses). Each real weekly sample stands in
for "one accumulated day" here -- a real, stated granularity mismatch with
`weavr.drift.TRAILING_WINDOW_SAMPLES = 14`'s own real-calendar-day
justification, not hidden. Once step 4's daily pipeline has been running
for a few weeks, this same code operates on literal daily history instead;
nothing here needs to change for that, since `weavr.drift` only ever sees
an ordered score series, indifferent to what real calendar time each entry
spans.

**Baseline overlaps the trailing window today, stated plainly.** With only
17-18 real historical samples total and a 14-sample trailing window, there
isn't enough real history to hold out a non-overlapping baseline period the
way a mature deployment would -- the baseline computed here
(`weavr.drift.compute_baseline_stats`) is fit from the *entire* real
sample series, the same series the trailing window is drawn from. This is
this project's own established small-sample caution
(`docs/tier2-hierarchical-baseline-results.md`'s own 57%-fallback finding,
`docs/phase5-regime-conditioned-results.md`'s own small-sample caveat)
applying here too: a "no drift detected" result from this run is
inconclusive-on-more-data, not proof today's pipeline is stable, exactly
the same honest distinction Phase 5 step 4 drew for its own go/no-go
result.

Checked directly, not just argued: whenever `n <= weavr.drift.
TRAILING_WINDOW_SAMPLES` (today's real case, `n=17` or `18`),
`trailing_window_value` and `compute_baseline_stats` are computed from the
exact same full sample series, so the rolling value collapses to precisely
the baseline mean -- drift *cannot* fire here no matter how large a real
shift is, not merely "is less sensitive." Proven in
`tests/test_run_daily_verification.py` (`TestRollingVerificationForLead::
test_baseline_overlapping_the_trailing_window_masks_a_shift_when_n_le_window`)
with a synthetic 500mm late-sample error injection that still scores
`is_drift=False`. This resolves on its own once step 4's daily pipeline has
accumulated more than `TRAILING_WINDOW_SAMPLES` real days of history to
hold a baseline period separate from the rolling window -- no code change
needed here for that, only more real data.

## Real historical spread this module's threshold is grounded in

Computed directly against `data/baseline_2020_jjas.zarr` (graphcast
precipitation, all 5 leads, per-sample domain-wide RMSE across the 18 real
weekly JJAS-2020 samples):

| Lead | n | mean (mm) | std (mm) | min (mm) | max (mm) |
|---|---|---|---|---|---|
| 24h  | 18 | 13.888 | 4.675 | 5.219 | 20.801 |
| 48h  | 18 | 13.493 | 3.867 | 5.182 | 21.578 |
| 72h  | 18 | 14.405 | 4.000 | 8.566 | 20.196 |
| 96h  | 17 | 15.038 | 3.837 | 9.942 | 23.260 |
| 120h | 17 | 14.672 | 3.906 | 9.267 | 21.160 |

`std/mean` sits at ~28-33% consistently across leads -- see
`src/weavr/drift.py`'s own module docstring for how this grounds
`TRAILING_WINDOW_SAMPLES`/`DRIFT_THRESHOLD_STD_MULTIPLIER`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from weavr import verify as V
from weavr.drift import (
    compute_baseline_stats,
    detect_drift,
    rescore_trailing_window,
    trailing_window_value,
)
from weavr.grid import align_forecast_to_imd_day
from weavr.score_io import FORCE_HELP, guard_result_overwrites, resolve_result_paths, write_rows_csv
from weavr.stores import DEFAULT_BASELINE_DAILY_STORES, open_multi_season, resolve_store_paths

PRECIP_VARIABLE = "total_precipitation_24hr"
PRECIP_M_TO_MM = 1000.0
LEAD_HOURS = [24, 48, 72, 96, 120]
# Same 3 precip-carrying sources scripts/run_tier0_baseline.py scores --
# pangu has no precipitation variable in its archive (docs/baseline-store.md).
PRECIP_SOURCE_NAMES = ["graphcast", "hres", "ifs_ens_mean"]


def _align_to_imd_day(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    """`weavr.grid.align_forecast_to_imd_day`, under this script's old name.

    This used to be a local copy of the rule, which is how five copies came to
    agree with each other while all being a day early (#106).
    """
    return align_forecast_to_imd_day(da, lead_hours)


def load_aligned(
    store: str | Sequence[str], source_name: str, lead_hours: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """One source's precipitation forecast and matching IMD obs, aligned by
    IMD day and dropped of any sample missing ground truth -- the same
    shape `scripts/run_tier0_baseline.py`'s own `equal_weight_mean_and_obs`
    produces, for one source instead of the equal-weight mean across all of
    them (this script scores each source's own rolling skill separately,
    since a drift in one source shouldn't be masked by an unaffected one).
    """
    paths = [store] if isinstance(store, (str, Path)) else list(store)
    # `open_multi_season` rather than a bare `open_zarr`, so both daily seasons
    # concatenate into one continuous series. That is the point of the v2 run:
    # with the seasons joined, the rolling window actually crosses the 2018 ->
    # 2020 GraphCast checkpoint change, which is what step 07 item 4 asks this
    # script to test. Run per season instead, the two windows never meet and the
    # checkpoint change is invisible by construction.
    fc = (
        open_multi_season(paths, group=source_name)[PRECIP_VARIABLE]
        .sel(prediction_timedelta=lead_hours)
        .load()
        * PRECIP_M_TO_MM
    )
    fc = _align_to_imd_day(fc, lead_hours)

    obs = open_multi_season(paths, group="imd_observed")["rain"].load()
    obs_aligned = obs.reindex(time=fc["sample"].values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    return fc.isel(sample=has_obs.values), obs_aligned.isel(sample=has_obs.values)


def rolling_verification_for_lead(
    store: str | Sequence[str], source_name: str, lead_hours: int
) -> dict:
    """One (source, lead)'s real per-sample RMSE series, rolling value, and
    drift flag -- see this module's own docstring for the real granularity
    caveats (weekly samples standing in for daily ones; baseline overlapping
    the trailing window) this run's numbers are subject to today.
    """
    forecast, obs = load_aligned(store, source_name, lead_hours)
    n = forecast.sizes["sample"]

    per_sample_rmse = [
        float(V.rmse(forecast.isel(sample=i), obs.isel(sample=i))) for i in range(n)
    ]

    baseline = compute_baseline_stats(per_sample_rmse, metric_name="rmse_mm")
    rolling_value = trailing_window_value(per_sample_rmse)
    # Cross-checked against rescore_trailing_window's own independent path
    # (domain-wide RMSE over the trailing sample slice directly, rather
    # than averaging each sample's own already-computed RMSE) -- the two
    # differ in general (RMSE doesn't commute with averaging per-sample
    # RMSEs), so both are reported rather than assumed identical.
    rolling_value_domain_wide = rescore_trailing_window(forecast, obs, metric_fn=V.rmse)
    drift = detect_drift(rolling_value, baseline)

    return {
        "source": source_name,
        "lead_hours": lead_hours,
        "n_samples": n,
        "baseline_mean_rmse_mm": baseline.mean,
        "baseline_std_rmse_mm": baseline.std,
        "baseline_n_samples": baseline.n_samples,
        "rolling_mean_of_per_sample_rmse_mm": rolling_value,
        "rolling_domain_wide_rmse_mm": rolling_value_domain_wide,
        "drift_threshold_mm": drift.threshold,
        "is_drift": drift.is_drift,
        "reason": drift.reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stores",
        nargs="+",
        default=None,
        help=(
            "One or more baseline stores, concatenated into one series "
            "(default: 2018 + 2020 daily, so the rolling window spans the "
            "GraphCast checkpoint change)"
        ),
    )
    parser.add_argument("--store", default=None, help="Legacy single store path")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for the per-day files, and the default parent for the output CSV(s).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Aggregated CSV path (default: <results-dir>/daily_verification.csv)",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    args = parser.parse_args()
    _paths = resolve_result_paths(
        args.results_dir,
        {
            "out_csv": "daily_verification.csv",
        },
        {
            "out_csv": args.out_csv,
        },
    )
    args.out_csv = _paths["out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    stores = resolve_store_paths(
        args.stores,
        args.store,
        DEFAULT_BASELINE_DAILY_STORES,
        "data/baseline_2020_jjas.zarr",
    )
    print(f"Rolling verification + drift detection against {stores}")
    print(
        "NOTE: standing in each real weekly JJAS-2020 sample for one "
        "accumulated day -- see this script's own module docstring."
    )

    rows = []
    for source_name in PRECIP_SOURCE_NAMES:
        for lead_hours in LEAD_HOURS:
            row = rolling_verification_for_lead(stores, source_name, lead_hours)
            rows.append(row)
            flag = "DRIFT" if row["is_drift"] else "ok"
            print(
                f"[{source_name:>12} lead {lead_hours:>3}h] {flag:>5} -- {row['reason']}"
            )

    out_path = write_rows_csv(args.out_csv, rows)

    n_drift = sum(1 for r in rows if r["is_drift"])
    print(f"\n{n_drift} of {len(rows)} (source, lead) cells flagged as drift.")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
