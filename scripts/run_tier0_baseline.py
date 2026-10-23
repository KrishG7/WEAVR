#!/usr/bin/env python3
"""Run the Tier 0 equal-weight baseline: Phase 1's actual exit criterion.

Computes the simplest possible blend -- an unweighted mean across all
forecast sources that carry a given variable -- and scores it with every
function in src/weavr/verify.py, against IMD gauges, using the split
strategy from src/weavr/splits.py. This is the number every later tier
(Phase 3's regional weights, Phase 4's hierarchical BMA/EMOS, ...) must beat
to justify its own added complexity. See docs/tier0-baseline-results.md for
the actual numbers this produces and every limitation found running it.

Scope, decided by what the data actually supports, not assumed:

- **Precipitation only.** The baseline store's only IMD ground truth is
  `imd_observed.rain` -- there is no IMD temperature observation in the
  store at all (build_baseline_store.py only ever pulled IMD's `rain`
  product). IMD does publish a gridded temperature product, but at 1 deg
  native resolution -- coarser than weavr's locked 0.25 deg grid, so
  `regrid_to_common` would (correctly) refuse it as upsampling, the same
  guard that already blocks a temperature ACC climatology (see
  docs/phase-1-data-requirements.md). Verifying 2m_temperature against real
  IMD ground truth is therefore not possible with data available today;
  Tier 0 scores precipitation (`total_precipitation_24hr`) only, and this is
  documented rather than silently narrowing scope.
- Of the 4 forecast sources, only graphcast/hres/ifs_ens_mean carry
  `total_precipitation_24hr` (pangu's archive has no precipitation variable
  at all, per docs/baseline-store.md) -- the equal-weight mean is over these
  3, not all 4.
- CRPS/Brier are not computed: no ensemble-shaped source exists in the store
  (deferred to Phase 2's lagged AI ensembles, per
  docs/phase-1-data-requirements.md).
- leave_one_year_out is not usable (the store is one season); every lead
  time uses seasonal_block_split instead, which this script states
  explicitly rather than silently falling back.

A real unit bug was caught running this, not assumed away: WeatherBench 2's
`total_precipitation_24hr` is in **meters** (the ECMWF/GRIB convention),
while IMD's `rain` is in **millimeters**. Scoring them directly against each
other without converting produces a "bias" of roughly -obs_mean -- which
looks like a plausible negative number, not an obvious crash -- because the
forecast is effectively ~0 in the wrong units. Caught by checking the actual
magnitudes (`fc.mean() ~ 0.006`, `obs.mean() ~ 5.7`) before trusting any
score, not by assuming matching units. Fixed by multiplying forecast
precipitation by 1000 before scoring (`PRECIP_M_TO_MM` below).

Usage:
    python scripts/run_tier0_baseline.py [--store PATH] [--climatology PATH]
                                          [--out-csv PATH]
                                          [--test-fraction F]
                                          [--neighborhood-size N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seeps_climatology import load_climatology  # noqa: E402

from weavr import verify as V  # noqa: E402
from weavr.grid import align_forecast_to_imd_day  # noqa: E402
from weavr.score_io import (  # noqa: E402
    FORCE_HELP,
    PerDayScoreWriter,
    guard_result_overwrites,
    per_day_scores,
    resolve_result_paths,
)
from weavr.splits import (  # noqa: E402
    InsufficientTimeBlocksError,
    iter_evaluation_folds,
    leave_one_year_out,
    seasonal_block_split,
)
from weavr.stores import open_multi_season, resolve_store_paths  # noqa: E402

PRECIP_VARIABLE = "total_precipitation_24hr"
PRECIP_M_TO_MM = 1000.0  # WeatherBench 2 precip is in meters; IMD's is in mm.
LEAD_HOURS = [24, 48, 72, 96, 120]
FORECAST_SOURCE_NAMES = ["graphcast", "pangu", "hres", "ifs_ens_mean"]
NEIGHBORHOOD_SIZE = 5  # grid cells (~1.25 deg edge) -- a representative mesoscale window.


def _align_to_imd_day(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    """Reindex a forecast DataArray from init_time to the IMD day its lead validates on.

    A thin name for `weavr.grid.align_forecast_to_imd_day`, kept because the
    other runners import it from here. The rule itself lives in one place
    only: an earlier copy here labelled the IMD day by its start date and
    paired every forecast with the day before the one it covers (#106).
    """
    return align_forecast_to_imd_day(da, lead_hours)


def equal_weight_mean_and_obs(
    sources: dict[str, xr.Dataset],
    obs: xr.Dataset,
    variable: str,
    lead_hours: int,
) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Build the equal-weight mean forecast and its matching obs, aligned by IMD day.

    Returns (mean_forecast, obs_aligned, contributing_source_names), both
    DataArrays indexed by a shared `sample` dim, with any sample missing IMD
    ground truth already dropped.
    """
    contributing = [name for name, ds in sources.items() if variable in ds.data_vars]
    aligned = []
    for name in contributing:
        da = sources[name][variable].sel(prediction_timedelta=lead_hours).load()
        if variable == PRECIP_VARIABLE:
            da = da * PRECIP_M_TO_MM
        aligned.append(_align_to_imd_day(da, lead_hours))

    stacked = xr.concat(aligned, dim="source")
    mean_forecast = stacked.mean(dim="source", skipna=True)

    obs_aligned = obs["rain"].reindex(time=mean_forecast["sample"].values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    return (
        mean_forecast.isel(sample=has_obs.values),
        obs_aligned.isel(sample=has_obs.values),
        contributing,
    )


def train_test_masks(
    sample_times: pd.DatetimeIndex, test_fraction: float
) -> tuple[str, np.ndarray, np.ndarray]:
    """The one split rule every Tier-0-comparable script uses.

    Extracted out of `score_lead` so scripts that need the *train* side too
    (e.g. scripts/run_single_source_baseline.py picking its best single
    member on train, scripts/run_independence_diagnostic.py computing error
    correlations on train only) get literally the same masks rather than
    re-deriving the rule and hoping it still matches.

    Returns `(split_kind, train_mask, test_mask)`.
    """
    split_kind = "leave_one_year_out"
    try:
        # Only meaningful with 2+ distinct years -- checked, not assumed:
        # this raises against today's single-season store, so the except
        # branch is what actually runs. Kept so this script automatically
        # starts using the more rigorous split the day a second season is
        # added to the store, with no code change needed here.
        train_mask, test_mask = next(iter(leave_one_year_out(sample_times)))
    except InsufficientTimeBlocksError:
        split_kind = "seasonal_block_split (single-season store; leave_one_year_out not usable)"
        train_mask, test_mask = seasonal_block_split(sample_times, test_fraction=test_fraction)
    return split_kind, train_mask, test_mask


def score_lead(
    mean_forecast: xr.DataArray,
    obs: xr.DataArray,
    climatology: xr.Dataset,
    test_fraction: float,
    thresholds: tuple[float, ...],
    neighborhood_size: int,
) -> list[dict]:
    n_samples = mean_forecast.sizes["sample"]
    sample_times = pd.DatetimeIndex(mean_forecast["sample"].values)
    climatology_mean = climatology["rain"].mean(dim="time", skipna=True)

    folds = list(iter_evaluation_folds(sample_times, test_fraction=test_fraction))
    rows: list[dict] = []
    test_forecasts = []
    test_obs_list = []

    for train_mask, test_mask, split_label in folds:
        test_forecast = mean_forecast.isel(sample=test_mask)
        test_obs = obs.isel(sample=test_mask)
        test_forecasts.append(test_forecast)
        test_obs_list.append(test_obs)

        fss_by_threshold = {
            t: float(
                V.fss(
                    test_forecast,
                    test_obs,
                    threshold=t,
                    neighborhood_size=neighborhood_size,
                )
            )
            for t in thresholds
        }
        contingency = V.contingency_scores(test_forecast, test_obs, thresholds=thresholds)

        split_kind = (
            "leave_one_year_out"
            if split_label != "seasonal_block_split"
            else "seasonal_block_split (single-season store; leave_one_year_out not usable)"
        )
        rows.append(
            {
                "n_samples": n_samples,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "split": split_kind,
                "fold": split_label,
                "rmse_mm": float(V.rmse(test_forecast, test_obs)),
                "bias_mm": float(V.bias(test_forecast, test_obs)),
                "acc": float(V.acc(test_forecast, test_obs, climatology_mean)),
                "seeps": float(
                    V.seeps(test_forecast, test_obs, climatology["rain"], climatology_dim="time")
                ),
                "fss": fss_by_threshold,
                "contingency": {
                    t: {k: float(v) for k, v in s.items()} for t, s in contingency.items()
                },
            }
        )

    if len(folds) > 1:
        pooled_forecast = xr.concat(test_forecasts, dim="sample")
        pooled_obs = xr.concat(test_obs_list, dim="sample")

        fss_pooled = {
            t: float(
                V.fss(
                    pooled_forecast,
                    pooled_obs,
                    threshold=t,
                    neighborhood_size=neighborhood_size,
                )
            )
            for t in thresholds
        }
        contingency_pooled = V.contingency_scores(
            pooled_forecast, pooled_obs, thresholds=thresholds
        )

        rows.append(
            {
                "n_samples": n_samples,
                "n_train": n_samples,
                "n_test": int(pooled_forecast.sizes["sample"]),
                "split": "leave_one_year_out",
                "fold": "pooled",
                "rmse_mm": float(V.rmse(pooled_forecast, pooled_obs)),
                "bias_mm": float(V.bias(pooled_forecast, pooled_obs)),
                "acc": float(V.acc(pooled_forecast, pooled_obs, climatology_mean)),
                "seeps": float(
                    V.seeps(
                        pooled_forecast,
                        pooled_obs,
                        climatology["rain"],
                        climatology_dim="time",
                    )
                ),
                "fss": fss_pooled,
                "contingency": {
                    t: {k: float(v) for k, v in s.items()}
                    for t, s in contingency_pooled.items()
                },
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="One or more baseline store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--store", default=None, help="Legacy single store path")
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for the per-day files, and the default parent for the output CSV(s).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Aggregated CSV path (default: <results-dir>/tier0_baseline.csv)",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--neighborhood-size", type=int, default=NEIGHBORHOOD_SIZE)
    args = parser.parse_args()
    _paths = resolve_result_paths(
        args.results_dir,
        {
            "out_csv": "tier0_baseline.csv",
        },
        {
            "out_csv": args.out_csv,
        },
    )
    args.out_csv = _paths["out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    store_paths = resolve_store_paths(
        specified_paths=args.baseline_stores,
        legacy_single_path=args.store,
    )
    sources = {
        name: open_multi_season(store_paths, group=name)
        for name in FORECAST_SOURCE_NAMES
    }
    obs = open_multi_season(store_paths, group="imd_observed").load()
    climatology = load_climatology(args.climatology).load()

    print(
        f"Tier 0 baseline: equal-weight mean, scored against {store_paths}'s imd_observed group"
    )
    print(f"Climatology: {args.climatology} ({climatology.sizes['time']} JJAS days)")
    print("Scope: precipitation only -- no matching-resolution IMD temperature ground truth")
    print("CRPS/Brier: not computed -- no ensemble-shaped source in this store (deferred, Phase 2)")

    rows: list[dict] = []
    per_day_writer = PerDayScoreWriter(args.results_dir)
    for lead_hours in LEAD_HOURS:
        mean_forecast, obs_aligned, contributing = equal_weight_mean_and_obs(
            sources, obs, PRECIP_VARIABLE, lead_hours
        )
        sample_times = pd.DatetimeIndex(mean_forecast["sample"].values)
        for _, test_mask, split_label in iter_evaluation_folds(
            sample_times, test_fraction=args.test_fraction
        ):
            per_day_writer.add(
                "tier0",
                lead_hours,
                per_day_scores(
                    mean_forecast.isel(sample=test_mask),
                    obs_aligned.isel(sample=test_mask),
                    fold=split_label,
                ),
            )

        lead_results = score_lead(
            mean_forecast,
            obs_aligned,
            climatology,
            args.test_fraction,
            V.IMD_RAIN_THRESHOLDS_MM,
            args.neighborhood_size,
        )
        for result in lead_results:
            result["lead_hours"] = lead_hours
            result["sources"] = "+".join(contributing)
            rows.append(result)
            print(f"[lead {lead_hours:>3}h | fold {result['fold']}] {result}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lead_hours",
        "sources",
        "fold",
        "split",
        "n_samples",
        "n_train",
        "n_test",
        "rmse_mm",
        "bias_mm",
        "acc",
        "seeps",
    ]
    for t in V.IMD_RAIN_THRESHOLDS_MM:
        fieldnames += [f"fss_{t}mm", f"pod_{t}mm", f"far_{t}mm", f"csi_{t}mm", f"ets_{t}mm"]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fieldnames if k in row}
            for t in V.IMD_RAIN_THRESHOLDS_MM:
                flat[f"fss_{t}mm"] = row["fss"][t]
                flat[f"pod_{t}mm"] = row["contingency"][t]["pod"]
                flat[f"far_{t}mm"] = row["contingency"][t]["far"]
                flat[f"csi_{t}mm"] = row["contingency"][t]["csi"]
                flat[f"ets_{t}mm"] = row["contingency"][t]["ets"]
            writer.writerow(flat)

    per_day_paths = per_day_writer.flush()
    print(f"\nWrote {out_path}")
    for path in per_day_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
