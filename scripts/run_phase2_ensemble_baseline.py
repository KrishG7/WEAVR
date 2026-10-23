#!/usr/bin/env python3
"""Score Phase 2's lagged AI ensembles with CRPS/Brier -- Phase 2's exit criterion.

The probabilistic counterpart to Phase 1's Tier 0 run
(scripts/run_tier0_baseline.py): where Tier 0 scored a deterministic
equal-weight mean, this scores the +/-4-starts, 12h-spaced lagged
pseudo-ensembles src/weavr/ensemble.py builds from GraphCast/Pangu, using
CRPS and Brier -- the two metrics Phase 1 could not compute at all, because
no ensemble-shaped source existed yet (docs/phase-1-data-requirements.md).

Scope, decided by what the data actually supports, not assumed:

- **Precipitation only, scored.** Exactly Tier 0's own limitation, still
  true here: the baseline store's only IMD ground truth is
  `imd_observed.rain`, and IMD's own gridded temperature product is 1 deg
  native -- coarser than weavr's locked 0.25 deg grid, so it can't back a
  temperature climatology or a temperature observation to score CRPS
  against. Building a temperature *ensemble* (Phase 2's actual job) doesn't
  change this -- there is still nothing to verify it against in this
  repo's data.
- **GraphCast + Pangu temperature ensembles are still built and counted**
  (per this step's own prompt), confirming build_lagged_ensemble works
  uniformly across every AI source/variable combination in the real
  fetched data -- just not scored. See the temperature section of
  docs/phase2-ensemble-baseline-results.md for the counts.
- **This does not re-fetch anything from GCS.** scripts/build_lagged_ensemble_store.py
  already pulled every (time, prediction_timedelta) pair a lagged ensemble
  needs into data/lagged_ensemble_inputs_2020_jjas.zarr -- including
  working around a real stall found running that fetch (see
  docs/baseline-store.md). Re-fetching the same data here via a live GCS
  source would just repeat that ~15-20 minute cost for identical output.
  Instead, `_reconstruct_raw_source` rebuilds an in-memory Dataset shaped
  like the raw WeatherBench 2 archives (the `(time, prediction_timedelta,
  ...)` contract src/weavr/ensemble.py's `build_lagged_ensemble` expects)
  directly from the already-fetched local store, then calls
  `build_lagged_ensemble` against it -- so this script genuinely exercises
  the same tested construction function (previously only exercised against
  synthetic data), just without paying for the network fetch twice.
- **Sample counts do not grow.** Collapsing 9 lagged members into one
  ensemble still verifies against exactly one IMD day per nominal forecast
  -- the same ~14 train / 3-4 test samples per lead Tier 0 had. Ensemble
  members add spread, not more independent verification days.
- Precipitation units: the same meters-to-millimeters conversion
  scripts/run_tier0_baseline.py established -- imported directly rather
  than re-derived, so a future unit-convention change only needs fixing
  once.

Usage:
    python scripts/run_phase2_ensemble_baseline.py
        [--lagged-store PATH] [--baseline-store PATH]
        [--out-csv PATH] [--test-fraction F]
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
from build_lagged_ensemble_store import _valid_combos  # noqa: E402
from run_tier0_baseline import PRECIP_M_TO_MM  # noqa: E402

from weavr import verify as V  # noqa: E402
from weavr.ensemble import NoLaggedMembersError, build_lagged_ensemble  # noqa: E402
from weavr.grid import imd_day_of_accumulation  # noqa: E402
from weavr.score_io import (  # noqa: E402
    FORCE_HELP,
    PerDayScoreWriter,
    guard_result_overwrites,
    per_day_scores,
    resolve_result_paths,
)
from weavr.splits import (  # noqa: E402,E501
    iter_evaluation_folds,
)
from weavr.stores import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    open_multi_season,
    resolve_store_paths,
)

PRECIP_VARIABLE = "total_precipitation_24hr"
TEMPERATURE_VARIABLE = "2m_temperature"
LEAD_HOURS = [24, 48, 72, 96, 120]
N_LAGS = 4
LAG_SPACING_HOURS = 12

# (source group name in the lagged-ensemble store, variable) pairs that are
# actually scored against IMD ground truth -- precipitation only, per the
# module docstring's Scope section.
SCORED_SOURCES = [("graphcast", PRECIP_VARIABLE)]

# Built (to confirm build_lagged_ensemble works against every real AI
# source/variable combination) but not scored -- no matching-resolution
# IMD ground truth exists for temperature.
UNSCORED_TEMPERATURE_SOURCES = [
    ("graphcast", TEMPERATURE_VARIABLE),
    ("pangu", TEMPERATURE_VARIABLE),
]


def _reconstruct_raw_source(dense: xr.Dataset, var: str) -> xr.Dataset:
    """Rebuild a raw, WeatherBench-2-shaped Dataset from an already-fetched
    lagged-ensemble-store group, so build_lagged_ensemble can run against it
    with no further network access.

    `dense` is shaped `(nominal_time, lead_hours, member_offset_hours,
    latitude, longitude)` -- scripts/build_lagged_ensemble_store.py's output.
    This inverts that: for every (nominal_time, lead, offset) combination
    that store fetched (recomputed via the same `_valid_combos` it uses, not
    reimplemented), it places that cell's already-fetched value at its real
    `(source_time, source_lead)` coordinate in a new `(time,
    prediction_timedelta, latitude, longitude)` Dataset -- exactly the shape
    `build_lagged_ensemble` expects from a live-opened source archive.
    """
    nominal_times = dense["nominal_time"].values
    lead_hours_list = [int(x) for x in dense["lead_hours"].values]
    offsets = [int(x) for x in dense["member_offset_hours"].values]
    combos = _valid_combos(nominal_times, lead_hours_list, offsets)

    source_times = sorted({c[3] for c in combos})
    source_leads = sorted({c[4] for c in combos})
    time_index = {t: idx for idx, t in enumerate(source_times)}
    lead_index = {lead: idx for idx, lead in enumerate(source_leads)}

    n_lat = dense.sizes["latitude"]
    n_lon = dense.sizes["longitude"]
    values = np.full((len(source_times), len(source_leads), n_lat, n_lon), np.nan)

    dense_values = dense[var].values  # (nominal, lead, offset, lat, lon)
    for i, j, k, source_time, source_lead in combos:
        cell = dense_values[i, j, k]
        if not np.all(np.isnan(cell)):
            values[time_index[source_time], lead_index[source_lead]] = cell

    return xr.Dataset(
        {var: (("time", "prediction_timedelta", "latitude", "longitude"), values)},
        coords={
            "time": np.array(source_times),
            "prediction_timedelta": np.array(source_leads),
            "latitude": dense["latitude"].values,
            "longitude": dense["longitude"].values,
        },
    )


def _valid_time_to_imd_day(nominal_time: np.datetime64, lead_hours: int) -> np.datetime64:
    """Map one nominal (init_time, lead) forecast's valid time to its IMD day.

    `weavr.grid.imd_day_of_accumulation` for a single forecast rather than a
    whole array (each ensemble here is built per nominal forecast).
    """
    valid_time = pd.Timestamp(nominal_time) + pd.Timedelta(hours=lead_hours)
    return imd_day_of_accumulation(valid_time)[0].to_datetime64()


def _exceedance_probability(ensemble: xr.DataArray, threshold: float) -> xr.DataArray:
    """Fraction of ensemble members at/above `threshold` -- the
    ensemble-mean-exceedance-probability `weavr.verify.brier_score` needs,
    since Brier takes a probability forecast, not a raw ensemble.

    A member missing from a short lagged window (NaN, per
    src/weavr/ensemble.py's padding) must not silently count as a
    non-exceedance: `NaN >= threshold` evaluates to `False`, not `NaN`, in
    numpy -- checked directly, not assumed -- so the boolean comparison is
    re-masked back to NaN wherever the source member was NaN, before
    `mean(skipna=True)` correctly excludes only those.
    """
    exceeds = (ensemble >= threshold).astype(float)
    return exceeds.where(~ensemble.isnull()).mean(dim="member", skipna=True)


def build_ensembles_for_lead(
    raw_ds: xr.Dataset, var: str, nominal_times: np.ndarray, lead_hours: int
) -> xr.DataArray:
    """Stack one lead's per-nominal-week lagged ensembles along a `sample`
    dim, labelled by the IMD day each forecast's valid time falls in (so the
    result can be reindexed against `imd_observed.rain` directly, matching
    scripts/run_tier0_baseline.py's convention).
    """
    members = []
    imd_days = []
    for nominal_time in nominal_times:
        try:
            ensemble = build_lagged_ensemble(
                raw_ds,
                var,
                nominal_time,
                lead_hours,
                n_lags=N_LAGS,
                lag_spacing_hours=LAG_SPACING_HOURS,
            )
        except NoLaggedMembersError:
            continue
        members.append(ensemble.drop_vars(["source_init_time", "source_lead_hours"]))
        imd_days.append(_valid_time_to_imd_day(nominal_time, lead_hours))

    return xr.concat(members, dim=pd.Index(imd_days, name="sample"))


def score_precip_lead(
    ensemble_mm: xr.DataArray,
    obs_rain: xr.DataArray,
    thresholds: tuple[float, ...],
    test_fraction: float,
    per_day_method: str | None = None,
    lead_hours: int | None = None,
    per_day_writer: PerDayScoreWriter | None = None,
) -> list[dict]:
    """Score one lagged-ensemble lead across evaluation folds (LOYO or single-season block)."""
    sample_times = pd.DatetimeIndex(ensemble_mm["sample"].values)

    obs_aligned = obs_rain.reindex(time=ensemble_mm["sample"].values).rename(time="sample")
    has_obs = ~obs_aligned.isnull().all(dim=["latitude", "longitude"])
    ensemble_mm = ensemble_mm.isel(sample=has_obs.values)
    obs_aligned = obs_aligned.isel(sample=has_obs.values)
    sample_times = sample_times[has_obs.values]

    folds = list(iter_evaluation_folds(sample_times, test_fraction=test_fraction))
    rows: list[dict] = []
    test_ensembles = []
    test_obs_list = []

    for train_mask, test_mask, split_label in folds:
        test_ensemble = ensemble_mm.isel(sample=test_mask)
        test_obs = obs_aligned.isel(sample=test_mask)
        test_ensembles.append(test_ensemble)
        test_obs_list.append(test_obs)

        brier_by_threshold = {
            t: float(
                V.brier_score(
                    _exceedance_probability(test_ensemble, t),
                    (test_obs >= t).astype(float),
                )
            )
            for t in thresholds
        }

        if per_day_method and lead_hours is not None and per_day_writer is not None:
            per_day_writer.add(
                per_day_method,
                lead_hours,
                per_day_scores(
                    test_ensemble.mean(dim="member", skipna=True),
                    test_obs,
                    fold=split_label,
                    ensemble=test_ensemble,
                    thresholds=thresholds,
                    probabilities={
                        t: _exceedance_probability(test_ensemble, t) for t in thresholds
                    },
                ),
            )

        n_members = float(test_ensemble.notnull().sum(dim="member").mean())
        spread_mm = float(V.ensemble_spread(test_ensemble))
        rmse_of_mean_mm = float(V.rmse(test_ensemble.mean(dim="member", skipna=True), test_obs))
        ratio = float(V.spread_skill_ratio(test_ensemble, test_obs))
        calibrated_ratio = V.calibrated_spread_skill_ratio(n_members)

        split_kind = (
            "leave_one_year_out"
            if split_label != "seasonal_block_split"
            else "seasonal_block_split (single-season store; leave_one_year_out not usable)"
        )

        rows.append(
            {
                "n_samples": int(ensemble_mm.sizes["sample"]),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "split": split_kind,
                "fold": split_label,
                "crps_mm": float(V.crps(test_ensemble, test_obs)),
                "brier": brier_by_threshold,
                "n_members": n_members,
                "spread_mm": spread_mm,
                "rmse_of_mean_mm": rmse_of_mean_mm,
                "spread_skill_ratio": ratio,
                "calibrated_spread_skill_ratio": calibrated_ratio,
            }
        )

    if len(folds) > 1:
        pooled_ensemble = xr.concat(test_ensembles, dim="sample")
        pooled_obs = xr.concat(test_obs_list, dim="sample")

        brier_pooled = {
            t: float(
                V.brier_score(
                    _exceedance_probability(pooled_ensemble, t),
                    (pooled_obs >= t).astype(float),
                )
            )
            for t in thresholds
        }
        n_members = float(pooled_ensemble.notnull().sum(dim="member").mean())
        spread_mm = float(V.ensemble_spread(pooled_ensemble))
        rmse_of_mean_mm = float(V.rmse(pooled_ensemble.mean(dim="member", skipna=True), pooled_obs))
        ratio = float(V.spread_skill_ratio(pooled_ensemble, pooled_obs))
        calibrated_ratio = V.calibrated_spread_skill_ratio(n_members)

        rows.append(
            {
                "n_samples": int(ensemble_mm.sizes["sample"]),
                "n_train": int(ensemble_mm.sizes["sample"]),
                "n_test": int(pooled_ensemble.sizes["sample"]),
                "split": "leave_one_year_out",
                "fold": "pooled",
                "crps_mm": float(V.crps(pooled_ensemble, pooled_obs)),
                "brier": brier_pooled,
                "n_members": n_members,
                "spread_mm": spread_mm,
                "rmse_of_mean_mm": rmse_of_mean_mm,
                "spread_skill_ratio": ratio,
                "calibrated_spread_skill_ratio": calibrated_ratio,
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lagged-stores",
        nargs="+",
        default=None,
        help="One or more lagged ensemble store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--lagged-store", default=None, help="Legacy single lagged store path")
    parser.add_argument(
        "--baseline-stores",
        nargs="+",
        default=None,
        help="One or more baseline store paths (multi-season; default: 2018 + 2020 daily)",
    )
    parser.add_argument("--baseline-store", default=None, help="Legacy single baseline store path")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for the per-day files, and the default parent for the output CSV(s).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Aggregated CSV path (default: <results-dir>/phase2_ensemble_baseline.csv)",
    )
    parser.add_argument("--force", action="store_true", help=FORCE_HELP)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    _paths = resolve_result_paths(
        args.results_dir,
        {
            "out_csv": "phase2_ensemble_baseline.csv",
        },
        {
            "out_csv": args.out_csv,
        },
    )
    args.out_csv = _paths["out_csv"]
    guard_result_overwrites(_paths.values(), force=args.force)

    lagged_paths = resolve_store_paths(
        args.lagged_stores,
        args.lagged_store,
        DEFAULT_LAGGED_DAILY_STORES,
        "data/lagged_ensemble_inputs_2020_jjas.zarr",
    )
    baseline_paths = resolve_store_paths(
        args.baseline_stores,
        args.baseline_store,
        DEFAULT_BASELINE_DAILY_STORES,
        "data/baseline_2020_jjas.zarr",
    )

    obs = open_multi_season(baseline_paths, group="imd_observed").load()

    print("Phase 2 ensemble baseline: lagged GraphCast/Pangu ensembles scored with CRPS/Brier")
    print(f"Lagged-ensemble store(s): {lagged_paths}")
    print("Scope: precipitation only scored -- no matching-resolution IMD temperature ground truth")

    rows = []
    per_day_writer = PerDayScoreWriter(args.results_dir)
    for lead_hours in LEAD_HOURS:
        for group, var in SCORED_SOURCES:
            dense = open_multi_season(lagged_paths, group=group).load()
            raw_ds = _reconstruct_raw_source(dense, var)
            ensemble = build_ensembles_for_lead(
                raw_ds, var, dense["nominal_time"].values, lead_hours
            )
            ensemble_mm = ensemble * PRECIP_M_TO_MM

            lead_results = score_precip_lead(
                ensemble_mm,
                obs["rain"],
                V.IMD_RAIN_THRESHOLDS_MM,
                args.test_fraction,
                per_day_method=f"phase2_lagged_{group}",
                lead_hours=lead_hours,
                per_day_writer=per_day_writer,
            )
            for result in lead_results:
                result["lead_hours"] = lead_hours
                result["source"] = group
                result["variable"] = var
                rows.append(result)
                print(f"[lead {lead_hours:>3}h | fold {result['fold']}] {group}/{var}: {result}")

    print("\nTemperature ensembles (built, not scored -- no matching-resolution IMD ground truth):")
    print("Spread magnitude only below -- descriptive, no error to compare against.")
    temperature_rows = []
    for group, var in UNSCORED_TEMPERATURE_SOURCES:
        dense = open_multi_season(lagged_paths, group=group).load()
        raw_ds = _reconstruct_raw_source(dense, var)
        for lead_hours in LEAD_HOURS:
            n_built = 0
            for nominal_time in dense["nominal_time"].values:
                try:
                    build_lagged_ensemble(
                        raw_ds, var, nominal_time, lead_hours, n_lags=N_LAGS,
                        lag_spacing_hours=LAG_SPACING_HOURS,
                    )
                    n_built += 1
                except NoLaggedMembersError:
                    pass

            ensemble = build_ensembles_for_lead(
                raw_ds, var, dense["nominal_time"].values, lead_hours
            )
            spread_k = float(V.ensemble_spread(ensemble))
            temperature_rows.append(
                {
                    "source": group,
                    "variable": var,
                    "lead_hours": lead_hours,
                    "n_built": n_built,
                    "spread_k": spread_k,
                }
            )
            print(
                f"  [lead {lead_hours:>3}h] {group}/{var}: {n_built} ensembles built, "
                f"spread={spread_k:.3f} K (descriptive only, no ground truth)"
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lead_hours",
        "source",
        "variable",
        "fold",
        "split",
        "n_samples",
        "n_train",
        "n_test",
        "crps_mm",
        "n_members",
        "spread_mm",
        "rmse_of_mean_mm",
        "spread_skill_ratio",
        "calibrated_spread_skill_ratio",
    ]
    for t in V.IMD_RAIN_THRESHOLDS_MM:
        fieldnames.append(f"brier_{t}mm")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fieldnames if k in row}
            for t in V.IMD_RAIN_THRESHOLDS_MM:
                flat[f"brier_{t}mm"] = row["brier"][t]
            writer.writerow(flat)

    print(f"\nWrote {out_path}")

    temperature_out_path = out_path.parent / "phase2_temperature_dispersion.csv"
    with temperature_out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "variable", "lead_hours", "n_built", "spread_k"]
        )
        writer.writeheader()
        writer.writerows(temperature_rows)
    print(f"Wrote {temperature_out_path}")

    per_day_writer.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
