#!/usr/bin/env python3
"""Reliability of the quantile-averaged extreme-rain probabilities (#75 follow-up 3).

Step 13 (#75) found that averaging the EMOS-CSG and BMA quantile functions
beats both parents on CRPS (H10), and `export_dashboard_example_grids.py
--combiner quantile_avg` can draw the extreme-probability map from it. Its
probabilities run about twice the EMOS map's, and nothing had measured whether
that is better or worse calibrated above 64.5 mm. That decides whether it can
become the dashboard default. This runner measures it the way step 12 measured
the EMOS tail:

  * the same leave-one-year-out folds and the same test cells
  * the same pooled GPD tail, fitted on the IMD archive minus the test year,
    spliced above u onto P(Y > u) read off the averaged quantile function
    (the construction `quantile_avg_exceedance_grids` draws on the map)
  * the same reliability bins, with moving-block bootstrap CIs, the same Brier
    (observed cells only, `obs >= t`), the same climatology reference (archive
    minus the test year) and the same expected-count SEDI

The EMOS source and Vincentization weight are the ones Tier 2b used for that
lead and fold (`results/tier2b_combined_train.csv`), not refitted, so this
scores the configuration H10 was decided on.

Cells whose (bin, region) has neither an EMOS nor a BMA fit are scored with
probability 0, as the map draws them; they are counted (`cells_fallback`) so
the reader can see how much of the score they carry.

Outputs: `results/quantile_avg_extreme.csv` (one row per lead, fold, threshold)
and `results/quantile_avg_reliability.csv` (one row per reliability bin, folds
pooled).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_seeps_climatology import load_climatology  # noqa: E402
from run_tail_repair_evt import (  # noqa: E402
    EXTREME_THRESHOLDS,
    climatology_exceedance,
    expected_counts,
    fit_tail_on_obs,
    reliability_day_sums,
    reliability_with_ci,
)
from run_tier2_hierarchical_baseline import (  # noqa: E402
    DEFAULT_BASELINE_DAILY_STORES,
    DEFAULT_IFS_DAILY_STORES,
    DEFAULT_LAGGED_DAILY_STORES,
    align_all_sources,
    load_graphcast_ensemble,
    load_hres_forecast,
    load_ifs_ensemble,
)

from weavr.bma import bma_predictive_quantiles, fit_hierarchical_bma  # noqa: E402
from weavr.emos import fit_emos_csg, predict_csgd_params  # noqa: E402
from weavr.quantile_mapping import climatology_excluding_years  # noqa: E402
from weavr.rain_bins import classify_rain_bin  # noqa: E402
from weavr.regions import assign_regions  # noqa: E402
from weavr.score_io import (  # noqa: E402
    exceedance_event,
    guard_result_overwrites,
    resolve_result_paths,
)
from weavr.splits import iter_evaluation_folds  # noqa: E402
from weavr.stacking import (  # noqa: E402
    EXCEEDANCE_LEVELS,
    exceedance_from_quantiles,
    predictive_quantiles_csgd,
    quantile_average,
)
from weavr.stores import open_multi_season, resolve_store_paths  # noqa: E402
from weavr.tail import DEFAULT_TAIL_THRESHOLD_U, spliced_exceedance_probability  # noqa: E402

LEAD_HOURS = [24, 48, 72, 96, 120]
ARM = "quantile_avg_gpd_tail"
#: Cells per block when the quantile functions are evaluated: 20k cells x 280
#: levels x 8 bytes is ~45 MB per array, so the largest (bin, region) group
#: never needs more than a few hundred MB.
CHUNK_CELLS = 20_000


def tier2b_config(train_csv: str | Path, lead_hours: int, fold: str) -> tuple[str, float]:
    """The EMOS source and Vincentization weight Tier 2b used for this lead and fold."""
    with open(train_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["lead_hours"]) == lead_hours and str(row["fold"]) == str(fold):
                return row["emos_source"], float(row["vincentization_weight"])
    raise ValueError(f"no Tier 2b train row for lead {lead_hours}h, fold {fold} in {train_csv}")


def quantile_avg_exceedance_cells(
    day_mean: dict[str, np.ndarray],
    day_spread: dict[str, np.ndarray | None],
    bins: np.ndarray,
    regions: np.ndarray,
    emos_results: dict,
    bma_results: dict,
    source: str,
    weight: float,
    tail_fit,
    thresholds,
    cell_mask: np.ndarray | None = None,
) -> tuple[dict[float, np.ndarray], np.ndarray, np.ndarray]:
    """`quantile_avg` exceedance probabilities on every cell with a full forecast.

    All arrays share one shape (`(days, lat, lon)` for the runner). Returns
    `(probabilities per threshold, fallback mask, finite mask)`: probability is
    NaN where a source forecast is missing, and 0.0 on fallback cells (neither
    parent fitted for the cell's bin and region), as the dashboard map draws
    them. Thresholds at or below u read the averaged quantile function; those
    above splice the GPD onto P(Y > u) read off it.

    `cell_mask` limits the work to the cells that will be scored (those IMD
    observes: ~560k of ~4.2M grid cells per fold). Every other cell stays NaN,
    which `score_probabilities` excludes anyway, so scores are unchanged and
    the BMA quantile inversion costs a seventh as much.
    """
    finite = np.all([np.isfinite(m) for m in day_mean.values()], axis=0)
    if cell_mask is not None:
        finite = finite & cell_mask
    u = tail_fit.threshold_u
    probs = {t: np.full(bins.shape, np.nan, dtype=float) for t in thresholds}
    fallback = np.zeros(bins.shape, dtype=bool)
    flat_bins, flat_regions = bins.reshape(-1), regions.reshape(-1)
    flat_mean = {k: v.reshape(-1) for k, v in day_mean.items()}
    flat_spread = {k: None if v is None else v.reshape(-1) for k, v in day_spread.items()}
    flat_finite = finite.reshape(-1)
    flat_probs = {t: probs[t].reshape(-1) for t in thresholds}  # views
    flat_fallback = fallback.reshape(-1)

    for (bin_label, region), bma_result in bma_results.items():
        group = np.flatnonzero((flat_bins == bin_label) & (flat_regions == region) & flat_finite)
        if group.size == 0:
            continue
        emos_result = emos_results.get(bin_label)
        no_emos = emos_result is None or emos_result.is_fallback
        if no_emos and bma_result.is_fallback:
            flat_fallback[group] = True
            for t in thresholds:
                flat_probs[t][group] = 0.0
            continue
        for start in range(0, group.size, CHUNK_CELLS):
            idx = group[start : start + CHUNK_CELLS]
            cell_mean = {k: v[idx] for k, v in flat_mean.items()}
            cell_spread = {k: None if v is None else v[idx] for k, v in flat_spread.items()}
            if emos_result is not None and cell_spread[source] is not None:
                loc, scale, shift = predict_csgd_params(
                    emos_result, cell_mean[source], cell_spread[source]
                )
                q_emos = predictive_quantiles_csgd(loc, scale, shift, EXCEEDANCE_LEVELS)
            else:
                q_emos = np.zeros((idx.size, EXCEEDANCE_LEVELS.size), dtype=float)
            q_bma = bma_predictive_quantiles(
                bma_result, cell_mean, cell_spread, EXCEEDANCE_LEVELS
            )
            q_avg = quantile_average(q_emos, q_bma, weight=weight)
            p_u = exceedance_from_quantiles(q_avg, EXCEEDANCE_LEVELS, u)
            for t in thresholds:
                flat_probs[t][idx] = (
                    exceedance_from_quantiles(q_avg, EXCEEDANCE_LEVELS, t)
                    if t <= u
                    else spliced_exceedance_probability(p_u, tail_fit, flat_regions[idx], t)
                )
    return probs, fallback, finite


def score_probabilities(
    probability: np.ndarray, obs: np.ndarray, climatology: np.ndarray, threshold: float
) -> dict[str, float]:
    """Brier, skill, SEDI and calibration-in-total on the cells IMD observes.

    `probability`, `obs` are `(days, lat, lon)`, `climatology` is `(lat, lon)`.
    A cell without an observation is excluded (F8): a probability there is
    scored against nothing. The Brier is the mean of per-day means, as every
    other runner reports it.
    """
    event = exceedance_event(obs, threshold)
    observed = np.isfinite(event) & np.isfinite(probability)
    p = np.where(observed, probability, np.nan)
    o = np.where(observed, event, np.nan)
    clim = np.broadcast_to(climatology[None], p.shape)
    day_brier = np.nanmean((p - o) ** 2, axis=(1, 2))
    day_clim = np.nanmean(np.where(observed, (clim - o) ** 2, np.nan), axis=(1, 2))
    brier, clim_brier = float(np.nanmean(day_brier)), float(np.nanmean(day_clim))
    # NaN (not 0) where unobserved: the helper skips NaN cells, whereas p = 0, o = 0
    # would count each as a correct negative and inflate SEDI.
    counts = expected_counts(p, o)
    h, m, fa, cn = (float(np.sum(counts[k])) for k in
                    ("hits", "misses", "false_alarms", "correct_negatives"))
    n_obs_cells = float(observed.sum())
    n_events = float(np.nansum(o))
    return {
        "brier": brier,
        "climatology_brier": clim_brier,
        "brier_skill_score": 1.0 - brier / clim_brier if clim_brier > 0 else float("nan"),
        "sedi": _sedi(h, m, fa, cn),
        "mean_forecast": float(np.nansum(p) / n_obs_cells) if n_obs_cells else float("nan"),
        "observed_frequency": n_events / n_obs_cells if n_obs_cells else float("nan"),
        "forecast_over_observed": (
            float(np.nansum(p) / n_events) if n_events > 0 else float("nan")
        ),
        "n_observed_cells": int(n_obs_cells),
        "n_events": int(n_events),
    }


def _sedi(h: float, m: float, fa: float, cn: float) -> float:
    """Symmetric extremal dependence index from pooled counts; NaN when degenerate."""
    if h + m <= 0 or fa + cn <= 0:
        return float("nan")
    hit_rate, false_rate = h / (h + m), fa / (fa + cn)
    if not (0.0 < hit_rate < 1.0 and 0.0 < false_rate < 1.0):
        return float("nan")
    lf, lh = np.log(false_rate), np.log(hit_rate)
    l1f, l1h = np.log(1 - false_rate), np.log(1 - hit_rate)
    return float((lf - lh - l1f + l1h) / (lf + lh + l1f + l1h))


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-stores", nargs="+", default=None)
    parser.add_argument("--lagged-stores", nargs="+", default=None)
    parser.add_argument("--ifs-ensemble-stores", nargs="+", default=None)
    parser.add_argument("--climatology", default="data/imd_seeps_climatology_jjas.zarr")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="default: <results-dir>/quantile_avg_extreme.csv")
    parser.add_argument("--reliability-out-csv", type=Path, default=None,
                        help="default: <results-dir>/quantile_avg_reliability.csv")
    parser.add_argument("--tier2b-train-csv", default="results/tier2b_combined_train.csv")
    parser.add_argument("--emos-source", default=None,
                        help="Override Tier 2b's EMOS source (rehearsals and cross-checks).")
    parser.add_argument("--weight", type=float, default=None,
                        help=("Override Tier 2b's Vincentization weight: the weight on BMA, "
                              "so 0.0 is EMOS-CSG only and 1.0 is BMA only."))
    parser.add_argument("--threshold-u", type=float, default=DEFAULT_TAIL_THRESHOLD_U)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--leads", nargs="+", type=int, default=None)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    paths = resolve_result_paths(
        args.results_dir,
        {"out_csv": "quantile_avg_extreme.csv",
         "reliability_out_csv": "quantile_avg_reliability.csv"},
        {"out_csv": args.out_csv, "reliability_out_csv": args.reliability_out_csv},
    )
    guard_result_overwrites(paths.values(), force=args.force)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    baseline_paths = resolve_store_paths(
        args.baseline_stores, None, DEFAULT_BASELINE_DAILY_STORES, "data/baseline_2020_jjas.zarr")
    lagged_paths = resolve_store_paths(
        args.lagged_stores, None, DEFAULT_LAGGED_DAILY_STORES,
        "data/lagged_ensemble_inputs_2020_jjas.zarr")
    ifs_paths = resolve_store_paths(
        args.ifs_ensemble_stores, None, DEFAULT_IFS_DAILY_STORES, "data/ifs_ens_2020_jjas.zarr")

    obs = open_multi_season(baseline_paths, group="imd_observed").load()
    climatology_rain = load_climatology(args.climatology).load()["rain"]
    region_labels = assign_regions(obs["latitude"].values, obs["longitude"].values)
    region_2d = region_labels.transpose("latitude", "longitude").values

    print(f"Quantile-averaged extreme-rain reliability (#75 follow-up 3), u={args.threshold_u} mm")
    rows: list[dict] = []
    sums: dict[tuple[float, int], list[dict[str, np.ndarray]]] = {}
    for lead_hours in args.leads or LEAD_HOURS:
        forecasts, obs_aligned = align_all_sources(
            load_graphcast_ensemble(lagged_paths, lead_hours),
            load_ifs_ensemble(ifs_paths, lead_hours),
            load_hres_forecast(baseline_paths, lead_hours),
            obs,
        )
        sample_times = pd.DatetimeIndex(obs_aligned["sample"].values)
        rain_bin_labels = classify_rain_bin(forecasts["graphcast"].mean(dim="member", skipna=True))
        bins_all = np.asarray(rain_bin_labels.values)
        obs_all = np.asarray(obs_aligned.values, dtype=float)
        folds = list(iter_evaluation_folds(sample_times, test_fraction=args.test_fraction))
        for train_mask, test_mask, fold in folds:
            if args.emos_source is not None and args.weight is not None:
                source, weight = args.emos_source, args.weight
            else:
                source, weight = tier2b_config(args.tier2b_train_csv, lead_hours, fold)
            emos_results = fit_emos_csg(
                forecasts[source], obs_aligned, rain_bin_labels, train_mask, source=source
            )
            bma_results = fit_hierarchical_bma(
                forecasts, obs_aligned, rain_bin_labels, region_labels, train_mask
            )
            tail_archive = climatology_excluding_years(climatology_rain, sample_times[test_mask])
            tail_fit = fit_tail_on_obs(tail_archive, region_labels, args.threshold_u)
            clim_p = climatology_exceedance(tail_archive, EXTREME_THRESHOLDS)

            test_idx = np.flatnonzero(test_mask)
            day_mean: dict[str, np.ndarray] = {}
            day_spread: dict[str, np.ndarray | None] = {}
            for name, da in forecasts.items():
                day = da.isel(sample=test_idx)
                if "member" in day.dims:
                    day_mean[name] = np.asarray(day.mean(dim="member", skipna=True).values, float)
                    day_spread[name] = np.asarray(
                        day.std(dim="member", ddof=1, skipna=True).values, float
                    )
                else:
                    day_mean[name] = np.asarray(day.values, float)
                    day_spread[name] = None
            regions3d = np.broadcast_to(region_2d[None], day_mean["graphcast"].shape)
            probs, fallback, _ = quantile_avg_exceedance_cells(
                day_mean, day_spread, bins_all[test_idx], regions3d,
                emos_results, bma_results, source, weight, tail_fit, EXTREME_THRESHOLDS,
                cell_mask=np.isfinite(obs_all[test_idx]),
            )
            test_obs = obs_all[test_idx]
            for t in EXTREME_THRESHOLDS:
                scores = score_probabilities(
                    probs[t], test_obs, np.asarray(clim_p[t].values, dtype=float), t
                )
                observed_fallback = int((fallback & np.isfinite(test_obs)).sum())
                rows.append({
                    "lead_hours": lead_hours, "fold": fold, "arm": ARM, "threshold": t,
                    "emos_source": source, "vincentization_weight": weight,
                    "threshold_u": args.threshold_u, "tail_shape_xi": float(tail_fit.shape_xi),
                    "n_test_days": int(test_idx.size),
                    "cells_fallback": observed_fallback, **scores,
                })
                event = exceedance_event(test_obs, t)
                sums.setdefault((t, lead_hours), []).append(
                    reliability_day_sums(
                        np.where(np.isfinite(event), probs[t], np.nan), event
                    )
                )
            print(f"[lead {lead_hours:>3}h | fold {fold}] source={source} w={weight:.2f}",
                  flush=True)

    pd.DataFrame(rows).to_csv(paths["out_csv"], index=False)
    tables = []
    for (t, lead_hours), parts in sums.items():
        merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        table = reliability_with_ci(merged, args.block_days, args.n_resamples, args.seed)
        tables.append(table.assign(arm=ARM, threshold=t, lead_hours=lead_hours))
    columns = ["lead_hours", "arm", "threshold", "bin", "p_lo", "p_hi", "n",
               "mean_forecast", "observed_frequency", "sum_forecast", "sum_observed",
               "observed_frequency_ci_lo", "observed_frequency_ci_hi"]
    pd.concat(tables, ignore_index=True)[columns].to_csv(paths["reliability_out_csv"], index=False)
    print(f"Wrote {paths['out_csv']} and {paths['reliability_out_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
