#!/usr/bin/env python3
"""Build the Phase 0 baseline store: one monsoon season of aligned forecasts + IMD rain.

This is Phase 0's literal exit criterion (see docs/phase-plan.md and
solving issues/01-phase-0-data-access-and-scoping/05-build-aligned-data-store.md):
one full JJAS (June-September) monsoon season of WeatherBench 2 forecasts,
regridded to the common 0.25 deg grid (weavr.grid), plus IMD gridded
rainfall as ground truth, sitting in one Zarr store.

WeatherBench 2 has GraphCast evaluation windows covering both JJAS 2018 and
JJAS 2020. Select the source-specific archive by --year and write each year
to its own resumable store. The default keeps the original 2020 weekly build;
--year 2018 defaults to daily 00 UTC initializations.

Known gap, documented rather than silently dropped: Pangu's WeatherBench 2
archive has no precipitation variable at all (2m_temperature only). It is
still included for temperature; its precipitation columns are absent from
the store, not zero-filled.

Sampling density: measured, not assumed. WeatherBench 2's native-resolution
stores are chunked at roughly one (init_time, lead_time) pair per chunk, and
a single chunk fetch from GCS takes seconds. The script samples five
representative lead times (24/48/72/96/120h). The 2020 default preserves its
weekly cadence; the 2018 default samples daily 00 UTC starts, matching the
2020 daily store built separately in step 05.
IMD's daily observations are pulled at full density regardless (they're
already one chunk per day, cheap regardless of cadence).

Usage:
    python scripts/build_baseline_store.py [--year 2018|2020] [--out PATH]
                                            [--lead-hours H [H ...]]
                                            [--start DATE] [--end DATE]
                                            [--init-cadence-days N]

    WEAVR_BASELINE_STORE_PATH can also set the output path.

Idempotent: writes a JSON manifest next to the store recording which source
succeeded; a source already marked "ok" is skipped on re-run unless --force
is passed.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from weavr.archives import archive_for, normalize_coordinates
from weavr.data.imd_gridded import fetch_year
from weavr.grid import SourceTooCoarseError, regrid_to_common

GCS_ANON = {"token": "anon"}

DEFAULT_YEAR = 2020
DEFAULT_START = f"{DEFAULT_YEAR}-06-01"
DEFAULT_END = f"{DEFAULT_YEAR}-09-30"
# Representative lead times reaching the prompt's stated 5-day (120h) minimum,
# rather than every 6h step -- see the sampling-density note in the module
# docstring for the measured chunk-fetch cost that forced this choice.
DEFAULT_LEAD_HOURS = [24, 48, 72, 96, 120]
DEFAULT_INIT_CADENCE_DAYS = 7  # weekly init sampling across the full JJAS range

INDIA_LAT_SLICE = slice(6.0, 39.0)
INDIA_LON_SLICE_0_360 = slice(66.0, 101.0)  # WeatherBench 2 native stores use 0-360 longitude


@dataclass
class ForecastSource:
    name: str
    year: int
    zarr_path: str
    variables: list[str]
    known_gaps: list[str] = field(default_factory=list)


def forecast_sources(year: int) -> list[ForecastSource]:
    """Select the same source set from the requested evaluation year."""
    specs: tuple[tuple[str, list[str], list[str]], ...] = (
        ("graphcast", ["2m_temperature", "total_precipitation_24hr"], []),
        ("pangu", ["2m_temperature"], ["no precipitation variable in this archive"]),
        ("hres", ["2m_temperature", "total_precipitation_24hr"], []),
        ("ifs_ens_mean", ["2m_temperature", "total_precipitation_24hr"], []),
    )
    return [
        ForecastSource(name, year, archive_for(name, year).path, variables, gaps)
        for name, variables, gaps in specs
    ]


FORECAST_SOURCES = forecast_sources(DEFAULT_YEAR)


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str))


def _lat_slice_for(ds: xr.Dataset, lat_dim: str, lo: float, hi: float) -> slice:
    """Build a lat slice respecting the coordinate's own direction.

    xarray's `.sel(dim=slice(lo, hi))` silently returns an empty selection
    (not an error) on a descending coordinate unless the slice bounds are
    given high-to-low. Confirmed this matters in practice: WeatherBench 2's
    native-resolution stores are ascending (-90..90) for graphcast/hres/
    ifs_ens, but Pangu's is descending (90..-90) -- a naive slice(lo, hi)
    silently dropped all of Pangu's latitude points until this check caught
    it in a smoke test.
    """
    values = ds[lat_dim].values
    return slice(lo, hi) if values[0] < values[-1] else slice(hi, lo)


def _weekly_init_times(ds: xr.Dataset, start: str, end: str, cadence_days: int) -> np.ndarray:
    """Pick one init time per `cadence_days` within [start, end], from what the source actually has.

    Filters to 00 UTC-only inits when present to guarantee consistent cycle alignment.
    Building the target dates independently of the source's own time index
    and then re-selecting can miss entirely if the source's init times don't
    fall exactly on those dates/hours (they may be offset, e.g. 00/12 UTC vs
    06/18 UTC). Sampling from the source's own index guarantees a hit.
    """
    all_times = ds.sel(time=slice(start, end)).time.values
    if len(all_times) == 0:
        return all_times
    # Filter to 00 UTC inits if present to guarantee consistent 00 UTC cycle sampling
    dt_times = pd.to_datetime(all_times)
    utc00_mask = dt_times.hour == 0
    if np.any(utc00_mask):
        utc00_times = all_times[utc00_mask]
        return utc00_times[::cadence_days]
    # Fallback for sources without 00 UTC
    hours_per_step = float(
        (all_times[1] - all_times[0]) / np.timedelta64(1, "h") if len(all_times) > 1 else 24.0
    )
    stride = max(1, round(cadence_days * 24 / hours_per_step))
    return all_times[::stride]


def _slice_source(
    source: ForecastSource,
    start: str,
    end: str,
    lead_hours: list[int],
    init_cadence_days: int,
) -> xr.Dataset:
    ds = xr.open_zarr(source.zarr_path, storage_options=GCS_ANON, consolidated=True)
    ds = ds[[v for v in source.variables if v in ds.data_vars]]
    ds = normalize_coordinates(ds, archive_for(source.name, source.year))

    sampled_times = _weekly_init_times(ds, start, end, init_cadence_days)
    ds = ds.sel({"time": sampled_times})
    # prediction_timedelta is stored as plain int hours (see units attr), not
    # a real timedelta64 dtype -- select with plain ints to match, and
    # "nearest" in case a source's lead-time grid doesn't hit these exactly.
    ds = ds.sel({"prediction_timedelta": lead_hours}, method="nearest")

    lat_slice = _lat_slice_for(ds, "latitude", INDIA_LAT_SLICE.start, INDIA_LAT_SLICE.stop)
    ds = ds.sel(latitude=lat_slice, longitude=INDIA_LON_SLICE_0_360)

    return ds


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Strip zarr-v2-era encoding (chunk shape, Blosc codec) copied from the source.

    Source stores are read as-is from GCS; their per-variable `.encoding`
    carries the *original, full-grid* chunk shape and a numcodecs Blosc
    compressor spec. Writing that encoding back out via the installed
    zarr-python v3 writer fails ("Expected a BytesBytesCodec. Got
    numcodecs.blosc.Blosc") and would in any case apply a chunk shape sized
    for the full global grid to a much smaller India-sliced array. Clearing
    it lets `to_zarr` pick fresh, correctly-sized chunking and a
    zarr-v3-native codec.
    """
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding = {}
    return ds


def build_forecast_group(
    source: ForecastSource,
    start: str,
    end: str,
    lead_hours: list[int],
    init_cadence_days: int,
) -> tuple[xr.Dataset, dict]:
    ds = _slice_source(source, start, end, lead_hours, init_cadence_days)
    n_times = int(ds.sizes.get("time", 0))
    batch_size = 25
    if n_times > batch_size:
        batches = []
        for i in range(0, n_times, batch_size):
            t_slice = ds.isel(time=slice(i, i + batch_size))
            print(
                f"  [{source.name}] loading init times "
                f"{i + 1}..{min(i + batch_size, n_times)} of {n_times}..."
            )
            batches.append(t_slice.load())
        ds = xr.concat(batches, dim="time")
    else:
        ds = ds.load()
    ds = _clear_encoding(ds)

    try:
        ds = regrid_to_common(ds)
    except SourceTooCoarseError:
        # Shouldn't happen for the native-resolution stores selected above,
        # but fail loudly rather than silently using un-regridded data if a
        # store path is ever swapped for a coarser one.
        raise

    info = {
        "status": "ok",
        "variables": list(ds.data_vars),
        "n_init_times": int(ds.sizes.get("time", 0)),
        "n_lead_steps": int(ds.sizes.get("prediction_timedelta", 0)),
        "init_cadence_days": init_cadence_days,
        "init_cycle": "00:00 UTC",
        "known_gaps": source.known_gaps,
        # WeatherBench 2's closest thing to a model-version identifier: this
        # path changes when WB2 republishes a model under a new date-range
        # (see docs/baseline-store.md's Phase 6 addition on why this is
        # recorded -- no store output otherwise distinguishes two different
        # archive vintages of the same named source).
        "source_archive_path": source.zarr_path,
    }
    return ds, info


def build_imd_group(
    year: int, start: str, end: str, imd_nc_path: str | None = None
) -> tuple[xr.Dataset, dict]:
    if imd_nc_path:
        ds = xr.open_dataset(imd_nc_path)
        ds = ds.rename(
            {"TIME": "time", "LATITUDE": "lat", "LONGITUDE": "lon", "RAINFALL": "rain"}
        )
        if ds["rain"].attrs.get("units") != "mm":
            raise ValueError("IMD NetCDF rainfall must have units='mm'")
    else:
        ds = fetch_year(year, var_type="rain", cache_dir="data/imd_cache")
    ds = ds.sel(time=slice(start, end))
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    ds = _clear_encoding(ds)

    # IMD's own product is already daily on IMD's own 03-03 UTC convention,
    # labelled by the date each window ends on (see weavr.grid)
    # (imdlib serves pre-aggregated daily values, not sub-daily) -- there is
    # no further resampling to do here. weavr.grid.resample_to_imd_day exists
    # for sources that arrive sub-daily or on a different day boundary; IMD
    # data is not one of those, so applying it here would be a no-op dressed
    # up as work. Documented rather than silently skipped.
    ds = regrid_to_common(ds)

    expected_days = pd.date_range(start, end, freq="D")
    actual_days = pd.DatetimeIndex(ds["time"].values)
    missing_days = sorted(set(expected_days.normalize()) - set(actual_days.normalize()))

    info = {
        "status": "ok",
        "variables": list(ds.data_vars),
        "n_days": int(ds.sizes.get("time", 0)),
        "missing_days": [str(d.date()) for d in missing_days],
        "source_archive_path": imd_nc_path or "https://imdpune.gov.in/cmpg/Griddata/rainfall.php",
    }
    return ds, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2018, 2020), default=DEFAULT_YEAR)
    parser.add_argument(
        "--imd-nc-path",
        default=None,
        help="Use a yearly IMD 0.25-degree NetCDF file instead of the IMD server",
    )
    parser.add_argument(
        "--out",
        default=None,
    )
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS
    )
    parser.add_argument("--init-cadence-days", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-pull sources already marked ok")
    args = parser.parse_args()

    args.start = args.start or f"{args.year}-06-01"
    args.end = args.end or f"{args.year}-09-30"
    args.init_cadence_days = args.init_cadence_days or (
        1 if args.year == 2018 else DEFAULT_INIT_CADENCE_DAYS
    )
    args.out = args.out or os.environ.get("WEAVR_BASELINE_STORE_PATH") or (
        f"data/baseline_{args.year}_jjas_daily.zarr"
        if args.year == 2018 else "data/baseline_2020_jjas.zarr"
    )

    store_path = Path(args.out)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)

    print(f"Building baseline store at {store_path}")
    print(
        f"Window: {args.start} .. {args.end}, init cadence {args.init_cadence_days}d, "
        f"lead hours {args.lead_hours}"
    )

    jobs: list[tuple[str, Callable[[], tuple[xr.Dataset, dict]]]] = [
        (
            source.name,
            functools.partial(
                build_forecast_group,
                source,
                args.start,
                args.end,
                args.lead_hours,
                args.init_cadence_days,
            ),
        )
        for source in forecast_sources(args.year)
    ]
    jobs.append(
        (
            "imd_observed",
            functools.partial(
                build_imd_group, args.year, args.start, args.end, args.imd_nc_path
            ),
        )
    )

    for group_name, job in jobs:
        if not args.force and manifest.get(group_name, {}).get("status") == "ok":
            print(f"[skip] {group_name}: already marked ok in manifest")
            continue

        print(f"[fetch] {group_name} ...")
        try:
            ds, info = job()
            mode: Literal["w", "a"] = "w" if not store_path.exists() else "a"
            ds.to_zarr(store_path, group=group_name, mode=mode)
            info["fetched_at"] = datetime.utcnow().isoformat() + "Z"
            manifest[group_name] = info
            print(f"[ok]   {group_name}: {info}")
        except Exception as exc:  # a source failing must not abort the others
            manifest[group_name] = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "fetched_at": datetime.utcnow().isoformat() + "Z",
            }
            print(f"[FAIL] {group_name}: {exc}", file=sys.stderr)
        finally:
            manifest["_meta"] = {
                "init_cadence_days": args.init_cadence_days,
                "init_cycle": "00:00 UTC",
                "start": args.start,
                "end": args.end,
                "year": args.year,
                "lead_hours": args.lead_hours,
            }
            _save_manifest(manifest_path, manifest)

    print("\n--- Summary ---")
    print(f"Window requested: {args.start} .. {args.end}")
    for group_name, info in manifest.items():
        if group_name.startswith("_"):
            continue
        status = info.get("status")
        if status == "ok":
            gaps = info.get("known_gaps") or info.get("missing_days") or "none"
            print(f"  {group_name}: OK, variables={info.get('variables')}, gaps={gaps}")
        else:
            print(f"  {group_name}: FAILED - {info.get('error')}")

    n_failed = sum(
        1 for k, i in manifest.items() if not k.startswith("_") and i.get("status") != "ok"
    )
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
