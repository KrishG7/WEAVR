#!/usr/bin/env python3
"""Fetch the real monsoon low-pressure-system catalogue and build a
per-day "monsoon depression (or stronger) presence" indicator for 2020 JJAS.

`kieranmrhunt/monsoon-low-atlas`'s v5.5.1 ERA5-derived South Asian
low-pressure-system catalogue (Zenodo DOI 10.5281/zenodo.22142640) --
checked directly, not assumed from the project's version numbering: a
newer v5.6 exists in that project's own GitHub repo, but its own release
manifest states `"zenodo_status": "not_published"`, so v5.5.1 is the
newest version actually fetchable. See docs/phase5-regime-covariate-scope.md.

The catalogue's own `imd_category`/`imd_label` columns (1="low", 2=
"depression", 3="deep_depression", 4="cyclonic_storm", 5=
"severe_cyclonic_storm") give an IMD-equivalent intensity classification
directly -- confirmed live against the real 2020 JJAS rows (2,235 hourly
positions, including 125 real "depression" and 57 "deep_depression"
rows). Issue #7 asks for "monsoon depression... presence" specifically,
not any low-pressure area, so a day counts as a depression day only when
at least one real hourly position that day has `imd_category >= 2`
(depression or stronger) -- a plain "low" (category 1, the weakest,
far more common stage) does not count.

A day is labelled using the same IMD day convention every other daily
series in this project uses (03:00 UTC to the next day's 03:00 UTC, labelled
by the date it ends on: `weavr.grid.imd_day_of_instant`) -- not a UTC
midnight-to-midnight day, so this covariate's own day boundaries line up with
imd_observed's.

Usage:
    python scripts/build_monsoon_depression_index.py [--out PATH]
        [--year Y] [--zenodo-record-id ID]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from weavr.grid import (
    GRID_LAT_MAX,
    GRID_LAT_MIN,
    GRID_LON_MAX,
    GRID_LON_MIN,
    imd_day_of_instant,
)

ZENODO_RECORD_ID = "22142640"
CATALOGUE_FILENAME = "lps_v5.5.1-era5-1940-2025-core.parquet"
# "depression" (2) or stronger, per imd_category's own real ordering --
# a plain "low" (1) does not count as depression presence, per this
# module's docstring.
MIN_DEPRESSION_IMD_CATEGORY = 2


def _zenodo_file_url(record_id: str, filename: str) -> str:
    return f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"


def fetch_lps_catalogue(
    out_dir: Path, record_id: str = ZENODO_RECORD_ID, filename: str = CATALOGUE_FILENAME
) -> Path:
    """Downloads the real catalogue (~60MB) to `out_dir`, skipping the
    download if it's already there -- this is a fixed, versioned archive
    (not a live-refreshed source), so a re-run doesn't need to re-fetch it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / filename
    if local_path.exists():
        print(f"Already have {local_path}, skipping download")
        return local_path

    url = _zenodo_file_url(record_id, filename)
    print(f"Fetching real LPS catalogue from {url}")
    response = requests.get(url, timeout=300.0)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    print(f"Wrote {local_path} ({len(response.content) / 1e6:.1f}MB)")
    return local_path


def load_india_season_positions(
    catalogue_path: Path, year: int
) -> pd.DataFrame:
    """Real hourly LPS positions restricted to weavr's own common India
    grid domain and one JJAS season -- the same lat/lon bounds
    (`weavr.grid.GRID_LAT_MIN/MAX`, `GRID_LON_MIN/MAX`) every other source
    in this project is aligned to, so this covariate's domain matches
    `imd_observed`'s own.
    """
    frame = pd.read_parquet(catalogue_path)
    in_season = (frame["year"] == year) & (frame["month"] >= 6) & (frame["month"] <= 9)
    in_domain = (
        (frame["lat"] >= GRID_LAT_MIN)
        & (frame["lat"] <= GRID_LAT_MAX)
        & (frame["lon"] >= GRID_LON_MIN)
        & (frame["lon"] <= GRID_LON_MAX)
    )
    return frame.loc[in_season & in_domain].copy()


def build_daily_presence(positions: pd.DataFrame, year: int) -> xr.Dataset:
    """One boolean per real IMD day (03 UTC to next 03 UTC) over the full
    JJAS season: whether any real position that day has `imd_category >=
    MIN_DEPRESSION_IMD_CATEGORY`. Every JJAS day is included, even one
    with zero real LPS rows in the domain (a real "no depression" day, not
    a gap) -- so the result aligns one-to-one with `imd_observed`'s own
    full daily calendar.
    """
    all_days = pd.date_range(f"{year}-06-01", f"{year}-09-30", freq="D")

    if positions.empty:
        imd_day = pd.Series([], dtype="datetime64[ns]")
        category = pd.Series([], dtype="int64")
    else:
        imd_day = pd.Series(imd_day_of_instant(positions["time"]), index=positions.index)
        category = positions["imd_category"]

    daily = pd.DataFrame({"imd_day": imd_day, "category": category})
    max_category_by_day = daily.groupby("imd_day")["category"].max()
    max_category_by_day = max_category_by_day.reindex(all_days, fill_value=0)

    depression_present = (max_category_by_day >= MIN_DEPRESSION_IMD_CATEGORY).to_numpy()

    return xr.Dataset(
        {
            "depression_present": (("time",), depression_present),
            "max_imd_category": (("time",), max_category_by_day.to_numpy().astype(np.int64)),
        },
        coords={"time": all_days},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/monsoon_depression_2020_jjas.zarr")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--zenodo-record-id", default=ZENODO_RECORD_ID)
    parser.add_argument("--cache-dir", default="data/.monsoon_low_atlas_cache")
    args = parser.parse_args()

    catalogue_path = fetch_lps_catalogue(
        Path(args.cache_dir), record_id=args.zenodo_record_id
    )

    positions = load_india_season_positions(catalogue_path, args.year)
    print(
        f"{args.year} JJAS: {len(positions)} real hourly positions in the India domain"
    )
    print(positions["imd_label"].value_counts() if not positions.empty else "none")

    dataset = build_daily_presence(positions, args.year)
    n_depression_days = int(dataset["depression_present"].sum())
    print(
        f"{dataset.sizes['time']} real IMD days, "
        f"{n_depression_days} with a depression (or stronger) present"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(out_path, mode="w")
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
