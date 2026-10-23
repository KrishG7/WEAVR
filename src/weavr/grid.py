"""The one common grid and time convention every data source gets aligned to.

Locked decision (see docs/grid-and-time-convention.md for the full
rationale): 0.25 degree lat/lon over India, and IMD's 08:30 IST -> 08:30 IST
24-hour accumulation window (03 UTC -> 03 UTC), **labelled by the date it
ends on** -- not UTC midnight-to-midnight, and not the date it starts on.

Every script that turns a time into an IMD day must go through
`imd_day_of_accumulation`, `imd_day_of_instant`, `align_forecast_to_imd_day`
or `resample_to_imd_day` rather than reimplementing the rule inline. Five
copies of it once agreed with each other and were all wrong by a day (#106),
which is the failure this single definition exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# --- Common grid ------------------------------------------------------------

# Resolution, in degrees. Matches IMD gridded rainfall's native resolution
# and the resolution WeatherBench 2's regridded products are served at.
GRID_RESOLUTION_DEG = 0.25

# Bounds confirmed against a real IMD gridded-rainfall pull (imdlib,
# var_type="rain", 2023): lat 6.5-38.5, lon 66.5-100.0 at 0.25 deg spacing.
# This is IMD's actual native grid, not the rough 6-38N/68-98E estimate a
# first pass at this decision might reach for — use these exact bounds so
# regridding onto IMD's own rainfall grid needs no further interpolation at
# the domain edges.
GRID_LAT_MIN = 6.5
GRID_LAT_MAX = 38.5
GRID_LON_MIN = 66.5
GRID_LON_MAX = 100.0

COMMON_LAT = np.round(
    np.arange(GRID_LAT_MIN, GRID_LAT_MAX + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG), 2
)
COMMON_LON = np.round(
    np.arange(GRID_LON_MIN, GRID_LON_MAX + GRID_RESOLUTION_DEG / 2, GRID_RESOLUTION_DEG), 2
)


class SourceTooCoarseError(ValueError):
    """Raised when a source dataset's native resolution is coarser than the common grid.

    Regridding a coarser source onto a finer grid is upsampling: it produces
    extra grid points that carry no additional information, which can hide
    real skill differences behind an illusion of matching resolution. Fail
    loudly instead.
    """


def _native_resolution(coord: xr.DataArray) -> float:
    diffs = np.abs(np.diff(np.sort(coord.values)))
    return float(np.median(diffs))


def regrid_to_common(
    ds: xr.Dataset,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.Dataset:
    """Regrid a source dataset onto the common 0.25 deg India grid.

    Uses linear interpolation (xarray's built-in `interp`), which is
    sufficient for a common evaluation grid at matched-order resolution.
    Swap for a conservative-remapping tool (e.g. xesmf) if a source needs
    area-weighted regridding (e.g. from a much coarser native grid) rather
    than point interpolation.

    Raises SourceTooCoarseError if the source's native resolution is coarser
    than the common grid, rather than silently upsampling.
    """
    if lat_dim not in ds.coords or lon_dim not in ds.coords:
        raise ValueError(f"Dataset missing expected coords: {lat_dim!r}, {lon_dim!r}")

    native_lat_res = _native_resolution(ds[lat_dim])
    native_lon_res = _native_resolution(ds[lon_dim])
    if native_lat_res > GRID_RESOLUTION_DEG or native_lon_res > GRID_RESOLUTION_DEG:
        raise SourceTooCoarseError(
            f"Source resolution (lat={native_lat_res:.3f}, lon={native_lon_res:.3f} deg) "
            f"is coarser than the common grid ({GRID_RESOLUTION_DEG} deg). "
            "Upsampling would hide skill differences due to resolution, not model quality — "
            "regrid conservatively with an area-weighted tool instead, or exclude this source."
        )

    return ds.interp({lat_dim: COMMON_LAT, lon_dim: COMMON_LON}, method="linear")


# --- IMD day convention ------------------------------------------------------

# IMD's daily accumulation window is 08:30 IST to 08:30 IST (03:00 UTC to
# 03:00 UTC), and IMD labels it by the date it *ends* on: the rainfall
# "recorded from 0830 hrs of 21-10-2026 to 0830 hrs of 22-10-2026" is the
# value for 22-10-2026 (IMD's rainfall-information page). So a day labelled
# "2024-06-15" covers 2024-06-14 03:00 UTC through 2024-06-15 03:00 UTC.
#
# This used to say the opposite -- labelled by the start date -- and every
# forecast in the project was scored against the IMD day before the one it
# covered, overlapping it by 3 of 24 hours (#106). The day-to-day correlation
# of every source against IMD peaks one day later than that pairing, at every
# lead, in both seasons.
IMD_DAY_END_HOUR_UTC = 3

_IMD_DAY_END = pd.Timedelta(hours=IMD_DAY_END_HOUR_UTC)
_HALF_DAY = pd.Timedelta(hours=12)
_ONE_DAY = pd.Timedelta(days=1)


def imd_day_of_accumulation(valid_time) -> pd.DatetimeIndex:
    """The IMD day a 24 h accumulation *ending* at `valid_time` is scored against.

    A forecast accumulation ending at `valid_time` covers
    `(valid_time - 24h, valid_time]`. The IMD day it belongs to is the one
    whose window overlaps it most, i.e. the IMD day ending nearest to
    `valid_time`: round `valid_time - 03 UTC` to the nearest calendar day.
    For the usual 00 UTC valid time that is the IMD day ending three hours
    later (a 21 h overlap); for 12 UTC it is the one ending nine hours earlier
    (15 h). Accepts a scalar or an array of datetimes.
    """
    shifted = pd.DatetimeIndex(np.atleast_1d(np.asarray(valid_time, dtype="datetime64[ns]")))
    return (shifted - _IMD_DAY_END + _HALF_DAY).floor("D")


def imd_day_of_instant(times) -> pd.DatetimeIndex:
    """The IMD day an instantaneous observation at `times` falls inside.

    IMD day `D` is the half-open window `(D-1 03 UTC, D 03 UTC]`, so an
    instant belongs to the day whose window *end* is the first 03 UTC at or
    after it: `ceil(t - 03 UTC)` to a calendar day. 00 UTC on 15 June is in
    IMD day 15 June; 06 UTC on 15 June is already in IMD day 16 June.
    """
    shifted = pd.DatetimeIndex(np.atleast_1d(np.asarray(times, dtype="datetime64[ns]")))
    return (shifted - _IMD_DAY_END).ceil("D")


def align_forecast_to_imd_day(da: xr.DataArray, lead_hours: int) -> xr.DataArray:
    """Reindex a forecast from its init time onto the IMD day its lead verifies on.

    `da` has a `time` coordinate of initialisation times and holds the
    accumulation ending at `init + lead_hours`. The result is indexed by a
    `sample` coordinate of IMD days, ready to join against `imd_observed`.
    """
    valid_time = pd.DatetimeIndex(da["time"].values) + pd.Timedelta(hours=lead_hours)
    return da.assign_coords(time=imd_day_of_accumulation(valid_time)).rename(time="sample")


def resample_to_imd_day(ds: xr.Dataset, time_dim: str = "time") -> xr.Dataset:
    """Resample/accumulate a sub-daily or differently-aligned dataset onto IMD's day.

    Sums each sub-daily *period-start* stamp into the 24 h window from 03:00
    UTC to the next 03:00 UTC that contains it, and labels the result with the
    date the window *ends* on -- IMD's own labelling.

    Implementation note: `Dataset.resample(..., offset=...)` does not shift
    calendar-day ("1D") bin boundaries in the xarray/pandas versions this was
    tested against — the offset is silently ignored, which would produce
    midnight-aligned bins while claiming to be 03 UTC-aligned. Shift the time
    coordinate back by the offset before resampling instead, so the
    calendar-day boundaries fall exactly on the desired 03 UTC/day.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset missing expected time dimension: {time_dim!r}")

    shifted = ds.assign_coords({time_dim: ds[time_dim] - _IMD_DAY_END})
    resampled = shifted.resample({time_dim: "1D"}).sum(skipna=True)

    # Bin labels are now midnight in shifted time, i.e. 03:00 UTC in real time
    # at the window's *start*. IMD labels a day by its end, one day later.
    new_labels = pd.DatetimeIndex(resampled[time_dim].values).normalize() + _ONE_DAY
    return resampled.assign_coords({time_dim: new_labels})
