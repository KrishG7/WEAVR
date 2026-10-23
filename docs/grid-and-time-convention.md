# Grid and time convention

Locked decision, implemented in [`src/weavr/grid.py`](../src/weavr/grid.py).
Every ingestion script must call `regrid_to_common` / `resample_to_imd_day`
from that module rather than reimplementing regridding or resampling
inline — this is exactly the kind of thing that silently diverges between
scripts if left to each author's judgement.

## Common grid: 0.25° lat/lon, bounds 6.5–38.5N / 66.5–100.0E

This matches IMD gridded rainfall's own native resolution and the
resolution WeatherBench 2's regridded products are served at, so the two
primary data sources ([docs/data-sources.md](data-sources.md)) need no
further interpolation to compare against each other.

The bounds are **not** the rough 6–38N/68–98E estimate a first pass at this
decision might reach for — they're taken directly from a real IMD
gridded-rainfall pull (`imdlib`, `var_type="rain"`, year 2023), which
returned exactly `lat 6.5–38.5, lon 66.5–100.0` at 0.25° spacing. Using
IMD's actual grid rather than an approximation means regridding onto it
needs no further interpolation at the domain edges.

`regrid_to_common()` refuses (`SourceTooCoarseError`) to regrid a source
whose native resolution is coarser than 0.25° — upsampling a coarser source
would create the illusion of matching resolution while hiding real skill
differences behind interpolation artifacts. If a genuinely coarser source
needs to be included later, it should go through an explicit, documented
area-weighted remap (e.g. `xesmf`), not silently through this function.

## Daily accumulation window: IMD's 08:30 IST → 08:30 IST, labelled by its end

This is IMD's own observation-day convention (08:30 IST = 03:00 UTC), not
UTC midnight-to-midnight, and IMD labels the window by the date it **ends**
on. IMD's rainfall-information page tags the rainfall "recorded from 0830 hrs
of 21-10-2026 to 0830 hrs of 22-10-2026" as the value for **22-10-2026**
(https://mausam.imd.gov.in/responsive/rainfallinformation_msd.php). So a day
labelled `2024-06-15` covers `2024-06-14T03:00 UTC` through
`2024-06-15T03:00 UTC`. Every other source (WeatherBench 2, ECMWF open data)
must be aligned onto this window, not the other way around — IMD is the
ground truth we verify against (§3.2/§3.6 of the research brief), so its
convention is the one that must not silently drift.

**Correction (#106).** This section used to say the window is labelled by the
date it *starts* on. That was wrong, and every forecast in the project was
paired with the IMD day before the one it covers -- a 3-hour overlap instead of
21. Nothing structural caught it: both labellings give one IMD day per
forecast. The data did: every source's day-to-day correlation with IMD peaked
one day after that pairing, at every lead, in both seasons, and re-pairing cut
raw GraphCast's 24 h RMSE by ~10%.

The rule lives in one place, `weavr.grid`, and every runner goes through it:

- `imd_day_of_accumulation(valid_time)` -- the IMD day a 24 h accumulation
  ending at `valid_time` is scored against: the one whose window overlaps it
  most. A 00 UTC valid time maps to the IMD day ending three hours later.
- `align_forecast_to_imd_day(da, lead_hours)` -- the same, for a forecast
  array indexed by init time; this is the pairing every runner uses.
- `imd_day_of_instant(t)` -- the IMD day an instantaneous observation falls
  in (`(D-1 03 UTC, D 03 UTC]`), used for 6-hourly depression positions.
- `resample_to_imd_day(ds)` -- sums a sub-daily series into IMD days.

`scripts/preflight_stores.py` checks the pairing on the real stores before any
long run: each source's correlation with IMD must peak at offset 0. That is the
check that would have caught #106 on day one.

`resample_to_imd_day()` sums (not averages) over each 24h window, and labels
the resulting day by the date the window *ends* on, as IMD does.

**Implementation note**: `xarray.Dataset.resample(freq="1D", offset=...)`
does *not* actually shift calendar-day bin boundaries in the xarray/pandas
versions this was tested against — the `offset` kwarg is silently ignored,
which would produce midnight-UTC-aligned bins while claiming to be
03-UTC-aligned. This was caught by a unit test (`test_grid.py`) checking an
exact 03:00 UTC-aligned day's sum, which failed on the naive
`resample(..., offset=...)` implementation. The fix shifts the time
coordinate back by the offset before resampling to `"1D"`, then relabels —
verified by hand against three separate cases (full days, exact boundary,
partial edge windows) in the test suite.

## Why this matters

Grid/time misalignment is a common source of silent errors: two sources
that look aligned (both "daily", both "0.25°") can still be comparing
different physical windows or different physical points, producing skill
differences that reflect the alignment bug rather than real model
performance. Locking this in one place, with tests that fail loudly on a
misimplementation, is cheaper than re-discovering the bug downstream in a
skill-score comparison.
