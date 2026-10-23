import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weavr.grid import (
    COMMON_LAT,
    COMMON_LON,
    GRID_RESOLUTION_DEG,
    SourceTooCoarseError,
    align_forecast_to_imd_day,
    imd_day_of_accumulation,
    imd_day_of_instant,
    regrid_to_common,
    resample_to_imd_day,
)


def _synthetic_grid(lat_res: float, lon_res: float) -> xr.Dataset:
    lat = np.arange(6.0, 39.0, lat_res)
    lon = np.arange(66.0, 101.0, lon_res)
    data = np.random.default_rng(0).random((len(lat), len(lon)))
    da = xr.DataArray(
        data, coords={"latitude": lat, "longitude": lon}, dims=["latitude", "longitude"]
    )
    return xr.Dataset({"x": da})


class TestRegridToCommon:
    def test_finer_source_regrids_to_common_shape(self):
        ds = _synthetic_grid(lat_res=0.1, lon_res=0.1)
        out = regrid_to_common(ds)

        assert out["x"].shape == (len(COMMON_LAT), len(COMMON_LON))
        assert out.latitude.min().item() == pytest.approx(COMMON_LAT.min())
        assert out.latitude.max().item() == pytest.approx(COMMON_LAT.max())

    def test_matching_resolution_source_passes_through(self):
        ds = _synthetic_grid(lat_res=GRID_RESOLUTION_DEG, lon_res=GRID_RESOLUTION_DEG)
        out = regrid_to_common(ds)
        assert out["x"].shape == (len(COMMON_LAT), len(COMMON_LON))

    def test_coarser_source_raises_instead_of_upsampling(self):
        ds = _synthetic_grid(lat_res=0.5, lon_res=0.5)
        with pytest.raises(SourceTooCoarseError):
            regrid_to_common(ds)

    def test_missing_coords_raises_value_error(self):
        da = xr.DataArray(np.zeros((3, 3)), dims=["y", "x"])
        with pytest.raises(ValueError):
            regrid_to_common(xr.Dataset({"v": da}))


class TestResampleToImdDay:
    def test_full_days_sum_to_24_hours(self):
        times = pd.date_range("2024-06-14T03:00", "2024-06-16T02:00", freq="1h")
        precip = xr.DataArray(np.ones(len(times)), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"precip": precip})

        out = resample_to_imd_day(ds)

        # Two full IMD days: [06-14T03:00, 06-15T03:00) and [06-15T03:00, 06-16T03:00)
        assert list(out["precip"].values) == [24.0, 24.0]

    def test_day_is_labelled_by_the_date_it_ends_on(self):
        # IMD labels the 08:30 IST -> 08:30 IST total by the date it ends on, so
        # the window 06-14T03:00 -> 06-15T03:00 UTC is IMD day 2024-06-15. The
        # start-date label this used to assert is the #106 bug.
        times = pd.date_range("2024-06-14T03:00", periods=24, freq="1h")
        precip = xr.DataArray(np.ones(24), coords={"time": times}, dims=["time"])
        ds = xr.Dataset({"precip": precip})

        out = resample_to_imd_day(ds)

        assert len(out["time"]) == 1
        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2024-06-15")
        assert out["precip"].values[0] == 24.0

    def test_partial_window_at_data_edges_sums_only_available_hours(self):
        # Data starts mid-window: the window [06-13T03:00, 06-14T03:00), IMD
        # day "2024-06-14", has data only in its last 3 hours.
        times = pd.date_range("2024-06-14T00:00", "2024-06-14T02:00", freq="1h")
        ds = xr.Dataset({"precip": xr.DataArray(np.ones(3), coords={"time": times}, dims=["time"])})

        out = resample_to_imd_day(ds)

        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2024-06-14")
        assert out["precip"].values[0] == 3.0

    def test_missing_time_dim_raises_value_error(self):
        ds = xr.Dataset({"v": xr.DataArray(np.zeros(3), dims=["x"])})
        with pytest.raises(ValueError):
            resample_to_imd_day(ds)


class TestImdDayOfAccumulation:
    """IMD day D is the window (D-1 03 UTC, D 03 UTC], labelled by its end (#106)."""

    @pytest.mark.parametrize(
        ("valid_time", "imd_day"),
        [
            # 00 UTC: the window (D-1 00, D 00] overlaps IMD day D for 21 h.
            ("2020-06-02T00:00", "2020-06-02"),
            # Ends exactly where an IMD day ends: that day, a full overlap.
            ("2020-06-02T03:00", "2020-06-02"),
            # 06 UTC: 21 h of overlap with the IMD day ending three hours earlier.
            ("2020-06-02T06:00", "2020-06-02"),
            # 12 UTC: 15 h with the day ending nine hours earlier, 9 h with the next.
            ("2020-06-02T12:00", "2020-06-02"),
            # 18 UTC: now the next IMD day overlaps more (15 h against 9 h).
            ("2020-06-02T18:00", "2020-06-03"),
        ],
    )
    def test_picks_the_imd_day_the_window_overlaps_most(self, valid_time, imd_day):
        assert imd_day_of_accumulation(np.datetime64(valid_time))[0] == pd.Timestamp(imd_day)

    def test_accepts_arrays(self):
        valid = pd.to_datetime(["2020-06-02T00:00", "2020-06-09T00:00"])
        out = imd_day_of_accumulation(valid.values)
        assert list(out) == list(pd.to_datetime(["2020-06-02", "2020-06-09"]))


class TestImdDayOfInstant:
    @pytest.mark.parametrize(
        ("instant", "imd_day"),
        [
            ("2020-07-10T00:00", "2020-07-10"),  # before 10 July's 03 UTC end
            ("2020-07-10T03:00", "2020-07-10"),  # exactly at the end: still 10 July
            ("2020-07-10T03:01", "2020-07-11"),  # just after: the next IMD day
            ("2020-07-10T12:00", "2020-07-11"),
        ],
    )
    def test_window_is_closed_at_its_03_utc_end(self, instant, imd_day):
        assert imd_day_of_instant(np.datetime64(instant))[0] == pd.Timestamp(imd_day)


class TestAlignForecastToImdDay:
    def test_each_lead_verifies_on_init_plus_lead_days(self):
        # 00 UTC inits: a 24 h forecast verifies on the IMD day after its init,
        # a 120 h one five days after. This is the pairing every runner uses.
        inits = pd.date_range("2020-06-01", periods=3, freq="D")
        da = xr.DataArray(np.arange(3.0), coords={"time": inits}, dims=["time"])
        for lead in (24, 48, 72, 96, 120):
            out = align_forecast_to_imd_day(da, lead)
            assert out.dims == ("sample",)
            assert list(pd.DatetimeIndex(out["sample"].values)) == list(
                inits + pd.Timedelta(days=lead // 24)
            )
