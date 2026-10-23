import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_tier0_baseline import PRECIP_M_TO_MM, _align_to_imd_day, score_lead  # noqa: E402


class TestAlignToImdDay:
    def test_maps_valid_time_to_the_imd_day_it_falls_within(self):
        # init 2020-06-01T00:00 UTC + 24h lead -> valid at 2020-06-02T00:00 UTC,
        # which falls inside the IMD day window 06-01T03:00 -> 06-02T03:00.
        # IMD labels that window by its end, so it is 2020-06-02 -- labelling
        # it 2020-06-01 was the one-day-early pairing of #106.
        times = pd.date_range("2020-06-01", periods=3, freq="7D")
        da = xr.DataArray(np.arange(3, dtype=float), coords={"time": times}, dims=["time"])

        result = _align_to_imd_day(da, lead_hours=24)

        assert result.dims == ("sample",)
        expected = pd.DatetimeIndex(times).normalize() + pd.Timedelta(days=1)
        assert list(pd.DatetimeIndex(result["sample"].values)) == list(expected)

    def test_longer_lead_can_cross_into_the_next_calendar_day(self):
        # init at 00:00 UTC + 96h (4 days) lands at 00:00 UTC 4 days later:
        # three more IMD days than the 24h case, no more and no less.
        times = pd.date_range("2020-06-01", periods=1, freq="D")
        da = xr.DataArray([1.0], coords={"time": times}, dims=["time"])

        result_24h = _align_to_imd_day(da, lead_hours=24)
        result_96h = _align_to_imd_day(da, lead_hours=96)

        assert pd.Timestamp(result_96h["sample"].values[0]) - pd.Timestamp(
            result_24h["sample"].values[0]
        ) == pd.Timedelta(days=3)


class TestPrecipUnitConversion:
    def test_meters_to_millimeters_constant(self):
        # WeatherBench 2's total_precipitation_24hr is in meters; IMD's rain
        # is in millimeters -- caught by comparing real magnitudes
        # (forecast ~0.006, obs ~5.7) before trusting any score, not assumed.
        assert PRECIP_M_TO_MM == 1000.0


class TestScoreLeadMultiSeason:
    def test_score_lead_multi_season_yields_fold_rows_and_pooled_row(self):
        times = pd.date_range("2018-06-01", "2018-06-10", freq="D").union(
            pd.date_range("2020-06-01", "2020-06-10", freq="D")
        )
        lat = np.array([10.0, 11.0])
        lon = np.array([75.0, 76.0])

        fc = xr.DataArray(
            np.ones((len(times), len(lat), len(lon))),
            coords={"sample": times, "latitude": lat, "longitude": lon},
            dims=["sample", "latitude", "longitude"],
        )
        ob = xr.DataArray(
            np.ones((len(times), len(lat), len(lon))) * 1.5,
            coords={"sample": times, "latitude": lat, "longitude": lon},
            dims=["sample", "latitude", "longitude"],
        )
        clim_coords = {
            "time": pd.date_range("2010-06-01", periods=5),
            "latitude": lat,
            "longitude": lon,
        }
        clim = xr.Dataset(
            {"rain": (("time", "latitude", "longitude"), np.ones((5, len(lat), len(lon))))},
            coords=clim_coords,
        )

        rows = score_lead(
            fc, ob, clim, test_fraction=0.2, thresholds=(7.5, 64.5), neighborhood_size=3
        )
        assert len(rows) == 3
        assert rows[0]["fold"] == "2018"
        assert rows[1]["fold"] == "2020"
        assert rows[2]["fold"] == "pooled"
        assert rows[2]["split"] == "leave_one_year_out"
        assert rows[2]["n_test"] == len(times)

    def test_score_lead_single_season_yields_single_row(self):
        times = pd.date_range("2020-06-01", "2020-06-10", freq="D")
        lat = np.array([10.0, 11.0])
        lon = np.array([75.0, 76.0])

        fc = xr.DataArray(
            np.ones((len(times), len(lat), len(lon))),
            coords={"sample": times, "latitude": lat, "longitude": lon},
            dims=["sample", "latitude", "longitude"],
        )
        ob = xr.DataArray(
            np.ones((len(times), len(lat), len(lon))) * 1.5,
            coords={"sample": times, "latitude": lat, "longitude": lon},
            dims=["sample", "latitude", "longitude"],
        )
        clim_coords = {
            "time": pd.date_range("2010-06-01", periods=5),
            "latitude": lat,
            "longitude": lon,
        }
        clim = xr.Dataset(
            {"rain": (("time", "latitude", "longitude"), np.ones((5, len(lat), len(lon))))},
            coords=clim_coords,
        )

        rows = score_lead(
            fc, ob, clim, test_fraction=0.2, thresholds=(7.5,), neighborhood_size=3
        )
        assert len(rows) == 1
        assert rows[0]["fold"] == "seasonal_block_split"

