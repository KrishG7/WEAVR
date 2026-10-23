import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_daily_verification import _align_to_imd_day, rolling_verification_for_lead  # noqa: E402


class TestAlignToImdDay:
    def test_shifts_init_time_by_lead_and_renames_to_sample(self):
        da = xr.DataArray(
            np.array([1.0, 2.0]),
            coords={"time": pd.to_datetime(["2020-06-01T00:00", "2020-06-08T00:00"])},
            dims=["time"],
        )

        aligned = _align_to_imd_day(da, lead_hours=24)

        # init 2020-06-01T00:00 + 24h = valid 2020-06-02T00:00, whose 24 h
        # window overlaps IMD day 2020-06-02 (06-01T03:00 -> 06-02T03:00,
        # labelled by its end) for 21 of 24 hours.
        expected = pd.to_datetime(["2020-06-02", "2020-06-09"])
        assert aligned.dims == ("sample",)
        np.testing.assert_array_equal(pd.DatetimeIndex(aligned["sample"].values), expected)


class TestRollingVerificationForLead:
    def test_real_shaped_store_produces_a_scored_row_with_no_drift_on_stable_data(self, tmp_path):
        lat = [10.0, 10.25]
        lon = [70.0, 70.25]
        times = pd.date_range("2020-06-01", periods=10, freq="7D")
        rng = np.random.default_rng(0)

        def _forecast_da(values):
            return xr.DataArray(
                values,
                coords={"time": times, "latitude": lat, "longitude": lon},
                dims=["time", "latitude", "longitude"],
            )

        # Values in meters (matching the real WeatherBench 2 convention this
        # script's own PRECIP_M_TO_MM converts from).
        fc_values = rng.normal(loc=0.005, scale=0.001, size=(10, 2, 2))
        fc = _forecast_da(fc_values).expand_dims(prediction_timedelta=[24])
        ds = xr.Dataset({"total_precipitation_24hr": fc})

        obs_values = fc_values * 1000.0 + rng.normal(scale=0.5, size=(10, 2, 2))
        obs_da = xr.DataArray(
            obs_values,
            # A 24 h forecast from day t verifies on IMD day t+1.
            coords={"time": times + pd.Timedelta(days=1), "latitude": lat, "longitude": lon},
            dims=["time", "latitude", "longitude"],
        )
        obs_ds = xr.Dataset({"rain": obs_da})

        store = tmp_path / "store.zarr"
        ds.to_zarr(store, group="graphcast", mode="w")
        obs_ds.to_zarr(store, group="imd_observed", mode="a")

        row = rolling_verification_for_lead(str(store), "graphcast", 24)

        assert row["source"] == "graphcast"
        assert row["lead_hours"] == 24
        assert row["n_samples"] == 10
        assert row["baseline_n_samples"] == 10
        # Stable synthetic data (same generating distribution throughout)
        # should not trip the drift flag.
        assert row["is_drift"] is False
        assert "threshold" in row["reason"]

    def test_baseline_overlapping_the_trailing_window_masks_a_shift_when_n_le_window(
        self, tmp_path
    ):
        """A real, documented limitation of this script's own current data
        scale (see its module docstring): with only `n <= 14`
        (`weavr.drift.TRAILING_WINDOW_SAMPLES`) real samples total, the
        trailing window covers the *entire* series, so `rolling_value`
        collapses to the exact same mean `compute_baseline_stats` computes
        -- drift can never fire no matter how large a late shift is, since
        there is no real history left to hold out as a baseline once the
        window covers everything. This is exactly why the module docstring
        states "no drift detected" is inconclusive-on-more-data at today's
        real sample count, not proof of stability -- proven here rather
        than merely asserted.
        """
        lat = [10.0, 10.25]
        lon = [70.0, 70.25]
        times = pd.date_range("2020-06-01", periods=10, freq="7D")

        fc_values = np.full((10, 2, 2), 0.005)
        obs_values = np.full((10, 2, 2), 5.0)
        obs_values[-2:] += 500.0  # a huge, real error shift on the trailing samples

        fc = xr.DataArray(
            fc_values,
            coords={"time": times, "latitude": lat, "longitude": lon},
            dims=["time", "latitude", "longitude"],
        ).expand_dims(prediction_timedelta=[24])
        ds = xr.Dataset({"total_precipitation_24hr": fc})
        obs_ds = xr.Dataset(
            {
                "rain": xr.DataArray(
                    obs_values,
                    # A 24 h forecast from day t verifies on IMD day t+1.
                    coords={
                        "time": times + pd.Timedelta(days=1),
                        "latitude": lat,
                        "longitude": lon,
                    },
                    dims=["time", "latitude", "longitude"],
                )
            }
        )

        store = tmp_path / "store.zarr"
        ds.to_zarr(store, group="graphcast", mode="w")
        obs_ds.to_zarr(store, group="imd_observed", mode="a")

        row = rolling_verification_for_lead(str(store), "graphcast", 24)

        assert row["n_samples"] == 10  # <= TRAILING_WINDOW_SAMPLES (14)
        assert row["rolling_mean_of_per_sample_rmse_mm"] == pytest.approx(
            row["baseline_mean_rmse_mm"]
        )
        assert row["is_drift"] is False
