import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_single_source_baseline import (  # noqa: E402
    FORECAST_SOURCE_NAMES,
    best_single_member_on_train,
    score_lead_all_sources,
    train_rmse_by_source,
)
from run_tier0_baseline import PRECIP_M_TO_MM  # noqa: E402

from weavr import verify as V  # noqa: E402

LAT = np.array([10.0, 11.0, 12.0])
LON = np.array([75.0, 76.0, 77.0])


def _synthetic_store(tmp_path: Path, n_init: int = 10, year: int = 2020) -> Path:
    """A tiny store with the same group layout as data/baseline_2020_jjas.zarr."""
    store = tmp_path / f"tiny_{year}.zarr"
    init_times = pd.date_range(f"{year}-06-01", periods=n_init, freq="D")
    leads = np.array([24, 48], dtype="int64")
    lead_days = leads // 24
    rng = np.random.default_rng(0)
    # Observations run past the last init by the longest lead, because a 24 h
    # forecast from 00 UTC on day t verifies on IMD day t+1 (IMD labels a day
    # by its 08:30 IST end, #106) and a 48 h one on t+2.
    obs_days = pd.date_range(f"{year}-06-01", periods=n_init + int(lead_days.max()), freq="D")
    obs_values = rng.uniform(0.0, 20.0, size=(obs_days.size, LAT.size, LON.size))

    obs = xr.Dataset(
        {"rain": (("time", "latitude", "longitude"), obs_values)},
        coords={"time": obs_days, "latitude": LAT, "longitude": LON},
    )
    obs.to_zarr(store, group="imd_observed", mode="w", consolidated=True)

    offsets_mm = {"graphcast": 1.0, "ifs_ens_mean": 3.0, "hres": 6.0}
    for name, offset in offsets_mm.items():
        # Each forecast is the observation on the day it verifies on, plus a
        # source-specific offset, so the offset is exactly its error.
        values = np.stack(
            [obs_values[days : days + n_init] + offset for days in lead_days], axis=1
        )
        source = xr.Dataset(
            {
                "total_precipitation_24hr": (
                    ("time", "prediction_timedelta", "latitude", "longitude"),
                    values / PRECIP_M_TO_MM,
                )
            },
            coords={
                "time": init_times,
                "prediction_timedelta": leads,
                "latitude": LAT,
                "longitude": LON,
            },
        )
        source.to_zarr(store, group=name, mode="a", consolidated=True)
    return store


def _synthetic_climatology(n_days: int = 30) -> xr.Dataset:
    rng = np.random.default_rng(1)
    times = pd.date_range("2019-06-01", periods=n_days, freq="D")
    return xr.Dataset(
        {
            "rain": (
                ("time", "latitude", "longitude"),
                rng.uniform(0.0, 20.0, size=(n_days, LAT.size, LON.size)),
            )
        },
        coords={"time": times, "latitude": LAT, "longitude": LON},
    )


class TestBestSingleMemberOnTrain:
    def test_picks_the_lowest_train_rmse(self):
        assert (
            best_single_member_on_train({"a": 3.0, "b": 1.5, "c": 9.0}) == "b"
        )

    def test_breaks_ties_by_name_so_the_choice_is_reproducible(self):
        assert best_single_member_on_train({"zeta": 2.0, "alpha": 2.0}) == "alpha"

    def test_rejects_an_empty_mapping_rather_than_returning_nothing(self):
        with pytest.raises(ValueError, match="no source to choose from"):
            best_single_member_on_train({})


class TestTrainRmseBySource:
    def test_scores_only_the_train_samples(self):
        samples = np.arange(4)
        obs = xr.DataArray(
            np.zeros((4, 1, 1)),
            coords={"sample": samples, "latitude": [0.0], "longitude": [0.0]},
            dims=["sample", "latitude", "longitude"],
        )
        # The two test samples carry a huge error; if the mask were ignored
        # the RMSE would be dominated by them.
        forecast = obs + xr.DataArray(
            np.array([2.0, 2.0, 100.0, 100.0]), coords={"sample": samples}, dims=["sample"]
        )
        train_mask = np.array([True, True, False, False])

        result = train_rmse_by_source({"a": forecast}, obs, train_mask)

        assert result["a"] == pytest.approx(2.0)


class TestScoreLeadAllSources:
    def test_produces_one_row_per_source_plus_the_train_selected_row(self, tmp_path):
        store = _synthetic_store(tmp_path)
        sources = {
            name: xr.open_zarr(store, group=name, consolidated=True)
            for name in FORECAST_SOURCE_NAMES
        }
        obs = xr.open_zarr(store, group="imd_observed", consolidated=True).load()

        rows = score_lead_all_sources(
            sources,
            obs,
            _synthetic_climatology(),
            lead_hours=24,
            test_fraction=0.2,
            thresholds=V.IMD_RAIN_THRESHOLDS_MM,
            neighborhood_size=3,
        )

        assert [row["source"] for row in rows] == [
            *FORECAST_SOURCE_NAMES,
            "best_single_member_on_train",
        ]
        assert all(row["lead_hours"] == 24 for row in rows)
        assert all(row["n_test"] >= 1 for row in rows)

    def test_selects_the_source_with_the_lowest_train_rmse_not_the_lowest_test_rmse(
        self, tmp_path
    ):
        store = _synthetic_store(tmp_path)
        sources = {
            name: xr.open_zarr(store, group=name, consolidated=True)
            for name in FORECAST_SOURCE_NAMES
        }
        obs = xr.open_zarr(store, group="imd_observed", consolidated=True).load()

        rows = score_lead_all_sources(
            sources,
            obs,
            _synthetic_climatology(),
            lead_hours=24,
            test_fraction=0.2,
            thresholds=V.IMD_RAIN_THRESHOLDS_MM,
            neighborhood_size=3,
        )

        best = rows[-1]
        assert best["selected_source"] == "graphcast"
        assert [row["source"] for row in rows if row["selected_on_train"]] == ["graphcast"]

    def test_applies_the_metres_to_millimetres_conversion(self, tmp_path):
        # graphcast was written as obs + 1 mm, in metres. With the
        # conversion its test RMSE is 1 mm; without it, the forecast is
        # ~0 and the RMSE would be the size of the observations (~11 mm).
        store = _synthetic_store(tmp_path)
        sources = {
            name: xr.open_zarr(store, group=name, consolidated=True)
            for name in FORECAST_SOURCE_NAMES
        }
        obs = xr.open_zarr(store, group="imd_observed", consolidated=True).load()

        rows = score_lead_all_sources(
            sources,
            obs,
            _synthetic_climatology(),
            lead_hours=24,
            test_fraction=0.2,
            thresholds=V.IMD_RAIN_THRESHOLDS_MM,
            neighborhood_size=3,
        )

        graphcast = next(row for row in rows if row["source"] == "graphcast")
        assert graphcast["rmse_mm"] == pytest.approx(1.0, abs=1e-6)

    def test_score_lead_all_sources_multi_season(self, tmp_path):
        from weavr.stores import open_multi_season

        s1 = _synthetic_store(tmp_path / "y1", year=2018)
        s2 = _synthetic_store(tmp_path / "y2", year=2020)
        sources = {
            name: open_multi_season([s1, s2], group=name)
            for name in FORECAST_SOURCE_NAMES
        }
        obs = open_multi_season([s1, s2], group="imd_observed").load()

        rows = score_lead_all_sources(
            sources,
            obs,
            _synthetic_climatology(),
            lead_hours=24,
            test_fraction=0.2,
            thresholds=V.IMD_RAIN_THRESHOLDS_MM,
            neighborhood_size=3,
        )
        assert len(rows) == 12
        folds = {r["fold"] for r in rows}
        assert folds == {"2018", "2020", "pooled"}
        sources_present = {r["source"] for r in rows}
        assert sources_present == {*FORECAST_SOURCE_NAMES, "best_single_member_on_train"}
