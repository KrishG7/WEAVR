import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from preflight_stores import alignment_problem, pairing_offset_correlations  # noqa: E402


def _stores(shift_days: int, n_days: int = 60, seed: int = 0):
    """A rain-like obs series and a 24 h forecast that sees it `shift_days` late.

    `shift_days = 0` is a correctly built store: the forecast initialised on
    day t is noisy truth for IMD day t+1, the day a 24 h forecast verifies on.
    `shift_days = -1` is the #106 failure seen from the store side -- each
    forecast actually describes the day before the one it is paired with.
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2020-06-01", periods=n_days + 2, freq="D")
    truth = rng.gamma(0.8, 8.0, size=(days.size, 3, 4))
    obs = xr.DataArray(truth, dims=("time", "latitude", "longitude"), coords={"time": days})
    inits = days[1 : 1 + n_days]
    verifies = np.arange(1, 1 + n_days) + 1 + shift_days
    fc = truth[verifies] + rng.normal(scale=4.0, size=(n_days, 3, 4))
    forecast = xr.DataArray(fc, dims=("time", "latitude", "longitude"), coords={"time": inits})
    return forecast, obs


def test_correct_pairing_peaks_at_offset_zero():
    forecast, obs = _stores(shift_days=0)
    corr = pairing_offset_correlations(forecast, obs, lead_hours=24)
    assert corr[0] > corr[-1] and corr[0] > corr[1]
    assert alignment_problem("store", corr) is None


def test_a_one_day_shift_is_caught():
    forecast, obs = _stores(shift_days=-1)
    corr = pairing_offset_correlations(forecast, obs, lead_hours=24)
    assert corr[-1] > corr[0]
    message = alignment_problem("store", corr)
    assert message is not None
    assert "offset -1" in message and "#106" in message


def test_no_skill_at_all_is_not_passed_as_aligned():
    # A forecast with no relationship to the observations cannot prove the
    # pairing is right, so the margin rule must not wave it through.
    _, obs = _stores(shift_days=0, seed=1)
    noise = xr.DataArray(
        np.random.default_rng(2).normal(size=(60, 3, 4)),
        dims=("time", "latitude", "longitude"),
        coords={"time": pd.date_range("2020-06-02", periods=60, freq="D")},
    )
    corr = pairing_offset_correlations(noise, obs, lead_hours=24)
    assert alignment_problem("store", corr) is not None
