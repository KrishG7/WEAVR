import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_quantile_avg_reliability import _sedi, score_probabilities  # noqa: E402


def _grid(values):
    return np.asarray(values, dtype=float).reshape(1, 1, -1)


class TestScoreProbabilities:
    def test_a_cell_without_an_observation_is_not_scored(self):
        # F8: the second cell has a probability but IMD observes nothing there.
        # Scored as "no event" it would add (0.8 - 0)^2 and halve the mean.
        obs = _grid([200.0, np.nan])
        p = _grid([1.0, 0.8])
        got = score_probabilities(p, obs, np.zeros((1, 2)), 115.6)
        assert got["brier"] == pytest.approx(0.0)
        assert got["n_observed_cells"] == 1

    def test_an_observation_exactly_on_the_threshold_is_an_event(self):
        got = score_probabilities(_grid([1.0]), _grid([115.6]), np.zeros((1, 1)), 115.6)
        assert got["brier"] == pytest.approx(0.0)
        assert got["n_events"] == 1

    def test_skill_against_climatology(self):
        obs = _grid([200.0, 0.0, 0.0, 0.0])
        clim = np.full((1, 4), 0.25)
        perfect = score_probabilities(_grid([1.0, 0.0, 0.0, 0.0]), obs, clim, 115.6)
        assert perfect["brier"] == pytest.approx(0.0)
        assert perfect["climatology_brier"] == pytest.approx(0.1875)  # o(1-o), o = 0.25
        assert perfect["brier_skill_score"] == pytest.approx(1.0)
        same = score_probabilities(np.broadcast_to(clim[None], obs.shape), obs, clim, 115.6)
        assert same["brier_skill_score"] == pytest.approx(0.0)

    def test_forecast_over_observed_is_calibration_in_total(self):
        obs = _grid([200.0, 0.0, 0.0, 0.0])
        got = score_probabilities(_grid([0.5, 0.5, 0.0, 0.0]), obs, np.zeros((1, 4)), 115.6)
        assert got["forecast_over_observed"] == pytest.approx(1.0)
        assert got["mean_forecast"] == pytest.approx(0.25)
        assert got["observed_frequency"] == pytest.approx(0.25)


def test_unobserved_cells_do_not_enter_the_contingency_counts():
    # The same 6 observed cells with and without 100 unobserved cells beside
    # them: SEDI must not move. Unobserved cells had been passed on as p=0, o=0,
    # which counts each as a correct negative and inflates SEDI.
    obs = np.array([200.0, 200.0, 0.0, 0.0, 0.0, 0.0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.0, 0.0])
    clim = np.zeros(6)
    base = score_probabilities(p.reshape(1, 1, -1), obs.reshape(1, 1, -1),
                               clim.reshape(1, -1), 115.6)
    padded = score_probabilities(
        np.concatenate([p, np.full(100, 0.0)]).reshape(1, 1, -1),
        np.concatenate([obs, np.full(100, np.nan)]).reshape(1, 1, -1),
        np.zeros((1, 106)), 115.6,
    )
    assert np.isfinite(base["sedi"])
    assert padded["sedi"] == pytest.approx(base["sedi"])


def test_sedi_is_nan_when_degenerate_and_one_for_a_perfect_forecast_limit():
    assert np.isnan(_sedi(0.0, 0.0, 5.0, 5.0))  # no events
    assert np.isnan(_sedi(5.0, 0.0, 1.0, 9.0))  # hit rate 1: undefined
    assert _sedi(9.0, 1.0, 1.0, 99.0) > 0.8


class TestCellMask:
    """Restricting to observed cells must not change a single scored probability."""

    @staticmethod
    def _inputs(monkeypatch):
        from types import SimpleNamespace

        import run_quantile_avg_reliability as m

        # Stub the model pieces: quantiles that depend on each cell's own mean,
        # so a cell's probability is a function of that cell alone.
        monkeypatch.setattr(m, "predict_csgd_params", lambda r, mean, sd: (mean, sd, 0.0))
        monkeypatch.setattr(
            m, "predictive_quantiles_csgd",
            lambda loc, scale, shift, lv: np.outer(loc + 1.0, lv / lv[-1]) * 300.0,
        )
        monkeypatch.setattr(
            m, "bma_predictive_quantiles",
            lambda r, mean, sd, lv: np.outer(mean["graphcast"] + 2.0, lv / lv[-1]) * 280.0,
        )
        monkeypatch.setattr(
            m, "spliced_exceedance_probability",
            lambda p_u, fit, region, y: p_u * 0.5,
        )
        shape = (2, 3, 4)
        rng = np.random.default_rng(0)
        mean = {"graphcast": rng.uniform(0.0, 5.0, shape)}
        spread = {"graphcast": rng.uniform(0.1, 1.0, shape)}
        bins = np.full(shape, "b", dtype=object)
        regions = np.full(shape, "r", dtype=object)
        fit = SimpleNamespace(threshold_u=64.5)
        fitted = SimpleNamespace(is_fallback=False)
        return m, mean, spread, bins, regions, {"b": fitted}, {("b", "r"): fitted}, fit

    def test_masked_run_equals_unmasked_on_observed_cells(self, monkeypatch):
        m, mean, spread, bins, regions, emos, bma, fit = self._inputs(monkeypatch)
        args = (mean, spread, bins, regions, emos, bma, "graphcast", 0.5, fit, [115.6])
        full, _, _ = m.quantile_avg_exceedance_cells(*args)
        observed = np.zeros(bins.shape, dtype=bool)
        observed[:, 1, :] = True
        part, _, _ = m.quantile_avg_exceedance_cells(*args, cell_mask=observed)
        assert np.isfinite(full[115.6]).all()
        assert np.array_equal(part[115.6][observed], full[115.6][observed])
        assert np.isnan(part[115.6][~observed]).all()
