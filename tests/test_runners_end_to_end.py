"""End-to-end rehearsal of every scoring runner, on miniature stores.

Why this file exists
--------------------
Two separate two-hour Tier 2 runs were lost, for two unrelated reasons, and
neither was caught by any test:

* `data/ifs_ens_2020_jjas_daily.zarr` was 3.4 MB of **zeros** with the right
  shape, dates and dtype. Zeros are not NaN, so every structural check passed.
* The by-bin CSV write raised `ValueError: dict contains fields not in
  fieldnames: 'rmse_mm'` *after* the numbers were computed, because fieldnames
  came from `rows[0]` and the pooled rows carry a column the per-fold rows do
  not. The per-day scores were still buffered and went with it.

Both are invisible to a unit test that calls one function, and far too
expensive to find by running the real 2.3 GB stores. `tests/synthetic_stores`
builds a schema-faithful replica of the whole store set at ~1/1000 scale, so
the real runners can be executed for real -- CLI, argparse, exit codes and all
-- in seconds.

Run this before any multi-hour regeneration. It is the rehearsal; the real
stores are the performance.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_stores import (  # noqa: E402
    SEASONS,
    SyntheticSpec,
    build_all,
    store_args,
)

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


#: Which store flags each runner actually accepts. Taken from the runners'
#: own argparse, because passing a flag a runner does not define is an
#: immediate exit 2 and tells you nothing about the code under test.
RUNNER_STORE_KINDS: dict[str, tuple[str, ...]] = {
    "run_tier0_baseline.py": ("baseline", "climatology"),
    "run_tier1_regional_baseline.py": ("baseline", "climatology"),
    "run_single_source_baseline.py": ("baseline", "climatology"),
    "run_independence_diagnostic.py": ("baseline",),
    "run_phase2_ensemble_baseline.py": ("baseline", "lagged"),
    "run_tier2_hierarchical_baseline.py": ("baseline", "lagged", "ifs_ens", "climatology"),
    "run_tier3_regime_conditioned_baseline.py": ("baseline", "lagged", "ifs_ens", "climatology"),
}


def run_script(
    script: str,
    paths: dict[str, Path],
    results_dir: Path,
    extra: list[str] | None = None,
    expect_ok: bool = True,
) -> subprocess.CompletedProcess:
    """Invoke a runner exactly as the chain does, and capture everything."""
    kinds = RUNNER_STORE_KINDS.get(script, ("baseline", "climatology"))
    cmd = [
        str(PYTHON),
        "-u",
        str(REPO / "scripts" / script),
        *store_args(paths, kinds=kinds),
    ]
    if "climatology" in kinds:
        cmd += ["--climatology", str(paths["climatology"])]
    cmd += ["--results-dir", str(results_dir), "--force", *(extra or [])]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800
    )
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"{script} exited {proc.returncode}\n"
            f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-3000:]}"
        )
    return proc


def test_replica_honours_the_units_contract(stores):
    """Forecast precipitation in METRES, observations in MILLIMETRES.

    Every runner multiplies the forecast by 1000 before scoring, because
    WeatherBench 2's `total_precipitation_24hr` is metres and IMD's `rain` is
    millimetres. Writing millimetres in the replica produces a `bias_mm` of
    roughly `+obs_mean` -- a plausible number, no crash -- which is how this
    mistake presented while building this suite, and it made every score in
    the rehearsal meaningless. Assert the contract so the replica cannot drift
    back.
    """
    import xarray as xr

    fc = xr.open_zarr(stores["baseline_2018"], group="graphcast")[
        "total_precipitation_24hr"
    ].values
    obs = xr.open_zarr(stores["baseline_2018"], group="imd_observed")["rain"].values
    ifs = xr.open_zarr(stores["ifs_ens_2018"])["total_precipitation_24hr"].values

    # Metres: a daily accumulation is O(0.001-0.3), never O(1-400).
    assert fc.mean() < 0.05, f"forecast mean {fc.mean()} looks like mm, not metres"
    assert ifs.mean() < 0.05, f"ifs_ens mean {ifs.mean()} looks like mm, not metres"
    # Millimetres: IMD daily totals are O(1-400).
    assert 0.5 < obs.mean() < 50.0, f"obs mean {obs.mean()} is not in mm"
    # And the two must be comparable after the x1000 conversion, or the blend
    # is dominated by whichever side has the larger numbers.
    ratio = obs.mean() / (fc.mean() * 1000.0)
    assert 0.2 < ratio < 5.0, (
        f"obs mean is {ratio:.2f}x the converted forecast mean; the sources are "
        "not on a comparable scale, so every score will be meaningless"
    )


def test_replica_sources_are_correlated_with_the_truth(stores):
    """Uncorrelated sources make the EMOS regression ill-posed.

    With no relationship between forecast and observation the fitted
    censored-gamma parameters run away and the rehearsal reports
    `emos_ifs_ens_rmse_mm` of 2e13. Real forecasts of the same rain agree to
    within tens of percent, and that correlation is what makes EMOS
    well-conditioned.
    """
    import xarray as xr

    obs = xr.open_zarr(stores["baseline_2018"], group="imd_observed")["rain"].values.ravel()
    graphcast = (
        xr.open_zarr(stores["baseline_2018"], group="graphcast")[
            "total_precipitation_24hr"
        ]
        .values[:, 0]
        .ravel()
    )
    hres = (
        xr.open_zarr(stores["baseline_2018"], group="hres")["total_precipitation_24hr"]
        .values[:, 0]
        .ravel()
    )
    assert np.corrcoef(obs, graphcast)[0, 1] > 0.5
    assert np.corrcoef(obs, hres)[0, 1] > 0.5
    assert np.corrcoef(graphcast, hres)[0, 1] > 0.5


@pytest.fixture(scope="module")
def stores(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("synthetic_stores")
    return build_all(root, SyntheticSpec())


@pytest.fixture(scope="module")
def tier2_out(stores, tmp_path_factory) -> Path:
    """Run Tier 2 once and share it: it is the slowest runner here."""
    out = tmp_path_factory.mktemp("tier2_results")
    run_script("run_tier2_hierarchical_baseline.py", stores, out)
    return out


# ---------------------------------------------------------------------------
# The runners
# ---------------------------------------------------------------------------

#: (script, aggregate filename, writes_per_day?)
#:
#: `run_independence_diagnostic.py` is False because it reports correlation
#: structure, not a forecast: it has no predictive distribution to score per
#: day. Asserting per-day files there would be asserting a change of purpose,
#: so the expectation is recorded rather than assumed.
AGGREGATE_RUNNERS = [
    ("run_tier0_baseline.py", "tier0_baseline.csv", True),
    ("run_tier1_regional_baseline.py", "tier1_regional_baseline.csv", True),
    ("run_single_source_baseline.py", "single_source_baseline.csv", True),
    ("run_phase2_ensemble_baseline.py", "phase2_ensemble_baseline.csv", True),
    ("run_tier3_regime_conditioned_baseline.py", "tier3_regime_conditioned_baseline.csv", True),
    ("run_independence_diagnostic.py", "independence_diagnostic.csv", False),
]


@pytest.mark.parametrize("script,filename,per_day", AGGREGATE_RUNNERS)
def test_runner_completes_and_writes_its_aggregate(
    stores, tmp_path, script, filename, per_day
):
    out = tmp_path / script.replace(".py", "")
    run_script(script, stores, out)
    produced = out / filename
    assert produced.exists(), (
        f"{script} exited 0 but did not write {filename}. "
        f"Present: {sorted(p.name for p in out.iterdir())}"
    )
    frame = pd.read_csv(produced)
    assert len(frame) > 0, f"{filename} is empty"


@pytest.mark.parametrize("script,filename,expects_per_day", AGGREGATE_RUNNERS)
def test_runner_writes_both_folds_to_per_day(
    stores, tmp_path, script, filename, expects_per_day
):
    """Every scoring runner must carry both seasons into its per-day files.

    This is the invariant that was silently broken before: a per-day file
    holding one fold looks exactly like a finished run.
    """
    out = tmp_path / f"{script.replace('.py', '')}_pd"
    run_script(script, stores, out)
    per_day = out / "per_day"
    files = sorted(per_day.glob("*.csv")) if per_day.exists() else []
    if not expects_per_day:
        assert not files, (
            f"{script} is recorded as writing no per-day files, but wrote "
            f"{[f.name for f in files]}"
        )
        return
    assert files, f"{script} wrote no per-day files at all"

    for f in files:
        frame = pd.read_csv(f)
        assert "fold" in frame.columns, f"{f.name} has no 'fold' column"
        folds = set(frame["fold"].astype(str))
        assert len(frame) > 0, f"{f.name} is empty"
        # A single-season run degrades to fold='seasonal_block_split'. Two
        # seasons must give one fold per year.
        assert folds == {"2018", "2020"}, (
            f"{f.name} folds={sorted(folds)}; expected one fold per season. "
            "A per-day file with a single fold is the PerDayScoreWriter "
            "overwrite bug, or one season's store failed silently."
        )


#: Columns whose units are millimetres, so a magnitude bound is meaningful.
#: Deliberately excludes `mse_mm2` (squared, so a bound calibrated in mm would
#: fire on a correct value) and any `*_se` / count column.
def _mm_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if c.endswith("_mm") and "mse" not in c and not c.endswith("_se_mm")
    ]


def test_aggregate_values_are_finite_and_plausible(tier2_out):
    """Numbers, not just files. A zero-filled store yields finite nonsense."""
    frame = pd.read_csv(tier2_out / "tier2_hierarchical_baseline.csv")
    mm_cols = _mm_columns(frame)
    assert mm_cols, f"no mm-valued columns found in {list(frame.columns)}"
    for col in mm_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().all():
            continue
        assert values.abs().max() < 200.0, (
            f"{col} max |value| = {values.abs().max():.1f}; the known BMA/EMOS "
            "blow-up is ~1540 against a true value near 5"
        )


def test_tier2_writes_every_per_day_file_for_all_leads(tier2_out):
    per_day = tier2_out / "per_day"
    leads = {24, 48, 72, 96, 120}
    for method in ("tier2_emos_graphcast", "tier2_emos_ifs_ens", "tier2_bma"):
        found = {
            int(f.name.split("__lead")[-1].split(".")[0])
            for f in per_day.glob(f"{method}__lead*.csv")
        }
        assert found == leads, f"{method}: per-day leads {sorted(found)}, expected {sorted(leads)}"


def test_tier2_by_bin_csv_carries_pooled_and_fold_rows(tier2_out):
    """The write that killed a two-hour run.

    Pooled rows carry an `rmse_mm` the per-fold rows do not. If the writer
    derives fieldnames from `rows[0]`, this file raises *after* the expensive
    work, and the buffered per-day scores are lost with it.
    """
    by_bin = tier2_out / "tier2_hierarchical_baseline_by_bin.csv"
    assert by_bin.exists(), "tier2 did not write its by-bin CSV"
    frame = pd.read_csv(by_bin)
    assert {"fold", "combiner", "bin"} <= set(frame.columns)
    folds = set(frame["fold"].astype(str))
    assert "pooled" in folds, f"by-bin folds={sorted(folds)}, expected a 'pooled' row set"
    assert {"2018", "2020"} <= folds, f"by-bin folds={sorted(folds)}"
    if "rmse_mm" in frame.columns:
        # Present for some rows and absent for others is the exact shape that
        # used to raise; the union-of-keys writer must have coped.
        assert frame["rmse_mm"].notna().any()


# ---------------------------------------------------------------------------
# Fault injection: each of the failures we actually hit
# ---------------------------------------------------------------------------


def test_all_zero_ifs_ens_store_is_caught_before_a_long_run(tmp_path):
    """The 2020 failure, reproduced deliberately.

    A store that is right in shape, dates and dtype, and empty. Zeros are not
    NaN, so a structural check passes it and 2h15m of scoring is spent against
    nothing.
    """
    spec = SyntheticSpec(zero_ifs_ens_years=(2020,))
    paths = build_all(tmp_path / "stores", spec)

    ifs = __import__("xarray").open_zarr(paths["ifs_ens_2020"])
    values = ifs["total_precipitation_24hr"].values
    assert values.shape == (spec.n_times, spec.n_members, 5, spec.n_lat, spec.n_lon)
    assert (values == 0).all(), "fault injection did not take effect"
    assert not np.isnan(values).any(), "the point is that zeros are not NaN"

    # And the guard the builder now applies must reject it.
    sys.path.insert(0, str(REPO / "scripts"))
    from build_ifs_ensemble_store import verify_written_store

    with pytest.raises(ValueError, match="non-zero"):
        verify_written_store(paths["ifs_ens_2020"])
    assert not paths["ifs_ens_2020"].exists(), "an empty store must be deleted"


def test_nan_observed_season_is_never_given_a_finite_score(tmp_path):
    """A season whose observations are all NaN must not acquire a number.

    The rehearsal found the real behaviour: `run_tier0_baseline.py` does not
    score the NaN season at all -- `has_obs` in `equal_weight_mean_and_obs`
    drops every sample whose ground truth is entirely null, and with one season
    left `iter_evaluation_folds` falls back to a single `seasonal_block_split`
    fold. So the run *succeeds*, reports no `2020` fold, and every number it
    does report is a real 2018 number.

    That is defensible (there is nothing to score against) and it is what the
    test pins: no finite score may be attributed to the NaN season, and the
    run must not claim a two-season LOYO it did not perform.

    Worth raising with the team separately: a store that is entirely null
    disappears without a warning, which is the same failure *shape* as the
    2020 IFS-ENS zeros -- a season quietly contributing nothing.
    """
    spec = SyntheticSpec(nan_observed_years=(2020,))
    paths = build_all(tmp_path / "stores", spec)
    out = tmp_path / "results"
    run_script("run_tier0_baseline.py", paths, out)
    frame = pd.read_csv(out / "tier0_baseline.csv")
    col = next(c for c in ("rmse_mm", "tier0_rmse_mm") if c in frame.columns)

    folds = set(frame["fold"].astype(str))
    assert "2020" not in folds, (
        "a season with no ground truth was scored as if it had some; "
        f"folds={sorted(folds)}"
    )
    # Whatever it did report must be finite, i.e. a genuine 2018 number.
    reported = pd.to_numeric(frame[col], errors="coerce").dropna()
    assert not reported.empty
    assert reported.abs().max() < 200.0, f"{col} is implausible: {reported.abs().max()}"


def test_single_season_store_set_does_not_claim_two_season_loyo(tmp_path):
    """One season present must not produce two folds.

    With a single year, `iter_evaluation_folds` returns one
    `seasonal_block_split` fold. That is a legitimate run, but it must be
    labelled as such -- presenting it as two-season LOYO is the failure.
    """
    spec = SyntheticSpec(seasons_written=(2020,))
    paths = build_all(tmp_path / "stores", spec)
    out = tmp_path / "results"
    run_script("run_tier0_baseline.py", paths, out)
    frame = pd.read_csv(out / "tier0_baseline.csv")
    folds = set(frame["fold"].astype(str))
    assert folds == {"seasonal_block_split"}, (
        f"single-season store set produced folds={sorted(folds)}"
    )
    per_day = pd.read_csv(sorted((out / "per_day").glob("*.csv"))[0])
    assert set(per_day["fold"].astype(str)) == {"seasonal_block_split"}


def test_missing_store_fails_loudly_rather_than_scoring_nan(tmp_path):
    """A store the runner needs but cannot find must be an error, not a NaN."""
    spec = SyntheticSpec(omit_stores=("ifs_ens",))
    paths = build_all(tmp_path / "stores", spec)
    out = tmp_path / "results"
    proc = run_script(
        "run_tier2_hierarchical_baseline.py", paths, out, expect_ok=False
    )
    assert proc.returncode != 0, (
        "Tier 2 exited 0 with no IFS-ENS store present; it must fail at "
        "resolution rather than score nan (that is what #98 added)"
    )
    combined = proc.stdout + proc.stderr
    assert (
        "ifs" in combined.lower()
    ), f"the error should name the missing store:\n{combined[-1500:]}"


def test_truncated_ifs_ens_store_is_detectable(tmp_path):
    """A structurally short store silently truncates the fold it feeds."""
    spec = SyntheticSpec(truncate_ifs_ens_times={2020: 3})
    paths = build_all(tmp_path / "stores", spec)
    import xarray as xr

    short = xr.open_zarr(paths["ifs_ens_2020"]).sizes["time"]
    full = xr.open_zarr(paths["ifs_ens_2018"]).sizes["time"]
    assert short < full, "fault injection did not take effect"


def test_heterogeneous_rows_do_not_break_the_shared_csv_writer(tmp_path):
    """Directly the `rmse_mm` crash, at the unit level."""
    from weavr.score_io import write_rows_csv

    rows = [
        {"lead_hours": 24, "fold": "2018", "crps_mm": 4.2, "mse_mm2": 30.0},
        {
            "lead_hours": 24,
            "fold": "pooled",
            "crps_mm": 4.3,
            "mse_mm2": 31.0,
            "rmse_mm": 5.57,
        },
    ]
    out = write_rows_csv(tmp_path / "by_bin.csv", rows)
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["lead_hours", "fold", "crps_mm", "mse_mm2", "rmse_mm"]
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def _write_valid_results(root: Path, season_days: int) -> None:
    """A complete, well-formed results set the gate should accept."""
    per_day = root / "per_day"
    per_day.mkdir(parents=True, exist_ok=True)
    methods = ("tier2_emos_graphcast", "tier2_emos_ifs_ens", "tier2_bma", "tier3_regime")
    for lead in (24, 48, 72, 96, 120):
        rows = []
        for year in (2018, 2020):
            # A 24 h forecast from 1 June verifies on IMD day 2 June, so the
            # first scored day moves with the lead and the last inits fall off.
            n = season_days - lead // 24
            first = pd.Timestamp(f"{year}-06-01") + pd.Timedelta(days=lead // 24)
            rows.append(
                pd.DataFrame(
                    {
                        "date": pd.date_range(first, periods=n).date,
                        "fold": [str(year)] * n,
                        "crps_mm": [4.2] * n,
                    }
                )
            )
        combined = pd.concat(rows, ignore_index=True)
        for method in methods:
            combined.to_csv(per_day / f"{method}__lead{lead}.csv", index=False)

    domain = pd.DataFrame(
        {
            "lead_hours": [24, 48, 72, 96, 120] * 2,
            "fold": ["2018"] * 5 + ["2020"] * 5,
            "emos_graphcast_crps_mm": [4.2] * 10,
            "bma_crps_mm": [4.3] * 10,
        }
    )
    domain.to_csv(root / "tier2_hierarchical_baseline.csv", index=False)
    tier3 = domain.rename(columns={"emos_graphcast_crps_mm": "regime_crps_mm"})
    tier3.to_csv(root / "tier3_regime_conditioned_baseline.csv", index=False)

    # Breakdown files. `by_bin` carries a combiner and a blank region for the
    # EMOS rows; `by_region` carries a region and bin but NO combiner column.
    # The gate has to cope with both shapes, and requiring the union of all
    # their columns is how one of the two used to be silently skipped.
    by_bin_rows = []
    by_region_rows = []
    for lead in (24, 48, 72, 96, 120):
        for region, crps in (
            ("WC", 32.0),
            ("CI", 27.0),
            ("NE1", 31.0),
            ("SI", 28.0),
            ("WI", 35.0),
            ("NE2", 23.0),
        ):
            row = {
                "lead_hours": lead,
                "fold": "pooled",
                "region": region,
                "bin": "heavy",
                "n_test_cells": 500,
                "crps_mm": crps,
                "rmse_mm": 55.0,
                "is_fallback": False,
            }
            by_region_rows.append(dict(row))
            by_bin_rows.append(dict(row, combiner="bma"))
            by_bin_rows.append({**row, "region": "", "combiner": "emos_graphcast"})
    pd.DataFrame(by_region_rows).to_csv(
        root / "tier2_hierarchical_baseline_by_region.csv", index=False
    )
    pd.DataFrame(by_bin_rows).to_csv(
        root / "tier2_hierarchical_baseline_by_bin.csv", index=False
    )


def test_result_gate_accepts_a_complete_two_fold_results_set(tmp_path):
    """The gate must PASS good output, not only reject bad.

    A gate that only ever fails is indistinguishable from a broken one, and the
    point of running it before a 3-hour job is to be told to proceed. So the
    accept path is pinned as hard as the reject path.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_step07_results as gate

    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    gate.RESULTS = results
    gate.PER_DAY = results / "per_day"
    gate.SEASON_DAYS = 8
    assert gate.main() == 0, f"the gate rejected a valid results set: {gate.failures}"


def _gate(results: Path, season_days: int = 8):
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_step07_results as gate

    gate.RESULTS = results
    gate.PER_DAY = results / "per_day"
    gate.SEASON_DAYS = season_days
    gate.failures = []
    gate.advisories = []
    gate.notes = []
    return gate


def test_result_gate_flags_a_degenerate_bin_region_fit(tmp_path):
    """The failure the domain-level check cannot see.

    Measured on the real run: `bma`/`heavy` reads CRPS 119.5mm for NE1 and
    101.3mm for SI against a 33.6mm median for the same bin and lead, with
    RMSE to 748mm. Those cell-days are ~0.2% of the total, so the domain CRPS
    is 4.6mm and every headline verdict is unaffected.
    """
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    by_bin = results / "tier2_hierarchical_baseline_by_bin.csv"
    frame = pd.read_csv(by_bin)
    # Lead 24 only, as on the real run: the same two regions are healthy at
    # every other lead, which is what makes this a fit failure rather than a
    # property of the bin.
    mask = (
        (frame["combiner"] == "bma")
        & (frame["bin"] == "heavy")
        & (frame["fold"] == "pooled")
        & (frame["lead_hours"] == 24)
    )
    frame.loc[mask & (frame["region"] == "NE1"), "crps_mm"] = 119.5
    frame.loc[mask & (frame["region"] == "SI"), "crps_mm"] = 101.3
    frame.to_csv(by_bin, index=False)

    gate = _gate(results)
    # Advisory, not blocking: the defect covers ~0.2% of cell-days and leaves
    # the domain CRPS intact, so it must not stop unrelated downstream work.
    assert gate.main() == 0, f"a breakdown defect blocked the gate: {gate.failures}"
    assert gate.failures == [], gate.failures
    flagged = [a for a in gate.advisories if "quantile for the same bin" in a]
    assert len(flagged) == 2, gate.advisories
    assert any("NE1" in a for a in flagged)
    assert any("SI" in a for a in flagged)


def test_result_gate_does_not_flag_a_uniformly_hard_bin(tmp_path):
    """A bin that is hard for every region is hard, not degenerate.

    This is the false-positive guard. A flat magnitude threshold would fire on
    any `heavy` row, because a bin targeting 64.5-115.6mm sits around 25-30mm
    CRPS by construction. The check compares a region against its own bin's
    median, so a uniformly high bin passes.
    """
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    by_bin = results / "tier2_hierarchical_baseline_by_bin.csv"
    frame = pd.read_csv(by_bin)
    mask = (frame["bin"] == "heavy") & (frame["fold"] == "pooled")
    frame.loc[mask, "crps_mm"] = 28.0
    frame.loc[mask, "rmse_mm"] = 55.0
    # every region identical -> no region is an outlier relative to the rest
    frame.to_csv(by_bin, index=False)

    gate = _gate(results)
    assert gate.main() == 0, f"a uniformly hard bin was flagged: {gate.advisories}"
    assert gate.advisories == [], gate.advisories


def test_result_gate_ignores_count_columns(tmp_path):
    """`*_n_test_cells` is a count, not a millimetre value.

    Regression: the first version of the magnitude check matched any column
    starting with `emos_`/`bma_`, so `emos_graphcast_n_test_cells` (1,128,500
    on the real run) tripped the 200mm bound and produced a false positive on
    the very first real output it saw.
    """
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    domain = results / "tier2_hierarchical_baseline.csv"
    frame = pd.read_csv(domain)
    frame["emos_graphcast_n_test_cells"] = 1_128_500
    frame.to_csv(domain, index=False)

    gate = _gate(results)
    assert gate.main() == 0, f"a count column tripped the magnitude bound: {gate.failures}"
    assert gate.advisories == [], gate.advisories


def test_result_gate_checks_by_region_which_has_no_combiner_column(tmp_path):
    """`by_region` has no `combiner`; the check must still run on it.

    Regression: requiring the union of `by_bin` and `by_region` columns made
    the guard reject one file and skip the other, which looks checked but is
    not.
    """
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    by_region = results / "tier2_hierarchical_baseline_by_region.csv"
    frame = pd.read_csv(by_region)
    assert "combiner" not in frame.columns, "fixture should mirror the real file"
    mask = (frame["bin"] == "heavy") & (frame["fold"] == "pooled")
    frame.loc[mask & (frame["region"] == "NE1") & (frame["lead_hours"] == 24), "crps_mm"] = 130.0
    frame.to_csv(by_region, index=False)

    gate = _gate(results)
    assert gate.main() == 0
    assert any(
        "by_region" in a and "quantile for the same bin" in a for a in gate.advisories
    ), gate.advisories


def test_result_gate_rejects_a_single_fold_per_day_file(tmp_path):
    """The gate must fail on the exact shape it was written to catch."""
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_step07_results as gate

    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    # Replace one file with a 2018-only version: the PerDayScoreWriter
    # overwrite bug, and the thing a 3-hour run is most likely to hide.
    pd.DataFrame(
        {
            "date": pd.date_range("2018-06-01", periods=8).date,
            "fold": ["2018"] * 8,
            "crps_mm": [4.2] * 8,
        }
    ).to_csv(results / "per_day" / "tier2_bma__lead24.csv", index=False)

    gate.RESULTS = results
    gate.PER_DAY = results / "per_day"
    gate.SEASON_DAYS = 8
    assert gate.main() == 1
    assert any("folds" in f for f in gate.failures), gate.failures


def test_result_gate_blocks_on_evidence_base_failures_even_with_advisories(tmp_path):
    """A blocking defect must still block, whatever else is advisory.

    The severity split must not become a way for a broken evidence base to
    pass: a per-day file missing a fold means the verdicts cannot be computed,
    and that outranks any number of breakdown advisories sitting alongside it.
    """
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    by_bin = results / "tier2_hierarchical_baseline_by_bin.csv"
    frame = pd.read_csv(by_bin)
    mask = (
        (frame["combiner"] == "bma")
        & (frame["bin"] == "heavy")
        & (frame["fold"] == "pooled")
        & (frame["lead_hours"] == 24)
    )
    frame.loc[mask & (frame["region"] == "NE1"), "crps_mm"] = 119.5
    frame.to_csv(by_bin, index=False)

    # ... and now break the evidence base as well.
    pd.DataFrame(
        {"date": pd.date_range("2018-06-01", periods=8).date, "fold": ["2018"] * 8}
    ).to_csv(results / "per_day" / "tier2_bma__lead24.csv", index=False)

    gate = _gate(results)
    assert gate.main() == 1, "a single-fold per-day file did not block"
    assert gate.advisories, "the breakdown advisory should still be reported"
    assert any("folds" in f for f in gate.failures), gate.failures


def test_result_gate_rejects_a_missing_ifs_ens_per_day_family(tmp_path):
    """A method that wrote no per-day file at all is a failure, not a pass."""
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_step07_results as gate

    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    for f in (results / "per_day").glob("tier3_*.csv"):
        f.unlink()

    gate.RESULTS = results
    gate.PER_DAY = results / "per_day"
    gate.SEASON_DAYS = 8
    assert gate.main() == 1
    assert any("tier3" in f for f in gate.failures), gate.failures


def test_expected_row_count_formula_matches_two_season_daily_cadence():
    """`122 - lead_days` per season under IMD's end-date day labelling (#106).

    The old one-day-early pairing gave 244/242/240/238/236 on the real
    122-day stores. Pinning the sequence means a change to the pairing
    assumption has to be deliberate -- as this one was.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import gate_step07_results as gate

    original = gate.SEASON_DAYS
    try:
        gate.SEASON_DAYS = 122
        assert [gate.expected_rows(lead) for lead in (24, 48, 72, 96, 120)] == [
            242,
            240,
            238,
            236,
            234,
        ]
    finally:
        gate.SEASON_DAYS = original


def test_seasons_constant_is_two():
    assert SEASONS == (2018, 2020)
