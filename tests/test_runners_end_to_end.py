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
import xarray as xr

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
    "run_tier2b_combined.py": ("baseline", "lagged", "ifs_ens", "climatology"),
    "run_tail_repair_qm.py": ("baseline", "climatology"),
    "run_tail_repair_qm_emos.py": ("baseline", "lagged", "ifs_ens", "climatology"),
    "run_tail_repair_evt.py": ("baseline", "lagged", "ifs_ens", "climatology"),
    "run_quantile_avg_reliability.py": ("baseline", "lagged", "ifs_ens", "climatology"),
}


#: Extra flags a runner needs on the synthetic stores. The replica's rain is too
#: light for 10 exceedances of 64.5 mm per region, so the EVT runner fits its
#: tail above a lower u (tw-CRPS is still scored at 64.5 and 115.6 mm, >= u).
RUNNER_EXTRA_ARGS: dict[str, list[str]] = {
    "run_tail_repair_evt.py": ["--threshold-u", "10"],
    # No Tier 2b train CSV exists for the synthetic stores, so name the config.
    "run_quantile_avg_reliability.py": [
        "--threshold-u", "10", "--emos-source", "graphcast", "--weight", "0.5",
        "--n-resamples", "50",
    ],
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
    cmd += ["--results-dir", str(results_dir), "--force"]
    cmd += [*RUNNER_EXTRA_ARGS.get(script, []), *(extra or [])]
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
    ("run_tier2b_combined.py", "tier2b_combined.csv", True),
    # Step 11: both QM runners, the member-level one with both sources.
    ("run_tail_repair_qm.py", "tail_repair_qm.csv", True),
    ("run_tail_repair_qm_emos.py", "tail_repair_qm_emos.csv", True),
    # Step 12: every arm, including step 11's QM variant and the IFS benchmark.
    ("run_tail_repair_evt.py", "tail_repair_evt.csv", True),
]

#: Step 13's arms. The two parents plus the four combinations; a run that
#: silently dropped one would still satisfy every check above, because a
#: shorter arm list is not an error anywhere else in the suite.
TIER2B_ARMS = (
    "emos_csg",
    "bma",
    "per_bin",
    "quantile_avg",
    "quantile_avg_fixed",
    "linear_pool",
)


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


def test_result_gate_blocks_a_fitted_bin_that_scores_zero(tmp_path):
    """An impossibly good score is as broken as an impossibly bad one.

    Measured on the #106 regeneration: EMOS-GraphCast's dry bin at 120 h, 2020
    fold, read CRPS 0.000 over 312,251 cells with a +399.8 mm bias -- a CSGD
    whose shape fell under the clip floor, so the closed form lost every digit
    and clipped to 0. The outlier check only looks upward and passed it.
    """
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    by_bin = results / "tier2_hierarchical_baseline_by_bin.csv"
    frame = pd.read_csv(by_bin)
    mask = (frame["combiner"] == "emos_graphcast") & (frame["lead_hours"] == 120)
    frame.loc[mask, "crps_mm"] = 0.0
    frame.to_csv(by_bin, index=False)

    gate = _gate(results)
    assert gate.main() == 1, "a zero-CRPS fitted bin did not block"
    assert any("cannot score zero" in f for f in gate.failures), gate.failures


def test_result_gate_allows_a_zero_crps_fallback_bin(tmp_path):
    """A point-mass fallback scoring 0 on dry cells is correct, not broken."""
    results = tmp_path / "results"
    _write_valid_results(results, season_days=8)
    by_bin = results / "tier2_hierarchical_baseline_by_bin.csv"
    frame = pd.read_csv(by_bin)
    mask = (frame["combiner"] == "emos_graphcast") & (frame["lead_hours"] == 120)
    frame.loc[mask, "crps_mm"] = 0.0
    frame.loc[mask, "is_fallback"] = True
    frame.to_csv(by_bin, index=False)

    gate = _gate(results)
    assert gate.main() == 0, gate.failures


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
    """`122 - lead_days` per season: IMD labels a day by the date its window ends.

    A forecast from init day `t` at lead `L` days is scored against IMD day
    `t + L`, and the observations end on the season's last init day, so the
    last `L` init days have nothing to score against (issue #106). Before the
    fix the labels sat one day earlier and the sequence was 244/242/240/238/236;
    pinning the new one here means a change to the cadence has to be deliberate.
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


def test_tier2b_scores_every_arm_and_writes_all_three_csvs(stores, tmp_path):
    """Step 13's four methods are the deliverable, so all six arms must appear.

    The suite's generic checks only assert that *some* rows were written. A
    fold loop that dropped the linear pool, or emitted only the parents, would
    pass both of those and still leave H10 unanswerable, because H10 asks
    whether a combination beats both parents.
    """
    out = tmp_path / "t2b_arms"
    run_script(
        "run_tier2b_combined.py", stores, out,
        extra=["--train-stride", "1", "--n-resamples", "50"],
    )
    frame = pd.read_csv(out / "tier2b_combined.csv")
    assert set(frame["arm"]) == set(TIER2B_ARMS), (
        f"tier2b_combined.csv arms={sorted(set(frame['arm']))}; "
        f"expected all of {sorted(TIER2B_ARMS)}"
    )
    assert (out / "tier2b_combined_train.csv").exists()
    assert (out / "tier2b_combined_paired.csv").exists()
    for column in ("pit_mean", "pit_mad_from_half", "pit_central_90_coverage"):
        assert frame[column].between(0.0, 1.0).all(), f"{column} missing or out of [0, 1]"

    per_day = sorted((out / "per_day").glob("tier2b_*.csv"))
    assert len(per_day) == len(TIER2B_ARMS) * 5, (
        f"wrote {len(per_day)} per-day files; expected "
        f"{len(TIER2B_ARMS)} arms x 5 leads"
    )
    for f in per_day:
        day = pd.read_csv(f)
        assert {"pit_mad_from_half", "pit_central_90_coverage"} <= set(day.columns), f.name
        # `.astype(str)` for the same reason as the generic test above: a
        # column of only these two values is read back as int64.
        folds = set(day["fold"].astype(str))
        assert folds == {"2018", "2020"}, f"{f.name} folds={sorted(folds)}"
        assert np.isfinite(day["crps_mm"].to_numpy()).any(), f"{f.name} has no finite CRPS"


def test_tier2b_h10_is_decided_not_skipped(stores, tmp_path):
    """H10 must reach a verdict, and the verdict must follow the pre-registered rule."""
    out = tmp_path / "t2b_h10"
    proc = run_script(
        "run_tier2b_combined.py", stores, out,
        extra=["--train-stride", "1", "--n-resamples", "50"],
    )
    assert "H10 PASS" in proc.stdout or "H10 FAIL" in proc.stdout, (
        "tier2b printed no H10 verdict. The plan's third deliverable is the "
        "verdict, so a run that scores arms without deciding one is unfinished."
    )

    paired = pd.read_csv(out / "tier2b_combined_paired.csv")
    assert {
        "arm", "vs_parent", "crps_difference_mm", "ci_low", "ci_high", "beats_parent"
    } <= set(paired.columns)
    for lead, group in paired.groupby("lead_hours"):
        assert set(group["vs_parent"]) == {"emos_csg", "bma"}, (
            f"lead {lead} compared against {sorted(set(group['vs_parent']))}; "
            "H10 requires beating each parent separately."
        )


def test_tier2b_default_weight_is_the_fixed_one(stores, tmp_path):
    """The fitted weight is opt-in, so a default run is the pre-registered 0.5.

    The fit lost a lead to 0.5 on the measured evidence (3/5 vs 4/5), so making
    it the default would put the weaker method in every future run.
    """
    out = tmp_path / "t2b_default_weight"
    run_script(
        "run_tier2b_combined.py", stores, out,
        extra=["--train-stride", "1", "--n-resamples", "50"],
    )
    train = pd.read_csv(out / "tier2b_combined_train.csv")
    assert np.allclose(train["vincentization_weight"], 0.5), (
        f"default weights were {sorted(set(train['vincentization_weight']))}; "
        "expected the pre-registered fixed 0.5 unless --fit-weight is passed"
    )
    # The fit is still recorded, so "we tried it" stays reproducible.
    assert "vincentization_train_crps" in train.columns
    assert "vincentization_weight_fitted" in train.columns


def test_tier2b_line_pool_is_comparable_to_quantile_averaging(stores, tmp_path):
    """The linear pool must land in the same CRPS neighbourhood as Vincentization.

    These two differ only in the space they average in, and the plan keeps the
    linear pool as a comparison arm. If it came out orders of magnitude worse,
    the cause would be the bracketed interpolation in `linear_pool_quantiles`
    rather than the pooling rule, and the arm would be quietly misleading.
    """
    out = tmp_path / "t2b_pool"
    run_script(
        "run_tier2b_combined.py", stores, out,
        extra=["--train-stride", "1", "--n-resamples", "50"],
    )
    frame = pd.read_csv(out / "tier2b_combined.csv")
    pooled = frame.groupby("arm")["crps_mm"].mean()
    for arm in ("linear_pool", "quantile_avg", "quantile_avg_fixed", "emos_csg", "bma"):
        assert np.isfinite(pooled[arm]), f"{arm} CRPS is not finite"
    assert pooled["linear_pool"] < 10 * max(pooled["emos_csg"], 1e-9), (
        f"linear_pool CRPS {pooled['linear_pool']:.4f} is far outside the "
        f"parent range (emos_csg {pooled['emos_csg']:.4f}, bma {pooled['bma']:.4f}); "
        "that is an inversion failure, not a pooling result."
    )


def test_tier2b_missing_store_fails_loudly(stores, tmp_path):
    """A missing IFS store must abort before any CSV claims to be scored."""
    partial = dict(stores)
    partial.pop("ifs_ens_2020")
    proc = run_script(
        "run_tier2b_combined.py", partial, tmp_path / "t2b_missing",
        extra=["--train-stride", "1"],
        expect_ok=False,
    )
    assert proc.returncode != 0
    assert not (tmp_path / "t2b_missing" / "tier2b_combined.csv").exists(), (
        "tier2b wrote its domain CSV despite a missing store; the partial run "
        "must not be readable as a finished one."
    )


def test_dashboard_probability_grid_fits_on_the_other_season(stores):
    """#74: the map is re-emitted under leave-one-year-out, not fitted in sample.

    With `fit_stores`, EMOS-CSG and the GPD tail come from 2018 while the
    displayed day comes from 2020, and the NPZ says so in `fitted_on`. A fit
    that quietly fell back to the display season would still draw a plausible
    map, so the provenance is what is pinned.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import export_dashboard_example_grids as export

    display = tuple(str(stores[f"{kind}_2020"]) for kind in ("baseline", "lagged", "ifs_ens"))
    fit = tuple(str(stores[f"{kind}_2018"]) for kind in ("baseline", "lagged", "ifs_ens"))

    loyo = export.build_example_probability_grids(*display, 24, fit_stores=fit)
    in_sample = export.build_example_probability_grids(*display, 24)

    assert loyo["fitted_on"] == fit[0]
    assert in_sample["fitted_on"] == display[0]
    assert loyo["sample_time"] == in_sample["sample_time"]
    assert loyo["sample_time"].startswith("2020")
    for slug in ("115p6", "204p5"):
        probability = np.asarray(loyo[f"probability_{slug}"])
        finite = probability[np.isfinite(probability)]
        assert finite.size and ((finite >= 0) & (finite <= 1)).all()
        assert set(np.unique(loyo[f"method_{slug}"])) <= {"csgd", "csgd+gpd_tail", "fallback"}
    assert not np.array_equal(
        np.asarray(loyo["probability_115p6"]), np.asarray(in_sample["probability_115p6"]),
        equal_nan=True,
    ), "fitting on 2018 produced the same map as fitting on 2020"


def test_dashboard_probability_grid_classifies_bins_like_the_fit(stores, monkeypatch):
    """The display day's bins come from the GraphCast mean, as the fit's do.

    The tail function picks each cell's bin from `forecast_values`. Passing
    the IFS mean there scored cells with a bin model fitted on cells chosen
    by a different forecast; the IFS mean and spread stay the predictors.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import export_dashboard_example_grids as export

    seen = []
    real = export.exceedance_probability_with_tail

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(export, "exceedance_probability_with_tail", spy)
    display = tuple(str(stores[f"{kind}_2020"]) for kind in ("baseline", "lagged", "ifs_ens"))
    fit = tuple(str(stores[f"{kind}_2018"]) for kind in ("baseline", "lagged", "ifs_ens"))
    grid = export.build_example_probability_grids(*display, 24, fit_stores=fit)

    forecasts, _ = export.align_all_sources(
        export.load_graphcast_ensemble(display[1], 24),
        export.load_ifs_ensemble(display[2], 24),
        export.load_hres_forecast(display[0], 24),
        xr.open_zarr(display[0], group="imd_observed", consolidated=True).load(),
    )
    index = export.latest_complete_sample_index(forecasts)
    graphcast_mean = forecasts["graphcast"].mean(dim="member", skipna=True).isel(sample=index)
    ifs_mean = forecasts["ifs_ens"].mean(dim="member", skipna=True).isel(sample=index)

    assert seen, "the export never called the tail function"
    for call in seen:
        np.testing.assert_array_equal(np.asarray(call["forecast_values"]), graphcast_mean.values)
        np.testing.assert_array_equal(np.asarray(call["ensemble_mean"]), ifs_mean.values)
    assert grid["sample_time"] == str(graphcast_mean["sample"].values)


def _tier2b_train_csv(path, weight, source="ifs_ens"):
    path.write_text(
        "lead_hours,fold,emos_source,vincentization_weight\n"
        f"24,2020,{source},{weight}\n24,2018,{source},{weight}\n"
    )
    return path


def test_dashboard_quantile_avg_at_weight_zero_is_the_emos_map(stores, tmp_path):
    """#75: a quantile average with weight 0 *is* EMOS-CSG, so the new path must
    reproduce the existing map wherever both use a fitted bin -- an end-to-end
    check of the quantiles, P(Y > u) read off them, and the GPD splice."""
    sys.path.insert(0, str(REPO / "scripts"))
    import export_dashboard_example_grids as export

    display = tuple(str(stores[f"{kind}_2020"]) for kind in ("baseline", "lagged", "ifs_ens"))
    fit = tuple(str(stores[f"{kind}_2018"]) for kind in ("baseline", "lagged", "ifs_ens"))
    csv_path = _tier2b_train_csv(tmp_path / "train.csv", 0.0)
    emos = export.build_example_probability_grids(*display, 24, fit_stores=fit)
    qavg = export.build_example_probability_grids(
        *display, 24, fit_stores=fit, combiner="quantile_avg", tier2b_train_csv=csv_path
    )
    assert qavg["combiner"] == "quantile_avg" and qavg["quantile_avg_weight"] == 0.0
    for slug in ("115p6", "204p5"):
        a = np.asarray(emos[f"probability_{slug}"])
        b = np.asarray(qavg[f"probability_{slug}"])
        both = (np.asarray(emos[f"method_{slug}"]) == "csgd+gpd_tail") & ~np.asarray(
            qavg[f"is_fallback_{slug}"]
        ) & (a > 1e-12)
        assert both.sum() > 20, "too few comparable cells to say anything"
        rel = np.abs(b[both] - a[both]) / a[both]
        assert np.median(rel) < 0.02, f"{slug}: median relative gap {np.median(rel):.3f}"


def test_dashboard_quantile_avg_is_bounded_ordered_and_labelled(stores, tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import export_dashboard_example_grids as export

    display = tuple(str(stores[f"{kind}_2020"]) for kind in ("baseline", "lagged", "ifs_ens"))
    fit = tuple(str(stores[f"{kind}_2018"]) for kind in ("baseline", "lagged", "ifs_ens"))
    grid = export.build_example_probability_grids(
        *display, 24, fit_stores=fit, combiner="quantile_avg",
        tier2b_train_csv=_tier2b_train_csv(tmp_path / "train.csv", 0.7, source="graphcast"),
    )
    assert grid["emos_source"] == "graphcast" and grid["quantile_avg_weight"] == 0.7
    p115 = np.asarray(grid["probability_115p6"])
    p204 = np.asarray(grid["probability_204p5"])
    for p in (p115, p204):
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert (p204 <= p115 + 1e-12).all(), "P(>204.5) must never exceed P(>115.6)"
    for slug in ("115p6", "204p5"):
        assert set(np.unique(grid[f"method_{slug}"])) <= {"quantile_avg+gpd_tail", "fallback"}
    assert (np.asarray(grid["method_115p6"]) == "quantile_avg+gpd_tail").any()


def test_dashboard_quantile_avg_needs_the_fit_season_and_its_tier2b_row(stores, tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import export_dashboard_example_grids as export

    display = tuple(str(stores[f"{kind}_2020"]) for kind in ("baseline", "lagged", "ifs_ens"))
    fit = tuple(str(stores[f"{kind}_2018"]) for kind in ("baseline", "lagged", "ifs_ens"))
    csv_path = _tier2b_train_csv(tmp_path / "train.csv", 0.5)
    with pytest.raises(ValueError, match="--fit-"):
        export.build_example_probability_grids(
            *display, 24, combiner="quantile_avg", tier2b_train_csv=csv_path
        )
    with pytest.raises(ValueError, match="no Tier 2b train row"):
        export.build_example_probability_grids(
            *display, 48, fit_stores=fit, combiner="quantile_avg", tier2b_train_csv=csv_path
        )


def test_dashboard_export_rejects_a_partial_fit_season(stores, tmp_path):
    result = subprocess.run(
        [str(PYTHON), str(REPO / "scripts" / "export_dashboard_example_grids.py"),
         "--fit-store", str(stores["baseline_2018"]),
         "--out-blend", str(tmp_path / "b.npz"), "--out-probability", str(tmp_path / "p.npz")],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 2
    assert "go together" in result.stderr


def test_qm_emos_scores_both_sources_and_every_arm(stores, tmp_path):
    """Step 11 item 4c: EMOS-CSG on QM-corrected IFS members as well as GraphCast.

    A run that silently dropped a source or an arm would still write a
    non-empty CSV, which is all the aggregate check above can see.
    """
    out = tmp_path / "qm_emos"
    run_script("run_tail_repair_qm_emos.py", stores, out)
    scores = pd.read_csv(out / "tail_repair_qm_emos.csv")
    assert set(scores["emos_source"]) == {"graphcast", "ifs_ens"}
    for source, group in scores.groupby("emos_source"):
        assert set(group["arm"]) == {
            "raw_members", "qm_members_same_period", "qm_members_climatology",
        }, source
    paired = pd.read_csv(out / "tail_repair_qm_emos_paired.csv")
    assert set(paired["emos_source"]) == {"graphcast", "ifs_ens"}
    assert "h7_lead_improved" in paired.columns
    names = {p.name.split("__lead")[0] for p in (out / "per_day").glob("*.csv")}
    assert {"qmemos_raw_members", "qmemos_ifs_ens_raw_members"} <= names


def test_evt_scores_every_arm_and_comparison(stores, tmp_path):
    """Step 12 item 3: the tail against the fallback, step 11's QM variant and
    raw IFS member counting, with tw-CRPS, the Brier@7.5 guard and H7 per lead."""
    out = tmp_path / "evt"
    run_script("run_tail_repair_evt.py", stores, out)
    scores = pd.read_csv(out / "tail_repair_evt.csv")
    assert set(scores["arm"]) == {
        "csgd_only", "csgd_gpd_tail", "qm_csgd_only", "qm_csgd_gpd_tail",
        "raw_ifs_ens_member_counting",
    }
    for column in ("twcrps_64.5_mm", "twcrps_115.6_mm", "brier_7.5"):
        assert scores[column].notna().all(), column
    paired = pd.read_csv(out / "tail_repair_evt_paired.csv")
    assert set(paired["comparison"]) == {
        "csgd_gpd_tail_vs_csgd_only", "csgd_gpd_tail_vs_raw_ifs_ens_member_counting",
        "csgd_gpd_tail_vs_qm_csgd_only", "qm_csgd_gpd_tail_vs_qm_csgd_only",
    }
    assert {"h7_lead_improved", "brier_7.5_degraded", "sedi_115.6_improved"} <= set(paired.columns)
    reliability = pd.read_csv(out / "tail_repair_evt_reliability.csv")
    assert {"observed_frequency_ci_lo", "observed_frequency_ci_hi"} <= set(reliability.columns)


def test_quantile_avg_reliability_writes_scores_and_a_reliability_table(stores, tmp_path):
    """#75 follow-up 3: every lead x fold x extreme threshold is scored, and the
    reliability table has the bins, counts and CI columns step 12's does."""
    out = tmp_path / "qavg_rel"
    run_script("run_quantile_avg_reliability.py", stores, out, extra=["--leads", "24"])
    scores = pd.read_csv(out / "quantile_avg_extreme.csv")
    assert set(scores["threshold"]) == {115.6, 204.5}
    assert set(scores["fold"].astype(str)) == {"2018", "2020"}
    assert (scores["arm"] == "quantile_avg_gpd_tail").all()
    assert scores["brier"].between(0.0, 1.0).all()
    assert scores["climatology_brier"].between(0.0, 1.0).all()
    table = pd.read_csv(out / "quantile_avg_reliability.csv")
    for column in ("n", "mean_forecast", "observed_frequency",
                   "observed_frequency_ci_lo", "observed_frequency_ci_hi"):
        assert column in table.columns, column
    counts = table.groupby("threshold")["n"].sum()
    assert (counts > 0).all()
    # Every observed forecast cell lands in exactly one bin.
    by_threshold = scores.groupby("threshold")["n_observed_cells"].sum()
    assert counts.to_dict() == by_threshold.to_dict()
