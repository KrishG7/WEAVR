#!/usr/bin/env python3
"""Gate: refuse to start follow-on work unless Step 07's outputs are actually sound.

Step 07's own history is a list of ways a run can look successful and not be:
a fold silently overwritten (244 -> 122 rows), a season scoring `nan` because
its store was missing, a BMA CRPS of 1540 where 4.9 belongs, a CSV whose
header does not match its rows. Each of those passed a "did it write a file"
check. This checks the numbers instead.

Exits 0 only if every check passes. Any failure prints the specific reason.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

#: The committed outputs by default. It used to default to the untracked
#: `.step07_v2` staging folder, so the documented command silently validated
#: a stale scratch run rather than what `results/` actually holds.
RESULTS = Path(
    os.environ.get("WEAVR_STEP07_RESULTS_DIR", "results")
).resolve()
PER_DAY = RESULTS / "per_day"
LEADS = [24, 48, 72, 96, 120]

#: Init times per season on the daily evidence base. Overridable so the gate
#: can be exercised against the miniature stores in
#: `tests/test_runners_end_to_end.py`, which use 8. The real value is asserted
#: directly by a test so it cannot drift silently.
SEASON_DAYS = int(os.environ.get("WEAVR_GATE_SEASON_DAYS", "122"))

#: Any CRPS/RMSE above this in a *domain* aggregate is a numerical failure,
#: not weather. The historical lead-72 BMA outlier was ~1540 against a true
#: value near 4.9. Domain CRPS on this evidence base runs 4.5-5.1mm, so 200mm
#: is ~40x headroom -- loose enough to never fire on a hard bin, tight enough
#: to catch the known failure.
ABSURD = 200.0

#: A (bin, region) row is an outlier when its CRPS exceeds this multiple of a
#: low quantile of the same (lead, combiner, bin) group across regions.
#: Measured on the real run the degenerate rows sit at 3.6x and 4.2x of the
#: 25th percentile, while the highest legitimate row (bma/heavy/WI at lead
#: 120) is 1.2x. 2.5x sits in that gap.
BIN_OUTLIER_RATIO = 2.5

#: Which quantile of the group is the reference. Deliberately NOT the median:
#: a fit that blows up *inflates* the group, so with few regions the outliers
#: drag the median toward themselves and hide. Measured: with 4 regions and 2
#: outliers the median lands at 65mm, the ratios fall to 1.8x and 1.6x, and
#: both degenerate rows slip under the threshold. The 25th percentile is
#: barely moved by a minority of inflating rows (28.4mm on the real data,
#: against a 33.6mm median).
BIN_REFERENCE_QUANTILE = 0.25

#: Minimum test cells before a per-bin CRPS is trusted enough to judge. Below
#: this the estimate is dominated by sampling noise, and those rows are already
#: flagged `is_fallback`.
MIN_BIN_CELLS = 50

#: Regions needed in a group before the comparison means anything. Five, not
#: three: with only three or four regions a pair of outliers is a large enough
#: fraction to move any central reference, which is exactly the case this check
#: exists to catch. Real BMA groups have six.
MIN_REGIONS_FOR_MEDIAN = 5

#: Largest daily 24h accumulation on the IMD grid, measured from
#: `imd_observed` (370.9mm in 2018, 259.9mm in 2020 at the sampled times). A
#: score cannot exceed the range of what it predicts.
MAX_OBSERVED_DAILY_MM = 400.0
PHYSICAL_MAX_MM = 3.0 * MAX_OBSERVED_DAILY_MM

#: Blocking: the evidence base itself cannot be trusted. The domain aggregate
#: and the per-day files are what every pre-registered verdict is computed
#: from, so a wrong fold count, a NaN, a missing method or an impossible
#: domain magnitude means the numbers must not be used. The gate exits 1.
failures: list[str] = []

#: Advisory: something is wrong that does NOT invalidate the domain numbers.
#:
#: Reported just as loudly, but the gate still exits 0. This exists because the
#: by-bin breakdown can carry a degenerate per-(bin, region) fit -- measured
#: at ~0.2% of scored cell-days on the real run -- while the cell-weighted
#: domain CRPS stays at 4.6mm and no verdict moves. Blocking on that would
#: stall unrelated downstream work (Step 12's EVT) behind a diagnostic detail
#: of the tier being reported, which is a worse failure than the one being
#: reported.
advisories: list[str] = []

notes: list[str] = []


def expected_rows(lead_hours: int) -> int:
    """Per-day rows across both folds at this lead.

    A 00 UTC init plus `lead_days` verifies on the IMD day `lead_days` later
    (IMD labels a day by its end, #106), so the last `lead_days` inits of each
    season verify past 30 September and have no observation: 242/240/238/236/234.

    Under the old one-day-early pairing this was `SEASON_DAYS - (lead_days - 1)`
    (244/242/240/238/236) -- the same count shape, one day longer, which is
    why the row count could never have caught that error.
    """
    lead_days = lead_hours // 24
    return 2 * (SEASON_DAYS - lead_days)


def check_domain_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        failures.append(f"missing domain CSV: {name}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"unreadable domain CSV {name}: {type(exc).__name__}: {exc}")
        return None

    if "fold" not in df.columns:
        failures.append(f"{name}: no 'fold' column")
        return None
    folds = set(df["fold"].astype(str))
    if not {"2018", "2020"} <= folds:
        failures.append(
            f"{name}: not two-season -- folds present {sorted(folds)}; "
            "a single fold here means the other season silently failed"
        )
    if "lead_hours" in df.columns:
        leads = set(df["lead_hours"].astype(int))
        missing = sorted(set(LEADS) - leads)
        if missing:
            failures.append(f"{name}: missing leads {missing}")
    return df


def check_finite(df: pd.DataFrame | None, name: str, prefixes: tuple[str, ...]) -> None:
    if df is None:
        return
    checked = 0
    for col in df.columns:
        if not any(col.startswith(p) for p in prefixes):
            continue
        # Only millimetre-valued columns get a magnitude bound. A column such
        # as `emos_graphcast_n_test_cells` is a *count* and legitimately runs
        # to six figures; bounding it at 200mm produced a false positive on
        # the first real run of this check.
        if not (col.endswith("_mm") or col.endswith("_crps")):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        checked += 1
        if vals.isna().any():
            n = int(vals.isna().sum())
            failures.append(f"{name}: column '{col}' has {n} NaN of {len(vals)} rows")
        finite = vals.dropna().abs()
        if not finite.empty and float(finite.max()) > ABSURD:
            worst = float(finite.max())
            failures.append(
                f"{name}: column '{col}' max |value| = {worst:.1f} > {ABSURD}; "
                "looks like the known BMA/EMOS blow-up, not weather"
            )
    if checked == 0:
        notes.append(f"{name}: no {prefixes} columns found to check")


def check_per_day(pattern: str) -> None:
    files = sorted(PER_DAY.glob(pattern))
    if not files:
        failures.append(f"no per-day files matching {pattern}")
        return
    for lead in LEADS:
        suffix = f"__lead{lead}.csv"
        match = [f for f in files if f.name.endswith(suffix)]
        if not match:
            failures.append(f"{pattern}: no per-day file for lead {lead}")
            continue
        for f in match:
            try:
                df = pd.read_csv(f)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"unreadable per-day {f.name}: {type(exc).__name__}")
                continue
            folds = set(df["fold"].astype(str)) if "fold" in df.columns else set()
            if not {"2018", "2020"} <= folds:
                failures.append(
                    f"{f.name}: folds {sorted(folds)} -- PerDayScoreWriter "
                    "overwrite bug, or one fold failed"
                )
            exp = expected_rows(lead)
            got = len(df)
            if got > exp:
                failures.append(f"{f.name}: {got} rows > expected {exp}")
            elif exp - got > 8:
                failures.append(
                    f"{f.name}: {got} rows, expected ~{exp} "
                    f"({exp - got} short -- a season may be missing)"
                )


def check_by_bin_outliers(df: pd.DataFrame, name: str) -> None:
    """Flag a (bin, region) row whose CRPS dwarfs its own bin's median.

    A flat magnitude bound cannot do this job, because CRPS legitimately
    scales with the bin: the `heavy` bin targets 64.5-115.6mm and sits around
    25-30mm CRPS, while `light` sits near 8mm. A threshold loose enough to
    catch a genuine blow-up would flag every heavy row.

    So the comparison is made *within* a (lead, combiner, bin) group, across
    regions. Every region faces the same target distribution and the same lead,
    so a row several times its own group's median is a degenerate fit rather
    than a hard bin.

    This exists because the domain-level gate passes while the by-bin table
    does not. Measured on the real two-season run: `bma`/`heavy` at lead 24
    reads CRPS 119.5mm for NE1 and 101.3mm for SI against a 33.6mm median for
    the same bin, with RMSE up to 748mm against an observed daily maximum near
    400mm. Those cell-days are ~0.2% of the scored total, so the cell-weighted
    domain CRPS is 4.6mm and every headline verdict is unaffected -- the failure
    is visible *only* here.

    Rows below `min_n` are skipped: a bin/region with a handful of test cells
    has a legitimately unstable CRPS, and those rows already carry
    `is_fallback=True`.
    """
    if df is None or df.empty:
        return
    # The two breakdown files differ in shape and neither column is common to
    # both: `by_bin` has `combiner` but its `region` is blank for the EMOS
    # rows, `by_region` has `region` and `bin` but no `combiner` at all.
    # Requiring the union of all three silently skipped one file or the other,
    # which is worse than useless -- it looks checked.
    required = {"fold", "region", "crps_mm", "n_test_cells"}
    if not required <= set(df.columns):
        missing = sorted(required - set(df.columns))
        failures.append(
            f"{name}: cannot check for degenerate per-region fits -- missing "
            f"column(s) {missing}. A breakdown with no region column cannot be "
            "checked at all, so this is a failure rather than a skip."
        )
        return
    if "combiner" not in df.columns:
        df = df.assign(combiner="(all)")

    pooled = df[df["fold"].astype(str) == "pooled"]
    if pooled.empty:
        pooled = df
    usable = pooled[pd.to_numeric(pooled["n_test_cells"], errors="coerce") >= MIN_BIN_CELLS]

    # `by_bin` groups by (lead, combiner, bin) and compares regions. `by_region`
    # has no bin column -- one row per (lead, fold, combiner, region) pooled
    # over bins -- so it groups by (lead, combiner) and compares regions
    # against their own median. Same idea, one dimension fewer.
    has_bin = "bin" in usable.columns and usable["bin"].astype(str).str.strip().ne("").any()
    keys = ["lead_hours", "combiner", "bin"] if has_bin else ["lead_hours", "combiner"]

    for group_key, group in usable.groupby(keys, dropna=False):
        lead, combiner = group_key[0], group_key[1]
        rain_bin = group_key[2] if has_bin else "(pooled over bins)"
        regions = group[group["region"].notna()]
        if len(regions) < MIN_REGIONS_FOR_MEDIAN:
            continue
        values = pd.to_numeric(regions["crps_mm"], errors="coerce").dropna()
        reference = float(values.quantile(BIN_REFERENCE_QUANTILE))
        if not (reference > 0):
            continue
        for row in regions.itertuples():
            value = float(row.crps_mm)
            if value > BIN_OUTLIER_RATIO * reference:
                advisories.append(
                    f"{name}: {combiner}/{rain_bin} region {row.region} at lead "
                    f"{lead} reads CRPS {value:.1f}mm against a {reference:.1f}mm "
                    f"{BIN_REFERENCE_QUANTILE:.0%} quantile for the same bin "
                    f"({value / reference:.1f}x, n={int(row.n_test_cells)}). "
                    "Regions share a target distribution at a fixed lead, so this "
                    "is a degenerate fit, not a hard bin. It is invisible in the "
                    "domain CRPS because it covers too few cell-days to move a "
                    "cell-weighted mean."
                )


def check_physical_bounds(df: pd.DataFrame, name: str) -> None:
    """No score can exceed the range of the quantity being predicted.

    An RMSE above the largest observed daily accumulation is arithmetically
    impossible, whatever the bin. This is the backstop for a blow-up large
    enough that the median-ratio rule above would also be dragged off.
    """
    if df is None or df.empty:
        return
    for col in df.columns:
        if not col.endswith("rmse_mm") and not col.endswith("crps_mm"):
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        worst = float(values.abs().max())
        if worst > PHYSICAL_MAX_MM:
            advisories.append(
                f"{name}: column '{col}' reaches {worst:.0f}mm, above the "
                f"{PHYSICAL_MAX_MM:.0f}mm physical ceiling. No daily accumulation "
                f"on this grid exceeds ~{MAX_OBSERVED_DAILY_MM:.0f}mm, so this "
                "cannot be a forecast error."
            )


def main() -> int:
    if not RESULTS.exists():
        print(f"GATE FAIL: {RESULTS} does not exist")
        return 1

    print("=== Step 07 output gate ===")

    # 1. The slow tiers must exist and be two-season.
    t2 = check_domain_csv("tier2_hierarchical_baseline.csv")
    t3 = check_domain_csv("tier3_regime_conditioned_baseline.csv")
    check_finite(t2, "tier2", ("emos_", "bma_"))
    check_finite(t3, "tier3", ("regime_",))

    # 2a. The by-bin and by-region breakdowns, which is where a degenerate
    #     per-(bin, region) fit shows up while the domain aggregate stays
    #     clean. A domain-level pass is NOT sufficient.
    for csv_name in (
        "tier2_hierarchical_baseline_by_bin.csv",
        "tier2_hierarchical_baseline_by_region.csv",
    ):
        path = RESULTS / csv_name
        if not path.exists():
            failures.append(f"missing breakdown CSV: {csv_name}")
            continue
        try:
            breakdown = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"unreadable {csv_name}: {type(exc).__name__}: {exc}")
            continue
        check_by_bin_outliers(breakdown, csv_name)
        check_physical_bounds(breakdown, csv_name)

    # 2. Per-day files must carry BOTH folds at every lead. This is the check
    #    that would have caught the original bug.
    check_per_day("tier2_emos_graphcast__lead*.csv")
    check_per_day("tier2_bma__lead*.csv")
    check_per_day("tier3_*.csv")

    # 3. Sanity: a completed run leaves no stray staging or partial files.
    for stray in ("tier2_hierarchical_baseline.csv.tmp",):
        if (RESULTS / stray).exists():
            failures.append(f"stray partial file present: {stray}")

    for n in notes:
        print(f"  note:  {n}")

    if advisories:
        print(f"  ADVISORY ({len(advisories)}) -- real defects, but not verdict-affecting:")
        for a in advisories:
            print(f"    ! {a}")
        print(
            "  These must be written up and investigated, but they live in the\n"
            "  breakdown tables and cover too few cell-days to move a\n"
            "  cell-weighted domain mean, so the pre-registered verdicts stand."
        )

    if failures:
        print(f"  GATE FAIL ({len(failures)} blocking problem(s)) -- do NOT trust the numbers:")
        for f in failures:
            print(f"    - {f}")
        return 1

    print(
        "  GATE PASS: tier2/tier3 are two-season, finite, plausible, and per-day complete."
    )
    if advisories:
        print("  Safe to start follow-on work, with the advisories above carried forward.")
    else:
        print("  Safe to start follow-on work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
