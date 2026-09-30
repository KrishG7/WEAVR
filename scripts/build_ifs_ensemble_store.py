#!/usr/bin/env python3
"""Build the Phase 4 real IFS 50-member ensemble store.

Phase 0/1 deliberately deferred this exact pull: `docs/baseline-store.md`'s
`ifs_ens_mean` row already flagged "full-ensemble access is a Phase 1+
decision if BMA/EMOS needs individual members," and
`docs/phase-1-data-requirements.md` measured the real cost of getting it
(~47s/chunk, ~2-2.5h/~30-35GB at Phase 0's sampling density) before
deferring it to whichever later phase actually needed individual members.
Phase 4 is that phase -- see `docs/phase4-data-and-combiner-scope.md` for
the full decision (a genuine cost/benefit tradeoff, put to the user via
`AskUserQuestion` rather than decided silently).

Real chunking confirmed live before writing this script (not re-quoted
from the earlier estimate): `gs://weatherbench2/datasets/ifs_ens/
2018-2022-1440x721.zarr` chunks every variable as one `(1, 50, 1, 721,
1440)` block per `(time, prediction_timedelta)` pair -- the full 50-member,
full-global-grid slab in a single I/O chunk, matching
docs/phase-1-data-requirements.md's own finding. A live single-chunk probe
measured **~34s and ~208MB** for one `(time, prediction_timedelta)` pair of
`total_precipitation_24hr` alone -- close to the earlier ~47s/chunk
estimate (that estimate covered 2 variables per chunk-pair; this store
pulls precipitation only, per this project's precipitation-only scope
throughout every prior tier, so real cost here is smaller: 18 timestamps x
5 leads = 90 chunks x ~208MB = ~18.7GB, ~50 minutes if fetched serially).

This project's exact 18 weekly JJAS-2020 timestamps are confirmed (checked
live, not re-derived) to all exist verbatim in the raw archive's own time
index -- selected directly by real value here (`.sel(time=<exact list>)`)
rather than re-running `build_baseline_store.py`'s `_weekly_init_times`
cadence logic against this archive independently, which risks a
subtly-different stride/anchor picking different timestamps. This
guarantees the new store's samples align 1:1 with
`data/baseline_2020_jjas.zarr` with no separate alignment logic needed
downstream.

Fetched in one batch per timestamp (5 leads x 50 members x full lat/lon
each, ~1GB/batch), not one giant vectorized `.sel()` over all 90 combos at
once -- following `scripts/build_lagged_ensemble_store.py`'s own
documented fix for the real stall a single mega-call caused there. Also
resumable: each fetched timestamp is cached to a small per-timestamp
NetCDF file in a staging directory as soon as it arrives, with an
idempotent manifest tracking which timestamps succeeded -- so a failure
partway through a ~50-minute fetch only needs to retry the timestamps that
actually failed (re-reading the rest from the staging cache, not
re-fetching from GCS), rather than wasting the whole run.

Usage:
    python scripts/build_ifs_ensemble_store.py [--out PATH]
        [--baseline-store PATH] [--lead-hours H [H ...]]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_baseline_store import (  # noqa: E402
    DEFAULT_LEAD_HOURS,
    GCS_ANON,
    INDIA_LAT_SLICE,
    INDIA_LON_SLICE_0_360,
    _clear_encoding,
    _lat_slice_for,
)

from weavr.archives import archive_for  # noqa: E402
from weavr.grid import SourceTooCoarseError, regrid_to_common  # noqa: E402

IFS_ENS_ZARR_PATH = archive_for("ifs_ens", 2020).path
PRECIP_VARIABLE = "total_precipitation_24hr"
N_MEMBERS = 50


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str))


def real_baseline_timestamps(baseline_store: str, group: str = "ifs_ens_mean") -> np.ndarray:
    """The timestamps the baseline store already uses, read directly from the real store.
    Guarantees the new store's samples align 1:1 with baseline store by construction.
    """
    try:
        ds = xr.open_zarr(baseline_store, group=group, consolidated=True)
        return ds["time"].values
    except Exception:
        import zarr

        r = zarr.open_group(baseline_store, mode="r")
        keys = list(r.group_keys())
        target = group if group in keys else (keys[0] if keys else None)
        if target is not None:
            ds = xr.open_zarr(baseline_store, group=target, consolidated=True)
        else:
            ds = xr.open_zarr(baseline_store, consolidated=True)
        coord = "time" if "time" in ds.coords else "nominal_time"
        return ds[coord].values


def fetch_one_timestamp(
    ds: xr.Dataset, timestamp: np.datetime64, lead_hours: list[int]
) -> xr.Dataset:
    """Fetch every requested lead's full 50-member, full-global-grid slab
    for one timestamp.

    Leads are fetched one at a time and concatenated, rather than as a single
    vectorized `.sel()` over all five. Both read the same bytes from the same
    chunks -- the archive chunks this variable as one
    `(1, 50, 1, 721, 1440)` slab per `(time, prediction_timedelta)` -- but the
    combined read forces five such slabs to be held and materialised together
    before anything is written, and that measured ~595 s per timestamp
    against ~105-140 s for the same work read lead-by-lead. Reading per lead
    also means a partial result survives a failure instead of the whole
    timestamp's worth of buffering being lost.
    """
    parts = []
    for lead in lead_hours:
        lead_indexer = xr.DataArray([lead], dims="prediction_timedelta")
        parts.append(ds.sel(time=timestamp, prediction_timedelta=lead_indexer).load())
    return xr.concat(parts, dim="prediction_timedelta")


def staging_path(staging_dir: Path, ts: np.datetime64) -> Path:
    ts_key = str(np.datetime_as_string(ts, unit="s")).replace(":", "")
    return staging_dir / f"{ts_key}.nc"


def fetch_and_stage_one_timestamp(
    ds: xr.Dataset,
    ts: np.datetime64,
    lead_hours: list[int],
    staging_dir: Path,
    manifest: dict,
    manifest_path: Path,
    lock: threading.Lock,
    per_timestamp_seconds: dict[str, float],
    force: bool = False,
    max_retries: int = 3,
    timeout_seconds: float = 300.0,
    worker_label: str = "",
) -> bool:
    ts_key = str(np.datetime_as_string(ts, unit="s"))
    staging_file = staging_path(staging_dir, ts)

    with lock:
        already_ok = manifest.get(ts_key, {}).get("status") == "ok" and staging_file.exists()
    if not force and already_ok:
        print(f"[skip] {ts_key}: already fetched and cached in staging")
        return True

    tag = f"[{worker_label}] " if worker_label else ""
    print(f"[fetch] {tag}{ts_key} ...")

    last_exc = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(fetch_one_timestamp, ds, ts, lead_hours)
            fetched = fut.result(timeout=timeout_seconds)
            pool.shutdown(wait=False)

            elapsed = time.time() - t0
            # Atomic staging write: write to unique .tmp file first, then atomic rename
            tmp_staging_file = staging_file.with_name(
                f"{staging_file.stem}.tmp.{os.getpid()}.{threading.get_ident()}.nc"
            )
            fetched.assign_coords(time=ts).to_netcdf(tmp_staging_file)
            tmp_staging_file.replace(staging_file)

            with lock:
                per_timestamp_seconds[ts_key] = elapsed
                manifest[ts_key] = {
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "attempts": attempt,
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                }
                manifest["_per_timestamp_seconds"] = per_timestamp_seconds
                _save_manifest(manifest_path, manifest)

            print(f"[ok]   {tag}{ts_key}: {elapsed:.1f}s (attempt {attempt})")
            return True
        except Exception as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            last_exc = exc
            print(f"[retry {attempt}/{max_retries}] {tag}{ts_key}: {exc}", file=sys.stderr)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 10))

    with lock:
        manifest[ts_key] = {
            "status": "failed",
            "error": str(last_exc),
            "attempts": max_retries,
            "traceback": traceback.format_exc(),
            "fetched_at": datetime.utcnow().isoformat() + "Z",
        }
        _save_manifest(manifest_path, manifest)
    print(f"[FAIL] {tag}{ts_key}: failed after {max_retries} attempts: {last_exc}", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2018, 2020), default=2020)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--init-times-from",
        "--baseline-store",
        dest="init_times_from",
        default=None,
        help=(
            "Path to a zarr store to read init times from "
            "(default: data/baseline_2020_jjas.zarr)"
        ),
    )
    parser.add_argument("--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker threads for chunk fetching (default: 1)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout in seconds for fetching a single timestamp (default: 300.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per timestamp (default: 3)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-fetch timestamps already marked ok"
    )
    args = parser.parse_args()
    args.out = args.out or os.environ.get("WEAVR_IFS_ENSEMBLE_STORE_PATH") or (
        f"data/ifs_ens_{args.year}_jjas_daily.zarr"
        if args.year == 2018 else "data/ifs_ens_2020_jjas.zarr"
    )
    args.init_times_from = args.init_times_from or (
        f"data/baseline_{args.year}_jjas_daily.zarr"
        if args.year == 2018 else "data/baseline_2020_jjas.zarr"
    )
    archive_path = archive_for("ifs_ens", args.year).path

    store_path = Path(args.out)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = store_path.with_suffix(".manifest.json")
    manifest = _load_manifest(manifest_path)
    staging_dir = store_path.parent / f"{store_path.name}.staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building real IFS 50-member ensemble store at {store_path}")
    print(f"Source: {archive_path}")

    ds = xr.open_zarr(archive_path, storage_options=GCS_ANON, consolidated=True)
    ds = ds[[PRECIP_VARIABLE]]
    timestamps = real_baseline_timestamps(args.init_times_from)
    lat_slice = _lat_slice_for(ds, "latitude", INDIA_LAT_SLICE.start, INDIA_LAT_SLICE.stop)
    ds = ds.sel(latitude=lat_slice, longitude=INDIA_LON_SLICE_0_360)
    ds = ds.rename({"number": "member"})

    print(
        f"Window: {len(timestamps)} timestamps (matching {args.init_times_from}'s own), "
        f"lead hours {args.lead_hours}, workers={args.workers}"
    )

    manifest["_source_archive_path"] = archive_path
    manifest["_meta"] = {
        "init_times_from": str(args.init_times_from),
        "lead_hours": args.lead_hours,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
    }
    _save_manifest(manifest_path, manifest)

    fetch_start = time.time()
    per_timestamp_seconds: dict[str, float] = manifest.get("_per_timestamp_seconds", {})
    lock = threading.Lock()

    needed_timestamps = []
    for ts in timestamps:
        ts_key = str(np.datetime_as_string(ts, unit="s"))
        staging_file = staging_path(staging_dir, ts)
        already_ok = manifest.get(ts_key, {}).get("status") == "ok" and staging_file.exists()
        if args.force or not already_ok:
            needed_timestamps.append(ts)
        else:
            print(f"[skip] {ts_key}: already fetched and cached in staging")

    print(f"Need to fetch {len(needed_timestamps)}/{len(timestamps)} timestamps...")

    if args.workers > 1 and needed_timestamps:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    fetch_and_stage_one_timestamp,
                    ds,
                    ts,
                    args.lead_hours,
                    staging_dir,
                    manifest,
                    manifest_path,
                    lock,
                    per_timestamp_seconds,
                    force=args.force,
                    max_retries=args.max_retries,
                    timeout_seconds=args.timeout_seconds,
                    worker_label=f"worker-{idx % args.workers + 1}",
                ): ts
                for idx, ts in enumerate(needed_timestamps)
            }
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    else:
        for idx, ts in enumerate(needed_timestamps):
            fetch_and_stage_one_timestamp(
                ds,
                ts,
                args.lead_hours,
                staging_dir,
                manifest,
                manifest_path,
                lock,
                per_timestamp_seconds,
                force=args.force,
                max_retries=args.max_retries,
                timeout_seconds=args.timeout_seconds,
                worker_label=f"{idx + 1}/{len(needed_timestamps)}",
            )

    total_elapsed = time.time() - fetch_start

    n_failed = sum(
        1 for k, v in manifest.items() if not k.startswith("_") and v.get("status") != "ok"
    )
    if n_failed:
        print(
            f"\n{n_failed} timestamp(s) failed -- not writing the final store. "
            "Re-run this script (successfully-fetched timestamps are cached in "
            f"{staging_dir} and will be skipped) to retry only the failures."
        )
        return 1

    print(f"\nCombining {len(timestamps)} timestamps from staging into final store...")
    fetched_datasets = [xr.open_dataset(staging_path(staging_dir, ts)) for ts in timestamps]
    combined = xr.concat(fetched_datasets, dim="time")
    combined = _clear_encoding(combined)

    try:
        combined = regrid_to_common(combined)
    except SourceTooCoarseError:
        raise

    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)
    combined.to_zarr(store_path, mode="w")

    real_total_bytes = sum(d[PRECIP_VARIABLE].nbytes for d in fetched_datasets)
    summary = {
        "n_timestamps": len(fetched_datasets),
        "n_lead_hours": len(args.lead_hours),
        "n_members": N_MEMBERS,
        "workers": args.workers,
        "total_elapsed_seconds": total_elapsed,
        "total_bytes_fetched": real_total_bytes,
        "mean_seconds_per_timestamp": total_elapsed / max(len(timestamps), 1),
    }
    manifest["_summary"] = summary
    _save_manifest(manifest_path, manifest)

    print("\n--- Summary ---")
    print(
        f"Fetched {len(fetched_datasets)} timestamps this run in {total_elapsed / 60:.1f} min "
        f"({real_total_bytes / 1e9:.2f} GB from staging) -- "
        f"{summary['mean_seconds_per_timestamp']:.1f}s/timestamp average."
    )
    print(f"Wrote {store_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
