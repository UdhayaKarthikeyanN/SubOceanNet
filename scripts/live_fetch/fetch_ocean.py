"""Fetch one Copernicus Marine ocean surface variable (SST/SSS/SLA/currents)
for the configured North Indian Ocean bounding box, using the *latest
actually available* timestep - never a guessed or faked "now".

Runs inside the isolated .venv-live (see requirements.txt in this folder),
never inside the main app's .venv. Invoked as a subprocess by
src/data/live/providers.py; never imported directly by the main app.

Usage:
    python fetch_ocean.py --var sst --dataset-id <copernicus dataset id> \
        --source-var thetao --lat-min 5 --lat-max 30 --lon-min 45 \
        --lon-max 105 --out-dir data/live_cache

On success prints exactly one JSON line to stdout:
    {"variable": "sst", "provider": "copernicus_marine",
     "observation_time": "...", "path": "..."}
On failure, prints a message to stderr and exits non-zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import emit_result, fail, load_dotenv_if_present, require_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, help="canonical SubOceanNet variable name, e.g. sst")
    ap.add_argument("--dataset-id", required=True, help="Copernicus Marine dataset id")
    ap.add_argument("--source-var", required=True, help="variable name inside that dataset, e.g. thetao")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--lookback-days", type=int, default=10,
                     help="how far back to search for the latest available timestep")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    load_dotenv_if_present()
    creds = require_env("COPERNICUS_MARINE_USERNAME", "COPERNICUS_MARINE_PASSWORD")

    try:
        import copernicusmarine
        import xarray as xr
    except ImportError as e:
        fail(f"copernicusmarine/xarray not installed in this environment: {e}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.lookback_days)
    out_name = f"_incoming_{args.var}.nc"
    out_path = out_dir / out_name

    try:
        copernicusmarine.subset(
            dataset_id=args.dataset_id,
            variables=[args.source_var],
            minimum_longitude=args.lon_min,
            maximum_longitude=args.lon_max,
            minimum_latitude=args.lat_min,
            maximum_latitude=args.lat_max,
            minimum_depth=0,
            maximum_depth=1,
            start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=now.strftime("%Y-%m-%dT%H:%M:%S"),
            output_directory=str(out_dir),
            output_filename=out_name,
            username=creds["COPERNICUS_MARINE_USERNAME"],
            password=creds["COPERNICUS_MARINE_PASSWORD"],
            overwrite=True,
            disable_progress_bar=True,
        )
    except Exception as e:  # noqa: BLE001 - surface any API/network error clearly
        fail(f"copernicusmarine.subset failed for {args.var} "
             f"({args.dataset_id}/{args.source_var}): {e}")
        return

    if not out_path.exists():
        fail(f"copernicusmarine.subset reported success but {out_path} was not written")
        return

    ds = xr.open_dataset(out_path)
    if "time" not in ds.dims or ds.sizes["time"] == 0:
        ds.close()
        fail(f"no timesteps returned for {args.var} in the last {args.lookback_days} days "
             f"- the dataset id/variable may not cover this region or date range")
        return
    if args.source_var not in ds:
        avail = list(ds.data_vars)
        ds.close()
        fail(f"variable '{args.source_var}' not found in downloaded file (has: {avail})")
        return

    latest = ds.isel(time=[-1])
    if "depth" in latest.dims:
        latest = latest.isel(depth=0, drop=True)
    da = latest[args.source_var].rename(args.var).load()  # force into memory
    obs_time = str(latest["time"].values[-1])
    ds.close()

    out_ds = da.to_dataset(name=args.var)
    out_ds.to_netcdf(out_path)
    out_ds.close()

    emit_result(variable=args.var, provider="copernicus_marine",
               observation_time=obs_time, path=str(out_path))


if __name__ == "__main__":
    main()
