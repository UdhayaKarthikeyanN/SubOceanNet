"""Bulk-fetch a historical date range of one Copernicus Marine ocean surface
variable (SST/SSS/SLA/currents) for the configured North Indian Ocean
bounding box - used to build a real training/reference dataset, as opposed
to fetch_ocean.py which only ever grabs the single latest timestep for the
live "current conditions" view.

Runs inside the isolated .venv-live (see requirements.txt in this folder),
never inside the main app's .venv. Invoked directly (not a subprocess of the
main app) as a one-off data-prep step - see scripts/build_real_training_data.py.

Usage:
    python fetch_ocean_historical.py --var sst \
        --dataset-id cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m \
        --source-var thetao --lat-min 5 --lat-max 30 --lon-min 45 --lon-max 105 \
        --start-date 2026-03-01 --end-date 2026-08-28 \
        --out-dir data/_train_staging/raw

    # full-depth reference truth (same dataset/source-var as sst, just depth-full):
    python fetch_ocean_historical.py --var reference_temperature \
        --dataset-id cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m \
        --source-var thetao --lat-min 5 --lat-max 30 --lon-min 45 --lon-max 105 \
        --start-date 2026-03-01 --end-date 2026-08-28 --full-depth --max-depth 1000 \
        --out-dir data/_train_staging

On success prints exactly one JSON line to stdout:
    {"variable": "...", "provider": "copernicus_marine",
     "start": "...", "end": "...", "path": "..."}
On failure, prints a message to stderr and exits non-zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import emit_result, fail, load_dotenv_if_present, require_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, help="output variable name (used only for the filename/log)")
    ap.add_argument("--dataset-id", required=True, help="Copernicus Marine dataset id")
    ap.add_argument("--source-var", required=True, help="variable name inside that dataset, e.g. thetao")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--full-depth", action="store_true",
                     help="fetch the depth range [0, --max-depth] instead of surface-only")
    ap.add_argument("--max-depth", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", default=None)
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
    out_name = args.out_name or f"{args.var}.nc"
    out_path = out_dir / out_name

    max_depth = args.max_depth if args.full_depth else 1.0

    try:
        copernicusmarine.subset(
            dataset_id=args.dataset_id,
            variables=[args.source_var],
            minimum_longitude=args.lon_min,
            maximum_longitude=args.lon_max,
            minimum_latitude=args.lat_min,
            maximum_latitude=args.lat_max,
            minimum_depth=0,
            maximum_depth=max_depth,
            start_datetime=f"{args.start_date}T00:00:00",
            end_datetime=f"{args.end_date}T00:00:00",
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
        fail(f"no timesteps returned for {args.var} in {args.start_date}..{args.end_date}")
        return
    if args.source_var not in ds:
        avail = list(ds.data_vars)
        ds.close()
        fail(f"variable '{args.source_var}' not found in downloaded file (has: {avail})")
        return
    n = ds.sizes["time"]
    first_t, last_t = str(ds["time"].values[0]), str(ds["time"].values[-1])
    ds.close()

    emit_result(variable=args.var, provider="copernicus_marine",
                observation_time=f"{first_t}..{last_t} (n={n})", path=str(out_path))


if __name__ == "__main__":
    main()
