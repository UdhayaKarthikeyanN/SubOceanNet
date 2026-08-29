"""Fetch one ERA5 10m wind component (via the Copernicus Climate Data Store /
CDS API) for the configured North Indian Ocean bounding box.

ERA5 is published with a multi-day latency (it is a reanalysis, not a live
feed), so "latest actually available" here means: walk backwards day by day
from today until the CDS API actually returns data, rather than assuming a
fixed lag. Never invents a timestamp that wasn't actually served.

Runs inside the isolated .venv-live (see requirements.txt in this folder),
never inside the main app's .venv. Invoked as a subprocess by
src/data/live/providers.py; never imported directly by the main app.

Usage:
    python fetch_wind.py --var wind_u --cds-variable 10m_u_component_of_wind \
        --lat-min 5 --lat-max 30 --lon-min 45 --lon-max 105 --out-dir data/live_cache

On success prints exactly one JSON line to stdout:
    {"variable": "wind_u", "provider": "era5",
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
    ap.add_argument("--var", required=True, help="canonical SubOceanNet variable name, e.g. wind_u")
    ap.add_argument("--cds-variable", required=True, help="ERA5 variable name, e.g. 10m_u_component_of_wind")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--lookback-days", type=int, default=10,
                     help="how many days back to search for the latest published day")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    load_dotenv_if_present()
    require_env("CDSAPI_URL", "CDSAPI_KEY")

    try:
        import cdsapi
        import xarray as xr
    except ImportError as e:
        fail(f"cdsapi/xarray not installed in this environment: {e}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"_incoming_{args.var}.nc"

    client = cdsapi.Client()
    area = [args.lat_max, args.lon_min, args.lat_min, args.lon_max]  # N, W, S, E

    today = datetime.now(timezone.utc).date()
    last_error: Exception | None = None
    for back in range(args.lookback_days):
        day = today - timedelta(days=back)
        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [args.cds_variable],
                    "year": f"{day.year:04d}",
                    "month": f"{day.month:02d}",
                    "day": f"{day.day:02d}",
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(out_path),
            )
            break
        except Exception as e:  # noqa: BLE001 - that day isn't published yet; try an earlier one
            last_error = e
            continue
    else:
        fail(f"no ERA5 data available for {args.cds_variable} in the last "
             f"{args.lookback_days} days: {last_error}")
        return

    ds = xr.open_dataset(out_path)
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    if time_dim not in ds.dims or ds.sizes[time_dim] == 0:
        ds.close()
        fail(f"downloaded ERA5 file has no '{time_dim}' steps")
        return
    if args.cds_variable not in ds and len(ds.data_vars) == 1:
        source_name = list(ds.data_vars)[0]  # ERA5 netcdf vars use short codes (e.g. u10), not the request name
    else:
        source_name = args.cds_variable

    # daily mean of the 4 synoptic hours pulled above
    da = ds[source_name].mean(dim=time_dim).rename(args.var).load()
    obs_time = str(ds[time_dim].values[-1])
    ds.close()

    out_ds = da.to_dataset(name=args.var)
    out_ds.to_netcdf(out_path)
    out_ds.close()

    emit_result(variable=args.var, provider="era5",
               observation_time=obs_time, path=str(out_path))


if __name__ == "__main__":
    main()
