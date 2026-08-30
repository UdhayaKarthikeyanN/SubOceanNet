"""Bulk-fetch a historical date range of one ERA5 10m wind component (via the
Copernicus Climate Data Store / CDS API) for the configured North Indian
Ocean bounding box - used to build a real training dataset, as opposed to
fetch_wind.py which only ever walks backward from today to find the latest
published single day.

Requests are chunked by calendar month (CDS API's own recommended pattern
for bulk pulls - one giant multi-month request is more likely to time out
or get queued for a very long time). Each day's value is the mean of the
same 4 synoptic hours fetch_wind.py uses, so the daily cadence matches
exactly.

Runs inside the isolated .venv-live (see requirements.txt in this folder),
never inside the main app's .venv. Invoked directly (not a subprocess of the
main app) as a one-off data-prep step - see scripts/build_real_training_data.py.

Usage:
    python fetch_wind_historical.py --var wind_u \
        --cds-variable 10m_u_component_of_wind \
        --lat-min 5 --lat-max 30 --lon-min 45 --lon-max 105 \
        --start-date 2026-03-01 --end-date 2026-08-28 \
        --out-dir data/_train_staging/raw

On success prints exactly one JSON line to stdout:
    {"variable": "wind_u", "provider": "era5",
     "observation_time": "...", "path": "..."}
On failure, prints a message to stderr and exits non-zero.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import date
from pathlib import Path

from common import emit_result, fail, load_dotenv_if_present, require_env


def _month_chunks(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = date(y, m, 1)
        last_day_of_month = calendar.monthrange(y, m)[1]
        last = date(y, m, last_day_of_month)
        chunk_start = max(first, start)
        chunk_end = min(last, end)
        yield y, m, chunk_start.day, chunk_end.day
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, help="canonical SubOceanNet variable name, e.g. wind_u")
    ap.add_argument("--cds-variable", required=True, help="ERA5 variable name, e.g. 10m_u_component_of_wind")
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
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

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        fail(f"--start-date {start} is after --end-date {end}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / f"_incoming_{args.var}_chunks"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    area = [args.lat_max, args.lon_min, args.lat_min, args.lon_max]  # N, W, S, E

    client = cdsapi.Client()
    chunk_paths = []
    for y, m, d0, d1 in _month_chunks(start, end):
        chunk_path = tmp_dir / f"{y:04d}-{m:02d}.nc"
        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [args.cds_variable],
                    "year": f"{y:04d}",
                    "month": f"{m:02d}",
                    "day": [f"{d:02d}" for d in range(d0, d1 + 1)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(chunk_path),
            )
            chunk_paths.append(chunk_path)
        except Exception as e:  # noqa: BLE001 - surface clearly, abort the whole range
            fail(f"CDS retrieve failed for {args.cds_variable} {y:04d}-{m:02d}: {e}")
            return

    if not chunk_paths:
        fail(f"no month chunks produced for {args.start_date}..{args.end_date}")
        return

    datasets = [xr.open_dataset(p) for p in chunk_paths]
    ds = xr.concat(datasets, dim="valid_time" if "valid_time" in datasets[0].dims else "time")
    for d in datasets:
        d.close()

    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    if time_dim not in ds.dims or ds.sizes[time_dim] == 0:
        fail(f"concatenated ERA5 file has no '{time_dim}' steps")
        return
    if args.cds_variable not in ds and len(ds.data_vars) == 1:
        source_name = list(ds.data_vars)[0]  # ERA5 netcdf vars use short codes (e.g. u10), not the request name
    else:
        source_name = args.cds_variable

    # daily mean of the 4 synoptic hours, one value per calendar day
    daily = ds[source_name].resample({time_dim: "1D"}).mean()
    daily = daily.rename(args.var).load()
    first_t, last_t = str(daily[time_dim].values[0]), str(daily[time_dim].values[-1])

    out_path = out_dir / f"{args.var}.nc"
    out_ds = daily.rename({time_dim: "time"}).to_dataset(name=args.var)
    out_ds.to_netcdf(out_path)
    out_ds.close()
    ds.close()

    for p in chunk_paths:
        p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    emit_result(variable=args.var, provider="era5",
                observation_time=f"{first_t}..{last_t}", path=str(out_path))


if __name__ == "__main__":
    main()
