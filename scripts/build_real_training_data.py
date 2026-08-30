"""One-off data-prep step: fetch a real historical window of Copernicus
Marine + ERA5 data and assemble it into the exact files src/data/loaders.py
expects for data_source.type == "netcdf":

    data/raw/{sst,sss,sla,cur_u,cur_v,wind_u,wind_v}.nc   (dims time,lat,lon)
    data/reference/temperature.nc                          (dims time,depth,lat,lon)

Two phases:
  1. Fetch (subprocess into the isolated .venv-live) - writes native-grid
     multi-day files into data/_train_staging/.
  2. Assemble (this process, main .venv) - regrids each onto the app's exact
     target grid with the existing interpolate_grid(), aligns every
     variable's time to daily-midnight timestamps (so xr.merge(join="inner")
     in load_input_dataset() actually finds the overlap), and for the
     reference file snaps the fetched depth levels onto the app's canonical
     15 depths_m.

Run from the project root with the MAIN venv (needs this project's own
xarray/scipy for interpolate_grid - never copernicusmarine/cdsapi, which stay
isolated in .venv-live and are only invoked as subprocesses here, exactly
like src/data/live/providers.py does for the live-mode fetchers):

    .venv\\Scripts\\python.exe scripts\\build_real_training_data.py \\
        --start-date 2026-03-01 --end-date 2026-08-28
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROJECT_ROOT, depths as cfg_depths, input_variable_names, load_config
from src.data.preprocessing import interpolate_grid

STAGING = PROJECT_ROOT / "data" / "_train_staging"
STAGING_RAW = STAGING / "raw"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

# canonical output var -> (dataset_id key under data_source.live.ocean.variables)
OCEAN_VARS = ("sst", "sss", "sla", "cur_u", "cur_v")
WIND_VARS = ("wind_u", "wind_v")


def _live_python(cfg: dict) -> Path:
    py = Path(cfg["data_source"]["live"].get("python", ".venv-live/Scripts/python.exe"))
    if not py.is_absolute():
        py = PROJECT_ROOT / py
    return py


def _run(args: list[str], label: str) -> None:
    print(f"[fetch] {label} ...", flush=True)
    proc = subprocess.run(args, cwd=str(PROJECT_ROOT / "scripts" / "live_fetch"),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[fetch] {label} FAILED:\n{(proc.stderr or proc.stdout)[-3000:]}")
    out = proc.stdout.strip().splitlines()
    print(f"[fetch] {label} OK: {out[-1] if out else '(no output)'}")


def fetch_all(cfg: dict, start: str, end: str) -> None:
    STAGING_RAW.mkdir(parents=True, exist_ok=True)
    py = _live_python(cfg)
    if not py.exists():
        raise SystemExit(f"isolated live-fetch interpreter not found at {py} - see README 'Live data' setup")
    region = cfg["region"]
    latlon_args = ["--lat-min", str(region["lat_min"]), "--lat-max", str(region["lat_max"]),
                   "--lon-min", str(region["lon_min"]), "--lon-max", str(region["lon_max"])]
    ocean_specs = cfg["data_source"]["live"]["ocean"]["variables"]
    wind_specs = cfg["data_source"]["live"]["wind"]["variables"]

    for name in OCEAN_VARS:
        spec = ocean_specs.get(name) or {}
        dataset_id = spec.get("dataset_id")
        if not dataset_id:
            raise SystemExit(f"no dataset_id configured for '{name}' under data_source.live.ocean.variables")
        _run([str(py), "fetch_ocean_historical.py",
              "--var", name, "--dataset-id", dataset_id, "--source-var", spec.get("variable", name),
              *latlon_args, "--start-date", start, "--end-date", end,
              "--out-dir", str(STAGING_RAW)], f"ocean/{name}")

    # reference truth: same product/variable as sst, full depth this time
    sst_spec = ocean_specs.get("sst") or {}
    if not sst_spec.get("dataset_id"):
        raise SystemExit("no dataset_id configured for 'sst' - needed as the reference-temperature source")
    max_depth = max(cfg_depths(cfg))
    _run([str(py), "fetch_ocean_historical.py",
          "--var", "reference_temperature", "--dataset-id", sst_spec["dataset_id"],
          "--source-var", sst_spec.get("variable", "thetao"),
          *latlon_args, "--start-date", start, "--end-date", end,
          "--full-depth", "--max-depth", str(max_depth),
          "--out-dir", str(STAGING), "--out-name", "reference_raw.nc"], "reference_temperature (full depth)")

    for name in WIND_VARS:
        cds_var = wind_specs.get(name)
        if not cds_var:
            raise SystemExit(f"no CDS variable configured for '{name}' under data_source.live.wind.variables")
        _run([str(py), "fetch_wind_historical.py",
              "--var", name, "--cds-variable", cds_var,
              *latlon_args, "--start-date", start, "--end-date", end,
              "--out-dir", str(STAGING_RAW)], f"wind/{name}")


def _target_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    region = cfg["region"]
    res = float(cfg["grid"]["resolution"])
    lat = np.arange(region["lat_min"], region["lat_max"] + 1e-9, res)
    lon = np.arange(region["lon_min"], region["lon_max"] + 1e-9, res)
    return lat, lon


def _floor_to_day(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds["time"] = ds["time"].dt.floor("D")
    return ds


def assemble(cfg: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    target_lat, target_lon = _target_grid(cfg)
    ocean_specs = cfg["data_source"]["live"]["ocean"]["variables"]

    for name in input_variable_names(cfg):
        staged = STAGING_RAW / f"{name}.nc"
        if not staged.exists():
            raise SystemExit(f"staged file missing: {staged} (fetch step didn't run or failed)")
        ds = xr.open_dataset(staged)
        if name in ocean_specs:
            source_var = ocean_specs[name].get("variable", name)
        else:
            source_var = name  # wind files are already written under the canonical name
        rename = {k: "lat" for k in ("latitude",) if k in ds.dims}
        rename.update({k: "lon" for k in ("longitude",) if k in ds.dims})
        if rename:
            ds = ds.rename(rename)
        if source_var in ds and source_var != name:
            ds = ds.rename({source_var: name})
        if "depth" in ds.dims:
            ds = ds.isel(depth=0, drop=True)
        ds = ds.sortby("lat").sortby("lon")
        ds = interpolate_grid(ds, target_lat, target_lon)
        ds = _floor_to_day(ds)
        ds = ds.sortby("time")
        out_path = RAW_DIR / f"{name}.nc"
        ds[[name]].astype("float32", casting="same_kind").to_netcdf(out_path)
        ds.close()
        print(f"[assemble] wrote {out_path} ({ds.sizes.get('time', '?')} days)")

    ref_staged = STAGING / "reference_raw.nc"
    if not ref_staged.exists():
        raise SystemExit(f"staged reference file missing: {ref_staged}")
    ref = xr.open_dataset(ref_staged)
    rename = {k: "lat" for k in ("latitude",) if k in ref.dims}
    rename.update({k: "lon" for k in ("longitude",) if k in ref.dims})
    if rename:
        ref = ref.rename(rename)
    sst_source_var = (ocean_specs.get("sst") or {}).get("variable", "thetao")
    if sst_source_var in ref:
        ref = ref.rename({sst_source_var: "temperature"})
    ref = ref.sortby("lat").sortby("lon").sortby("depth")
    ref = interpolate_grid(ref, target_lat, target_lon)
    target_depths = np.asarray(sorted(cfg_depths(cfg)), dtype=float)
    ref = ref.sel(depth=target_depths, method="nearest")
    ref = ref.assign_coords(depth=target_depths)  # snap to exact canonical values
    ref = _floor_to_day(ref)
    ref = ref.sortby("time")
    out_path = REFERENCE_DIR / "temperature.nc"
    ref[["temperature"]].astype("float32", casting="same_kind").to_netcdf(out_path)
    n_time = ref.sizes.get("time", "?")
    ref.close()
    print(f"[assemble] wrote {out_path} ({n_time} days x {len(target_depths)} depths)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--skip-fetch", action="store_true",
                     help="reuse whatever is already staged in data/_train_staging/ and only re-run assembly")
    args = ap.parse_args()

    cfg = load_config()
    if not args.skip_fetch:
        fetch_all(cfg, args.start_date, args.end_date)
    else:
        print("[fetch] skipped (--skip-fetch) - reusing existing staged files")
    assemble(cfg)
    print("[done] data/raw/*.nc and data/reference/temperature.nc are ready.")


if __name__ == "__main__":
    main()
