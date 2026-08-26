"""Dataset loaders: synthetic demo files or real NetCDF (swap via config only)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from ..config import get_config, input_variable_names


def synthetic_dir(cfg=None) -> Path:
    """Directory holding the synthetic demo dataset (config-overridable)."""
    cfg = cfg or get_config()
    p = Path(cfg.get("paths", {}).get("synthetic_dir", "data/synthetic"))
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    return p


def _standardize(ds: xr.Dataset, time_coord: str, depth_coord: str) -> xr.Dataset:
    def _has(name):
        return name in ds.coords or name in ds.dims or name in ds.variables

    ren = {}
    for cand, target in ((time_coord, "time"), ("latitude", "lat"), ("nav_lat", "lat"),
                         ("longitude", "lon"), ("nav_lon", "lon"),
                         (depth_coord, "depth")):
        if _has(cand) and cand != target and target not in ren.values():
            ren[cand] = target
    if ren:
        ds = ds.rename(ren)
    ds = ds.sortby("lat", "lon")
    # decode times to datetime64
    if np.issubdtype(ds["time"].dtype, np.number):
        ds["time"] = xr.conventions.decode_cf_variable("time", ds["time"])
    return ds


def load_input_dataset(cfg=None) -> xr.Dataset:
    """Load the 7 surface input variables as one canonical Dataset."""
    cfg = cfg or get_config()
    names = input_variable_names(cfg)
    src = cfg["data_source"]
    if src["type"] == "synthetic":
        path = synthetic_dir(cfg) / "inputs.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"Synthetic dataset not found at {path}. Run: python -m src.data.generate_synthetic"
            )
        ds = xr.open_dataset(path, engine="netcdf4")
        missing = [n for n in names if n not in ds]
        if missing:
            raise ValueError(f"Synthetic inputs.nc missing variables: {missing}")
        return _standardize(ds, "time", "depth")[names]

    # ---- real NetCDF mode: data/raw/<variable>.nc --------------------------
    raw = Path(src["netcdf"]["input_dir"])
    if not raw.is_absolute():
        raw = Path(cfg["_root"]) / raw
    if not raw.exists():
        raise FileNotFoundError(f"data_source=netcdf but input dir not found: {raw}")
    tcoord = src["netcdf"].get("time_coord", "time")
    datasets = []
    for name in names:
        p = raw / f"{name}.nc"
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file {p}")
        dsi = xr.open_dataset(p, engine="netcdf4")
        var_candidates = [v for v in dsi.data_vars if v.lower().replace("_", "") == name]
        if not var_candidates:
            var_candidates = list(dsi.data_vars)[:1]
        da = dsi[var_candidates[0]]
        da.name = name
        datasets.append(da.to_dataset())
    merged = xr.merge(datasets, join="inner")
    return _standardize(merged, tcoord, src["netcdf"].get("depth_coord", "depth"))[names]


def load_reference_dataset(cfg=None) -> tuple[xr.Dataset, str]:
    """Reference truth for validation.

    Priority: real GLORYS/ARGO file(s) under data/reference/, else the clean
    synthetic truth. Returns (dataset with 'temperature' (time, depth, lat, lon),
    label).
    """
    cfg = cfg or get_config()
    ref_cfg = cfg.get("data_source", {}).get("netcdf", {})
    ref_dir = Path(ref_cfg.get("reference_dir", "data/reference"))
    if not ref_dir.is_absolute():
        ref_dir = Path(cfg["_root"]) / ref_dir
    if ref_dir.exists():
        files = sorted(ref_dir.glob("*.nc"))
        if files:
            ds = xr.open_mfdataset(files, combine="by_coords")
            ds = _standardize(ds, ref_cfg.get("time_coord", "time"),
                              ref_cfg.get("depth_coord", "depth"))
            if "temperature" not in ds:
                cand = [v for v in ds.data_vars if "temp" in v.lower()]
                ds = ds.rename({cand[0]: "temperature"} if cand else {})
            if "temperature" in ds:
                return ds[["temperature"]], "glorys_argo"
    path = synthetic_dir(cfg) / "truth.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"No reference data: neither {ref_dir} nor synthetic truth at {path}"
        )
    ds = xr.open_dataset(path, engine="netcdf4")
    ds = _standardize(ds, "time", "depth")
    return ds[["temperature"]], "SYNTHETIC REFERENCE - DEMO MODE"


def available_date_range(cfg=None) -> tuple[str, str]:
    cfg = cfg or get_config()
    src_type = cfg["data_source"]["type"]
    try:
        if src_type == "synthetic":
            man = synthetic_dir(cfg) / "manifest.json"
            if man.exists():
                m = json.loads(man.read_text())
                return m["date_range"][0], m["date_range"][1]
        ds = load_input_dataset(cfg)
        t = ds["time"].values
        return str(np.datetime_as_string(np.datetime64(t[0], "D"), unit="D")), \
               str(np.datetime_as_string(np.datetime64(t[-1], "D"), unit="D"))
    except FileNotFoundError:
        tr = cfg["time_range"]
        return str(tr["start"]), str(tr["end"])


def nearest_date_index(ds: xr.Dataset, date_str: str) -> int:
    target = np.datetime64(date_str, "ns")
    times = ds["time"].values.astype("datetime64[ns]")
    idx = int(np.argmin(np.abs(times - target)))
    return idx
