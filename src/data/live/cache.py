"""On-disk cache for live-fetched surface variables.

Layout (under data_source.live.cache_dir, default data/live_cache):
    <var>.nc         rolling window of the last N fetched days for one
                     variable, dims (time, lat, lon)
    manifest.json    per-variable status snapshot - read by /api/live/status

Only ever imported by the main app (never by scripts/live_fetch/*.py, which
run in the isolated .venv-live and write plain single-timestep NetCDF files
that this module then merges in).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr


def cache_dir(cfg: dict) -> Path:
    live = cfg["data_source"].get("live", {})
    p = Path(live.get("cache_dir", "data/live_cache"))
    if not p.is_absolute():
        p = Path(cfg["_root"]) / p
    return p


def manifest_path(cfg: dict) -> Path:
    return cache_dir(cfg) / "manifest.json"


def load_manifest(cfg: dict) -> dict:
    p = manifest_path(cfg)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(cfg: dict, manifest: dict) -> None:
    d = cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path(cfg).with_suffix(".tmp.json")
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp, manifest_path(cfg))


def variable_cache_path(cfg: dict, name: str) -> Path:
    return cache_dir(cfg) / f"{name}.nc"


def append_timestep(cfg: dict, name: str, da: xr.DataArray, max_timesteps: int = 14) -> None:
    """Merge one new (time, lat, lon) slice into the rolling cache file for
    `name`. Same-day duplicates are replaced by the newer fetch; only the
    most recent `max_timesteps` are kept (bounded disk use)."""
    d = cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = variable_cache_path(cfg, name)

    if "time" not in da.dims:
        da = da.expand_dims("time")

    if path.exists():
        with xr.open_dataarray(path) as existing:
            combined = xr.concat([existing.load(), da], dim="time")
    else:
        combined = da

    _, first_idx = np.unique(combined["time"].values, return_index=True)
    combined = combined.isel(time=np.sort(first_idx))
    if combined.sizes["time"] > max_timesteps:
        combined = combined.isel(time=slice(-max_timesteps, None))

    tmp = path.with_suffix(".tmp.nc")
    combined.to_dataset(name=name).to_netcdf(tmp)
    combined.close() if hasattr(combined, "close") else None
    os.replace(tmp, path)


def read_variable_cache(cfg: dict, name: str) -> xr.DataArray | None:
    """Return the cached (time, lat, lon) array for `name`, or None if
    nothing has ever been fetched/synthesized for it yet."""
    path = variable_cache_path(cfg, name)
    if not path.exists():
        return None
    with xr.open_dataarray(path) as da:
        return da.load()
