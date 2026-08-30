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

from ..netcdf_lock import NETCDF_IO_LOCK as _IO_LOCK


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


def append_timestep(cfg: dict, name: str, da: xr.DataArray, max_timesteps: int = 14,
                    source: str = "real") -> None:
    """Merge one new (time, lat, lon) slice into the rolling cache file for
    `name`. Same-day duplicates are replaced by the newer fetch; only the
    most recent `max_timesteps` are kept (bounded disk use).

    `source` ("real" or "synthetic") is persisted per-timestep so callers
    picking "the latest day" can prefer a real entry over a same-or-later
    dated synthetic-fallback one - see manager.py's _latest_real_only()."""
    d = cache_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    path = variable_cache_path(cfg, name)

    if "time" not in da.dims:
        da = da.expand_dims("time")
    da = da.assign_coords(source=("time", [source] * da.sizes["time"]))

    with _IO_LOCK:
        if path.exists():
            with xr.open_dataarray(path) as existing:
                existing = existing.load()
                if "source" not in existing.coords:  # cache written before this field existed
                    existing = existing.assign_coords(
                        source=("time", ["real"] * existing.sizes["time"]))
                # new fetch (da) FIRST: np.unique(..., return_index=True) below
                # keeps the first occurrence of each duplicate time value, so
                # ordering here is what actually makes "newer fetch wins" on a
                # same-day re-fetch - existing-first would silently keep the
                # stale entry instead (this was backwards before).
                combined = xr.concat([da, existing], dim="time")
        else:
            combined = da

        _, first_idx = np.unique(combined["time"].values, return_index=True)
        # first_idx indexes into `combined` in concat order (new fetch first),
        # not chronological order - sort by the actual time values, not by
        # index position, so "latest" (isel(time=-1) elsewhere) stays correct
        # regardless of which side of the concat a given day came from.
        combined = combined.isel(time=first_idx).sortby("time")
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
    with _IO_LOCK, xr.open_dataarray(path) as da:
        return da.load()


def clear_cache(cfg: dict) -> list[str]:
    """Delete every cached variable file (and any leftover .tmp.nc from an
    interrupted write) plus the manifest. Returns the variable names that
    had a cache file removed. Safe to call while a background refresh is
    in flight - append_timestep() recreates a variable's file from scratch
    the next time it writes, same as if it had never been fetched."""
    d = cache_dir(cfg)
    removed: list[str] = []
    with _IO_LOCK:
        if d.exists():
            for p in d.glob("*.nc"):
                p.unlink(missing_ok=True)
                removed.append(p.stem)
            for p in d.glob("*.tmp.nc"):
                p.unlink(missing_ok=True)
            manifest_path(cfg).unlink(missing_ok=True)
    return removed
