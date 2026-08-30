"""Preprocessing pipeline (F2) - each step is an isolated, testable function.

    1. load_inputs            (via src.data.loaders)
    2. quality_control        range checks + spike detection
    3. fill_missing           linear-in-time + bilinear-in-space gap fill,
                              land mask preserved/tracked separately
    4. interpolate_grid       regrid to exact 0.25-deg target grid
    5. to_daily               align to daily resolution
    6. normalization          z-score scalars persisted to JSON (inference parity)
    7. crop_region            geographic crop to the user selection

Orchestrator: preprocess_for_inference() -> model-ready tensor + masks + QC report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from ..config import depths as cfg_depths, get_config, input_variable_names
from .landmask import points_in_polygon
from .loaders import load_input_dataset, nearest_date_index

# Ocean-physics variables (as opposed to wind_u/wind_v, an atmospheric
# reanalysis with valid data over land too) - used to decide the output
# land/ocean mask. See preprocess_for_inference().
OCEAN_DOMAIN_VARS = {"sst", "sss", "sla", "cur_u", "cur_v"}


# ---------------------------------------------------------------------------
# Region geometry
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """User-selected region: GeoJSON polygon ring [[lon,lat], ...] (open or closed)."""
    ring: list[list[float]]
    lon_min: float = field(default=float("inf"))
    lon_max: float = field(default=float("-inf"))
    lat_min: float = field(default=float("inf"))
    lat_max: float = field(default=float("-inf"))

    def __post_init__(self):
        lons = [p[0] for p in self.ring]
        lats = [p[1] for p in self.ring]
        self.lon_min, self.lon_max = min(lons), max(lons)
        self.lat_min, self.lat_max = min(lats), max(lats)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.lon_min, self.lat_min, self.lon_max, self.lat_max


def parse_region(geojson: dict) -> Region:
    """Accept a GeoJSON Polygon Feature/geometry (or raw ring list)."""
    if geojson is None or not isinstance(geojson, dict):
        raise ValueError("region must be a GeoJSON Polygon object")
    geom = geojson.get("geometry", geojson)
    gtype = geom.get("type")
    if gtype == "FeatureCollection":
        feats = geom.get("features") or []
        if not feats:
            raise ValueError("empty region")
        geom = feats[0].get("geometry", feats[0])
        gtype = geom.get("type")
    if gtype == "Polygon":
        coords = geom["coordinates"][0]
    elif gtype == "MultiPolygon":
        coords = geom["coordinates"][0][0]
    elif gtype == "LineString":  # degenerate draw
        raise ValueError("region too small: draw a rectangle or closed polygon")
    elif "ring" in geom:  # internal convenience
        coords = geom["ring"]
    else:
        raise ValueError(f"unsupported region geometry type: {gtype}")
    ring = [[float(p[0]), float(p[1])] for p in coords]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    # drop consecutive duplicates
    dedup = [ring[0]]
    for p in ring[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    if len(dedup) < 3:
        raise ValueError("region too small: need at least 3 distinct vertices")
    return Region(ring=dedup)


def region_cell_mask(lats: np.ndarray, lons: np.ndarray, region: Region) -> np.ndarray:
    """Boolean (ny, nx): True where the cell centre lies inside the polygon."""
    lon2, lat2 = np.meshgrid(np.asarray(lons, float), np.asarray(lats, float))
    return points_in_polygon(lon2, lat2, region.ring)


# ---------------------------------------------------------------------------
# Steps 2-5
# ---------------------------------------------------------------------------

def quality_control(da: xr.DataArray, valid_min: float, valid_max: float,
                    spike_sigma: float = 6.0) -> tuple[xr.DataArray, dict]:
    """Range checks + temporal spike detection on one variable."""
    report = {"range_violations": 0, "spikes_removed": 0}
    vals = da.values
    bad = ~np.isfinite(vals) | (vals < valid_min) | (vals > valid_max)
    report["range_violations"] = int((~np.isfinite(vals)).sum())  # pre-existing NaNs
    vals = np.where((vals < valid_min) | (vals > valid_max), np.nan, vals)

    if vals.shape[0] >= 3:
        med = pd.DataFrame(vals.reshape(vals.shape[0], -1)) \
            .rolling(5, center=True, min_periods=2).median().to_numpy().reshape(vals.shape)
        resid = np.abs(vals - med)
        finite = np.isfinite(resid)
        # robust spread (MAD) so the outliers themselves don't hide the threshold
        mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid[finite])))) if finite.any() else 0.0
        sigma = 1.4826 * mad
        thresh = max(spike_sigma * sigma, 5.0 * _var_scale_floor(da.name))
        spikes = finite & (resid > thresh)
        report["spikes_removed"] = int(spikes.sum())
        vals[spikes] = np.nan
    report["total_invalid"] = int(~np.isfinite(vals).sum())
    out = da.copy(data=vals.astype(np.float32))
    return out, report


def _var_scale_floor(name: str) -> float:
    return {
        "sst": 1.0, "sss": 0.5, "sla": 0.15,
        "cur_u": 0.4, "cur_v": 0.4, "wind_u": 5.0, "wind_v": 5.0,
    }.get(name, 1.0)


def fill_missing_spatial(arr: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                         land_mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Bilinear (linear Delaunay) interpolation over gaps, nearest fallback
    for points outside the convex hull of valid data (e.g. land-adjacent).
    Land cells are never treated as source data."""
    ny, nx = arr.shape
    la = np.asarray(lats, float); lo = np.asarray(lons, float)
    valid = np.isfinite(arr) if land_mask is None else (np.isfinite(arr) & ~land_mask)
    if not valid.any():
        return arr.astype(np.float32), 0
    if valid.all():
        return arr.astype(np.float32), 0

    yy, xx = np.meshgrid(la, lo, indexing="ij")
    pts = np.column_stack([yy[valid], xx[valid]])
    vals = arr[valid].astype(np.float64)
    query = np.column_stack([yy.ravel(), xx.ravel()])

    est = griddata(pts, vals, query, method="linear").reshape(ny, nx)
    still = ~np.isfinite(est) & ~valid
    if still.any():
        est[still] = griddata(pts, vals,
                              np.column_stack([yy[still], xx[still]]),
                              method="nearest")
    filled = int((~valid & np.isfinite(est) &
                  ~(land_mask.astype(bool) if land_mask is not None else False)).sum())
    return est.astype(np.float32), filled


def fill_missing(ds: xr.Dataset, lats=None, lons=None) -> tuple[xr.Dataset, dict]:
    """Gap-filling: linear in time, bilinear in space. Land mask is tracked,
    never 'filled' into the output mask (it is returned separately)."""
    report = {}
    lats = np.asarray(lats if lats is not None else ds["lat"].values, float)
    lons = np.asarray(lons if lons is not None else ds["lon"].values, float)
    out = {}
    for name in ds.data_vars:
        da = ds[name]
        arr = da.values.astype(np.float32)
        # land = NaN at every time step; gaps = transient NaN
        land = np.isfinite(arr).sum(axis=0) == 0 if arr.ndim == 3 else \
            (np.isfinite(arr).sum(axis=0) == 0)[0]
        # 1. temporal linear fill on ocean cells
        if arr.ndim == 3 and arr.shape[0] > 1:
            df = pd.DataFrame(arr.reshape(arr.shape[0], -1))
            df = df.interpolate(axis=0, limit_direction="both")
            arr_t = df.to_numpy().reshape(arr.shape)
        else:
            arr_t = arr
        gaps_time = int(((~np.isfinite(arr)) & np.isfinite(arr_t)).sum())
        # 2. spatial bilinear fill per step (skips land)
        filled_total = 0
        if arr_t.ndim == 3:
            for t in range(arr_t.shape[0]):
                arr_t[t], n = fill_missing_spatial(arr_t[t], lats, lons, land)
                filled_total += n
        else:
            arr_t, filled_total = fill_missing_spatial(arr_t, lats, lons, land)
        out[name] = xr.DataArray(
            arr_t.astype(np.float32), dims=da.dims, coords=da.coords, name=name
        )
        report[name] = {
            "land_cells": int(np.asarray(land).sum()),
            "gaps_filled_in_time": gaps_time,
            "spatial_cells_filled": filled_total,
        }
    filled_ds = xr.Dataset(out)
    filled_ds.attrs["land_mask"] = json.dumps(
        np.asarray(land).astype(np.int8).tolist()
    ) if np.asarray(land).size < 40000 else ""
    return filled_ds, report


def interpolate_grid(ds: xr.Dataset, target_lat: np.ndarray,
                     target_lon: np.ndarray) -> xr.Dataset:
    """Regrid onto the exact target grid (bilinear via xarray/scipy)."""
    cur_lat = np.asarray(ds["lat"].values, float)
    cur_lon = np.asarray(ds["lon"].values, float)
    same = (
        len(cur_lat) == len(target_lat)
        and len(cur_lon) == len(target_lon)
        and np.allclose(cur_lat, target_lat, atol=1e-5)
        and np.allclose(cur_lon, target_lon, atol=1e-5)
    )
    if same:
        return ds
    # fill_value=nan (never "extrapolate"): live-fetched provider grids don't
    # always cover the domain edges exactly (e.g. Copernicus Marine's native
    # 0.0833deg grid doesn't land exactly on lat=30.0), and extrapolating
    # fabricates a full row/column of invented values there - including over
    # land, since linear extrapolation has no concept of land/no-data. NaN
    # lets the normal land-aware gap-fill step handle any real edge gaps
    # instead of silently inventing data - consistent with this pipeline's
    # "never show a value we can't stand behind" rule (see OCEAN_DOMAIN_VARS
    # above).
    return ds.interp(lat=target_lat, lon=target_lon, method="linear",
                     kwargs={"fill_value": np.nan})


def to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Align to daily resolution (interpolating weekly/monthly sources)."""
    t = pd.to_datetime(ds["time"].values)
    if len(t) > 1 and (t[1] - t[0]) == pd.Timedelta(days=1) and \
       bool((np.diff(t.asi8()) == pd.Timedelta(days=1).value).all()):
        return ds
    return ds.resample(time="1D").interpolate("linear")


# ---------------------------------------------------------------------------
# Step 6 - normalization scalars
# ---------------------------------------------------------------------------

def compute_scalars(inputs: xr.Dataset, temperature: xr.DataArray | None = None) -> dict:
    scalars = {}
    for name in inputs.data_vars:
        v = inputs[name].values
        m = float(np.nanmean(v))
        s = float(np.nanstd(v))
        scalars[name] = {"mean": m, "std": s if s > 1e-6 else 1.0}
    if temperature is not None:
        tv = np.asarray(temperature.values, dtype=np.float64)
        axes = (0, 2, 3) if tv.ndim == 4 else (0, 1, 2)
        m = np.nanmean(tv, axis=axes).tolist()
        s = np.nanstd(tv, axis=axes).tolist()
        scalars["temperature"] = {
            "mean": [float(x) for x in m],
            "std": [float(max(si, 1e-4)) for si in s],
        }
    return scalars


def save_scalars(scalars: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(scalars, indent=2))


def load_scalars(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize(arr: np.ndarray, scalar: dict) -> np.ndarray:
    std = float(scalar.get("std", 1.0))
    return ((arr - float(scalar.get("mean", 0.0))) / (std if std > 1e-6 else 1.0)).astype(np.float32)


def denormalize(arr: np.ndarray, scalar: dict) -> np.ndarray:
    std = scalar.get("std", 1.0)
    mean = scalar.get("mean", 0.0)
    if isinstance(std, (list, tuple)):
        std = np.asarray(std, dtype=arr.dtype).reshape(-1, *([1] * (arr.ndim - 1)))
        mean = np.asarray(mean, dtype=arr.dtype).reshape(-1, *([1] * (arr.ndim - 1)))
    return (np.asarray(arr) * std + mean)


# ---------------------------------------------------------------------------
# Step 7 + orchestration
# ---------------------------------------------------------------------------

def stack_variables(ds: xr.Dataset, order: list[str]) -> xr.DataArray:
    """Stack variables into a DataArray with dims (time, lat, lon, variable)."""
    ordered = ds[order]
    da = ordered.to_array(dim="variable").transpose("time", "lat", "lon", "variable")
    return da


def crop_region(ds: xr.Dataset, region: Region) -> xr.Dataset:
    return ds.sel(
        lat=slice(region.lat_min, region.lat_max),
        lon=slice(region.lon_min, region.lon_max),
    )


def preprocess_for_inference(cfg: dict, date_str: str, region: Region,
                             scalars: dict | None = None,
                             apply_qc: bool = True,
                             ds: xr.Dataset | None = None) -> dict:
    """Full pipeline for one date + region -> normalized (7,H,W) tensor.

    Returns dict with keys: tensor, ocean_mask (True=ocean), lats, lons,
    qc_report, date_resolved, variable_order.
    """
    order = input_variable_names(cfg)
    specs = {v["name"]: v for v in cfg["input_variables"]}
    res = float(cfg["grid"]["resolution"])

    ds = ds if ds is not None else load_input_dataset(cfg)
    tidx = nearest_date_index(ds, date_str)
    resolved = str(np.datetime_as_string(np.datetime64(ds["time"].values[tidx], "D"), unit="D"))

    crop = crop_region(ds, region)
    if crop.sizes.get("lat", 0) == 0 or crop.sizes.get("lon", 0) == 0:
        raise ValueError("selected region does not intersect the data domain")

    lats = crop["lat"].values.astype(float)
    lons = crop["lon"].values.astype(float)

    # exact-grid enforcement (no-op when already aligned)
    target_lat = np.arange(5.0, 30.0 + 1e-9, res)
    target_lat = target_lat[(target_lat >= lats.min() - 1e-6) & (target_lat <= lats.max() + 1e-6)]
    target_lon = np.arange(45.0, 105.0 + 1e-9, res)
    target_lon = target_lon[(target_lon >= lons.min() - 1e-6) & (target_lon <= lons.max() + 1e-6)]
    if len(target_lat) and len(target_lon):
        crop = interpolate_grid(crop, target_lat, target_lon)
        lats = crop["lat"].values.astype(float)
        lons = crop["lon"].values.astype(float)

    qc = {"variables": {}, "date_requested": date_str, "date_resolved": resolved}
    chans = []
    ocean_land_flags = []  # land_local from OCEAN-domain variables only (see below)
    for name in order:
        da = crop[name].isel(time=tidx) if "time" in crop[name].dims else crop[name]
        arr = np.asarray(da.values, dtype=np.float32)
        spec = specs[name]
        n_bad_before = int((~np.isfinite(arr)).sum())
        if apply_qc:
            arr_qc, rep = quality_control(da.expand_dims("time"), spec["valid_min"],
                                          spec["valid_max"])
            arr = np.asarray(arr_qc.isel(time=0).values, dtype=np.float32)
            rep.pop("range_violations", None)
        else:
            arr = np.where((arr < spec["valid_min"]) | (arr > spec["valid_max"]),
                           np.nan, arr)
            rep = {}
        # land = permanently-missing cells (per this variable's own domain)
        land_local = ~_ocean_from_domain(ds, name, lats, lons, tidx)
        arr, n_filled = fill_missing_spatial(arr, lats, lons, land_local)
        if name in OCEAN_DOMAIN_VARS:
            ocean_land_flags.append(land_local)
        qc["variables"][name] = {
            "missing_before": n_bad_before,
            "filled_or_interpolated": n_filled,
            **rep,
        }
        sc = (scalars or {}).get(name, {"mean": 0.0, "std": 1.0})
        chans.append(normalize(arr, sc))

    tensor = np.stack(chans, axis=0)  # (7,H,W)
    # Output land/ocean mask must come from OCEAN variables (sst/sss/sla/
    # currents), never wind: ERA5 winds are an atmospheric reanalysis with
    # valid data over land too, so in live mode wind's own "land" flag is
    # ~always False - using it (e.g. by taking whichever variable happened
    # to be last in the loop) would predict "ocean" temperatures over land.
    # A cell counts as land here if ANY ocean-domain variable lacks data
    # there (conservative: never show a prediction we can't stand behind).
    land_local = np.logical_or.reduce(ocean_land_flags) if ocean_land_flags else land_local
    ocean_mask = ~land_local
    return {
        "tensor": tensor.astype(np.float32),
        "ocean_mask": ocean_mask,
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "qc_report": qc,
        "variable_order": order,
        "depths": cfg_depths(cfg),
    }


_OCEAN_CACHE: dict = {}


def _ocean_from_domain(ds: xr.Dataset, var: str, lats, lons, tidx: int) -> np.ndarray:
    """Ocean mask (True=ocean) for the crop, derived from the full-domain mask.

    A cell is 'ocean' if it is finite at ANY of a few sampled time steps
    (robust against single-date dropouts in real data)."""
    key = (id(ds), var)
    dom = _OCEAN_CACHE.get(key)
    if dom is None:
        da = ds[var]
        if "time" in da.dims:
            nt = da.sizes["time"]
            idx = np.unique(np.linspace(0, nt - 1, min(nt, 5)).astype(int))
            sub = np.asarray(da.isel(time=idx).values)   # backend reads ~5 planes
            dom = np.isfinite(sub).any(axis=0)
        else:
            dom = np.isfinite(np.asarray(da.values))
        _OCEAN_CACHE[key] = dom
    la = np.asarray(ds["lat"].values, float); lo = np.asarray(ds["lon"].values, float)
    iy = np.searchsorted(la, np.asarray(lats)); ix = np.searchsorted(lo, np.asarray(lons))
    iy = np.clip(iy, 0, dom.shape[0] - 1); ix = np.clip(ix, 0, dom.shape[1] - 1)
    return dom[np.ix_(iy, ix)]
