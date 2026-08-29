"""SubOceanNet FastAPI backend (F5).

Run:  uvicorn backend.main:app --port 8000   (from the project root)

All endpoints are JSON and documented at /docs.
Long predictions run as background jobs polled via GET /api/jobs/{id}.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date as dt_date
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import bundle_dir, depths as cfg_depths, get_config, input_variable_names
from src.data import loaders as data_loaders
from src.data.landmask import ISLAND_POINTS, coastlines_geojson
from src.data.netcdf_lock import NETCDF_IO_LOCK
from src.data.preprocessing import Region, parse_region, region_cell_mask
from src.inference import Predictor, json_safe

# ---------------------------------------------------------------------------
# Job store + training state
# ---------------------------------------------------------------------------

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=2)
TRAIN_STATE: dict[str, Any] = {"status": "idle", "progress": 0, "message": ""}
JOB_TTL_S = 1800


def _set_job(job_id: str, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw, updated_at=time.time())


def _cleanup_jobs():
    now = time.time()
    with JOBS_LOCK:
        for jid in [j for j, v in JOBS.items() if now - v.get("updated_at", 0) > JOB_TTL_S]:
            JOBS.pop(jid, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_bootstrap, name="oe-bootstrap", daemon=True)
    t.start()
    if get_config()["data_source"]["type"] == "live":
        from src.data.live.manager import get_manager
        get_manager(get_config()).start_background_refresh()
    yield


def _bootstrap():
    """First-run convenience: generate synthetic data + train a demo model."""
    cfg = get_config()
    try:
        if not Predictor.available(cfg):
            from src.data.loaders import synthetic_dir
            need_data = not (synthetic_dir(cfg) / "inputs.nc").exists() \
                and cfg["data_source"]["type"] == "synthetic"
            TRAIN_STATE.update(status="training", progress=0,
                               message="preparing demo environment")
            if need_data:
                def gcb(p, m):
                    TRAIN_STATE.update(progress=int(p * 0.5), message=f"generating dataset: {m}")
                from src.data.generate_synthetic import generate_with_config
                generate_with_config(cfg, progress_cb=gcb)
            def tcb(p, m):
                TRAIN_STATE.update(progress=50 + int(p * 0.5), message=f"training demo model: {m}")
            from src.train import run_training
            run_training(cfg, auto=True, quiet=True, progress_cb=tcb)
            Predictor.reset()
            TRAIN_STATE.update(status="ready", progress=100, message="demo model ready")
        else:
            Predictor.get(cfg)  # warm load
            TRAIN_STATE.update(status="ready", progress=100, message="model loaded")
    except Exception as exc:  # pragma: no cover
        traceback.print_exc()
        TRAIN_STATE.update(status="error", progress=0, message=str(exc))


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    date: str = Field(..., description="ISO date, e.g. 2023-06-15")
    region_geojson: dict = Field(..., description="GeoJSON Polygon selection")
    mc_passes: int | None = Field(None, ge=1, le=32)


class JobOut(BaseModel):
    job_id: str
    status: str
    stage: str = ""
    progress: int = 0


app = FastAPI(
    title="SubOceanNet API",
    description=(
        "Reconstructs 3D subsurface ocean temperature in the North Indian Ocean "
        "from 7 satellite surface observations via an AI encoder-decoder. "
        "Pipeline: INPUT VARIABLES -> AI EMBEDDING -> TEMPERATURE PREDICTION."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
_DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173",
                         "http://localhost:4173", "http://127.0.0.1:4173"]
_cors_env = os.environ.get("SUBOCEANNET_CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or _DEFAULT_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def abort(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def require_model() -> Predictor:
    if TRAIN_STATE["status"] == "error":
        abort(503, f"model unavailable: {TRAIN_STATE['message']}")
    if not Predictor.available():
        abort(503, "model is warming up - the demo model is training on first run; "
                   "retry shortly (see GET /api/model/info)")
    return Predictor.get()


def valid_region(geojson: dict, cfg=None, enforce_cap: bool = True) -> tuple[Region, np.ndarray]:
    try:
        region = parse_region(geojson)
    except ValueError as e:
        abort(400, str(e))
    lat = np.arange(cfg["region"]["lat_min"], cfg["region"]["lat_max"] + 1e-9,
                    float(cfg["grid"]["resolution"]))
    lon = np.arange(cfg["region"]["lon_min"], cfg["region"]["lon_max"] + 1e-9,
                    float(cfg["grid"]["resolution"]))
    inside_bbox = region.lat_min >= lat.min() - 1e-6 and region.lon_min >= lon.min() - 1e-6 \
        and region.lat_max <= lat.max() + 1e-6 and region.lon_max <= lon.max() + 1e-6
    lats = lat[(lat >= region.lat_min) & (lat <= region.lat_max)]
    lons = lon[(lon >= region.lon_min) & (lon <= region.lon_max)]
    if len(lats) < 2 or len(lons) < 2:
        abort(400, "selected region is too small (need at least a 2x2 cell block)")
    mask = region_cell_mask(lats, lons, region)
    n_cells = int(mask.sum())
    cap = int(cfg["inference"].get("max_region_cells", 10000))
    if enforce_cap and n_cells > cap:
        abort(400, f"selected region has {n_cells} cells; limit is {cap} - "
                   f"please select a smaller area")
    return region, mask


def valid_date(date_str: str | None, cfg=None) -> str:
    start, end = data_loaders.available_date_range(cfg or get_config())
    if date_str is None or date_str == "":
        return end
    try:
        d = dt_date.fromisoformat(date_str)
    except ValueError:
        abort(400, f"invalid date '{date_str}' - expected YYYY-MM-DD")
    if not (dt_date.fromisoformat(start) <= d <= dt_date.fromisoformat(end)):
        abort(400, f"date {date_str} outside available range [{start} .. {end}]")
    return date_str


_DS_CACHE: dict = {}


def input_dataset(cfg):
    key = cfg["data_source"]["type"]
    if key == "live":
        return data_loaders.load_input_dataset(cfg)  # refreshed in the background; never cache
    ds = _DS_CACHE.get(key)
    if ds is None:
        ds = data_loaders.load_input_dataset(cfg)
        _DS_CACHE[key] = ds
    return ds


# ---------------------------------------------------------------------------
# Meta / health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.get("/api/meta")
def meta():
    cfg = get_config()
    start, end = data_loaders.available_date_range(cfg)
    res = float(cfg["grid"]["resolution"])
    return {
        "project": cfg["project"],
        "version": cfg["version"],
        "bounds": {"lat_min": cfg["region"]["lat_min"], "lat_max": cfg["region"]["lat_max"],
                   "lon_min": cfg["region"]["lon_min"], "lon_max": cfg["region"]["lon_max"]},
        "grid": {"resolution": res,
                 "ny": int(round((cfg["region"]["lat_max"] - cfg["region"]["lat_min"]) / res)) + 1,
                 "nx": int(round((cfg["region"]["lon_max"] - cfg["region"]["lon_min"]) / res)) + 1},
        "depths": cfg_depths(cfg),
        "variables": [
            {"name": v["name"], "long_name": v["long_name"], "units": v["units"],
             "valid_min": v["valid_min"], "valid_max": v["valid_max"]}
            for v in cfg["input_variables"]
        ],
        "date_range": [start, end],
        "presets": cfg["presets"],
        "mc_passes": int(cfg["inference"]["mc_passes"]),
        "max_region_cells": int(cfg["inference"]["max_region_cells"]),
        "data_source": cfg["data_source"]["type"],
        "demo_mode": cfg["data_source"]["type"] == "synthetic",
        "land_polygons": coastlines_geojson(),
        "islands": [{"lat": la, "lon": lo, "name": nm} for lo, la, nm in ISLAND_POINTS],
        "model_status": TRAIN_STATE["status"],
    }


@app.get("/api/model/info")
def model_info():
    cfg = get_config()
    bdir = bundle_dir(cfg)
    info: dict[str, Any] = {
        "status": TRAIN_STATE["status"],
        "progress": TRAIN_STATE["progress"],
        "message": TRAIN_STATE["message"],
        "checkpoint_path": str(bdir),
        "available": Predictor.available(cfg),
    }
    meta_path = bdir / "metadata.json"
    if meta_path.exists():
        md = json.loads(meta_path.read_text())
        info.update({
            "version": md.get("model_version"),
            "encoder_type": md.get("encoder_type"),
            "architecture_summary": md.get("architecture_summary"),
            "parameters": md.get("parameters"),
            "depths_m": md.get("depths_m"),
            "input_variables": md.get("input_variables"),
            "trained_range": md.get("trained_range"),
            "val_rmse_degC": md.get("val_rmse_degC"),
            "trained_at": md.get("trained_at"),
            "demo_mode": md.get("demo_mode"),
            "history": md.get("history"),
        })
    info["outputs"] = ["temperature @ " + ",".join(str(z) for z in cfg_depths(cfg)) + " m"]
    return json.loads(json.dumps(info, default=str))


# ---------------------------------------------------------------------------
# Live data pipeline
# ---------------------------------------------------------------------------

@app.get("/api/live/status")
def live_status():
    """Per-variable live-data health: REAL / CACHED REAL / SYNTHETIC FALLBACK,
    provider, observation time, last fetch attempt, data age, and any error.
    Meaningful in any data_source.type - it reports live_mode_active: false
    (with an empty variables map) when live mode isn't the active source."""
    from src.data.live.manager import status_payload

    cfg = get_config()
    return json_safe(status_payload(cfg))


@app.get("/api/live/latest")
def live_latest():
    """The latest calendar day common to all 7 live variables, plus a quick
    per-variable value summary and status. Only meaningful in live mode;
    returns 409 otherwise so the frontend can distinguish "not live" from
    "live but not ready yet"."""
    cfg = get_config()
    if cfg["data_source"]["type"] != "live":
        abort(409, "data_source.type is not 'live' - this endpoint has nothing to report")
    from src.data.live.manager import latest_payload

    return json_safe(latest_payload(cfg))


# ---------------------------------------------------------------------------
# Input layers
# ---------------------------------------------------------------------------

def _parse_json_param(raw: str | None, name: str) -> dict | None:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON in '{name}': {e}")

@app.get("/api/layers/{variable}")
def layer(variable: str, date: str | None = Query(None), region: str | None = Query(None)):
    """Sparse grid raster for one surface input variable."""
    cfg = get_config()
    names = input_variable_names(cfg)
    specs = {v["name"]: v for v in cfg["input_variables"]}
    if variable not in names:
        abort(404, f"unknown variable '{variable}'; available: {names}")
    d = valid_date(date, cfg)

    reg = None
    if region:
        reg, _ = valid_region(_parse_json_param(region, "region"), cfg, enforce_cap=False)
    try:
        ds = input_dataset(cfg)
    except (FileNotFoundError, ValueError) as e:
        abort(503, f"input data not ready: {e}")
    da = ds[variable]
    tidx = data_loaders.nearest_date_index(ds, d)
    arr = da.isel(time=tidx)
    if reg is not None:
        arr = arr.sel(lat=slice(reg.lat_min, reg.lat_max), lon=slice(reg.lon_min, reg.lon_max))
    vals = np.asarray(arr.values, dtype=np.float64)

    # decimate huge selections
    ny, nx = vals.shape
    sy = max(1, int(np.ceil(ny / 220)))
    sx = max(1, int(np.ceil(nx / 220)))
    vals_d = vals[::sy, ::sx]
    lats = np.asarray(arr["lat"].values)[::sy].astype(float)
    lons = np.asarray(arr["lon"].values)[::sx].astype(float)

    finite = np.isfinite(vals)
    spec = specs[variable]
    if finite.any():
        lo_p, hi_p = np.percentile(vals[finite], [2, 98])
    else:
        lo_p, hi_p = spec["valid_min"], spec["valid_max"]

    payload = {
        "variable": variable,
        "long_name": spec["long_name"],
        "units": spec["units"],
        "date": str(np.datetime_as_string(np.datetime64(ds["time"].values[tidx], "D"), unit="D")),
        "source": cfg["data_source"]["type"],
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "values": [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                   for row in vals_d],
        "legend": {"min": float(lo_p), "max": float(hi_p)},
        "stage": "input",
    }
    return json_safe(payload)


# ---------------------------------------------------------------------------
# Prediction (job based)
# ---------------------------------------------------------------------------

@app.post("/api/predict", response_model=JobOut)
def predict(req: PredictRequest):
    """Start an async prediction job. Poll GET /api/jobs/{job_id}."""
    cfg = get_config()
    predictor = require_model()
    region, mask = valid_region(req.region_geojson, cfg)
    valid_date(req.date, cfg)

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"job_id": job_id, "status": "queued", "stage": "queued",
                        "progress": 0, "created_at": time.time(), "updated_at": time.time()}
    _cleanup_jobs()

    def run():
        _set_job(job_id, status="running", stage="preprocessing", progress=8)
        try:
            def cb(pct, msg):
                stage = ("embedding" if pct < 90 else "decoding")
                _set_job(job_id, progress=min(int(pct), 99), stage=stage, message=msg)
            result = predictor.predict_region(req.date, region, mc_passes=req.mc_passes,
                                              progress_cb=cb)
            result["region_cells_selected"] = int(mask.sum())
            _set_job(job_id, status="done", stage="done", progress=100, result=result)
        except HTTPException as e:
            _set_job(job_id, status="error", stage="error", message=str(e.detail))
        except Exception as e:
            traceback.print_exc()
            _set_job(job_id, status="error", stage="error", message=f"{type(e).__name__}: {e}")

    EXECUTOR.submit(run)
    return JobOut(job_id=job_id, status="queued", stage="queued", progress=0)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            abort(404, f"unknown job '{job_id}'")
        out = {k: v for k, v in job.items() if k != "result"}
        if job.get("status") == "done":
            out["result"] = job["result"]
        return out


# ---------------------------------------------------------------------------
# Profile / timeseries
# ---------------------------------------------------------------------------

def _check_point(lat: float, lon: float, cfg):
    r = cfg["region"]
    if not (r["lat_min"] <= lat <= r["lat_max"] and r["lon_min"] <= lon <= r["lon_max"]):
        abort(400, f"point ({lat}, {lon}) outside domain "
                   f"[{r['lat_min']}..{r['lat_max']}N, {r['lon_min']}..{r['lon_max']}E]")


@app.get("/api/profile")
def profile(lat: float, lon: float, date: str | None = Query(None)):
    cfg = get_config()
    predictor = require_model()
    _check_point(lat, lon, cfg)
    d = valid_date(date, cfg)
    try:
        res = predictor.predict_points([(lat, lon)], d)
    except (FileNotFoundError, ValueError) as e:
        abort(503, f"input data not ready: {e}")
    pt = res["points"][0]

    # reference overlay (clean truth / GLORYS)
    ref_out, ref_label = None, None
    try:
        ref_ds, ref_label = data_loaders.load_reference_dataset(cfg)
        with NETCDF_IO_LOCK:  # ref_ds is lazy - this is where the actual read happens
            tidx = data_loaders.nearest_date_index(ref_ds, res["date"])
            da = ref_ds["temperature"].isel(time=tidx).sel(
                lat=float(pt["snapped_lat"]), lon=float(pt["snapped_lon"]), method="nearest")
            ref_vals = np.asarray(da.values, dtype=np.float64)
        ref_out = [None if not np.isfinite(v) else round(float(v), 3) for v in ref_vals]
    except FileNotFoundError as e:
        ref_label = None
    except Exception:
        ref_out, ref_label = None, None

    return {
        "lat": pt["lat"], "lon": pt["lon"],
        "snapped_lat": pt["snapped_lat"], "snapped_lon": pt["snapped_lon"],
        "date": res["date"], "depths": res["depths"], "units": "degC",
        "temperature": pt["temperature"], "uncertainty": pt["uncertainty"],
        "reference_temperature": ref_out,
        "reference_type": ref_label,
        "stage": "prediction",
    }


@app.get("/api/timeseries")
def timeseries(lat: float = Query(None), lon: float = Query(None),
               depth: float = Query(...), start: str | None = Query(None),
               end: str | None = Query(None), stride_days: int = Query(10, ge=1, le=60),
               region: str | None = Query(None)):
    """Predicted temperature through time at a point (or polygon mean)."""
    cfg = get_config()
    predictor = require_model()
    rng = cfg["time_range"]
    start = valid_date(start or rng["start"], cfg)
    end = valid_date(end or rng["end"], cfg)
    if dt_date.fromisoformat(start) > dt_date.fromisoformat(end):
        abort(400, "start date is after end date")

    reg = None
    if region:
        reg, _ = valid_region(_parse_json_param(region, "region"), cfg)
        if lat is None or lon is None:
            lat, lon = float(np.mean([reg.lat_min, reg.lat_max])), \
                float(np.mean([reg.lon_min, reg.lon_max]))
    if lat is None or lon is None:
        abort(400, "provide lat/lon (or a region for area-mean series)")
    _check_point(lat, lon, cfg)

    prog = {"p": 0}

    def cb(p, m):
        prog["p"] = p

    try:
        ts = predictor.timeseries(lat, lon, start, end, stride_days=stride_days,
                                  region=reg, progress_cb=cb)
    except (FileNotFoundError, ValueError) as e:
        abort(503, f"input data not ready: {e}")
    temps = np.asarray([[np.nan if v is None else v for v in row] for row in ts["temperature"]],
                       dtype=np.float64)
    try:
        k = int(np.argmin(np.abs(np.asarray(ts["depths"]) - float(depth))))
    except (TypeError, ValueError):
        abort(400, f"invalid depth '{depth}'")

    series = [None if not np.isfinite(v) else round(float(v), 3) for v in temps[:, k]]

    # reference overlay
    ref_series, ref_label = None, None
    try:
        ref_ds, ref_label = data_loaders.load_reference_dataset(cfg)
        refs = []
        with NETCDF_IO_LOCK:  # ref_ds is lazy - this is where the actual reads happen
            times = ref_ds["time"].values.astype("datetime64[D]")
            for ds_ in ts["dates"]:
                target = np.datetime64(ds_, "D")
                tidx = int(np.argmin(np.abs(times - target)))
                da = ref_ds["temperature"].isel(time=tidx).isel(depth=k)
                if reg is not None:
                    da = da.sel(lat=slice(reg.lat_min, reg.lat_max),
                                lon=slice(reg.lon_min, reg.lon_max))
                    v = float(np.nanmean(np.asarray(da.values, dtype=np.float64)))
                else:
                    v = float(da.sel(lat=lat, lon=lon, method="nearest").values)
                refs.append(round(v, 3))
        ref_series, ref_label = refs, ref_label
    except Exception:
        ref_series, ref_label = None, None

    return {
        "lat": lat, "lon": lon, "depth_m": ts["depths"][k], "area_mean": bool(reg is not None),
        "dates": ts["dates"], "temperature": series,
        "reference_temperature": ref_series, "reference_type": ref_label,
        "units": "degC", "all_depths_matrix": ts["temperature"],
        "stage": "prediction",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@app.get("/api/validation")
def validation(region: str = Query(..., description="GeoJSON Polygon (URL-encoded JSON)"),
               date: str | None = Query(None), start: str | None = Query(None),
               end: str | None = Query(None), scatter_depth: float = Query(100)):
    """Metrics vs reference over region/time; JSON-cached by (dates, region)."""
    from src.validation.metrics import (band_skill, cache_key, load_cache,
                                        per_depth_metrics, save_cache, scatter_samples)

    cfg = get_config()
    predictor = require_model()
    reg, mask = valid_region(_parse_json_param(region, "region"), cfg, enforce_cap=False)
    avail_start, avail_end = data_loaders.available_date_range(cfg)
    start = valid_date(start or date or avail_end, cfg)
    end = valid_date(end or start, cfg)
    if dt_date.fromisoformat(start) > dt_date.fromisoformat(end):
        abort(400, "start date is after end date")

    key = cache_key({"r": sorted(map(tuple, reg.ring)), "s": start, "e": end,
                     "v": predictor.meta.get("model_version")})
    cached = load_cache(cfg["paths"]["cache_dir"], key)
    if cached:
        return cached

    # sample up to 8 days across the window (single MC pass each for speed)
    dr = pd.date_range(start, end, freq="D")
    if len(dr) > 8:
        dr = dr[np.unique(np.linspace(0, len(dr) - 1, 8).astype(int))]
    preds, refs = [], []
    ref_ds, ref_label = data_loaders.load_reference_dataset(cfg)
    with NETCDF_IO_LOCK:  # ref_ds is lazy - just the initial coordinate read
        ref_times = ref_ds["time"].values.astype("datetime64[D]")
    for d in dr:
        ds_ = d.strftime("%Y-%m-%d")
        # NOT locked: predict_region() does its own model inference (slow,
        # CPU-bound) plus its own brief internal NETCDF_IO_LOCK use for live
        # mode (see src/data/live/cache.py) - holding this loop's lock across
        # it would needlessly serialize inference across concurrent requests.
        res = predictor.predict_region(ds_, reg, mc_passes=1)
        pred_field = np.asarray(res["temperature"], dtype=np.float64)
        target = np.datetime64(res["date"], "D")
        with NETCDF_IO_LOCK:  # ref_ds is lazy - this is where the actual reads happen
            tidx = int(np.argmin(np.abs(ref_times - target)))
            ref_crop = ref_ds["temperature"].isel(time=tidx).sel(
                lat=slice(reg.lat_min, reg.lat_max), lon=slice(reg.lon_min, reg.lon_max))
            ref_arr = np.asarray(ref_crop.values, dtype=np.float64)
            if ref_arr.shape != pred_field.shape:
                ref_da = ref_ds["temperature"].isel(time=tidx).interp(
                    lat=np.asarray(res["lats"]), lon=np.asarray(res["lons"]))
                ref_arr = np.asarray(ref_da.values, dtype=np.float64)
        preds.append(pred_field)
        refs.append(ref_arr)

    P = np.concatenate(preds, axis=0)     # (ndays*nz, H, W) stacked along batch axis
    R = np.concatenate(refs, axis=0)
    metrics = []
    nz = len(cfg_depths(cfg))
    for k in range(nz):
        rows = per_depth_metrics(P[k::nz], R[k::nz])
        m = rows[0]
        m["depth"] = cfg_depths(cfg)[k]
        metrics.append(m)
    bands = band_skill(metrics, cfg_depths(cfg))
    payload = {
        "date_start": str(dr[0].date()), "date_end": str(dr[-1].date()),
        "days_sampled": len(dr), "region_cells": int(mask.sum()),
        "reference_type": ref_label,
        "demo_mode": cfg["data_source"]["type"] == "synthetic",
        "metrics": metrics, "bands": bands,
        "scatter": {"depth": scatter_depth,
                    "points": scatter_samples(P, R, cfg_depths(cfg), scatter_depth)},
        "skill_note": ("SYNTHETIC REFERENCE - DEMO MODE" if ref_label ==
                       "SYNTHETIC REFERENCE - DEMO MODE"
                       else "Validated against reference dataset under data/reference/"),
    }
    save_cache(cfg["paths"]["cache_dir"], key, json.loads(json.dumps(payload, default=float)))
    return json.loads(json.dumps(payload, default=float))


# ---------------------------------------------------------------------------
# Optional static hosting of a built frontend
# ---------------------------------------------------------------------------
_frontend_dist = get_config()["_root"] + "/frontend/dist"
if os.path.isdir(_frontend_dist):  # pragma: no cover
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")

