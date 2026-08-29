"""Live-data orchestration: per-variable fetch-or-cache-or-fallback with a
clear REAL / CACHED REAL / SYNTHETIC FALLBACK status for each of the 7
inputs, refreshed on a background cadence.

Design principle: HTTP request handling never blocks on network I/O. Real
fetches (subprocess calls into the isolated .venv-live, see providers.py)
only ever happen from the background refresh thread. A request that finds
an empty cache gets an instant synthetic snapshot instead of waiting on a
live fetch - see ensure_not_empty().
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import cache as live_cache
from . import providers
from .status import CACHED_REAL, REAL, SYNTHETIC_FALLBACK

OCEAN_VARS = ("sst", "sss", "sla", "cur_u", "cur_v")
WIND_VARS = ("wind_u", "wind_v")
ALL_VARS = OCEAN_VARS + WIND_VARS

# Copernicus Marine subset requests are slow (~2-3 min each even alone) and
# all 5 ocean variables share one account - firing them all at once causes
# contention severe enough to blow the per-request timeout rather than any
# real error. Cap how many run at once instead of raising the timeout.
MAX_CONCURRENT_OCEAN_FETCHES = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regrid_to_target(da: xr.DataArray, cfg: dict, name: str) -> xr.DataArray:
    """Bilinear-regrid one fetched variable onto the app's exact target
    grid, reusing the same interpolate_grid() the netcdf/synthetic paths
    already rely on - a no-op when the source already matches."""
    from ..preprocessing import interpolate_grid

    region = cfg["region"]
    res = float(cfg["grid"]["resolution"])
    target_lat = np.arange(region["lat_min"], region["lat_max"] + 1e-9, res)
    target_lon = np.arange(region["lon_min"], region["lon_max"] + 1e-9, res)
    regridded = interpolate_grid(da.to_dataset(name=name), target_lat, target_lon)
    return regridded[name]


def _hours_since(iso_or_ts) -> float | None:
    if not iso_or_ts:
        return None
    try:
        t = pd.Timestamp(iso_or_ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        return round((now - t).total_seconds() / 3600.0, 2)
    except (ValueError, TypeError):
        return None


class LiveDataManager:
    """One instance per process (see get_manager())."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._ocean_fetch_slots = threading.Semaphore(MAX_CONCURRENT_OCEAN_FETCHES)
        self._status: dict[str, dict] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._refreshing = False

    # -- status ---------------------------------------------------------------
    def status_snapshot(self) -> dict:
        with self._lock:
            snap = {k: dict(v) for k, v in self._status.items()}
        for name, info in snap.items():
            info["data_age_hours"] = _hours_since(info.get("observation_time"))
        return snap

    def _set_status(self, name: str, status: dict) -> None:
        with self._lock:
            self._status[name] = status

    # -- fallback synthesis -----------------------------------------------------
    def _synthesize_fallback(self, name: str) -> None:
        """Last-resort single-day synthetic snapshot so the app keeps
        working end-to-end even with zero live connectivity. Reuses the
        existing synthetic generator's physics - no separate model."""
        from datetime import date as _date

        from src.data.generate_synthetic import SyntheticOcean

        cfg = self.cfg
        res = float(cfg["grid"]["resolution"])
        region = cfg["region"]
        lat = np.arange(region["lat_min"], region["lat_max"] + 1e-9, res)
        lon = np.arange(region["lon_min"], region["lon_max"] + 1e-9, res)
        depths = cfg["depths_m"]
        today = _date.today()
        seed = (abs(hash(name)) % 10_000) + 1
        ocean = SyntheticOcean(lat, lon, depths, today, today, seed=seed)
        fields, _truth = ocean.render_day(0)
        arr = np.asarray(fields[name])
        if arr.ndim == 3 and arr.shape[0] == 1:  # land-mask broadcast can add a leading axis
            arr = arr[0]
        da = xr.DataArray(
            arr[np.newaxis, :, :], dims=("time", "lat", "lon"),
            coords={"time": [np.datetime64(pd.Timestamp(today), "ns")], "lat": lat, "lon": lon},
            name=name,
        )
        max_ts = int(cfg["data_source"]["live"].get("max_cached_timesteps", 14))
        live_cache.append_timestep(cfg, name, da, max_timesteps=max_ts)

    # -- per-variable refresh ---------------------------------------------------
    def _refresh_one(self, name: str, force: bool = False) -> dict:
        cfg = self.cfg
        live = cfg["data_source"]["live"]
        max_age_h = float(live.get("max_age_hours", 48))
        max_ts = int(live.get("max_cached_timesteps", 14))
        refresh_interval_h = float(live.get("refresh_interval_minutes", 180)) / 60.0

        # Providers publish at most once/day - re-fetching within the
        # refresh interval just re-downloads the same day's data. If the
        # on-disk cache was already successfully fetched more recently than
        # that, skip the network round-trip entirely (this is what makes
        # repeated app restarts instant instead of re-paying a multi-minute
        # live fetch every time - see README "Live data mode"). A manual
        # "recache now" request (force=True) bypasses this so it actually
        # re-fetches instead of silently no-op'ing.
        cache_path = live_cache.variable_cache_path(cfg, name)
        if cache_path.exists() and not force:
            age_since_fetch_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
            if age_since_fetch_h < refresh_interval_h:
                cached = live_cache.read_variable_cache(cfg, name)
                if cached is not None and cached.sizes.get("time", 0) > 0:
                    last_t = pd.Timestamp(cached["time"].values[-1])
                    stale = bool((_hours_since(last_t) or 0.0) > max_age_h)
                    return {
                        "source": CACHED_REAL, "provider": None,
                        "observation_time": last_t.isoformat(),
                        "fetched_at": _now_iso(), "stale": stale, "error": None,
                    }

        is_ocean = name in OCEAN_VARS
        fetch_fn = providers.fetch_ocean_variable if is_ocean else providers.fetch_wind_variable
        slots = self._ocean_fetch_slots if is_ocean else None

        try:
            if slots is not None:
                slots.acquire()
            try:
                result = fetch_fn(cfg, name)
            finally:
                if slots is not None:
                    slots.release()
            obs_day = pd.Timestamp(result["observation_time"]).normalize()
            with xr.open_dataarray(result["path"]) as raw:
                da = raw.load()
            # providers (Copernicus Marine, ERA5/CDS) return latitude/longitude,
            # not lat/lon - normalize here, the single choke point every
            # fetched variable passes through, so the cache and the merged
            # dataset always use consistent dimension names regardless of
            # which variables are REAL vs SYNTHETIC FALLBACK this cycle
            rename = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.dims}
            if rename:
                da = da.rename(rename)
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            # providers deliver their own native grid (e.g. CMEMS physics at
            # 0.0833deg vs the app's 0.25deg target) - regrid onto the exact
            # target grid now so every cached variable shares identical
            # lat/lon coordinate VALUES. xr.merge(join="inner") in
            # get_dataset() matches on those values, not just array length,
            # so mismatched native grids would otherwise merge to ~nothing.
            da = _regrid_to_target(da, cfg, name)
            da = da.expand_dims(time=[np.datetime64(obs_day, "ns")])
            live_cache.append_timestep(cfg, name, da, max_timesteps=max_ts)
            try:
                Path(result["path"]).unlink(missing_ok=True)
            except OSError:
                pass
            return {
                "source": REAL, "provider": result.get("provider"),
                "observation_time": str(result["observation_time"]),
                "fetched_at": _now_iso(), "stale": False, "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - any provider/subprocess failure -> fallback
            cached = live_cache.read_variable_cache(cfg, name)
            if cached is not None and cached.sizes.get("time", 0) > 0:
                last_t = pd.Timestamp(cached["time"].values[-1])
                age_h = _hours_since(last_t) or 0.0
                return {
                    "source": CACHED_REAL, "provider": None,
                    "observation_time": last_t.isoformat(),
                    "fetched_at": _now_iso(), "stale": bool(age_h > max_age_h),
                    "error": str(exc),
                }
            self._synthesize_fallback(name)
            return {
                "source": SYNTHETIC_FALLBACK, "provider": "synthetic_fallback",
                "observation_time": _now_iso(), "fetched_at": _now_iso(),
                "stale": False, "error": str(exc),
            }

    # -- bulk refresh (background thread) ----------------------------------------
    def refresh_all(self, force: bool = False) -> bool:
        """Fetch all 7 variables concurrently. Guarded by a non-blocking
        lock: if a refresh is already in flight (background thread overlaps
        a manual call, or one cycle runs longer than the interval), this
        call is skipped rather than racing the same _incoming_<var>.nc temp
        files, which corrupts the on-disk cache. Returns True if a refresh
        cycle actually ran, False if one was already in progress."""
        if not self._refresh_lock.acquire(blocking=False):
            return False
        self._refreshing = True
        try:
            with ThreadPoolExecutor(max_workers=len(ALL_VARS)) as ex:
                futures = {ex.submit(self._refresh_one, name, force): name for name in ALL_VARS}
                for fut, name in futures.items():
                    try:
                        status = fut.result()
                    except Exception as exc:  # pragma: no cover - _refresh_one already catches
                        status = {"source": SYNTHETIC_FALLBACK, "provider": None,
                                  "observation_time": _now_iso(), "fetched_at": _now_iso(),
                                  "stale": False, "error": str(exc)}
                    self._set_status(name, status)
            return True
        finally:
            self._refreshing = False
            self._refresh_lock.release()

    def is_refreshing(self) -> bool:
        return self._refreshing

    def trigger_manual_refresh(self) -> bool:
        """Kick off a forced (cache-bypassing) refresh of all 7 variables in
        a background thread and return immediately - a real fetch can take
        minutes, far too long for an HTTP request to hold open. Callers
        poll status_payload()/is_refreshing() (see /api/live/status) to
        watch it land. Returns False without starting anything if a refresh
        (background-cadence or another manual one) is already in flight."""
        if self._refreshing:
            return False
        t = threading.Thread(
            target=lambda: self.refresh_all(force=True), name="oe-live-manual-refresh", daemon=True
        )
        t.start()
        return True

    def clear_cache(self) -> list[str]:
        """Delete all cached live data now. Immediately backfills an
        instant synthetic snapshot per variable (same safety net as
        ensure_not_empty) so the app keeps working while the next real
        fetch - background cadence or a manual refresh - catches up.
        Returns the variable names whose cache file was removed."""
        removed = live_cache.clear_cache(self.cfg)
        with self._lock:
            self._status = {}
        self.ensure_not_empty()
        return removed

    def ensure_not_empty(self) -> None:
        """Fast, non-blocking guarantee that every variable's cache has at
        least one timestep, synthesizing instantly if needed. Real fetches
        only ever happen on the background refresh cadence, never here."""
        for name in ALL_VARS:
            if live_cache.read_variable_cache(self.cfg, name) is None:
                self._synthesize_fallback(name)
                with self._lock:
                    if name not in self._status:
                        self._status[name] = {
                            "source": SYNTHETIC_FALLBACK, "provider": "synthetic_fallback",
                            "observation_time": _now_iso(), "fetched_at": _now_iso(),
                            "stale": False, "error": "no live fetch has completed yet",
                        }

    # -- dataset assembly ---------------------------------------------------------
    def get_dataset(self) -> xr.Dataset:
        """Canonical (time, lat, lon) dataset with all 7 variables, in the
        same shape src/data/loaders.py expects from netcdf mode.

        Providers have different latencies in practice (ERA5 winds often
        land a day or more after altimetry-derived SLA/currents), so this
        does NOT require every variable to share an exact calendar day -
        that would make the live dataset go empty routinely, which is
        exactly what a "live" mode must not do. Instead each variable
        contributes its own most recent cached snapshot, and all of them
        are stacked under one composite "latest" time label (the newest
        observation time among the 7). This is standard practice for
        real-time multi-source composites: a same-day match is nice to
        have, not a requirement for a usable "current conditions" view."""
        self.ensure_not_empty()
        per_var, latest_obs = {}, None
        for name in ALL_VARS:
            da = live_cache.read_variable_cache(self.cfg, name)
            if da is None or da.sizes.get("time", 0) == 0:
                continue
            per_var[name] = da.isel(time=-1, drop=True)
            t = pd.Timestamp(da["time"].values[-1])
            if latest_obs is None or t > latest_obs:
                latest_obs = t
        if not per_var:
            return xr.Dataset()
        composite_time = np.datetime64(latest_obs, "ns")
        arrays = [da.expand_dims(time=[composite_time]).to_dataset(name=name)
                 for name, da in per_var.items()]
        return xr.merge(arrays, join="inner")

    # -- background thread ----------------------------------------------------------
    def start_background_refresh(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        interval_s = float(self.cfg["data_source"]["live"].get("refresh_interval_minutes", 180)) * 60

        def loop():
            while not self._stop.is_set():
                try:
                    self.refresh_all()
                except Exception:  # pragma: no cover - defensive; refresh_all already guards per-var
                    traceback.print_exc()
                self._stop.wait(interval_s)

        self._thread = threading.Thread(target=loop, name="oe-live-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


_MANAGER: LiveDataManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_manager(cfg: dict) -> LiveDataManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = LiveDataManager(cfg)
        return _MANAGER


def reset_manager() -> None:
    """Test-only: drop the singleton so a fresh config takes effect."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.stop()
        _MANAGER = None


# ---------------------------------------------------------------------------
# Payloads for the /api/live/* endpoints
# ---------------------------------------------------------------------------

def status_payload(cfg: dict) -> dict:
    live_cfg = cfg["data_source"].get("live", {})
    mgr = get_manager(cfg)
    return {
        "mode": cfg["data_source"]["type"],
        "live_mode_active": cfg["data_source"]["type"] == "live",
        "refresh_interval_minutes": live_cfg.get("refresh_interval_minutes"),
        "max_age_hours": live_cfg.get("max_age_hours"),
        "refreshing": mgr.is_refreshing(),
        "variables": mgr.status_snapshot(),
    }


def latest_payload(cfg: dict) -> dict:
    mgr = get_manager(cfg)
    ds = mgr.get_dataset()
    status = mgr.status_snapshot()
    latest_date = None
    if ds.sizes.get("time", 0) > 0:
        latest_date = str(np.datetime_as_string(np.datetime64(ds["time"].values[-1], "D"), unit="D"))

    variables: dict[str, dict] = {}
    for name in ALL_VARS:
        info = dict(status.get(name, {"source": SYNTHETIC_FALLBACK, "error": "not yet initialized"}))
        if name in ds and ds.sizes.get("time", 0) > 0:
            arr = np.asarray(ds[name].isel(time=-1).values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                info["min"] = round(float(finite.min()), 4)
                info["max"] = round(float(finite.max()), 4)
                info["mean"] = round(float(finite.mean()), 4)
        variables[name] = info

    return {
        "mode": cfg["data_source"]["type"],
        "latest_common_date": latest_date,
        "note": ("providers with different latencies (e.g. ERA5 winds) only overlap on "
                 "days every one of the 7 variables actually has data for"),
        "variables": variables,
    }
