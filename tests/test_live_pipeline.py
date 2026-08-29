"""Mocked tests for the live-data pipeline (F8): per-variable fetch-or-cache-
or-fallback, REAL / CACHED REAL / SYNTHETIC FALLBACK status, the /api/live/*
endpoints, and reuse of the existing preprocessing/inference code on
live-mode data.

None of these tests touch the network or the isolated .venv-live - the
provider layer (src/data/live/providers.py) is mocked throughout. They
build on the same tiny `test_env` fixture the rest of the suite uses (see
conftest.py), just with a private deep copy that flips data_source.type to
"live" so the shared synthetic/netcdf-mode tests are entirely unaffected.
"""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr


def _live_cfg(test_env: dict, tmp_path: Path) -> dict:
    """A live-mode copy of the shared tiny test config, isolated to its own
    cache dir per test so tests never share cached state."""
    cfg = copy.deepcopy(test_env)
    cfg["data_source"]["type"] = "live"
    cfg["data_source"]["live"] = {
        "cache_dir": str(tmp_path / "live_cache"),
        "python": str(Path(cfg["_root"]) / ".venv-live" / "Scripts" / "python.exe"),
        "refresh_interval_minutes": 180,
        "max_age_hours": 48,
        "max_cached_timesteps": 5,
        "fetch_timeout_s": 30,
        "ocean": {"provider": "copernicus_marine", "variables": {
            "sst": {"dataset_id": "", "variable": "thetao"},
            "sss": {"dataset_id": "", "variable": "so"},
            "sla": {"dataset_id": "", "variable": "zos"},
            "cur_u": {"dataset_id": "", "variable": "uo"},
            "cur_v": {"dataset_id": "", "variable": "vo"},
        }},
        "wind": {"provider": "era5", "variables": {
            "wind_u": "10m_u_component_of_wind", "wind_v": "10m_v_component_of_wind",
        }},
    }
    return cfg


def _fake_fetch_factory(cfg: dict, obs_date: str = "2026-08-20"):
    """A stand-in for providers.fetch_ocean_variable/fetch_wind_variable
    that writes a single-timestep NetCDF file the way the real isolated
    subprocess scripts would, with zero network access."""
    specs = {v["name"]: v for v in cfg["input_variables"]}

    def fake(cfg_: dict, name: str) -> dict:
        r = cfg["region"]
        res = float(cfg["grid"]["resolution"])
        lat = np.arange(r["lat_min"], r["lat_max"] + 1e-9, res)
        lon = np.arange(r["lon_min"], r["lon_max"] + 1e-9, res)
        spec = specs[name]
        lo, hi = float(spec["valid_min"]), float(spec["valid_max"])
        mid, spread = (lo + hi) / 2, (hi - lo) / 8  # comfortably inside the valid range
        arr = np.random.default_rng(abs(hash(name)) % 1000).normal(
            mid, spread, (len(lat), len(lon))).astype("f4")
        da = xr.DataArray(arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name=name)
        out_dir = Path(cfg["data_source"]["live"]["cache_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"_incoming_{name}.nc"
        da.to_dataset(name=name).to_netcdf(p)
        provider = "era5" if name in ("wind_u", "wind_v") else "copernicus_marine"
        return {"variable": name, "provider": provider,
                "observation_time": f"{obs_date}T00:00:00", "path": str(p)}
    return fake


@pytest.fixture()
def live_cfg(test_env, tmp_path):
    from src.data.live import manager as mgr_mod

    mgr_mod.reset_manager()
    cfg = _live_cfg(test_env, tmp_path)
    yield cfg
    mgr_mod.reset_manager()


# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------

def test_cache_dedups_same_day_and_caps_rolling_window(live_cfg):
    from src.data.live import cache as live_cache

    lat = np.arange(5.0, 30.001, 5.0)
    lon = np.arange(45.0, 105.001, 5.0)
    for day in ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"]:
        arr = np.full((len(lat), len(lon)), 20.0, dtype="f4")
        da = xr.DataArray(
            arr[np.newaxis], dims=("time", "lat", "lon"),
            coords={"time": [np.datetime64(day, "ns")], "lat": lat, "lon": lon}, name="sst",
        )
        live_cache.append_timestep(live_cfg, "sst", da, max_timesteps=2)

    cached = live_cache.read_variable_cache(live_cfg, "sst")
    assert cached.sizes["time"] == 2  # rolling window capped at 2
    dates = [str(t)[:10] for t in cached["time"].values]
    assert dates == ["2026-01-02", "2026-01-03"]  # duplicate day collapsed, oldest dropped


def test_read_variable_cache_none_when_never_written(live_cfg):
    from src.data.live import cache as live_cache

    assert live_cache.read_variable_cache(live_cfg, "sst") is None


# ---------------------------------------------------------------------------
# providers.py (subprocess bridge - real code path, no network)
# ---------------------------------------------------------------------------

def test_fetch_ocean_variable_requires_dataset_id(live_cfg):
    from src.data.live import providers

    with pytest.raises(providers.FetchError, match="no dataset_id configured"):
        providers.fetch_ocean_variable(live_cfg, "sst")


def test_fetch_wind_variable_requires_cds_variable(live_cfg):
    from src.data.live import providers

    live_cfg["data_source"]["live"]["wind"]["variables"] = {}
    with pytest.raises(providers.FetchError, match="no CDS variable configured"):
        providers.fetch_wind_variable(live_cfg, "wind_u")


def test_missing_isolated_interpreter_reports_clear_error(live_cfg):
    from src.data.live import providers

    live_cfg["data_source"]["live"]["python"] = str(Path(live_cfg["_root"]) / "does-not-exist" / "python.exe")
    live_cfg["data_source"]["live"]["ocean"]["variables"]["sst"]["dataset_id"] = "some-dataset"
    with pytest.raises(providers.FetchError, match="isolated live-fetch interpreter not found"):
        providers.fetch_ocean_variable(live_cfg, "sst")


# ---------------------------------------------------------------------------
# manager.py - the core fetch-or-cache-or-fallback state machine
# ---------------------------------------------------------------------------

def test_unconfigured_providers_fall_back_to_synthetic(live_cfg):
    from src.data.live import manager as mgr_mod
    from src.data.live.status import SYNTHETIC_FALLBACK

    mgr = mgr_mod.get_manager(live_cfg)
    mgr.refresh_all()
    snap = mgr.status_snapshot()

    assert set(snap) == set(mgr_mod.ALL_VARS)
    assert all(v["source"] == SYNTHETIC_FALLBACK for v in snap.values())
    assert all(v["error"] for v in snap.values())
    assert all(v["data_age_hours"] is not None for v in snap.values())

    ds = mgr.get_dataset()
    assert ds.sizes["time"] >= 1
    assert set(ds.data_vars) == set(mgr_mod.ALL_VARS)


def test_successful_fetch_marks_real_and_writes_cache(live_cfg):
    from src.data.live import manager as mgr_mod, providers
    from src.data.live.status import REAL

    mgr = mgr_mod.get_manager(live_cfg)
    fake = _fake_fetch_factory(live_cfg)
    with mock.patch.object(providers, "fetch_ocean_variable", side_effect=fake), \
         mock.patch.object(providers, "fetch_wind_variable", side_effect=fake):
        mgr.refresh_all()

    snap = mgr.status_snapshot()
    assert all(v["source"] == REAL for v in snap.values())
    assert all(v["error"] is None for v in snap.values())
    assert all(v["provider"] in ("copernicus_marine", "era5") for v in snap.values())

    from src.data.live import cache as live_cache
    for name in mgr_mod.ALL_VARS:
        da = live_cache.read_variable_cache(live_cfg, name)
        assert da is not None and da.sizes["time"] == 1

    # the temporary single-timestep files the fetch wrote are cleaned up
    leftovers = [f for f in os.listdir(live_cfg["data_source"]["live"]["cache_dir"])
                 if f.startswith("_incoming")]
    assert leftovers == []


def test_provider_failure_falls_back_to_cached_real_and_flags_stale(live_cfg):
    from src.data.live import cache as live_cache
    from src.data.live import manager as mgr_mod, providers
    from src.data.live.status import CACHED_REAL

    mgr = mgr_mod.get_manager(live_cfg)
    old_fetch = _fake_fetch_factory(live_cfg, obs_date="2020-01-01")  # far enough back to be stale
    with mock.patch.object(providers, "fetch_ocean_variable", side_effect=old_fetch), \
         mock.patch.object(providers, "fetch_wind_variable", side_effect=old_fetch):
        mgr.refresh_all()

    # age the on-disk cache past refresh_interval_minutes so the next
    # refresh_all() actually attempts a fetch instead of trusting the
    # still-fresh cache (see _refresh_one's freshness-skip optimization)
    cutoff = time.time() - live_cfg["data_source"]["live"]["refresh_interval_minutes"] * 60 - 60
    for name in mgr_mod.ALL_VARS:
        path = live_cache.variable_cache_path(live_cfg, name)
        os.utime(path, (cutoff, cutoff))

    def failing(cfg_, name):
        raise providers.FetchError("simulated network outage")

    with mock.patch.object(providers, "fetch_ocean_variable", side_effect=failing), \
         mock.patch.object(providers, "fetch_wind_variable", side_effect=failing):
        mgr.refresh_all()

    snap = mgr.status_snapshot()
    assert all(v["source"] == CACHED_REAL for v in snap.values())
    assert all(v["stale"] is True for v in snap.values())
    assert all("simulated network outage" in (v["error"] or "") for v in snap.values())


def test_fresh_cache_skips_refetch_and_reports_cached_real(live_cfg):
    """The optimization this test protects: once a variable has been
    fetched successfully, a second refresh within refresh_interval_minutes
    must reuse the on-disk cache instead of hitting the network again -
    this is what makes repeated app restarts fast instead of re-paying a
    multi-minute live fetch every time."""
    from src.data.live import manager as mgr_mod, providers
    from src.data.live.status import CACHED_REAL, REAL

    mgr = mgr_mod.get_manager(live_cfg)
    fake = _fake_fetch_factory(live_cfg)
    calls = {"n": 0}

    def counting_fake(cfg_, name):
        calls["n"] += 1
        return fake(cfg_, name)

    with mock.patch.object(providers, "fetch_ocean_variable", side_effect=counting_fake), \
         mock.patch.object(providers, "fetch_wind_variable", side_effect=counting_fake):
        mgr.refresh_all()
        assert all(v["source"] == REAL for v in mgr.status_snapshot().values())
        first_call_count = calls["n"]
        assert first_call_count == len(mgr_mod.ALL_VARS)

        mgr.refresh_all()  # cache is seconds old - must not hit the network again
        assert calls["n"] == first_call_count, "second refresh_all() re-fetched instead of reusing the fresh cache"
        assert all(v["source"] == CACHED_REAL for v in mgr.status_snapshot().values())


def test_live_dataset_flows_through_existing_preprocessing_and_inference(live_cfg):
    """The whole point of the live layer: its output must be usable by the
    SAME preprocessing + Predictor code the synthetic/netcdf modes already
    use, unmodified."""
    from src.data.live import manager as mgr_mod, providers
    from src.data.preprocessing import Region, preprocess_for_inference
    from src.inference import Predictor

    fake = _fake_fetch_factory(live_cfg)
    with mock.patch.object(providers, "fetch_ocean_variable", side_effect=fake), \
         mock.patch.object(providers, "fetch_wind_variable", side_effect=fake):
        mgr_mod.get_manager(live_cfg).refresh_all()

    pred = Predictor.get(live_cfg)
    region = Region(ring=[[52, 12], [57, 12], [57, 16], [52, 16]])
    pre = preprocess_for_inference(live_cfg, "2026-08-20", region, scalars=pred.scalars, ds=pred.ds)
    assert pre["tensor"].shape[0] == 7

    res = pred.predict_region("2026-08-20", region, mc_passes=1)
    assert len(res["temperature"]) == len(live_cfg["depths_m"])
    assert len(res["uncertainty"]) == len(live_cfg["depths_m"])


def test_load_input_dataset_composites_mismatched_provider_days(live_cfg):
    """Real providers routinely disagree on their latest available day (e.g.
    ERA5 winds lagging days behind altimetry-derived SLA/currents) - this
    must NOT make the live dataset empty. Each variable contributes its own
    latest snapshot, stacked under one composite "latest" date label (the
    newest observation among the 7), so the app stays usable even when no
    two providers agree on a calendar day."""
    from src.data.live import manager as mgr_mod, providers
    from src.data.loaders import load_input_dataset

    def fake_distinct_day(cfg_, name):
        day = f"2026-01-{mgr_mod.ALL_VARS.index(name) + 1:02d}"
        return _fake_fetch_factory(live_cfg, obs_date=day)(cfg_, name)

    mgr = mgr_mod.get_manager(live_cfg)
    with mock.patch.object(providers, "fetch_ocean_variable", side_effect=fake_distinct_day), \
         mock.patch.object(providers, "fetch_wind_variable", side_effect=fake_distinct_day):
        mgr.refresh_all()

    ds = load_input_dataset(live_cfg)
    assert ds.sizes["time"] == 1
    assert set(ds.data_vars) == set(mgr_mod.ALL_VARS)
    # composite label is the NEWEST of the 7 per-variable days (wind_v is last in ALL_VARS -> 2026-01-07)
    resolved_day = str(ds["time"].values[0])[:10]
    assert resolved_day == f"2026-01-{len(mgr_mod.ALL_VARS):02d}"


# ---------------------------------------------------------------------------
# /api/live/* endpoints
# ---------------------------------------------------------------------------

def test_live_status_endpoint_reports_per_variable(live_cfg, monkeypatch):
    import backend.main as backend_main
    from fastapi.testclient import TestClient
    from src.data.live import providers
    from src.data.live.manager import get_manager

    monkeypatch.setattr(backend_main, "get_config", lambda: live_cfg)
    fake = _fake_fetch_factory(live_cfg)
    monkeypatch.setattr(providers, "fetch_ocean_variable", fake)
    monkeypatch.setattr(providers, "fetch_wind_variable", fake)

    # populate deterministically before the app's own background refresh
    # thread starts (avoids racing it for the same on-disk cache files)
    get_manager(live_cfg).refresh_all()

    with TestClient(backend_main.app) as client:
        r = client.get("/api/live/status")
        assert r.status_code == 200
        body = r.json()
        assert body["live_mode_active"] is True
        assert set(body["variables"]) == {"sst", "sss", "sla", "cur_u", "cur_v", "wind_u", "wind_v"}
        for v in body["variables"].values():
            assert v["source"] == "REAL"
            assert "data_age_hours" in v
            assert v["error"] is None


def test_live_latest_endpoint_returns_summary(live_cfg, monkeypatch):
    import backend.main as backend_main
    from fastapi.testclient import TestClient
    from src.data.live import providers
    from src.data.live.manager import get_manager

    monkeypatch.setattr(backend_main, "get_config", lambda: live_cfg)
    fake = _fake_fetch_factory(live_cfg)
    monkeypatch.setattr(providers, "fetch_ocean_variable", fake)
    monkeypatch.setattr(providers, "fetch_wind_variable", fake)
    get_manager(live_cfg).refresh_all()

    with TestClient(backend_main.app) as client:
        body = client.get("/api/live/latest").json()
        assert body["mode"] == "live"
        assert body["latest_common_date"] is not None
        assert set(body["variables"]) == {"sst", "sss", "sla", "cur_u", "cur_v", "wind_u", "wind_v"}
        for v in body["variables"].values():
            assert "source" in v


def test_live_latest_endpoint_409_outside_live_mode(test_env, monkeypatch):
    import backend.main as backend_main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_main, "get_config", lambda: test_env)  # type: synthetic
    with TestClient(backend_main.app) as client:
        r = client.get("/api/live/latest")
        assert r.status_code == 409


def test_live_status_endpoint_inert_outside_live_mode(test_env, monkeypatch):
    """/api/live/status must never error just because live mode isn't on -
    the frontend polls it unconditionally to decide whether to show the
    live-data badge at all."""
    import backend.main as backend_main
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_main, "get_config", lambda: test_env)
    with TestClient(backend_main.app) as client:
        r = client.get("/api/live/status")
        assert r.status_code == 200
        assert r.json()["live_mode_active"] is False
