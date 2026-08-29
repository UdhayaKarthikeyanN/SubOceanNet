"""Subprocess bridge to the isolated live-fetch environment.

copernicusmarine/cdsapi are deliberately NOT importable from the main app's
.venv - copernicusmarine pulls in zarr, which requires numpy>=2, conflicting
with this project's pinned numpy==1.26.4 / torch==2.5.1 stack (see
scripts/live_fetch/requirements.txt for the full rationale). Each fetch
instead runs as a short-lived subprocess using the isolated interpreter at
data_source.live.python, which prints one JSON result line to stdout and
writes a small single-timestep NetCDF file. src/data/live/manager.py loads
that file with the main app's own xarray/netCDF4 install and deletes it.

Nothing in this module ever logs or returns credential values - they are
read only inside the subprocess, from .env.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class FetchError(RuntimeError):
    """A live fetch failed (bad config, missing credentials, network error,
    provider error, or timeout). Callers should catch this and fall back to
    cached or synthetic data - it must never propagate to an HTTP response."""


def _script_path(cfg: dict, name: str) -> Path:
    return Path(cfg["_root"]) / "scripts" / "live_fetch" / name


def _interpreter(cfg: dict) -> Path:
    live = cfg["data_source"].get("live", {})
    py = Path(live.get("python", ".venv-live/Scripts/python.exe"))
    if not py.is_absolute():
        py = Path(cfg["_root"]) / py
    return py


def _run(cfg: dict, script_name: str, extra_args: list[str]) -> dict[str, Any]:
    py_path = _interpreter(cfg)
    if not py_path.exists():
        raise FetchError(
            f"isolated live-fetch interpreter not found at {py_path}. Set it up with:\n"
            f"  python -m venv .venv-live\n"
            f"  .venv-live\\Scripts\\python.exe -m pip install -r scripts\\live_fetch\\requirements.txt"
        )
    script = _script_path(cfg, script_name)
    live = cfg["data_source"]["live"]
    region = cfg["region"]
    out_dir = Path(live.get("cache_dir", "data/live_cache"))
    if not out_dir.is_absolute():
        out_dir = Path(cfg["_root"]) / out_dir

    args = [str(py_path), str(script),
            "--lat-min", str(region["lat_min"]), "--lat-max", str(region["lat_max"]),
            "--lon-min", str(region["lon_min"]), "--lon-max", str(region["lon_max"]),
            "--out-dir", str(out_dir)] + extra_args
    timeout = int(live.get("fetch_timeout_s", 300))
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                              cwd=cfg["_root"])
    except subprocess.TimeoutExpired as e:
        raise FetchError(f"{script_name} timed out after {timeout}s") from e
    except OSError as e:
        raise FetchError(f"failed to launch {script_name}: {e}") from e

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise FetchError(f"{script_name} failed (exit {proc.returncode}): {detail}")

    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise FetchError(f"{script_name} produced no output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise FetchError(f"{script_name} produced unparsable output: {proc.stdout[-500:]}") from e


def fetch_ocean_variable(cfg: dict, name: str) -> dict[str, Any]:
    """Returns {"variable","provider","observation_time","path"} for one of
    sst/sss/sla/cur_u/cur_v. Raises FetchError on any failure."""
    spec = cfg["data_source"]["live"]["ocean"]["variables"].get(name) or {}
    dataset_id = spec.get("dataset_id")
    if not dataset_id:
        raise FetchError(
            f"no dataset_id configured for '{name}' under "
            f"data_source.live.ocean.variables in configs/config.yaml"
        )
    return _run(cfg, "fetch_ocean.py", [
        "--var", name, "--dataset-id", dataset_id,
        "--source-var", spec.get("variable", name),
    ])


def fetch_wind_variable(cfg: dict, name: str) -> dict[str, Any]:
    """Returns {"variable","provider","observation_time","path"} for wind_u
    or wind_v. Raises FetchError on any failure."""
    cds_var = cfg["data_source"]["live"]["wind"].get("variables", {}).get(name)
    if not cds_var:
        raise FetchError(
            f"no CDS variable configured for '{name}' under "
            f"data_source.live.wind.variables in configs/config.yaml"
        )
    return _run(cfg, "fetch_wind.py", ["--var", name, "--cds-variable", cds_var])
