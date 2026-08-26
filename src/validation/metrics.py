"""Validation layer (F7): per-depth RMSE / MAE / Bias / Pearson r / R2
against a reference dataset, plus depth-band skill scores and JSON caching.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

BANDS = {
    "mixed_layer": (0, 50),
    "thermocline": (50, 200),
    "deep": (200, 10_000),
}


def per_depth_metrics(pred: np.ndarray, ref: np.ndarray) -> list[dict]:
    """pred/ref: (nz, ny, nx) or (N, nz). Metrics computed over finite pairs."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if p.ndim == 3 and r.ndim == 3:
        p = p.reshape(p.shape[0], -1)
        r = r.reshape(r.shape[0], -1)
    rows = []
    for k in range(p.shape[0]):
        pk, rk = p[k], r[k]
        ok = np.isfinite(pk) & np.isfinite(rk)
        n = int(ok.sum())
        if n < 2:
            rows.append({"depth": None, "rmse": None, "mae": None, "bias": None,
                         "pearson_r": None, "r2": None, "n": n})
            continue
        d = pk[ok] - rk[ok]
        rmse = float(np.sqrt(np.mean(d ** 2)))
        mae = float(np.mean(np.abs(d)))
        bias = float(np.mean(d))
        if np.std(rk[ok]) > 1e-9 and np.std(pk[ok]) > 1e-9:
            pr = float(np.corrcoef(pk[ok], rk[ok])[0, 1])
            r2 = float(1.0 - np.sum(d ** 2) / np.sum((rk[ok] - rk[ok].mean()) ** 2))
        else:
            pr = r2 = None
        rows.append({"depth": k, "rmse": round(rmse, 4), "mae": round(mae, 4),
                     "bias": round(bias, 4),
                     "pearson_r": None if pr is None else round(pr, 4),
                     "r2": None if r2 is None else round(r2, 4), "n": n})
    return rows


def band_skill(metrics: list[dict], depths: list[int]) -> dict:
    """Aggregate metrics by vertical band: mixed layer / thermocline / deep."""
    out = {}
    for name, (z0, z1) in BANDS.items():
        idx = [i for i, z in enumerate(depths) if z0 <= z < z1 and i < len(metrics)]
        vals = [metrics[i] for i in idx if metrics[i]["rmse"] is not None]
        if not vals:
            out[name] = {"depth_range_m": [z0, min(z1, max(depths))], "rmse": None}
            continue
        w = np.array([m["n"] for m in vals], dtype=float)
        def avg(key):
            v = [m[key] for m in vals if m.get(key) is not None]
            return round(float(np.average(v, weights=w[:len(v)])), 4) if v else None
        out[name] = {
            "depth_range_m": [min(depths[i] for i in idx), max(depths[i] for i in idx)],
            "rmse": avg("rmse"), "mae": avg("mae"), "bias": avg("bias"),
            "pearson_r": avg("pearson_r"), "r2": avg("r2"),
        }
    return out


def scatter_samples(pred: np.ndarray, ref: np.ndarray, depths: list[int],
                    depth: float, max_points: int = 1500) -> list[list[float]]:
    """Downsampled (predicted, reference) pairs for the density scatter plot."""
    try:
        k = int(np.argmin(np.abs(np.asarray(depths) - float(depth))))
    except (TypeError, ValueError):
        return []
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    pk = p[k].reshape(-1); rk = r[k].reshape(-1)
    ok = np.isfinite(pk) & np.isfinite(rk)
    pk, rk = pk[ok], rk[ok]
    if len(pk) == 0:
        return []
    if len(pk) > max_points:
        sel = np.random.default_rng(0).choice(len(pk), max_points, replace=False)
        pk, rk = pk[sel], rk[sel]
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(pk, rk)]


def cache_key(parts: dict) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def load_cache(cache_dir: str | Path, key: str) -> dict | None:
    p = Path(cache_dir) / f"validation_{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def save_cache(cache_dir: str | Path, key: str, payload: dict) -> Path:
    cdir = Path(cache_dir); cdir.mkdir(parents=True, exist_ok=True)
    p = cdir / f"validation_{key}.json"
    payload = {**payload, "cache_key": key, "cached_at": datetime.utcnow().isoformat() + "Z"}
    p.write_text(json.dumps(payload))
    return p
