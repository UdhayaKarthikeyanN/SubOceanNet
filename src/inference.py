"""Inference engine (F4): bundle loading -> preprocessing -> encoder -> decoder
-> denormalized 15-depth temperature field + MC-dropout uncertainty.

CLI:
    python src/inference.py --date 2023-06-15 --region '{"type":"Polygon","coordinates":[[[52,5],[75,5],[75,20],[52,20],[52,5]]]}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import bundle_dir, get_config, load_config
    from src.data import loaders as data_loaders
    from src.data.preprocessing import Region, denormalize, preprocess_for_inference
    from src.models import TemperatureModel, build_model, enable_mc_dropout
else:
    from src.config import bundle_dir, get_config, load_config
    from src.data import loaders as data_loaders
    from src.data.preprocessing import Region, denormalize, preprocess_for_inference
    from src.models import TemperatureModel, build_model, enable_mc_dropout


def json_safe(obj):
    """Recursively convert numpy arrays / NaN to JSON-safe structures."""
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


class Predictor:
    """Lazy singleton wrapper around the trained bundle."""

    _lock = threading.Lock()
    _instance: "Predictor | None" = None
    _mtime: float = 0.0

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or get_config()
        bdir = bundle_dir(self.cfg)
        self.bdir = bdir
        self.meta = json.loads((bdir / "metadata.json").read_text())
        self.scalars = json.loads((bdir / "scalars.json").read_text())
        import torch
        ckpt = torch.load(bdir / "checkpoint.pt", map_location="cpu", weights_only=False)
        self.model: TemperatureModel = build_model(self.cfg)
        # honour encoder_type recorded at train time if config drifted
        self.model.input_variables = ckpt.get("input_variables") or self.meta["input_variables"]
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        import torch as _t
        self.torch = _t
        self.patch = int(self.meta.get("patch_size", self.cfg["inference"]["patch_size"]))
        self.depths = list(self.meta["depths_m"])
        self.temp_scalar = self.scalars["temperature"]
        self._ds = None  # lazy cached input dataset handle

    @property
    def ds(self):
        """Cached (lazy) xarray handle on the 7 surface inputs."""
        if self._ds is None:
            self._ds = data_loaders.load_input_dataset(self.cfg)
        return self._ds

    # -- availability ------------------------------------------------------
    @classmethod
    def bundle_mtime(cls, cfg=None) -> float:
        try:
            p = bundle_dir(cfg or get_config()) / "checkpoint.pt"
            return p.stat().st_mtime if p.exists() else 0.0
        except OSError:
            return 0.0

    @classmethod
    def available(cls, cfg=None) -> bool:
        return cls.bundle_mtime(cfg) > 0.0

    @classmethod
    def get(cls, cfg: dict | None = None) -> "Predictor":
        mtime = cls.bundle_mtime(cfg)
        with cls._lock:
            if cls._instance is None or mtime != cls._mtime:
                cls._instance = Predictor(cfg)
                cls._mtime = mtime
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None
            cls._mtime = 0.0

    # -- core tensor ops ---------------------------------------------------
    def _patches_full_grid(self, tensor: np.ndarray) -> np.ndarray:
        r = (self.patch - 1) // 2
        padded = np.pad(tensor, ((0, 0), (r, r), (r, r)), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(padded, self.patch, axis=1)
        win = np.lib.stride_tricks.sliding_window_view(win, self.patch, axis=2)
        # win: (7, H, W, k, k) -> (H*W, 7, k, k) with patch centred on each cell
        h, w = tensor.shape[1], tensor.shape[2]
        return np.ascontiguousarray(
            win.transpose(1, 2, 0, 3, 4).reshape(h * w, tensor.shape[0], self.patch, self.patch)
        )

    def _run_model(self, patches: np.ndarray, mc_passes: int,
                   progress_cb=None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean (N,nz), std (N,nz)) in normalized space."""
        torch = self.torch
        bs = int(self.cfg["inference"].get("batch_size", 1024))
        n = len(patches)
        acc_sum = np.zeros((n, len(self.depths)), dtype=np.float64)
        acc_sq = np.zeros_like(acc_sum)
        with torch.no_grad():
            for p_i in range(max(int(mc_passes), 1)):
                enable_mc_dropout(self.model)          # stochastic passes only
                for s in range(0, n, bs):
                    xb = torch.from_numpy(patches[s:s + bs])
                    out = self.model(xb).numpy()
                    acc_sum[s:s + len(out)] += out
                    acc_sq[s:s + len(out)] += out.astype(np.float64) ** 2
                if progress_cb and mc_passes > 1:
                    progress_cb(int(100 * (p_i + 1) / mc_passes),
                                f"MC pass {p_i + 1}/{mc_passes}")
        mean = acc_sum / mc_passes
        var = np.maximum(acc_sq / mc_passes - mean ** 2, 0.0)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    def _denorm_profiles(self, mean_n: np.ndarray, std_n: np.ndarray):
        t_std = np.asarray(self.temp_scalar["std"], dtype=np.float32)[None, :]
        t_mean = np.asarray(self.temp_scalar["mean"], dtype=np.float32)[None, :]
        return mean_n * t_std + t_mean, std_n * t_std

    # -- public API ----------------------------------------------------------
    def _preprocess(self, date_str: str, region: Region) -> dict:
        return preprocess_for_inference(self.cfg, date_str, region,
                                        scalars=self.scalars, ds=self.ds)

    def predict_region(self, date_str: str, region: Region, mc_passes: int | None = None,
                       progress_cb=None) -> dict:
        t0 = time.time()
        pre = self._preprocess(date_str, region)
        if progress_cb:
            progress_cb(15, f"preprocessed {pre['tensor'].shape[1]}x{pre['tensor'].shape[2]} cells")
        patches = self._patches_full_grid(pre["tensor"])
        if progress_cb:
            progress_cb(25, f"embedded {len(patches)} surface patches")
        mean_n, std_n = self._run_model(
            patches, int(mc_passes or self.cfg["inference"]["mc_passes"]),
            progress_cb=(lambda p, m: progress_cb(25 + int(0.65 * p), m))
            if progress_cb else None)
        mean_c, std_c = self._denorm_profiles(mean_n, std_n)

        ny, nx = len(pre["lats"]), len(pre["lons"])
        ocean = pre["ocean_mask"]
        temp = mean_c.reshape(ny, nx, -1).transpose(2, 0, 1)     # (nz, H, W)
        unc = std_c.reshape(ny, nx, -1).transpose(2, 0, 1)
        temp[:, ~ocean] = np.nan
        unc[:, ~ocean] = np.nan

        result = {
            "date": pre["qc_report"]["date_resolved"],
            "date_requested": date_str,
            "depths": self.depths,
            "units": "degC",
            "lats": pre["lats"],
            "lons": pre["lons"],
            # ONLY temperature is ever emitted (plus its uncertainty layer):
            "temperature": temp,
            "uncertainty": unc,
            "ocean_cells": int(ocean.sum()),
            "total_cells": int(ny * nx),
            "runtime_s": round(time.time() - t0, 2),
            "model_version": self.meta.get("model_version", "?"),
            "demo_mode": bool(self.meta.get("demo_mode", True)),
            "qc_summary": {
                "filled_or_interpolated": sum(
                    v.get("filled_or_interpolated", 0)
                    for v in pre["qc_report"]["variables"].values()),
                "missing_cells_before_fill": sum(
                    v.get("missing_before", 0)
                    for v in pre["qc_report"]["variables"].values()),
            },
        }
        return json_safe(result)

    def predict_points(self, points: list[tuple[float, float]], date_str: str,
                       mc_passes: int | None = None) -> dict:
        """Profile(s) at specific lat/lon point(s) for one date.

        Each point is served from a small (patch-sized) neighbourhood crop so
        the cost is independent of domain size.
        """
        res_deg = float(self.cfg["grid"]["resolution"])
        r_cells = (self.patch - 1) // 2
        half = r_cells * res_deg

        tensors, ocean_centers, snapped = [], [], []
        for la, lo in points:
            region = Region(ring=[[lo - half, la - half], [lo + half, la - half],
                                  [lo + half, la + half], [lo - half, la + half]])
            pre = self._preprocess(date_str, region)
            lats = np.asarray(pre["lats"]); lons = np.asarray(pre["lons"])
            iy = int(np.argmin(np.abs(lats - la))); ix = int(np.argmin(np.abs(lons - lo)))
            t = pre["tensor"]
            padded = np.pad(t, ((0, 0), (r_cells, r_cells), (r_cells, r_cells)), mode="edge")
            tensors.append(padded[:, iy: iy + self.patch, ix: ix + self.patch])
            ocean_centers.append(bool(pre["ocean_mask"][iy, ix]))
            snapped.append((float(lats[iy]), float(lons[ix])))

        patches = np.stack(tensors, axis=0).astype(np.float32)
        mean_n, std_n = self._run_model(patches,
                                        int(mc_passes or self.cfg["inference"]["mc_passes"]))
        mean_c, std_c = self._denorm_profiles(mean_n, std_n)

        out = []
        for k, ((la, lo), (slat, slon)) in enumerate(zip(points, snapped)):
            prof = mean_c[k].copy(); uncf = std_c[k].copy()
            if not ocean_centers[k]:
                prof[:] = np.nan; uncf[:] = np.nan
            out.append({
                "lat": float(la), "lon": float(lo),
                "snapped_lat": slat, "snapped_lon": slon,
                "temperature": prof.tolist(), "uncertainty": uncf.tolist(),
                "ocean": ocean_centers[k],
            })
        first_date = None
        try:
            import numpy as _np
            tvals = self.ds["time"].values.astype("datetime64[D]")
            target = _np.datetime64(date_str, "D")
            first_date = str(_np.datetime_as_string(tvals[int(np.argmin(np.abs(tvals - target)))], unit="D"))
        except Exception:
            first_date = date_str
        return {
            "date": first_date, "depths": self.depths, "units": "degC",
            "points": json_safe(out),
            "model_version": self.meta.get("model_version", "?"),
        }

    def timeseries(self, lat: float, lon: float, start: str, end: str,
                   stride_days: int = 7, region: Region | None = None,
                   progress_cb=None) -> dict:
        """Predicted temperature time series at a point (all depths).

        With ``region`` set, returns the polygon-mean series instead.
        """
        dates_all, _ = data_loaders.available_date_range(self.cfg)
        dr = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"),
                       np.timedelta64(int(stride_days), "D"))
        cap = int(self.cfg["inference"].get("max_timeseries_dates", 120))
        if len(dr) > cap:
            step = int(np.ceil(len(dr) / cap))
            dr = dr[::step]
        temps, dates_out = [], []
        total = len(dr)
        for i, d in enumerate(dr):
            ds_ = str(np.datetime_as_string(d, unit="D"))
            if region is not None:
                res = self.predict_region(ds_, region, mc_passes=1)
                field = np.asarray(res["temperature"], dtype=np.float64)
                vals = np.nanmean(field.reshape(len(self.depths), -1), axis=1)
            else:
                res = self.predict_points([(float(lat), float(lon))], ds_, mc_passes=1)
                vals = np.asarray(res["points"][0]["temperature"])
            temps.append([None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
                          for v in vals])
            dates_out.append(ds_)
            if progress_cb:
                progress_cb(int(100 * (i + 1) / total), f"{ds_} ({i + 1}/{total})")
        return {
            "lat": lat, "lon": lon, "area_mean": region is not None,
            "dates": dates_out, "depths": self.depths,
            "temperature": temps, "units": "degC",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="OceanEmbed inference")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--region", required=True,
                    help="GeoJSON Polygon as JSON string, or @path/to/file.json")
    ap.add_argument("--out", default=None, help="write full result JSON here")
    ap.add_argument("--summary-only", action="store_true",
                    help="print per-depth regional means instead of full grid")
    args = ap.parse_args(argv)

    region_raw = args.region
    if region_raw.startswith("@"):
        region_raw = Path(region_raw[1:]).read_text()
    geojson = json.loads(region_raw)
    region = Region(ring=geojson["coordinates"][0])

    pred = Predictor.get()
    res = pred.predict_region(args.date, region)
    if args.summary_only:
        field = np.asarray(res["temperature"], dtype=float)
        means = np.nanmean(field.reshape(len(res["depths"]), -1), axis=1)
        print(json.dumps({"date": res["date"], "depths_m": res["depths"],
                          "regional_mean_degC": json_safe(means.tolist())}, indent=2))
    else:
        payload = json.dumps(res)
        print(f"[inference] cells={res['ocean_cells']} depths={len(res['depths'])} "
              f"runtime={res['runtime_s']}s output_keys="
              f"{sorted(k for k in res.keys() if isinstance(res[k], (list, dict)))[:6]}")
        print(f"[inference] temperature grid shape "
              f"({len(res['temperature'])}, {len(res['lats'])}, {len(res['lons'])})")
    if args.out:
        Path(args.out).write_text(json.dumps(json_safe(res)))
        print(f"[inference] wrote {args.out}")
    return res


if __name__ == "__main__":
    main()
