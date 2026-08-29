"""Physically-plausible synthetic North Indian Ocean generator (demo mode).

Generates daily 0.25-degree surface satellite fields (the 7 INPUT variables)
plus clean subsurface temperature truth (15 depth levels) with:

* seasonal monsoon SST / wind reversal cycles,
* Arabian Sea summer upwelling off Somalia/Oman,
* fresher Bay of Bengal than Arabian Sea (+ Ganges plume),
* west -> east thermocline deepening and BoB barrier-layer effect,
* mesoscale eddies drifting westward (signature in SLA, SST and thermocline depth),
* wind-driven + geostrophic surface currents,
* land masking via the shared coarse coastline polygons.

Outputs (NetCDF, compressed):
    data/synthetic/inputs.nc    vars(sst, sss, sla, cur_u, cur_v, wind_u, wind_v)(time, lat, lon)
    data/synthetic/truth.nc     temperature(time, depth, lat, lon)  <- CLEAN reference
    data/synthetic/manifest.json

Deterministic given --seed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import zoom as _zoom
except ImportError:  # pragma: no cover
    _zoom = None

import netCDF4

if __package__ in (None, ""):  # allow `python src/data/generate_synthetic.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import load_config
    from src.data.landmask import build_land_mask
else:
    from src.config import load_config
    from src.data.landmask import build_land_mask

INPUT_ORDER = ["sst", "sss", "sla", "cur_u", "cur_v", "wind_u", "wind_v"]
OMEGA = 7.2921e-5
G = 9.81


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-(np.asarray(x, dtype=float))))


class SyntheticOcean:
    def __init__(self, lat, lon, depths, start: date, end: date, seed: int = 42):
        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        self.depths = list(depths)
        self.start, self.end = start, end
        self.ny, self.nx = len(lat), len(lon)
        self.lon2, self.lat2 = np.meshgrid(self.lon, self.lat)
        self.land = build_land_mask(self.lat, self.lon)
        self.east = (self.lon2 - 45.0) / 60.0                      # 0 (west) -> 1 (east)

        rng = np.random.default_rng(seed)
        # --- mesoscale eddy population (drifts westward, beta-gyre-ish) ------
        n_ed = 34
        self.ed_lat = rng.uniform(6.5, 28.5, n_ed)
        self.ed_lon0 = rng.uniform(47.0, 103.0, n_ed)
        self.ed_speed = rng.uniform(0.04, 0.09, n_ed)
        self.ed_amp = rng.uniform(0.35, 1.0, n_ed) * rng.choice([-1.0, 1.0], n_ed)
        self.ed_sig = rng.uniform(1.4, 3.0, n_ed)
        # --- slowly-varying synoptic noise patterns --------------------------
        self.pats = [self._smooth_pattern(rng) for _ in range(4)]
        self.freqs = rng.uniform(1.5, 6.0, 4)      # cycles/year
        self.phases = rng.uniform(0, 2 * np.pi, 4)

    def _smooth_pattern(self, rng) -> np.ndarray:
        if _zoom is None:
            return rng.normal(size=(self.ny, self.nx))
        k = max(3, self.ny // 7), max(3, self.nx // 7)
        field = _zoom(rng.normal(size=k), (self.ny / k[0], self.nx / k[1]), order=1)
        pad_y = self.ny - field.shape[0]
        pad_x = self.nx - field.shape[1]
        field = np.pad(field, ((0, max(pad_y, 0)), (0, max(pad_x, 0))))[: self.ny, : self.nx]
        sd = field.std()
        return field / (sd if sd > 1e-6 else 1.0)

    def _noise(self, t: int, idx) -> np.ndarray:
        acc = np.zeros((self.ny, self.nx))
        for k, i in enumerate(idx):
            acc += float(np.sin(2 * np.pi * self.freqs[k] * t / 365.25 + self.phases[k])) * self.pats[i]
        return acc / max(len(idx), 1)

    def _eddy_fields(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        sla = np.zeros((self.ny, self.nx))
        for la0, lo0, spd, amp, sig in zip(
            self.ed_lat, self.ed_lon0, self.ed_speed, self.ed_amp, self.ed_sig
        ):
            lo_t = lo0 - spd * t
            d2 = (self.lat2 - la0) ** 2 + (((self.lon2 - lo_t + 180) % 360) - 180) ** 2
            sla += amp * np.exp(-d2 / (2 * sig * sig))
        return sla, sla  # same field drives ssh & sst & thermocline anomalies

    def render_day(self, t: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Render one day. Returns ({var: (ny,nx)}, truth (nz,ny,nx))."""
        d = self.start + timedelta(days=int(t))
        doy = d.timetuple().tm_yday
        phi = 2 * np.pi * doy / 365.25
        monsoon = np.clip(np.sin(phi - 1.9), 0.0, None) ** 0.8   # peaks Aug
        winter = np.clip(-np.sin(phi - 1.9), 0.0, None) ** 0.8   # peaks Jan

        ed_sla, _ = self._eddy_fields(t)
        n_a = self._noise(t, (0, 1))
        n_b = self._noise(t, (2, 3))

        bob = _sigmoid((self.lon2 - 76.0) / 3.0) * _sigmoid((self.lat2 - 6.0) / 1.5)

        # ---------------- SST (seasonal + upwelling + eddies) ---------------
        sst = (
            28.9
            - 0.42 * (self.lat2 - 5.0)
            + 1.1 * np.sin(phi - 1.35) * (0.55 + 0.45 * self.east)     # seasonal warming
            - 3.2 * monsoon * np.exp(-(((self.lon2 - 53.0) ** 2) / 16.0 + ((self.lat2 - 10.5) ** 2) / 10.0))
            - 1.6 * winter * ((self.lat2 - 5.0) / 25.0)
            + 0.55 * ed_sla
            + 0.25 * n_a
        )
        # ---------------- SSS (BoB fresh, AS salty, Ganges plume) -----------
        sss = (
            36.6
            - 0.5 * (self.lat2 - 5.0) / 25.0
            - 3.1 * bob
            - 2.4 * np.exp(-(((self.lon2 - 89.0) ** 2) / 8.0 + ((self.lat2 - 19.5) ** 2) / 5.0)) * (0.45 + 0.55 * monsoon)
            - 0.5 * monsoon * bob
            + 0.9 * _sigmoid((58.0 - self.lon2) / 5.0)                 # AS evaporation
            + 0.07 * n_b
        )
        # ---------------- SLA ------------------------------------------------
        sla = (
            0.05 * np.sin(phi - 2.2) * (0.4 + 0.6 * bob)
            + 0.09 * ed_sla
            + 0.022 * (sst - 28.5)                                     # steric
            + 0.006 * n_b
        )
        # ---------------- Winds (monsoon reversal) ---------------------------
        theta = (np.pi / 4.0) * (monsoon - winter)                     # +45deg SW-monsoon, -135deg NE-monsoon
        spd = 4.5 + 3.8 * monsoon + 1.6 * winter
        wind_u = spd * np.cos(theta) + 1.4 * n_b
        wind_v = spd * np.sin(theta) + 1.4 * n_a
        # ---------------- Currents (Ekman + geostrophy + Somali jet) ---------
        # NOTE: np.gradient returns derivatives per DEGREE; convert to per-metre
        M_PER_DEG = 111_320.0
        f = 2 * OMEGA * np.sin(np.deg2rad(np.maximum(self.lat2, 4.0)))
        dsla_dy, dsla_dx = np.gradient(sla, self.lat, self.lon)
        dsla_dy /= M_PER_DEG
        dsla_dx /= M_PER_DEG
        cur_u = (
            0.030 * wind_u
            + (-G / np.maximum(np.abs(f), 1.2e-5)) * dsla_dy * np.sign(f)
            + 1.5 * monsoon * np.exp(-(((self.lon2 - 54.0) ** 2) / 20.0 + ((self.lat2 - 11.0) ** 2) / 24.0))
        )
        cur_v = 0.030 * wind_v + (G / np.maximum(np.abs(f), 1.2e-5)) * dsla_dx \
            + 0.7 * monsoon * np.exp(-(((self.lon2 - 54.5) ** 2) / 14.0 + ((self.lat2 - 9.0) ** 2) / 18.0))
        # ---------------- Subsurface temperature -----------------------------
        wind_mag = np.sqrt(wind_u ** 2 + wind_v ** 2)
        D = 26.0 + 48.0 * self.east + 10.0 * (wind_mag / 12.0) + 9.0 * bob + 85.0 * sla
        D = np.clip(D, 14.0, 220.0)
        p_exp = 1.2
        surf_term = sst - 18.0

        nz = len(self.depths)
        truth = np.empty((nz, self.ny, self.nx), dtype=np.float32)
        for k, z in enumerate(self.depths):
            td_z = 3.0 + 15.0 * np.exp(-z / 380.0)
            decay = np.exp(-((z / D) ** p_exp))
            # eddies depress/isotherm-shift the thermocline (decay with depth);
            # faint deep variability keeps deep levels physically plausible
            eddy_t = 5.5 * sla * np.exp(-z / 300.0)
            deep_wiggle = 0.30 * n_a * np.exp(-z / 650.0)
            truth[k] = (td_z + surf_term * decay + eddy_t + deep_wiggle).astype(np.float32)

        inputs = {
            "sst": sst.astype(np.float32),
            "sss": sss.astype(np.float32),
            "sla": sla.astype(np.float32),
            "cur_u": cur_u.astype(np.float32),
            "cur_v": cur_v.astype(np.float32),
            "wind_u": wind_u.astype(np.float32),
            "wind_v": wind_v.astype(np.float32),
        }
        if self.land.any():
            land3 = self.land[np.newaxis, :, :]
            for k in inputs:
                inputs[k] = np.where(land3, np.float32("nan"), inputs[k])
            truth = np.where(land3, np.float32("nan"), truth)
        return inputs, truth


def generate(start: date, end: date, out_dir: Path, resolution: float,
             depths: list[int], seed: int = 42, progress_cb=None) -> dict:
    lat = np.arange(5.0, 30.0 + 1e-9, resolution)
    lon = np.arange(45.0, 105.0 + 1e-9, resolution)
    ocean = SyntheticOcean(lat, lon, depths, start, end, seed=seed)
    ndays = (end - start).days + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    f_in = out_dir / "inputs.nc"
    f_tr = out_dir / "truth.nc"

    nc_in = netCDF4.Dataset(f_in, "w", format="NETCDF4")
    nc_tr = netCDF4.Dataset(f_tr, "w", format="NETCDF4")
    try:
        for nc, with_depth in ((nc_in, False), (nc_tr, True)):
            nc.createDimension("time", ndays)
            nc.createDimension("lat", ocean.ny)
            nc.createDimension("lon", ocean.nx)
            v = nc.createVariable("time", "f8", ("time",))
            v.units = "days since 1970-01-01"
            v.standard_name = "time"
            v[:] = netCDF4.date2num(
                [datetime(d.year, d.month, d.day) for d in (start + timedelta(days=i) for i in range(ndays))],
                v.units,
            )
            v = nc.createVariable("lat", "f4", ("lat",)); v.units = "degrees_north"; v[:] = lat
            v = nc.createVariable("lon", "f4", ("lon",)); v.units = "degrees_east"; v[:] = lon
            if with_depth:
                nc.createDimension("depth", len(depths))
                v = nc.createVariable("depth", "f4", ("depth",))
                v.units = "m"; v.positive = "down"; v[:] = np.array(depths, dtype=np.float32)

        in_vars = {}
        for name in INPUT_ORDER:
            in_vars[name] = nc_in.createVariable(
                name, "f4", ("time", "lat", "lon"), zlib=True, complevel=1,
                chunksizes=(1, ocean.ny, ocean.nx),
            )
        tr_var = nc_tr.createVariable(
            "temperature", "f4", ("time", "depth", "lat", "lon"), zlib=True, complevel=1,
            chunksizes=(1, 1, ocean.ny, ocean.nx),
        )
        tr_var.units = "degC"
        tr_var.long_name = "sea water temperature (clean synthetic truth)"

        t0 = _time.time()
        for t in range(ndays):
            fields, prof = ocean.render_day(t)
            for name in INPUT_ORDER:
                in_vars[name][t, :, :] = fields[name]
            tr_var[t, :, :, :] = prof
            if progress_cb:
                progress_cb(int(100 * (t + 1) / ndays), f"generating day {t + 1}/{ndays}")
            elif (t % 60 == 0 and t > 0) or t == ndays - 1:
                print(f"  synthetic: {t + 1}/{ndays} days "
                      f"({_time.time() - t0:.0f}s)", flush=True)
    finally:
        nc_in.close(); nc_tr.close()

    land_fraction = float(ocean.land.mean())
    manifest = {
        "type": "synthetic",
        "created": datetime.utcnow().isoformat() + "Z",
        "seed": int(seed),
        "resolution": float(resolution),
        "date_range": [start.isoformat(), end.isoformat()],
        "ndays": ndays,
        "lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 105.0,
        "depths_m": list(depths),
        "input_variables": INPUT_ORDER,
        "land_fraction": land_fraction,
        "files": {"inputs": str(f_in.name), "truth": str(f_tr.name)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def generate_with_config(cfg: dict, progress_cb=None) -> dict:
    ds = cfg["data_source"]
    if ds["type"] != "synthetic":
        raise ValueError("data_source.type is not 'synthetic'; refusing to overwrite real data")
    from src.data.loaders import synthetic_dir
    out_dir = synthetic_dir(cfg)
    start = date.fromisoformat(str(cfg["time_range"]["start"]))
    end = date.fromisoformat(str(cfg["time_range"]["end"]))
    return generate(start, end, out_dir, cfg["grid"]["resolution"],
                    cfg["depths_m"], seed=cfg.get("training", {}).get("seed", 42),
                    progress_cb=progress_cb)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic NIO dataset")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--start", default=None, help="override start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="override end date YYYY-MM-DD")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    import os
    if args.config:
        os.environ["SUBOCEANNET_CONFIG"] = args.config
    cfg = load_config()
    if args.start:
        cfg["time_range"]["start"] = args.start
    if args.end:
        cfg["time_range"]["end"] = args.end
    if args.seed:
        cfg.setdefault("training", {})["seed"] = args.seed
    print(f"[generate_synthetic] range {cfg['time_range']['start']} .. {cfg['time_range']['end']}")
    manifest = generate_with_config(cfg)
    print(f"[generate_synthetic] done -> {manifest['files']}")
    return manifest


if __name__ == "__main__":
    main()
