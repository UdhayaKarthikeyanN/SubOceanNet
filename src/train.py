"""Train the surface->profile demo model (F4).

Pairs are (spatial patch of the 7 surface variables) -> (temperature profile at
the centre cell). Train/validation split strictly BY TIME: the last
``training.holdout_last_days`` days are held out (e.g. all of 2023).

Artifacts written to one bundle directory (default data/checkpoints/oceanembed_demo):
    checkpoint.pt   model weights
    scalars.json    normalization means/stds (inputs + temperature)
    metadata.json   input vars, depths, version, val metrics, history summary
    history.json / loss_curve.csv

Usage:
    python src/train.py            # config-driven full run
    python src/train.py --auto     # quick first-run demo profile
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import bundle_dir, depths as cfg_depths, input_variable_names, load_config
    from src.data.generate_synthetic import INPUT_ORDER
    from src.models import TemperatureModel, build_model, count_parameters
else:
    from src.config import bundle_dir, depths as cfg_depths, input_variable_names, load_config
    from src.data.generate_synthetic import INPUT_ORDER
    from src.models import TemperatureModel, build_model, count_parameters


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SurfaceProfileDataset(Dataset):
    """Random (patch -> profile) pairs drawn from a preloaded day-subset."""

    def __init__(self, inputs_nc: Path, truth_nc: Path, day_indices: np.ndarray,
                 patch_size: int, order: list[str], scalars: dict,
                 noise_sigma: float = 0.0, deterministic: bool = False,
                 center_stride: int = 5, max_centers: int | None = None,
                 seed: int = 42):
        import xarray as xr

        self.patch = patch_size
        self.r = (patch_size - 1) // 2
        self.noise_sigma = noise_sigma
        self.deterministic = deterministic
        rng = np.random.default_rng(seed)

        ds_in = xr.open_dataset(inputs_nc)
        ds_tr = xr.open_dataset(truth_nc)
        self.inputs = ds_in[order].isel(time=day_indices) \
            .to_array(dim="variable").transpose("time", "lat", "lon", "variable") \
            .values.astype(np.float32)                       # (T, lat, lon, 7)
        self.inputs = np.transpose(self.inputs, (0, 3, 1, 2))  # (T, 7, H, W)
        self.truth = ds_tr["temperature"].isel(time=day_indices) \
            .values.astype(np.float32)                        # (T, nz, H, W)
        self.ndays, _, self.ny, self.nx = self.inputs.shape[0], len(ds_tr["depth"]), \
            self.inputs.shape[2], self.inputs.shape[3]

        ocean = np.isfinite(self.truth[:, 0]).all(axis=0)      # (H, W) True=ocean
        self.ocean_ij = np.argwhere(ocean)

        # per-variable input normalization (parity with inference pipeline)
        self.in_mean = np.asarray([scalars[v]["mean"] for v in order], dtype=np.float32)[:, None, None]
        self.in_std = np.asarray([max(scalars[v].get("std", 1.0), 1e-6) for v in order],
                                 dtype=np.float32)[:, None, None]

        t_mean = np.asarray(scalars["temperature"]["mean"], dtype=np.float32)
        t_std = np.asarray(scalars["temperature"]["std"], dtype=np.float32)
        self.t_mean = t_mean[:, None]
        self.t_std = t_std[:, None]

        # padded input cube for fast patch extraction
        self.padded = np.pad(self.inputs, ((0, 0), (0, 0), (self.r, self.r), (self.r, self.r)),
                             mode="edge")

        if deterministic:
            sel = self.ocean_ij[::center_stride]
            if max_centers and len(sel) > max_centers:
                step = max(1, len(sel) // max_centers)
                sel = sel[::step][:max_centers]
            days = np.arange(self.ndays)
            d_idx, c_idx = np.meshgrid(days, np.arange(len(sel)), indexing="ij")
            self.samples = np.stack([d_idx.ravel(), sel[c_idx.ravel(), 0],
                                     sel[c_idx.ravel(), 1]], axis=1)
        else:
            self.samples = None  # sampled lazily in __getitem__
            self._rng = rng
            self._epoch_len = min(len(self.ocean_ij), 20000) * self.ndays

        ds_in.close(); ds_tr.close()

    def set_epoch_seed(self, epoch: int):
        if not self.deterministic:
            self._rng = np.random.default_rng(1000 + epoch)

    def __len__(self):
        return len(self.samples) if self.samples is not None else self._epoch_len

    def _patch_at(self, t: int, y: int, x: int) -> np.ndarray:
        y0, x0 = y, x   # padding shifts index by r inside `padded`
        return self.padded[t, :, y0: y0 + 2 * self.r + 1, x0: x0 + 2 * self.r + 1]

    def __getitem__(self, idx: int):
        if self.samples is not None:
            t, y, x = self.samples[idx % len(self.samples)]
        else:
            t = int(self._rng.integers(0, self.ndays))
            c = self.ocean_ij[self._rng.integers(0, len(self.ocean_ij))]
            y, x = int(c[0]), int(c[1])
        patch = (self._patch_at(t, y, x) - self.in_mean) / self.in_std
        target = self.truth[t, :, y, x].copy()
        if self.noise_sigma > 0:                      # ARGO-like instrument noise
            target = target + self._rng.normal(0, self.noise_sigma, size=target.shape)
        target_n = (target - self.t_mean[:, 0]) / self.t_std[:, 0]
        patch = np.nan_to_num(patch, nan=0.0)
        return torch.from_numpy(patch.astype(np.float32)), \
               torch.from_numpy(target_n.astype(np.float32))


# ---------------------------------------------------------------------------
# Scalars / persistence
# ---------------------------------------------------------------------------

def ensure_scalars(cfg: dict, force: bool = False) -> tuple[dict, Path]:
    import xarray as xr

    bdir = bundle_dir(cfg)
    spath = bdir / "scalars.json"
    if spath.exists() and not force:
        return json.loads(spath.read_text()), spath

    root = Path(cfg["_root"])
    from src.data.loaders import synthetic_dir
    syn = synthetic_dir(cfg)
    inputs_path = syn / "inputs.nc"
    truth_path = syn / "truth.nc"
    if not inputs_path.exists():
        raise FileNotFoundError(
            f"Dataset missing ({inputs_path}). Run: python -m src.data.generate_synthetic")
    from src.data.preprocessing import compute_scalars, save_scalars
    ds_in = xr.open_dataset(inputs_path)[input_variable_names(cfg)]
    ds_tr = xr.open_dataset(truth_path)["temperature"]
    scalars = compute_scalars(ds_in, ds_tr)
    bdir.mkdir(parents=True, exist_ok=True)
    save_scalars(scalars, spath)
    return scalars, spath


def save_bundle(cfg: dict, model: TemperatureModel, scalars: dict, meta: dict) -> Path:
    from src.data.preprocessing import save_scalars
    bdir = bundle_dir(cfg)
    bdir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "encoder_type": type(model.encoder).__name__,
        "latent_dim": model.encoder.latent_dim,
        "depths": model.depths,
        "input_variables": model.input_variables,
    }, bdir / "checkpoint.pt")
    save_scalars(scalars, bdir / "scalars.json")
    (bdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return bdir


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def run_training(cfg: dict, auto: bool = False, epochs: int | None = None,
                 steps_per_epoch: int | None = None, device: str = "cpu",
                 progress_cb=None, quiet: bool = False) -> dict:
    t_start = time.time()
    tcfg = dict(cfg["training"])
    if auto:
        tcfg.update(cfg["training"].get("auto", {}))
    if epochs is not None:
        tcfg["epochs"] = epochs
    if steps_per_epoch is not None:
        tcfg["steps_per_epoch"] = steps_per_epoch

    seed = int(tcfg.get("seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)

    root = Path(cfg["_root"])
    from src.data.loaders import synthetic_dir
    syn = synthetic_dir(cfg)
    inputs_nc = syn / "inputs.nc"
    truth_nc = syn / "truth.nc"
    if not inputs_nc.exists():
        raise FileNotFoundError(
            f"Dataset missing ({inputs_nc}). Run: python -m src.data.generate_synthetic")

    import xarray as xr
    n_time = xr.open_dataset(inputs_nc).sizes["time"]
    holdout = int(tcfg.get("holdout_last_days", 365))
    stride = int(tcfg.get("preload_day_stride", 4))
    split_t0 = max(n_time - holdout, int(n_time * 0.75))
    train_days = np.arange(0, split_t0, max(stride // 2, 1))
    val_days = np.arange(split_t0, n_time, stride)

    scalars, _ = ensure_scalars(cfg)
    order = input_variable_names(cfg)
    patch = int(cfg["inference"]["patch_size"])

    def cb(pct, msg):
        if progress_cb:
            progress_cb(int(pct), msg)
        if not quiet and msg:
            print(f"  [{pct:3d}%] {msg}", flush=True)

    cb(2, "loading training windows")
    train_ds = SurfaceProfileDataset(
        inputs_nc, truth_nc, train_days, patch, order, scalars,
        noise_sigma=float(tcfg.get("target_noise_sigma", 0.12)), seed=seed)
    val_ds = SurfaceProfileDataset(
        inputs_nc, truth_nc, val_days, patch, order, scalars,
        noise_sigma=0.0, deterministic=True,
        center_stride=int(tcfg.get("val_center_stride", 6)), max_centers=400)

    model = build_model(cfg)
    model.input_variables = order
    model.version = str(cfg.get("version", "1"))
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1.5e-3)),
                            weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(tcfg["epochs"]), 1))

    batch = int(tcfg.get("batch_size", 64))
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0,
                        generator=g, drop_last=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_rmse_c": [], "lr": []}
    total_epochs = int(tcfg["epochs"])
    steps_target = int(tcfg.get("steps_per_epoch", 300))

    for epoch in range(total_epochs):
        train_ds.set_epoch_seed(epoch)
        model.train()
        running, seen = 0.0, 0
        for step, (x, y) in enumerate(loader):
            if step >= steps_target:
                break
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            running += float(loss) * x.size(0)
            seen += x.size(0)
        sched.step()
        train_loss = running / max(seen, 1)

        # ---- validation (clean truth, degrees C) ----
        model.eval()
        vloader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
        vloss_n, se, cnt, nsamp = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for x, y in vloader:
                p = model(x.to(dev)).cpu().numpy()
                yn = y.numpy()
                vloss_n += float(((p - yn) ** 2).sum())
                pc = p * val_ds.t_std[:, 0] + val_ds.t_mean[:, 0]
                yc = yn * val_ds.t_std[:, 0] + val_ds.t_mean[:, 0]
                se += float(((pc - yc) ** 2).sum())
                cnt += pc.size
                nsamp += len(x)
        val_loss = vloss_n / max(cnt, 1)
        val_rmse_c = float(np.sqrt(se / max(cnt, 1)))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vloss_n)
        history["val_rmse_c"].append(val_rmse_c)
        history["lr"].append(float(sched.get_last_lr()[0]))
        pct = 10 + int(85 * (epoch + 1) / total_epochs)
        cb(pct, f"epoch {epoch + 1}/{total_epochs} "
                f"train={train_loss:.4f} val_rmse={val_rmse_c:.3f} degC")

    # ---- persist bundle ----
    cb(96, "saving checkpoint bundle")
    tr_dates = (date.fromisoformat(str(cfg["time_range"]["start"])) +
                timedelta(days=int(train_days[0])),
                date.fromisoformat(str(cfg["time_range"]["start"])) +
                timedelta(days=int(train_days[-1])))
    meta = {
        "version": str(cfg.get("version", "1")),
        "model_version": f"{cfg.get('project','oceanembed')}-{cfg.get('version','1')}",
        "encoder_type": type(model.encoder).__name__,
        "architecture_summary": model.architecture_summary(),
        "parameters": count_parameters(model),
        "depths_m": cfg_depths(cfg),
        "input_variables": order,
        "patch_size": patch,
        "trained_range": [tr_dates[0].isoformat(), tr_dates[1].isoformat()],
        "val_rmse_degC": history["val_rmse_c"][-1] if history["val_rmse_c"] else None,
        "demo_mode": bool(cfg["data_source"]["type"] == "synthetic"),
        "history": history,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    bdir = save_bundle(cfg, model, scalars, meta)
    (bdir / "history.json").write_text(json.dumps(history, indent=2))
    with open(bdir / "loss_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_loss_norm", "val_rmse_degC", "lr"])
        for i in range(len(history["epoch"])):
            w.writerow([history["epoch"][i], history["train_loss"][i],
                        history["val_loss"][i], history["val_rmse_c"][i], history["lr"][i]])
    cb(100, f"bundle saved -> {bdir}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="OceanEmbed trainer")
    ap.add_argument("--config", default=None)
    ap.add_argument("--auto", action="store_true", help="quick first-run demo profile")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.config:
        os.environ["OCEANEMBED_CONFIG"] = args.config
    cfg = load_config()
    print(f"[train] source={cfg['data_source']['type']} auto={args.auto}")
    meta = run_training(cfg, auto=args.auto, epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch, device=args.device)
    print(f"[train] done. val_rmse={meta['val_rmse_degC']:.3f} degC")
    return meta


if __name__ == "__main__":
    main()
