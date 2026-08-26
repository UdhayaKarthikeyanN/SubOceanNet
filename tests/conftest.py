"""Shared fixtures: a miniature OceanEmbed environment (tiny grid/dates,
real generator + real trainer) so API/model/preprocessing tests stay fast."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _tiny_config(root: Path) -> dict:
    return {
        "project": "oceanembed-test",
        "version": "0.0.0",
        "region": {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 105.0},
        "grid": {"resolution": 1.0},
        "depths_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
        "input_variables": [
            {"name": "sst", "long_name": "Sea Surface Temperature", "units": "degC",
             "valid_min": -2.0, "valid_max": 40.0},
            {"name": "sss", "long_name": "Sea Surface Salinity", "units": "psu",
             "valid_min": 25.0, "valid_max": 40.0},
            {"name": "sla", "long_name": "Sea Level Anomaly", "units": "m",
             "valid_min": -1.5, "valid_max": 1.5},
            {"name": "cur_u", "long_name": "Surface Current U", "units": "m/s",
             "valid_min": -4.0, "valid_max": 4.0},
            {"name": "cur_v", "long_name": "Surface Current V", "units": "m/s",
             "valid_min": -4.0, "valid_max": 4.0},
            {"name": "wind_u", "long_name": "Wind U", "units": "m/s",
             "valid_min": -25.0, "valid_max": 25.0},
            {"name": "wind_v", "long_name": "Wind V", "units": "m/s",
             "valid_min": -25.0, "valid_max": 25.0},
        ],
        "data_source": {
            "type": "synthetic",
            "netcdf": {"input_dir": str(root / "data" / "raw"),
                       "reference_dir": str(root / "data" / "reference"),
                       "time_coord": "time", "depth_coord": "depth"},
        },
        "time_range": {"start": "2023-01-01", "end": "2023-01-22"},
        "model": {
            "encoder_type": "cnn", "latent_dim": 32,
            "cnn_channels": [8, 16],
            "vit": {"embed_dim": 24, "patch_size": 3, "num_layers": 1, "num_heads": 2,
                    "mlp_ratio": 2.0},
            "decoder_hidden": [64, 32], "dropout": 0.25,
        },
        "inference": {"patch_size": 9, "mc_passes": 3, "batch_size": 64,
                      "max_region_cells": 500, "max_timeseries_dates": 30},
        "training": {
            "holdout_last_days": 5, "preload_day_stride": 1, "val_center_stride": 4,
            "epochs": 2, "steps_per_epoch": 40, "batch_size": 32, "lr": 0.004,
            "weight_decay": 0.0, "target_noise_sigma": 0.05, "seed": 7,
            "auto": {"epochs": 2, "steps_per_epoch": 40, "preload_day_stride": 1},
        },
        "paths": {"checkpoints_dir": str(root / "data" / "checkpoints"),
                  "bundle_name": "test_bundle", "scalars_file": "scalars.json",
                  "cache_dir": str(root / "data" / "cache"),
                  "synthetic_dir": str(root / "data" / "synthetic")},
        "api": {"host": "127.0.0.1", "port": 8000},
        "presets": [{"id": "box", "name": "Box", "bbox": [52.0, 5.0, 75.0, 27.0]}],
    }


@pytest.fixture(scope="session")
def test_env(tmp_path_factory):
    """Generates the tiny synthetic dataset + trains a tiny demo bundle once."""
    root = tmp_path_factory.mktemp("oe")
    cfg_dict = _tiny_config(root)
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    os.environ["OCEANEMBED_CONFIG"] = str(cfg_path)

    from src.config import load_config
    cfg = load_config()

    from src.data.generate_synthetic import generate_with_config
    generate_with_config(cfg)

    from src.train import run_training
    t0 = time.time()
    run_training(cfg, epochs=cfg["training"]["epochs"],
                 steps_per_epoch=cfg["training"]["steps_per_epoch"], quiet=True)
    print(f"\n[test_env] tiny train took {time.time() - t0:.1f}s")
    return cfg


@pytest.fixture()
def api_client(test_env):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as client:  # context manager runs startup (warm-load)
        # wait for bootstrap to mark ready (bundle already trained)
        deadline = time.time() + 60
        while time.time() < deadline:
            r = client.get("/api/model/info")
            if r.status_code == 200 and r.json().get("status") == "ready":
                break
            time.sleep(0.4)
        yield client
