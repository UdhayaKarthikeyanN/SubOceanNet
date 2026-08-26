"""Unit tests for the F2 preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def cfg(test_env):
    return test_env


def test_quality_control_spikes_and_range(cfg):
    import xarray as xr
    from src.data.preprocessing import quality_control

    t = np.arange(10)
    vals = 28.0 + 0.1 * np.sin(t)
    vals[5] = 45.0            # out of range
    rng = np.random.default_rng(0)
    series = (vals + rng.normal(0, 0.02, 10)).astype(np.float32)
    da = xr.DataArray(
        series[:, None],                       # (time, x)
        dims=("time", "x"),
        coords={"time": np.arange(10), "x": [0]},
        name="sst",
    )
    # inject a spike at t=3
    da[3, 0] = 38.0
    out, report = quality_control(da, -2.0, 40.0)
    arr = out.isel(x=0).values
    assert np.isnan(arr).sum() >= 1          # range violation removed
    assert report["spikes_removed"] >= 1     # spike detected
    assert abs(float(np.nanmean(arr)) - 28.0) < 0.5


def test_fill_missing_spatial_recovers_smooth_field():
    from src.data.preprocessing import fill_missing_spatial

    lats = np.arange(4, dtype=float)
    lons = np.arange(6, dtype=float)
    lo2, la2 = np.meshgrid(lons, lats)
    field = 20 + 0.5 * la2 + 0.1 * lo2
    hole = field.copy()
    hole[1:3, 2:4] = np.nan                   # interior gap
    filled, n = fill_missing_spatial(hole, lats, lons)
    assert n > 0 and np.isfinite(filled).all()
    err = np.abs(filled[1:3, 2:4] - field[1:3, 2:4])
    assert err.max() < 0.35                    # bilinear recovery is close


def test_normalization_round_trip(cfg):
    from src.data.preprocessing import normalize, denormalize

    sc = {"mean": 27.5, "std": 2.1}
    x = np.array([[26.0, 28.0], [30.5, 24.9]], dtype=np.float32)
    z = normalize(x, sc)
    back = denormalize(z, sc)
    assert np.allclose(back, x, atol=1e-5)


def test_region_parse_and_cell_mask():
    from src.data.preprocessing import Region, parse_region, region_cell_mask

    ring = [[50.0, 10.0], [56.0, 10.0], [56.0, 14.0], [50.0, 14.0], [50.0, 10.0]]
    reg = parse_region({"type": "Polygon", "coordinates": [ring]})
    assert isinstance(reg, Region)
    assert (reg.lat_min, reg.lat_max, reg.lon_min, reg.lon_max) == (10, 14, 50, 56)

    lats = np.arange(5.0, 31.0, 1.0)
    lons = np.arange(45.0, 106.0, 1.0)
    mask = region_cell_mask(lats, lons, reg)
    # half-open convention: centres on the min-lat/min-lon boundary count as
    # inside -> lat rows 10..13 (4) x lon cols 50..55 (6)
    assert mask.sum() == 24
    assert mask.shape == (len(lats), len(lons))


def test_parse_region_rejects_degenerate():
    from src.data.preprocessing import parse_region

    with pytest.raises(ValueError):
        parse_region({"type": "LineString",
                      "coordinates": [[50, 10], [51, 10]]})


def test_stack_variables_order_and_dims(cfg):
    from src.data.loaders import load_input_dataset
    from src.config import input_variable_names
    from src.data.preprocessing import stack_variables, crop_region

    ds = load_input_dataset(cfg)
    crop = crop_region(ds, __import__(
        "src.data.preprocessing", fromlist=["Region"]).Region(
        ring=[[50, 10], [54, 10], [54, 13], [50, 13]]))
    order = input_variable_names(cfg)
    da = stack_variables(crop, order)
    assert list(da.dims) == ["time", "lat", "lon", "variable"]
    assert da.sizes["variable"] == 7


def test_preprocess_for_inference_shapes_and_mask(cfg):
    from src.config import input_variable_names
    from src.data.preprocessing import Region, preprocess_for_inference

    pre = preprocess_for_inference(
        cfg, "2023-01-05", Region(ring=[[52, 12], [57, 12], [57, 16], [52, 16]]))
    tensor = pre["tensor"]
    order = input_variable_names(cfg)
    assert tensor.shape[0] == len(order) == 7
    assert tensor.shape[1] == len(pre["lats"])
    assert tensor.shape[2] == len(pre["lons"])
    assert pre["ocean_mask"].shape == tensor.shape[1:]
    assert pre["ocean_mask"].any()             # ocean cells exist in that box
    assert set(pre["qc_report"]["variables"]) <= set(order)
