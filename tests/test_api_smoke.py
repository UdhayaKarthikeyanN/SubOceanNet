"""API smoke tests: full pipeline through FastAPI endpoints (F5)."""
from __future__ import annotations

import json

BOX = {"type": "Polygon",
       "coordinates": [[[55, 12], [58, 12], [58, 15], [55, 15], [55, 12]]]}
TINY = {"type": "Polygon",
        "coordinates": [[[56, 13], [57, 13], [57, 14], [56, 14], [56, 13]]]}

FORBIDDEN_KEYS = ("sst", "sss", "sla", "cur_u", "cur_v", "wind_u", "wind_v")


def test_health(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_meta_contract(api_client):
    m = api_client.get("/api/meta").json()
    assert len(m["depths"]) == 15
    assert [v["name"] for v in m["variables"]] == [
        "sst", "sss", "sla", "cur_u", "cur_v", "wind_u", "wind_v"]
    assert m["grid"]["resolution"] > 0
    assert m["date_range"][0] <= m["date_range"][1]
    assert "features" in m["land_polygons"]


def test_layer_endpoint(api_client):
    meta = api_client.get("/api/meta").json()
    d = meta["date_range"][1]
    r = api_client.get(f"/api/layers/sst?date={d}")
    assert r.status_code == 200
    layer = r.json()
    assert layer["units"] == "degC"
    assert len(layer["values"]) == len(layer["lats"]) > 0
    flat = [v for row in layer["values"] for v in row if v is not None]
    assert any(20 < v < 40 for v in flat)


def test_layer_unknown_variable_404(api_client):
    r = api_client.get("/api/layers/oxygen?date=2023-01-05")
    assert r.status_code == 404


def test_invalid_date_rejected(api_client):
    r = api_client.get("/api/layers/sst?date=1999-01-01")
    assert r.status_code == 400
    r = api_client.get("/api/profile?lat=15&lon=65&date=not-a-date")
    assert r.status_code == 400


def test_predict_job_flow_and_temperature_only_outputs(api_client):
    started = api_client.post(
        "/api/predict",
        json={"date": "2023-01-10", "region_geojson": BOX},
    )
    assert started.status_code == 200
    job_id = started.json()["job_id"]

    result = None
    import time
    deadline = time.time() + 90
    while time.time() < deadline:
        st = api_client.get(f"/api/jobs/{job_id}").json()
        if st["status"] == "done":
            result = st["result"]
            break
        if st["status"] == "error":
            raise AssertionError(st.get("message"))
        time.sleep(0.4)
    assert result is not None, "prediction did not finish in time"

    nz = 15
    ny, nx = len(result["lats"]), len(result["lons"])
    assert len(result["temperature"]) == nz
    assert len(result["uncertainty"]) == nz
    assert all(len(row) == nx for row in result["temperature"][0])
    assert all(v >= 0 for row in result["uncertainty"] for line in row
               for v in ([line] if isinstance(line, (int, float)) else []))

    # HARD CONSTRAINT: outputs contain ONLY temperature + uncertainty.
    # 1) no input variable name may appear as any key in the payload
    def keys_of(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield str(k).lower()
                yield from keys_of(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys_of(v)
    all_keys = set(keys_of(result))
    for key in FORBIDDEN_KEYS:
        assert key not in all_keys
    # 2) the only gridded output fields are temperature & uncertainty
    grids = {k for k, v in result.items()
             if isinstance(v, list) and v and isinstance(v[0], list)
             and isinstance(v[0][0], list)}
    assert grids == {"temperature", "uncertainty"}

    # unknown job -> 404
    assert api_client.get("/api/jobs/does-not-exist").status_code == 404


def test_profile_endpoint(api_client):
    r = api_client.get("/api/profile?lat=14.2&lon=66.5&date=2023-01-08")
    assert r.status_code == 200
    p = r.json()
    assert len(p["depths"]) == 15
    assert len(p["temperature"]) == 15
    assert p["reference_type"] and "SYNTHETIC" in p["reference_type"].upper()


def test_timeseries_endpoint(api_client):
    r = api_client.get(
        "/api/timeseries?lat=13.0&lon=67.0&depth=50&stride_days=5")
    assert r.status_code == 200
    ts = r.json()
    assert len(ts["dates"]) >= 3
    assert len(ts["all_depths_matrix"]) == len(ts["dates"])
    assert ts["depth_m"] == 50


def test_validation_endpoint_metrics(api_client):
    # /api/validation always reads from the fixed 2025-2026 timeseries
    # history archive (see configs/config.yaml "timeseries_history"),
    # independent of data_source.type - so the date must fall in that
    # range, not the tiny test config's synthetic 2023 range.
    r = api_client.get("/api/validation", params={"region": json.dumps(BOX),
                                                  "date": "2025-06-15"})
    assert r.status_code == 200
    v = r.json()
    assert v["reference_type"].upper().startswith("SYNTHETIC")
    assert len(v["metrics"]) == 15
    for m in v["metrics"]:
        assert set(m) >= {"depth", "rmse", "mae", "bias", "pearson_r", "r2", "n"}
    assert set(v["bands"]) >= {"mixed_layer", "thermocline", "deep"}
    assert len(v["scatter"]["points"]) > 10


def test_empty_and_invalid_regions(api_client):
    # degenerate polygon -> 400
    bad = {"type": "Polygon", "coordinates": [[[55, 12], [55.01, 12], [55, 12.01]]]}
    r = api_client.post("/api/predict", json={"date": "2023-01-10",
                                              "region_geojson": bad})
    assert r.status_code == 400
    # outside domain -> 400
    out = {"type": "Polygon", "coordinates": [[[-30, -30], [-29, -30], [-29, -29]]]}
    r = api_client.post("/api/predict", json={"date": "2023-01-10",
                                              "region_geojson": out})
    assert r.status_code == 400
