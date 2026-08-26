# OceanEmbed

**Satellite-based 3D subsurface ocean temperature reconstruction for the North Indian Ocean.**

OceanEmbed learns a mapping from **7 satellite-observed surface variables** to
**subsurface temperature at 15 standard depth levels** using an AI
encoder–decoder architecture:

```
INPUT VARIABLES            AI EMBEDDING                 TEMPERATURE PREDICTION
(7 surface fields)   →    CNN/ViT encoder → latent     decoder → T(z) at
SST · SSS · SLA ·         vector (dim 256)             15 depths + MC-dropout
currents U,V ·                                         uncertainty maps
winds U,V                                              (ONLY temperature is output)
```

* **Domain:** 5°N–30°N, 45°E–105°E · daily · 0.25° × 0.25° grid (101 × 241)
* **Depths:** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
* Runs fully offline on localhost — no cloud services.

---

## Quick start (Windows)

```powershell
# one-time bootstrap: venv, deps, synthetic data, demo model, frontend deps
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

# every day afterwards - one command up, one command down:
bin\start.cmd        # starts uvicorn (:8000) + vite (:5173) in minimized windows
bin\stop.cmd         # stops both (safe: only kills OceanEmbed processes)
```

Open **http://localhost:5173**. Backend docs: **http://127.0.0.1:8000/docs**.

Custom ports (e.g. when 8000 is taken):

```bat
bin\start.cmd -ApiPort 8017 -WebPort 5199
bin\stop.cmd  -ApiPort 8017 -WebPort 5199
```

Prefer visible server windows / manual control?

```powershell
# terminal A - backend API (FastAPI)
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# terminal B - frontend (Vite dev server)
cd frontend
npm.cmd run dev
```

On Linux/macOS use `scripts/setup.sh`, then `bin/start.sh` / `bin/stop.sh`
(logs land in `.run/*.log`; pass custom ports as `bin/start.sh 8017 5199`).

> **First run:** if you skip the setup script's data/training steps, the backend
> auto-generates the synthetic dataset and trains a small demo model in a
> background thread on first startup — the UI shows a "warming up" chip until
> it finishes (~1–2 min CPU).

---

## Demo mode vs real data (config-only swap)

Everything lives in `configs/config.yaml`.

### Demo mode (default)

```yaml
data_source:
  type: synthetic        # <- physically plausible generated ocean
```

`src/data/generate_synthetic.py` writes `data/synthetic/{inputs.nc,truth.nc}`
with realistic NIO physics: seasonal monsoon SST/wind reversal, Arabian Sea
summer upwelling, fresher Bay of Bengal (+ Ganges plume), west→east thermocline
deepening, barrier-layer effect in the BoB, westward-drifting mesoscale eddies,
wind-driven + geostrophic currents, ARGO-like noise added to training targets,
and land masking from coarse coastlines. Validation then runs against the clean
synthetic truth and every screen labels it **"SYNTHETIC REFERENCE — DEMO MODE."**

### Real data (GLORYS12V / satellite / ARGO)

1. Export per-variable daily files on a regular lat/lon grid into `data/raw/`
   named exactly after the config variables:

   ```
   data/raw/sst.nc  data/raw/sss.nc  data/raw/sla.nc
   data/raw/cur_u.nc data/raw/cur_v.nc data/raw/wind_u.nc data/raw/wind_v.nc
   ```

   Each file needs dims `(time, lat, lon)` (`latitude/longitude/nav_lat/nav_lon`
   are auto-renamed).
2. Put reference truth in `data/reference/*.nc` with a `temperature`
   variable of dims `(time, depth, lat, lon)`.
3. Flip one line — **no code changes**:

```yaml
data_source:
  type: netcdf           # was: synthetic
  netcdf:
    input_dir: data/raw
    reference_dir: data/reference
```

4. Re-train so scalars/checkpoint match the new distribution:
   `.venv\Scripts\python.exe src\train.py`

---

## Architecture & code map

```
oceanembed/
├── configs/config.yaml          # region, grid, depths, model, training, paths
├── backend/main.py              # FastAPI app (all endpoints, job queue, CORS)
├── src/
│   ├── config.py                # YAML loader (env OCEANEMBED_CONFIG overrides path)
│   ├── data/
│   │   ├── generate_synthetic.py  # physics-based demo generator (F4)
│   │   ├── loaders.py             # synthetic | NetCDF input/reference loading
│   │   ├── preprocessing.py       # F2 pipeline: QC, gap-fill, regrid, normalize…
│   │   └── landmask.py            # shared coastline polygons (mask + basemap)
│   ├── models/
│   │   ├── base.py                # BaseEncoder ABC, TemperatureModel, factory
│   │   ├── cnn_encoder.py         # default encoder 7→32→64→128 → latent 256
│   │   ├── vit_encoder.py         # ViT stub (patch embed + transformer), TODOs marked
│   │   └── decoder.py             # MLP head latent → exactly 15 temperatures
│   ├── train.py                   # time-split training, bundle writer
│   ├── inference.py               # Predictor: preprocess → encode → decode → denorm
│   └── validation/metrics.py      # RMSE/MAE/Bias/r/R² per depth + bands + cache
├── frontend/src/
│   ├── components/  MapCanvas (Leaflet draw+raster), PipelineBanner, DepthSlider…
│   ├── views/       Inputs, Prediction, Volume3D (Plotly), Profiles, TimeSeries, Validation
│   ├── api/client.ts                # typed REST client + job polling
│   └── state/AppContext.tsx         # shared date/region/prediction state
├── bin/                         # start.cmd / stop.cmd (+ .sh) - daily drivers
├── scripts/setup.ps1|sh
├── tests/                       # pytest: preprocessing, shapes, API smoke
└── data/                            # raw/ synthetic/ reference/ checkpoints/ cache/
```

### Model contract (hard constraint)

Input tensor `(B, 7, H, W)` — always the 7 surface variables, never anything else.
The single decoder head outputs `(B, 15)` = temperature at the configured depth
levels. SST/SSS/SLA/currents/winds can never appear in predictions; this is
enforced by architecture and asserted by tests (`test_model_shapes.py`,
`test_api_smoke.py` greps the prediction JSON).

Uncertainty: Monte-Carlo dropout — ≥5 stochastic passes (default 8) with only
dropout layers active produce per-depth std-dev maps exposed as an uncertainty
layer on every output surface.

---

## API overview (interactive docs at http://127.0.0.1:8000/docs)

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | bounds, grid, depths, variables, date range, presets, coastline GeoJSON |
| `GET /api/layers/{variable}?date=&region=` | sparse raster for an INPUT layer |
| `POST /api/predict` `{date, region_geojson}` | starts a prediction **job** → `job_id` |
| `GET /api/jobs/{id}` | poll status/stage/progress/result (15 rasters + uncertainty) |
| `GET /api/profile?lat=&lon=&date=` | T(z) at a point + reference overlay |
| `GET /api/timeseries?lat=&lon=&depth=` | predicted series (point or polygon mean) |
| `GET /api/validation?region=&date=` | per-depth RMSE/MAE/Bias/r/R² + band skill (cached) |
| `GET /api/model/info` | architecture summary, training history, version, warm-up status |

Errors are meaningful HTTP codes: `400` invalid date/region/too-large selection,
`404` unknown variable/job, `503` model still warming up or missing checkpoint.

CLI inference without the server:

```powershell
.venv\Scripts\python.exe src\inference.py --date 2023-06-15 `
  --region '{"type":"Polygon","coordinates":[[[52,8],[70,8],[70,20],[52,20],[52,8]]]}' --summary-only
```

---

## Training

```powershell
# quick demo profile (what first-run uses)
.venv\Scripts\python.exe src\train.py --auto

# full config-driven training
.venv\Scripts\python.exe src\train.py
```

* Pairs = (9×9 patch of the 7 inputs) → profile at centre cell.
* Split **strictly by time**: last `training.holdout_last_days` (365) days held out.
* Logs: `data/checkpoints/<bundle>/loss_curve.csv`, `history.json`.
* Bundle = `checkpoint.pt` + `scalars.json` (normalization parity) + `metadata.json`
  (input vars, depths, version, metrics) — everything inference needs in one folder.

Tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\
```

---

## Using the UI

1. **Select a region** — draw a rectangle/polygon on the Leaflet map or click a
   preset (Arabian Sea / Bay of Bengal / Equatorial Indian Ocean). Selection
   stats show bounds + approximate cell count.
2. **Pick a date** within the available range (header picker).
3. **Input Layers tab** — toggle any of the 7 overlays; hover the map for values.
   Pipeline banner highlights stage 01.
4. **Prediction Maps tab** — *Run prediction*. Progress shows preprocessing →
   embedding → decoding stages. Use the **depth slider / animate sweep**, toggle
   **Temperature ↔ Uncertainty ±σ**. Banner highlights embedding → prediction.
5. **3D Volume** — rotatable depth-stacked slices (or experimental isosurface).
6. **Vertical Profiles** — enable picking, drop up to 5 points on the map;
   dashed lines = reference truth; whiskers = ±2σ MC-dropout.
7. **Time Series** — point or polygon-mean temperature through time per depth.
8. **Validation** — 15-depth metric table (RMSE/MAE/Bias/r/R²), band skill cards
   (mixed layer / thermocline / deep), RMSE bars, prediction-vs-reference density
   scatter. Results cached by (dates, region-hash).

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "cannot reach the API" | start uvicorn (terminal A); check port 8000 |
| 503 “model is warming up” | wait ~1 min; watch the header chip / `/api/model/info` |
| `FileNotFoundError ... inputs.nc` | `python -m src.data.generate_synthetic` (or rerun setup) |
| torch wheel fails to install | requires Python 3.10–3.12; create venv with `py -3.11` |
| npm blocked by execution policy | call `npm.cmd` explicitly |
| port 8000 already in use by another app | run uvicorn on e.g. 8017 and start the dev server with `$env:OCEANEMBED_API="http://127.0.0.1:8017"; npm.cmd run dev` (the Vite proxy honours that env var) |

## Non-goals

Operational forecasting, non-NIO regions, sub-0.25° super-resolution, real-time
satellite ingestion, GPU requirement, authentication.
