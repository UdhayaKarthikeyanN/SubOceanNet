# SubOceanNet

**Satellite-based 3D subsurface ocean temperature reconstruction for the North Indian Ocean.**

SubOceanNet learns a mapping from **7 satellite-observed surface variables** to
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
bin\stop.cmd         # stops both (safe: only kills SubOceanNet processes)
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

### Live data (Copernicus Marine + ERA5, auto-refreshed)

A third mode fetches the same 7 variables from real providers on a
background schedule instead of static files. It reuses the trained
demo model and the entire preprocessing/inference pipeline unchanged - only
`src/data/loaders.py` gains a third branch.

1. **One-time isolated environment** (kept separate from `.venv` on purpose -
   the `copernicusmarine` client needs `numpy>=2`, which conflicts with this
   project's pinned `numpy==1.26.4`/`torch==2.5.1`; see the comment at the
   top of `scripts/live_fetch/requirements.txt`):

   ```powershell
   python -m venv .venv-live
   .venv-live\Scripts\python.exe -m pip install -r scripts\live_fetch\requirements.txt
   ```

2. **Credentials**: copy `.env.example` to `.env` and fill in a free
   [Copernicus Marine](https://data.marine.copernicus.eu) account and a free
   [CDS API](https://cds.climate.copernicus.eu/how-to-api) token. `.env` is
   only ever read inside the isolated fetch subprocess - the main app
   process never sees these values, and no API response includes them.

3. **Dataset IDs**: `configs/config.yaml` under `data_source.live.ocean.variables`
   ships with empty `dataset_id` fields on purpose - Copernicus Marine's
   catalog changes over time, so none is guessed here. Find current ones with:

   ```powershell
   .venv-live\Scripts\copernicusmarine.exe describe --contains <keyword>
   ```

   A variable left with an empty `dataset_id` (or wind variable with no CDS
   mapping) simply falls back to cached/synthetic data for that one
   variable - it never blocks the other six or the app as a whole.

4. **Flip the mode**:

   ```yaml
   data_source:
     type: live
   ```

5. Start the app as usual (`bin\start.cmd`). A background thread refreshes
   every `data_source.live.refresh_interval_minutes` (default 180). Every
   HTTP request is served from the on-disk cache (`data/live_cache/`,
   default) - no request ever blocks on a live fetch.

**Data is predownloaded, not fetched per-run.** Providers publish at most
once a day, so re-fetching more often than `refresh_interval_minutes` just
re-downloads the same day's data - `_refresh_one()` in
`src/data/live/manager.py` skips the network call entirely when the on-disk
cache is already newer than that interval, and reports it as `CACHED REAL`
instead. In practice this means: the first run of the day pays the real
fetch cost once (a few minutes, since Copernicus Marine subset requests
take ~2-3 min each), and every restart after that - for the rest of the
interval window - starts from cache in well under a second. You never need
to fetch on every `bin\start.cmd`.

**Status labels** (never invented - always reflects what actually happened):

| Label | Meaning |
|---|---|
| `REAL` | fetched this refresh cycle |
| `CACHED REAL` | reusing a previous successful fetch (flagged `stale` past `max_age_hours`) |
| `SYNTHETIC FALLBACK` | no live data has ever succeeded for that variable - a single synthetic day fills the gap so the app keeps working |

Check `GET /api/live/status` (per-variable diagnostics) and
`GET /api/live/latest` (latest common day across all 7 + value summary) any
time - the header shows a compact `LIVE DATA · <status>` badge with the same
detail, click to expand.

**Known limitation - provider latency mismatch**: ERA5 (winds) is a
reanalysis with several days of publication latency; Copernicus Marine
ocean products are usually much fresher. The 7 variables are merged on
exact calendar-day overlap, so the *usable* live date range is bounded by
whichever provider is slowest that day - check `/api/live/status` to see
each variable's own freshness before assuming "live" means "today."

**Known limitation - no live reference/validation data**: the Validation
tab still compares against the synthetic truth in live mode (there is no
ARGO/GLORYS reference-ingestion pipeline here) - metrics there remain
labelled `SYNTHETIC REFERENCE — DEMO MODE` regardless of `data_source.type`.

---

## Architecture & code map

```
suboceannet/
├── configs/config.yaml          # region, grid, depths, model, training, paths
├── backend/main.py              # FastAPI app (all endpoints, job queue, CORS)
├── src/
│   ├── config.py                # YAML loader (env SUBOCEANNET_CONFIG overrides path)
│   ├── data/
│   │   ├── generate_synthetic.py  # physics-based demo generator (F4)
│   │   ├── loaders.py             # synthetic | NetCDF | live input/reference loading
│   │   ├── preprocessing.py       # F2 pipeline: QC, gap-fill, regrid, normalize…
│   │   ├── landmask.py            # shared coastline polygons (mask + basemap)
│   │   └── live/                  # F8: live-data manager, cache, status (see below)
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
├── scripts/
│   ├── setup.ps1|sh
│   └── live_fetch/              # isolated-venv fetch scripts (Copernicus Marine + ERA5)
├── tests/                       # pytest: preprocessing, shapes, API smoke, live pipeline
└── data/                        # raw/ synthetic/ reference/ checkpoints/ cache/ live_cache/
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
| `GET /api/live/status` | per-variable REAL / CACHED REAL / SYNTHETIC FALLBACK, provider, observation time, data age (any mode - reports `live_mode_active: false` when not in live mode) |
| `GET /api/live/latest` | latest calendar day common to all 7 live variables + value summary (live mode only - `409` otherwise) |

Errors are meaningful HTTP codes: `400` invalid date/region/too-large selection,
`404` unknown variable/job, `409` live-only endpoint called outside live mode,
`503` model still warming up or missing checkpoint.

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

## Running in production (single server, real domain)

By default the frontend runs via Vite's dev server and CORS only allows
`localhost`. To serve everything from one FastAPI process behind your own
domain:

```powershell
# 1. build the frontend to static files
cd frontend
npm.cmd run build          # writes frontend/dist
cd ..

# 2. tell the backend which origin(s) may call it (comma-separated)
$env:SUBOCEANNET_CORS_ORIGINS = "https://your-domain.example.com"

# 3. run uvicorn only - it now also serves the built frontend at /
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

`backend/main.py` auto-mounts `frontend/dist` at `/` whenever that folder
exists, so step 3 alone replaces both dev servers. Put a reverse proxy
(nginx/Caddy) in front for TLS, and run uvicorn under a process supervisor
(systemd unit, NSSM/Task Scheduler on Windows, or a container) instead of a
bare terminal so it restarts on crash/reboot.

Two things this repo does **not** provide yet, by design (see Non-goals):
authentication and a job queue that survives multiple worker processes
(`JOBS` in `backend/main.py` is an in-memory dict — fine for one process,
not for `--workers N`). Add both before exposing this to the public internet.

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "cannot reach the API" | start uvicorn (terminal A); check port 8000 |
| 503 “model is warming up” | wait ~1 min; watch the header chip / `/api/model/info` |
| `FileNotFoundError ... inputs.nc` | `python -m src.data.generate_synthetic` (or rerun setup) |
| torch wheel fails to install | requires Python 3.10–3.12; create venv with `py -3.11` |
| npm blocked by execution policy | call `npm.cmd` explicitly |
| port 8000 already in use by another app | run uvicorn on e.g. 8017 and start the dev server with `$env:SUBOCEANNET_API="http://127.0.0.1:8017"; npm.cmd run dev` (the Vite proxy honours that env var) |
| live mode: every variable shows `SYNTHETIC FALLBACK` | check `GET /api/live/status` for the `error` field per variable - usually a missing/blank `dataset_id`, missing `.env` credentials, or `.venv-live` not set up yet |
| live mode: `latest_common_date` is null / far in the past | providers have different latency (ERA5 winds especially) - the 7 variables only overlap on days *all* of them have data; check each variable's `data_age_hours` in `/api/live/status` |

## Non-goals

Operational forecasting, non-NIO regions, sub-0.25° super-resolution, real-time
satellite ingestion, GPU requirement, authentication.
