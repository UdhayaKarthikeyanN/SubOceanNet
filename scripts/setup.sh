#!/usr/bin/env bash
# One-command bootstrap for OceanEmbed (Linux/macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_DATA=0; SKIP_TRAIN=0; SKIP_NPM=0
for a in "$@"; do case "$a" in
  --skip-data) SKIP_DATA=1;; --skip-train) SKIP_TRAIN=1;; --skip-npm) SKIP_NPM=1;; esac
done

echo "== OceanEmbed setup =="
python3 --version
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip -q
echo "-> installing python dependencies"
"$PY" -m pip install -r requirements.txt

if [ "$SKIP_NPM" -eq 0 ]; then
  echo "-> installing frontend dependencies"
  ( cd frontend && npm install )
fi

if [ "$SKIP_DATA" -eq 0 ] && [ ! -f data/synthetic/inputs.nc ]; then
  echo "-> generating synthetic dataset"
  "$PY" -m src.data.generate_synthetic
fi
if [ "$SKIP_TRAIN" -eq 0 ] && [ "$SKIP_DATA" -eq 0 ] \
   && [ ! -f data/checkpoints/oceanembed_demo/checkpoint.pt ]; then
  echo "-> training demo model"
  "$PY" src/train.py --auto
fi

echo "== Setup complete =="
echo "Start the app:    bin/start.sh       (stop with bin/stop.sh)"
echo "Or manually -     Terminal A: .venv/bin/python -m uvicorn backend.main:app --port 8000"
echo "                  Terminal B: cd frontend && npm run dev"
