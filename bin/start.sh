#!/usr/bin/env bash
# SubOceanNet launcher (Linux/macOS): backend + frontend dev servers.
# Usage: bin/start.sh [api_port] [web_port]
API_PORT="${1:-8000}"
WEB_PORT="${2:-5173}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/.run"; mkdir -p "$RUN"

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; } || return 1; }

# data source mode (informational only - config.yaml is the source of truth)
CONFIG_PATH="${SUBOCEANNET_CONFIG:-$ROOT/configs/config.yaml}"
DATA_MODE="unknown"
if [ -f "$CONFIG_PATH" ]; then
    DATA_MODE="$(grep -m1 -E '^\s*type:\s*\S+' "$CONFIG_PATH" | sed -E 's/^\s*type:\s*//')"
fi
echo "[mode]  data_source.type = $DATA_MODE"
if [ "$DATA_MODE" = "live" ]; then
    [ -x "$ROOT/.venv-live/bin/python" ] || echo "        WARNING: .venv-live not set up - live fetches will fall back to SYNTHETIC FALLBACK (see README 'Live data mode')"
    [ -f "$ROOT/.env" ] || echo "        WARNING: .env not found - copy .env.example and fill in credentials"
fi

if port_busy "$API_PORT"; then
    echo "[skip] port $API_PORT busy - assuming backend is up"
else
    PY="$ROOT/.venv/bin/python"
    [ -x "$PY" ] || { echo "ERROR: $PY missing - run scripts/setup.sh first"; exit 1; }
    ( cd "$ROOT" && nohup "$PY" -m uvicorn backend.main:app \
        --host 127.0.0.1 --port "$API_PORT" > "$RUN/backend.log" 2>&1 & echo $! > "$RUN/backend.pid" )
    echo "[ok]   backend  http://127.0.0.1:$API_PORT (logs: .run/backend.log)"
fi

if [ "${SKIP_FRONTEND:-0}" != "1" ]; then
    if port_busy "$WEB_PORT"; then
        echo "[skip] port $WEB_PORT busy - assuming frontend is up"
    else
        [ -d "$ROOT/frontend/node_modules" ] || { echo "ERROR: run scripts/setup.sh first"; exit 1; }
        ( cd "$ROOT/frontend" && SUBOCEANNET_API="http://127.0.0.1:$API_PORT" \
            nohup npm run dev -- --port "$WEB_PORT" > "$RUN/frontend.log" 2>&1 & echo $! > "$RUN/frontend.pid" )
        echo "[ok]   frontend http://localhost:$WEB_PORT (logs: .run/frontend.log)"
    fi
fi
echo "Stop with: bin/stop.sh $API_PORT $WEB_PORT"
