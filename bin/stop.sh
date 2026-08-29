#!/usr/bin/env bash
# SubOceanNet shutdown (Linux/macOS): kills pids recorded in .run/, then any
# listener on the given ports whose cwd lies inside this project tree.
# Usage: bin/stop.sh [api_port] [web_port]
API_PORT="${1:-8000}"
WEB_PORT="${2:-5173}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for name in backend frontend; do
    f="$ROOT/.run/$name.pid"
    if [ -f "$f" ]; then
        pid="$(cat "$f" 2>/dev/null)"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            cwd_target=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo "")
            case "$cwd_target" in
                "$ROOT"*) kill "$pid" 2>/dev/null || true
                          echo "[stopped] $name pid=$pid" ;;
                *) echo "[skip] pid $pid is not inside this project - not touching it" ;;
            esac
        fi
        rm -f "$f"
    fi
done

for pair in "backend $API_PORT" "frontend $WEB_PORT"; do
    set -- $pair; tag=$1; port=$2
    if command -v lsof >/dev/null 2>&1; then
        for pid in $(lsof -t -i TCP:"$port" -s TCP:LISTEN 2>/dev/null); do
            cwd_target=$(readlink "/proc/$pid/cwd" 2>/dev/null || echo "")
            case "$cwd_target" in
                "$ROOT"*) kill "$pid" 2>/dev/null || true
                          echo "[stopped] $tag(pid $pid on port $port)" ;;
                *) echo "[skip] port $port owned by pid $pid outside this project" ;;
            esac
        done
    fi
done
echo "SubOceanNet stopped."
