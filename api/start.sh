#!/usr/bin/env bash
# Runs redis + celery worker + uvicorn and streams ALL logs into this one terminal.
# Prefixes lines with [celery] and [uvicorn] so you can tell them apart.
# Ctrl+C stops everything cleanly.

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
: "${STABLEAVATAR_ROOT:=$(pwd)}"
export STABLEAVATAR_ROOT

CELERY_PID=""
UVICORN_PID=""

cleanup() {
    echo
    echo "[start] shutting down..."
    [ -n "$UVICORN_PID" ] && kill "$UVICORN_PID" 2>/dev/null || true
    [ -n "$CELERY_PID" ] && kill "$CELERY_PID" 2>/dev/null || true
    redis-cli shutdown nosave 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[start] stopped."
    exit 0
}
trap cleanup INT TERM

echo "[start] launching redis..."
redis-server --daemonize yes --save "" --appendonly no

echo "[start] launching celery worker (model load ~60s)..."
stdbuf -oL -eL celery -A api.celery_app.celery_app worker \
    --concurrency=1 --pool=solo --loglevel=info 2>&1 \
    | sed -u 's/^/[celery]  /' &
CELERY_PID=$!

echo "[start] launching uvicorn on 0.0.0.0:8000..."
stdbuf -oL -eL uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 \
    | sed -u 's/^/[uvicorn] /' &
UVICORN_PID=$!

echo "[start] both processes started. Ctrl+C to stop."
wait -n "$CELERY_PID" "$UVICORN_PID"
cleanup
