#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

: "${API_KEY:?API_KEY env var must be set}"
: "${STABLEAVATAR_ROOT:=$(pwd)}"
export STABLEAVATAR_ROOT

echo "[start] launching redis..."
redis-server --daemonize yes --save "" --appendonly no

echo "[start] launching celery worker (loads model, takes ~60s)..."
celery -A api.celery_app.celery_app worker \
    --concurrency=1 --pool=solo --loglevel=info \
    > /workspace/celery.log 2>&1 &
CELERY_PID=$!
echo "[start] celery PID=$CELERY_PID (logs: /workspace/celery.log)"

echo "[start] launching uvicorn on 0.0.0.0:8000..."
exec uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
