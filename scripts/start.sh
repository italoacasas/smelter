#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

source "$SCRIPT_DIR/load-config.sh"

BASE="http://localhost:$PORT"
DEADLINE=$((SECONDS + STARTUP_TIMEOUT))

echo "==> Starting inference service..."
docker compose up -d

echo "==> Waiting for container health at $BASE/health"
until curl -sf "$BASE/health" >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    echo "Error: container health check did not become ready within ${STARTUP_TIMEOUT}s."
    echo "Run 'scripts/logs.sh' to inspect startup logs."
    exit 1
  fi
  sleep 2
done

echo "==> Container is healthy. Waiting for model readiness at $BASE/v1/models"
until curl -sf "$BASE/v1/models" >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    echo "Error: model API did not become ready within ${STARTUP_TIMEOUT}s."
    echo "Model: $MODEL_ID"
    echo "Run 'scripts/logs.sh' to inspect startup logs."
    exit 1
  fi
  sleep 2
done

echo "==> Service ready at $BASE"
echo "    Model: $MODEL_ID"
echo "    Run 'scripts/health-check.sh' for full API verification."
