#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

source "$SCRIPT_DIR/load-config.sh"

ACTIVE_INSTANCES_LIST="$ACTIVE_INSTANCES"

echo "==> Starting workload: $ACTIVE_WORKLOAD"
docker compose -f "$COMPOSE_FILE" down --remove-orphans >/dev/null 2>&1 || true
docker compose -f "$COMPOSE_FILE" up -d $ACTIVE_INSTANCES_LIST

for instance in $ACTIVE_INSTANCES_LIST; do
  DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
  export INSTANCE="$instance"
  source "$SCRIPT_DIR/load-config.sh"
  BASE="http://localhost:$PORT"

  echo "==> Waiting for $INSTANCE_NAME container health at $BASE/health"
  until curl -sf "$BASE/health" >/dev/null 2>&1; do
    if (( SECONDS >= DEADLINE )); then
      echo "Error: container health check did not become ready within ${STARTUP_TIMEOUT}s."
      echo "Instance: $INSTANCE_NAME"
      echo "Run 'scripts/logs.sh' to inspect startup logs."
      exit 1
    fi
    sleep 2
  done

  echo "==> $INSTANCE_NAME is healthy. Waiting for model readiness at $BASE/v1/models"
  until curl -sf "$BASE/v1/models" >/dev/null 2>&1; do
    if (( SECONDS >= DEADLINE )); then
      echo "Error: model API did not become ready within ${STARTUP_TIMEOUT}s."
      echo "Instance: $INSTANCE_NAME"
      echo "Model: $MODEL_ID"
      echo "Run 'scripts/logs.sh' to inspect startup logs."
      exit 1
    fi
    sleep 2
  done

  echo "==> Ready: $INSTANCE_NAME at $BASE (${MODEL_ID})"
done

unset INSTANCE

echo "==> Workload ready: $ACTIVE_WORKLOAD"
echo "    Run 'scripts/health-check.sh' for full API verification."
