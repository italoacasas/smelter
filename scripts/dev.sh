#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

export SMELTER_LOG_LEVEL_OVERRIDE=info
source "$SCRIPT_DIR/load-config.sh"

echo "==> Starting workload in dev mode: $ACTIVE_WORKLOAD"
docker compose -f "$COMPOSE_FILE" down --remove-orphans >/dev/null 2>&1 || true
docker compose -f "$COMPOSE_FILE" up --abort-on-container-exit $ACTIVE_INSTANCES
