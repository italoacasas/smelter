#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

source "$SCRIPT_DIR/load-config.sh"

echo "==> Stopping workload: $ACTIVE_WORKLOAD"
docker compose -f "$COMPOSE_FILE" down
echo "==> Workload stopped."
