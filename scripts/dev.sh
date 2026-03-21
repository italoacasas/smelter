#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

source "$SCRIPT_DIR/load-config.sh"
export LOG_LEVEL=info

echo "==> Starting inference service (dev mode, foreground)..."
docker compose up --abort-on-container-exit
