#!/usr/bin/env bash
# Reads the active workload, renders compose, and exports runtime env vars.
# Source this from other scripts: source "$(dirname "$0")/load-config.sh"

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(python3 "$PROJ_DIR/scripts/export_runtime_env.py")"
