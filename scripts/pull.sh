#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="$(python3 - "$PROJECT_DIR/hardware.json" <<'PY'
import json
import sys
from pathlib import Path

hardware_path = Path(sys.argv[1])
hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
print(hardware.get("docker_image", "lmsysorg/sglang:v0.5.10.post1-cu130-runtime"))
PY
)"

cd "$PROJECT_DIR"
docker pull "$IMAGE"
