#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

source "$SCRIPT_DIR/load-config.sh"

declare -A SEEN_MODELS=()
TARGET_INSTANCES="${INSTANCE:-$ACTIVE_INSTANCES}"
DOWNLOAD_PYTHON="$(cat <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
repo_dir = Path("/root/.cache/huggingface/hub") / ("models--" + model_id.replace("/", "--"))

removed = 0
if repo_dir.exists():
    files = [path for path in repo_dir.rglob("*") if path.is_file() and path.stat().st_size == 0]
    for path in files:
        path.unlink()
    removed = len(files)

if removed:
    print(f"==> Removed {removed} zero-byte cache files for {model_id}")
else:
    print(f"==> Cache clean for {model_id}")

snapshot_download(model_id)
PY
)"

for instance in $TARGET_INSTANCES; do
  export INSTANCE="$instance"
  source "$SCRIPT_DIR/load-config.sh"

  if [[ -n "${SEEN_MODELS[$MODEL_ID]:-}" ]]; then
    continue
  fi
  SEEN_MODELS[$MODEL_ID]=1

  echo "==> Downloading model for $INSTANCE_NAME: $MODEL_ID"
  echo "    This may take a while..."

  docker run --rm -i \
    -v "$PROJECT_DIR/models:/root/.cache/huggingface" \
    -e "MODEL_ID=$MODEL_ID" \
    ${HF_TOKEN:+--env "HF_TOKEN=$HF_TOKEN"} \
    "$DOCKER_IMAGE" \
    python3 -c "$DOWNLOAD_PYTHON"
done

unset INSTANCE

echo "==> Model download completed."
