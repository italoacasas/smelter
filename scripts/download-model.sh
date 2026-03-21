#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

source "$SCRIPT_DIR/load-config.sh"

echo "==> Downloading model: $MODEL_ID"
echo "    This may take a while..."

docker run --rm \
  -v "$PROJECT_DIR/models:/root/.cache/huggingface" \
  ${HF_TOKEN:+--env "HF_TOKEN=$HF_TOKEN"} \
  lmsysorg/sglang:v0.5.9-cu130-runtime \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

echo "==> Model downloaded to $PROJECT_DIR/models"
