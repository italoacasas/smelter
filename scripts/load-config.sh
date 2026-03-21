#!/usr/bin/env bash
# Reads models.json + hardware.json + .active and exports config as env vars.
# Source this from other scripts: source "$(dirname "$0")/load-config.sh"

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_FILE="$PROJ_DIR/.active"
MODELS_JSON="$PROJ_DIR/models.json"
HARDWARE_JSON="$PROJ_DIR/hardware.json"

if [ ! -f "$MODELS_JSON" ]; then
    echo "Error: models.json not found." >&2
    exit 1
fi

if [ ! -f "$HARDWARE_JSON" ]; then
    echo "Error: hardware.json not found." >&2
    exit 1
fi

if [ ! -f "$ACTIVE_FILE" ]; then
    echo "Error: no active config. Run: make use MODEL=<name> HARDWARE=<name>" >&2
    exit 1
fi

ACTIVE_MODEL="$(sed -n '1p' "$ACTIVE_FILE")"
ACTIVE_HARDWARE="$(sed -n '2p' "$ACTIVE_FILE")"

# Use python to parse JSON and emit env vars
eval "$(python3 -c "
import json, sys

models = json.loads(open('$MODELS_JSON').read())
hardware = json.loads(open('$HARDWARE_JSON').read())

if '$ACTIVE_MODEL' not in models:
    print(f'echo \"Error: unknown model \$ACTIVE_MODEL\" >&2; exit 1')
    sys.exit()
if '$ACTIVE_HARDWARE' not in hardware:
    print(f'echo \"Error: unknown hardware profile \$ACTIVE_HARDWARE\" >&2; exit 1')
    sys.exit()

model_cfg = models['$ACTIVE_MODEL']
model_shared = models.get('_shared', {})
hw = hardware['$ACTIVE_HARDWARE']
hw_model = hw.get('models', {}).get('$ACTIVE_MODEL', {})

# Merge extra_args: model args + hardware per-model args
model_args = model_cfg.get('extra_args', [])
hw_args = hw_model.get('extra_args', [])
all_extra = model_args + hw_args

# Model exports
print(f'export MODEL_ID={model_cfg[\"model_id\"]!r}')
print(f'export EXTRA_LAUNCH_ARGS={\" \".join(all_extra)!r}')
print(f'export PORT={model_shared.get(\"port\", 11435)}')
print(f'export LOG_LEVEL={model_shared.get(\"log_level\", \"warning\")!r}')
print(f'export STARTUP_TIMEOUT={model_shared.get(\"startup_timeout\", 600)}')

# Hardware exports (model-specific values from hw.models.<model>)
print(f'export MEM_FRACTION_STATIC={hw_model[\"mem_fraction_static\"]}')
print(f'export CONTEXT_LENGTH={hw_model[\"context_length\"]}')
print(f'export SHM_SIZE={hw[\"shm_size\"]!r}')
print(f'export TP={hw[\"tp\"]}')
print(f'export EP={hw[\"ep\"]}')
print(f'export GPU_COUNT={hw[\"gpu_count\"]}')
print(f'export ATTENTION_BACKEND={hw[\"attention_backend\"]!r}')
print(f'export CHUNKED_PREFILL_SIZE={hw[\"chunked_prefill_size\"]}')
print(f'export NUM_CONTINUOUS_DECODE_STEPS={hw[\"num_continuous_decode_steps\"]}')
")"

# HF_TOKEN from environment if set
export HF_TOKEN="${HF_TOKEN:-}"
