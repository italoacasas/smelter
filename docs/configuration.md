# Configuration

## Overview

Configuration is split into two JSON files and one selector:

- **`models.json`** — model definitions (`model_id`, `extra_args`) and shared settings (`port`, `log_level`, `startup_timeout`) under `_shared`
- **`hardware.json`** — hardware profiles: GPU runtime settings and per-model tuning (`mem_fraction_static`, `context_length`, `extra_args`)
- **`.active`** — two-line file (model name, hardware name), written by `make use`, git-ignored

`scripts/load-config.sh` reads all three and exports env vars consumed by `docker-compose.yml` and all scripts. `EXTRA_LAUNCH_ARGS` is the concatenation of model `extra_args` (model-intrinsic flags like `--tool-call-parser`, `--reasoning-parser`) + hardware per-model `extra_args` (hardware-dependent flags like `--kv-cache-dtype`, `--quantization`).

## Adding a New Model

Add a top-level key to `models.json`:

```json
{
  "my-model": {
    "model_id": "org/Model-Name",
    "extra_args": ["--tool-call-parser", "hermes"]
  }
}
```

Then add per-model tuning to each hardware profile in `hardware.json`:

```json
{
  "rtx-pro-6000": {
    "models": {
      "my-model": {
        "mem_fraction_static": 0.85,
        "context_length": 32768
      }
    }
  }
}
```

## Adding a New Hardware Profile

Add a top-level key to `hardware.json` with all runtime settings and per-model tuning:

```json
{
  "my-gpu": {
    "description": "1x NVIDIA RTX 4090",
    "gpu_info": { "name": "NVIDIA RTX 4090", "count": 1, "vram_gb": 24, "arch": "Ada Lovelace" },
    "shm_size": "24g",
    "tp": 1,
    "ep": 1,
    "gpu_count": 1,
    "attention_backend": "flashinfer",
    "chunked_prefill_size": 8192,
    "num_continuous_decode_steps": 8,
    "models": {
      "nemotron-cascade-2": {
        "mem_fraction_static": 0.70,
        "context_length": 32768
      }
    }
  }
}
```

## Switching Model or Hardware

```bash
make use MODEL=nemotron-cascade-2 HARDWARE=rtx-pro-6000
make stop && make start
```

Clients use the `MODEL_ID` value (from `models.json`) as the model name in API calls.

## Scripts

| Script                           | Description                                                       |
| -------------------------------- | ----------------------------------------------------------------- |
| `scripts/use.py`                 | Switch active model and hardware profile                          |
| `scripts/load-config.sh`         | Read config files and export env vars (sourced by other scripts)  |
| `scripts/start.sh`               | Start the inference service                                       |
| `scripts/stop.sh`                | Stop the inference service                                        |
| `scripts/dev.sh`                 | Start in foreground with verbose logging                          |
| `scripts/download-model.sh`      | Pre-download model weights                                        |
| `scripts/health-check.sh`        | Check container health, API availability, and model readiness     |
| `scripts/chat.sh`                | Interactive chat session using the OpenAI-compatible API          |
| `scripts/logs.sh`                | Tail service logs                                                 |
| `scripts/benchmark.py`           | Benchmark warm requests and optional cold starts                  |
| `scripts/gpu-tuning-matrix.py`   | Benchmark matrix over mem_fraction_static and context_length      |
| `scripts/refresh-moe-configs.sh` | Refresh repo-backed MoE config files with before/after benchmarks |
