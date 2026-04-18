# Configuration

## Overview

Smelter now resolves runtime config from four JSON files plus one selector:

- `models.json` — model definitions and shared defaults
- `hardware.json` — fixed host facts and shared container defaults
- `instances.json` — runnable server instances
- `workloads.json` — named sets of instances
- `.active` — one-line file storing the active workload name

`scripts/load-config.sh` sources `scripts/export_runtime_env.py`, which validates the config, renders `.smelter/compose.generated.yml`, and exports env vars for the active workload or a specific `INSTANCE`.

## `models.json`

`models.json` contains:

- `_shared.log_level`
- `_shared.startup_timeout`
- one entry per model with:
  - `model_id`
  - optional `extra_args`

Example:

```json
{
  "_shared": {
    "log_level": "warning",
    "startup_timeout": 600
  },
  "qwen36-35b-a3b-fp8": {
    "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
    "extra_args": ["--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder"]
  }
}
```

Model `extra_args` are model-intrinsic launch flags. They are merged with any instance-level `extra_args`.

## `hardware.json`

`hardware.json` stores fixed host and container-wide defaults. It is no longer a model selector and no longer stores per-model tuning.

Current fields:

- `description`
- `docker_image`
- `gpu_info`
- `shm_size`
- `gpu_count`

Example:

```json
{
  "description": "2x NVIDIA RTX Pro 6000 Blackwell server",
  "docker_image": "lmsysorg/sglang:v0.5.10.post1-cu130-runtime",
  "gpu_info": {
    "name": "NVIDIA RTX Pro 6000 Blackwell Workstation Edition",
    "count": 2,
    "vram_gb": 192,
    "arch": "Blackwell"
  },
  "shm_size": "64g",
  "gpu_count": 2
}
```

Only change this file when the actual host or repo-wide runtime image changes.

## `instances.json`

`instances.json` defines each runnable server process. Every instance must declare:

- `model`
- `port`
- `gpu_ids`
- `tp`
- `ep`
- `mem_fraction_static`
- `context_length`
- `attention_backend`
- `chunked_prefill_size`
- `num_continuous_decode_steps`
- optional `extra_args`

Example:

```json
{
  "qwen36": {
    "model": "qwen36-35b-a3b-fp8",
    "port": 11435,
    "gpu_ids": [0],
    "tp": 1,
    "ep": 1,
    "mem_fraction_static": 0.8,
    "context_length": 131072,
    "attention_backend": "flashinfer",
    "chunked_prefill_size": 16384,
    "num_continuous_decode_steps": 4
  }
}
```

Instance `extra_args` are the right place for instance-specific launch flags.

## `workloads.json`

`workloads.json` groups instances into named deployment shapes:

```json
{
  "qwen36": ["qwen36"]
}
```

Validation rules:

- workloads must reference known instances
- ports must be unique inside a workload
- GPU assignments must not overlap inside a workload
- `tp` and `ep` must not exceed the number of GPUs assigned to the instance

## `.active`

`.active` now stores the active workload name, not the active model.

```text
qwen36
```

Set it with:

```bash
make use WORKLOAD=qwen36
```

## Active Runtime Exports

After `source scripts/load-config.sh`, Smelter exports workload-level values:

- `ACTIVE_WORKLOAD`
- `ACTIVE_INSTANCES`
- `COMPOSE_FILE`
- `DOCKER_IMAGE`
- `LOG_LEVEL`
- `STARTUP_TIMEOUT`

When `INSTANCE=<name>` is set as well, it also exports:

- `INSTANCE_NAME`
- `MODEL_NAME`
- `MODEL_ID`
- `PORT`
- `GPU_IDS`
- `MEM_FRACTION_STATIC`
- `CONTEXT_LENGTH`
- `TP`
- `EP`
- `ATTENTION_BACKEND`
- `CHUNKED_PREFILL_SIZE`
- `NUM_CONTINUOUS_DECODE_STEPS`
- `EXTRA_LAUNCH_ARGS`

## Current Checked-In Workloads

`qwen36`

- `qwen36`
- model `Qwen/Qwen3.6-35B-A3B-FP8`
- port `11435`
- GPU `0`

## Adding a Model

1. Add the model entry to `models.json`.
2. Add one or more runnable instances in `instances.json`.
3. Reference those instances from `workloads.json`.
4. Select a workload with `make use WORKLOAD=<name>`.
5. Download weights with `make download`.

## Common Commands

```bash
make use WORKLOAD=qwen36
make download
make start
INSTANCE=qwen36 make health
INSTANCE=qwen36 make bench
```

## Tooling Notes

- `make help` lists all targets with their required inputs.
- `make chat` requires `INSTANCE=<name>`.
- `make bench` requires `INSTANCE=<name>`.
- `scripts/gpu-tuning-matrix.py` requires `INSTANCE=<name>` and temporarily mutates that instance in `instances.json`.
- `make refresh-moe-configs` targets one instance. If the active workload has multiple instances, pass `INSTANCE=<name>`.
- `benchmarks/latest.json` is keyed by instance name.
