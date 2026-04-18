# Architecture

Smelter is a single-host GPU inference service built around SGLang and Docker. It now runs one active workload at a time, where a workload can contain one or more model instances with separate ports and GPU assignments.

## Runtime Shape

- one fixed host configuration in `hardware.json`
- one active workload selected by `.active`
- one or more active instances from `instances.json`
- one generated Compose file at `.smelter/compose.generated.yml`
- one local Hugging Face cache in `models/`
- one repo-backed MoE config directory in `sglang-moe-configs/`

The checked-in host is a 2x RTX Pro 6000 server. The checked-in runtime image is `lmsysorg/sglang:v0.5.10.post1-cu130-runtime`.

## Workloads And Instances

An instance is one SGLang server process with:

- one model
- one port
- one GPU allocation
- one set of runtime tuning values

A workload is just a named set of instances that start together.

Current workloads:

- `qwen36`
  - `qwen36` (`Qwen/Qwen3.6-35B-A3B-FP8`)

## Startup Flow

1. `make use WORKLOAD=<name>` writes the workload name to `.active`.
2. `scripts/load-config.sh` calls `scripts/export_runtime_env.py`.
3. `scripts/export_runtime_env.py` validates `models.json`, `hardware.json`, `instances.json`, `workloads.json`, and `.active`.
4. The same loader renders `.smelter/compose.generated.yml` for the active workload.
5. `scripts/start.sh` runs `docker compose -f .smelter/compose.generated.yml up -d ...`.
6. Smelter waits for each instance to pass `/health` and then `/v1/models`.
7. `scripts/health-check.sh` verifies the expected OpenAI-compatible and Ollama-compatible endpoints.

The generated Compose file is the runtime source of truth. The checked-in `docker-compose.yml` is no longer the file used by the repo scripts.

## Configuration Contract

`models.json`

- shared defaults under `_shared`
  - `log_level`
  - `startup_timeout`
- per-model entries
  - `model_id`
  - optional `extra_args`

`hardware.json`

- fixed host facts and shared container defaults
  - `description`
  - `docker_image`
  - `gpu_info`
  - `shm_size`
  - `gpu_count`

`instances.json`

- per-instance runtime values
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

`workloads.json`

- workload name -> ordered list of instance names

`.active`

- one line: active workload name

Launch args are merged in this order:

1. model-level `extra_args` from `models.json`
2. instance-level `extra_args` from `instances.json`

## Validation Rules

The config loader fails fast when:

- a model entry is missing required fields
- an instance references an unknown model
- a workload references an unknown instance
- two instances in the same workload reuse a port
- two instances in the same workload overlap on GPU assignment
- `tp` or `ep` exceeds the number of GPUs assigned to the instance
- the active workload is unknown

## Compose Rendering

For each active instance, the renderer emits one Compose service with:

- service name equal to the instance name
- `container_name` `smelter-<instance>`
- host networking
- host IPC
- `models/` mounted at `/root/.cache/huggingface`
- `sglang-moe-configs/` mounted read-only at `/sglang-moe-configs`
- NVIDIA device reservation for the instance GPU list
- one `python3 -m sglang.launch_server ...` command built from merged config

This is how Smelter supports both:

- a large single-instance workload using both GPUs
- a multi-instance workload with one model per GPU

## API Surface

Each running instance exposes:

- `/health`
- `/v1/*` OpenAI-compatible APIs
- `/api/*` Ollama-compatible APIs

The health check script currently verifies:

- `/health`
- `/v1/models`
- `/api/tags`
- `/api/show`
- `/api/generate`
- `/v1/chat/completions`

## Storage And Persistence

- `models/` persists Hugging Face downloads across restarts
- `sglang-moe-configs/` persists repo-managed MoE kernel config files
- `.smelter/compose.generated.yml` is regenerated from config on each load
- `benchmarks/results/` stores timestamped benchmark artifacts
- `benchmarks/latest.json` stores the latest benchmark summary per instance

## Operational Tooling

- `make download` downloads all unique models required by the active workload, or one `INSTANCE`
- `make start`, `make stop`, `make restart`, `make dev`, and `make logs` operate on the active workload
- `make health` checks every instance in the active workload by default, or one `INSTANCE`
- `make chat` requires `INSTANCE=<name>`
- `make bench` requires `INSTANCE=<name>` and runs the quick `sglang.bench_serving` profile
- `scripts/gpu-tuning-matrix.py` sweeps `mem_fraction_static` and `context_length` for one `INSTANCE`
- `make refresh-moe-configs` refreshes MoE config files for one target instance and compares before/after benchmark output
- `make help` lists every target and its inputs

The default quick benchmark is:

- `8` prompts
- `1024` input tokens
- `1024` output tokens
- concurrency `1`

Override that with `BENCH_*` environment variables when needed.

## Deliberate Scope

- no hardware-profile selection
- no dynamic scheduler across workloads
- no per-instance image selection yet
- no reverse proxy, TLS, auth, or orchestration layer
- no in-container persistence for MoE config edits
