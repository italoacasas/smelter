# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Smelter — a GPU inference service: SGLang in Docker serving one model at a time via Ollama-compatible and OpenAI-compatible APIs.

## Common Commands

```bash
make use MODEL=nemotron-cascade-2 HARDWARE=rtx-pro-6000  # Switch active model + hardware (writes .active, requires restart)
make start                # Start service (waits for readiness)
make stop                 # Stop service
make dev                  # Start in foreground with verbose logging
make health               # Verify all API endpoints
make logs                 # Tail container logs
make chat                 # Interactive chat session
make download             # Pre-download model weights
make pull                 # Pull latest Docker image
make benchmark            # Fixed benchmark (100 prompts, 1024 in/out, concurrency 1)
make benchmark LABEL=after-fp8-kv  # Same benchmark with a descriptive label
make refresh-moe-configs  # Refresh MoE kernel configs with before/after benchmarks
```

## Configuration

Two JSON files, one selector:

- **`models.json`** — model definitions (`model_id`, `extra_args`) and shared settings (`port`, `log_level`, `startup_timeout`) under `_shared`
- **`hardware.json`** — hardware profiles: GPU runtime settings, per-model tuning (`mem_fraction_static`, `context_length`, `extra_args`)
- **`.active`** — two-line file (model name, hardware name), written by `make use`, git-ignored

`scripts/load-config.sh` reads all three and exports env vars consumed by `docker-compose.yml` and all scripts.

## Layout

- **docker-compose.yml** — single `sglang` service (container: `smelter`); all tunable values are env vars
- **scripts/** — operational tooling (start, stop, health, chat, benchmark, config tools)
- **models/** — HF model cache (git-ignored)
- **sglang-moe-configs/** — Triton MoE kernel configs, mounted read-only into the container
- **benchmarks/results/** — timestamped JSONL benchmark reports

## Key Conventions

- `docs/architecture.md` is a **living current-state record** — update it when changes affect runtime, API, config, or operations
- No Python package manager — scripts use only stdlib plus what's in the SGLang Docker image
- Model definitions live in `models.json`, hardware tuning in `hardware.json`
- `MODEL_ID` is both the HF source and the client-facing model name
- MoE config edits go in `sglang-moe-configs/`, not inside the container

## Benchmarking

`scripts/benchmark.py` runs a fixed benchmark (100 random prompts, 1024 input/output tokens, concurrency 1) for consistent before/after comparison. Results go to `benchmarks/results/`.

`scripts/gpu-tuning-matrix.py` sweeps `MEM_FRACTION_STATIC` and `CONTEXT_LENGTH` combinations.

## SGLang Model Cookbooks

When adding or tuning models, check the SGLang cookbooks for recommended launch flags, TP sizing, and quantization:

- [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook)

## MoE Config Refresh

`scripts/refresh_moe_configs.py` (wrapped by `scripts/refresh-moe-configs.sh`) benchmarks before and after refreshing MoE configs from an upstream seed profile, and rolls back if the post-refresh benchmark fails. Re-run after changing SGLang image, Triton version, CUDA stack, GPU driver, or model.
