# Architecture

GPU inference service: SGLang in Docker serving one model at a time via Ollama-compatible and OpenAI-compatible APIs.

## How It Works

1. Pick a model and hardware profile: `make use MODEL=<name> HARDWARE=<name>`
2. Start the service: `make start`
3. SGLang loads the model and serves it on port `11435` (default)

## Configuration

```
models.json            → model definitions (HF model ID, launch flags) + shared settings
hardware.json          → hardware profiles (GPU settings, per-model memory/context tuning)
.active                → current selection (model + hardware), written by `make use`
scripts/load-config.sh → reads the above, exports env vars for docker-compose
```

## Layout

```
docker-compose.yml     → single sglang service, all values from env vars
scripts/               → start, stop, health, chat, benchmark, config tools
models/                → HF model cache (git-ignored)
sglang-moe-configs/    → Triton MoE kernel configs, mounted read-only into container
benchmarks/results/    → timestamped JSONL benchmark reports
```

## Design Choices

- Single host, single container, one model at a time
- No auth, TLS, reverse proxy, or orchestration — just the inference engine
- Hardware-specific tuning (`mem_fraction_static`, `context_length`) lives in `hardware.json`, not with model definitions, because it depends on GPU VRAM
- MoE configs are repo-backed and mounted in, not written inside the container
- Downloaded models are cached in `models/` (git-ignored) and persist across restarts and model switches

## MoE Kernel Configs

Triton MoE kernel config files live in `sglang-moe-configs/` and are mounted read-only into the container via `SGLANG_MOE_CONFIG_DIR`. Edit files in the repo, not inside the container.

`make refresh-moe-configs` benchmarks the current configs, refreshes from the best upstream seed profile, benchmarks again, and rolls back if performance regresses. Re-run after changing SGLang image, Triton version, CUDA stack, GPU driver, or model.

```bash
make refresh-moe-configs                                                       # Refresh and benchmark
make refresh-moe-configs REFRESH_MOE_ARGS="--list"                             # List upstream profiles
make refresh-moe-configs REFRESH_MOE_ARGS="--source-device NVIDIA_B200"        # Use specific seed
```

## Benchmarking

```bash
make benchmark                                              # Warm benchmark
make benchmark LABEL=after-fp8-kv                           # With a label
./scripts/benchmark.py --cold-start --label baseline        # Cold-start + warm
```

Results are saved to `benchmarks/results/` as timestamped JSONL files.

`scripts/gpu-tuning-matrix.py` sweeps `MEM_FRACTION_STATIC` and `CONTEXT_LENGTH` combinations by temporarily modifying `hardware.json` and benchmarking each config.
