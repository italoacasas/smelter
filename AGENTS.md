# AGENTS.md

Guidance for coding agents working on this repo. Also read by Claude Code via `CLAUDE.md` (pointer).

## What Smelter is

Single-host GPU inference service built on SGLang and Docker. One active **workload** runs at a time. A workload is a named set of **instances**; each instance is one SGLang server process with a dedicated port and GPU assignment.

Checked-in host: 2x NVIDIA RTX Pro 6000 Blackwell.
Checked-in image: set in `hardware.json` (`docker_image` field).

## Config surface

Four JSON files plus one selector. Keep them in sync:

| File | Purpose |
| --- | --- |
| `models.json` | Model definitions. `_shared` holds defaults (log level, startup timeout). Each entry: `model_id` (HF repo) and optional `extra_args`. |
| `hardware.json` | Fixed host facts: GPU info, shm size, GPU count, runtime `docker_image`. |
| `instances.json` | Runnable server instances. Required keys: `model`, `port`, `gpu_ids`, `tp`, `ep`, `mem_fraction_static`, `context_length`, `attention_backend`, `chunked_prefill_size`, `num_continuous_decode_steps`. Optional `extra_args`. |
| `workloads.json` | Named lists of instance names. Port and GPU overlap between instances in a workload is rejected. |
| `.active` | One line: the active workload name. Written by `make use`. |

Full schema enforcement lives in `scripts/smelter_config.py`. Any config key added to the JSON files must be validated there.

## How config flows

```
JSON files + .active
  -> scripts/smelter_config.py: load_state() validates
  -> render_compose() writes .smelter/compose.generated.yml
  -> scripts/export_runtime_env.py emits shell exports
  -> scripts/load-config.sh sources those exports
  -> shell scripts (start/stop/logs/health/chat/download) use the env vars
```

`scripts/benchmark.py` and `scripts/use.py` call `smelter_config` directly and do not go through the env-var layer.

## Common commands

Run `make help` for the full list. Daily ops: `use`, `start`, `stop`, `restart`, `logs`, `health`, `chat`, `bench`, `download`.

Inputs:
- `WORKLOAD=<name>` for `use`
- `INSTANCE=<name>` — only required for `chat` / `bench` when the active workload has more than one instance; optional filter for `logs`, `health`, `download`, `refresh-moe-configs`
- `LABEL=<string>` for `bench`
- `BENCH_NUM_PROMPTS` / `BENCH_INPUT_LEN` / `BENCH_OUTPUT_LEN` / `BENCH_CONCURRENCY` tune the benchmark

## Adding a new model

1. Add the model to `models.json` with `model_id` and any cookbook `extra_args`.
2. Add one or more instances to `instances.json` referencing that model.
3. Add or update a workload in `workloads.json`.
4. `make use WORKLOAD=<name>` then `make download` then `make start`.
5. Before deciding on flags, check the SGLang cookbook at <https://cookbook.sglang.io/>.

See `docs/plan-qwen36-35b-a3b.md` for a worked example.

## Guardrails

- **Per-instance env vars don't exist.** The compose `environment` block is hardcoded in `smelter_config.py` inside `render_compose()`. To add an env var, patch that dict — it applies to every instance.
- **MoE config edits go in `sglang-moe-configs/`**, mounted read-only at `/sglang-moe-configs`. Never edit inside a container.
- **`extra_args` compose**: the model-level list is concatenated with the instance-level list at launch; duplicate flags will pass both to SGLang.
- **Port and GPU overlap** within a workload is rejected by validation — check `workloads.json` first if `make use` fails.
- **No per-hardware config profiles.** The repo assumes a fixed host; `hardware.json` is a single object, not a map.
- **Benchmarks write to `benchmarks/results/*.jsonl`** and update `benchmarks/latest.json` keyed by instance name. Do not hand-edit `latest.json`.
- **Don't reintroduce hardware-profile selection or per-instance compose overrides** — those were deliberately removed.

## Documentation rules

- `docs/architecture.md` is the current-state record for runtime behavior, configuration, storage layout, and operations. Update it in the same change as architecture-impacting code.
- `docs/configuration.md` mirrors the config-file contract. Keep field names and required keys aligned with `smelter_config.py`.
- Future / exploratory plans belong in separate files under `docs/plan-*.md`, not in `architecture.md`.
- Remove stale statements instead of appending conflicting notes.
- When a `make` target is added, renamed, or removed, update its `##` help annotation and any table in `README.md` / docs that lists it.
