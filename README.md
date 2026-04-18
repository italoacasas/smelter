# Smelter

GPU inference service for a fixed Linux host, built around SGLang and Docker. Smelter now runs one active workload at a time, where a workload can contain one or more model instances on different ports and GPUs.

## Platform

Linux only. SGLang's Docker runtime requires the NVIDIA Container Toolkit.

## Prerequisites

- Linux host
- NVIDIA GPU with drivers installed (`nvidia-smi` must work)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

```bash
make pull
make use WORKLOAD=qwen36
make download
make start
make health
```

Target a specific instance when chatting or benchmarking:

```bash
INSTANCE=qwen36 make chat
INSTANCE=qwen36 make bench
```

## Current Runtime Model

Smelter assumes one fixed server:

- 2x NVIDIA RTX Pro 6000
- one active workload selected in `.active`
- one or more instances inside that workload
- one generated Compose file at `.smelter/compose.generated.yml`

The checked-in rollout:

- `qwen36`
  - `qwen36` (`Qwen/Qwen3.6-35B-A3B-FP8`) on port `11435`, GPU `0`

## Configuration

Runtime config is split across four JSON files and one selector:

- `models.json` — model definitions and shared defaults
- `hardware.json` — fixed host facts and shared container settings
- `instances.json` — runnable server instances
- `workloads.json` — named sets of instances
- `.active` — active workload name

See [docs/configuration.md](docs/configuration.md) for the full contract.

## Runtime Image

The repo uses the stable generic SGLang runtime image:

- `lmsysorg/sglang:v0.5.10.post1-cu130-runtime`

That image is used for the whole rollout.

## APIs

Each running instance exposes:

- `/health`
- `/v1/*` OpenAI-compatible APIs
- `/api/*` Ollama-compatible APIs

## Make Targets

Run `make help` for the full list. Common targets:

| Target | Description |
| --- | --- |
| `make use WORKLOAD=<name>` | Select the active workload |
| `make start` | Start the active workload and wait for readiness |
| `make stop` | Stop the active workload |
| `make restart` | Stop then start the active workload |
| `make dev` | Start the active workload in the foreground |
| `make logs` | Tail logs for the active workload, or one `INSTANCE` |
| `make health` | Verify endpoints for the active workload, or one `INSTANCE` |
| `make download` | Download all unique models in the active workload, or one `INSTANCE` |
| `make chat` | Interactive chat for one `INSTANCE` |
| `make bench` | Run the quick benchmark for one `INSTANCE` |
| `make refresh-moe-configs` | Refresh MoE configs for one target instance |
| `make pull` | Pull the configured runtime image |

## Troubleshooting

- Service will not start: run `make logs`; render the compose file with `scripts/render-compose.py` if you need to inspect it.
- A workload is invalid: run `make use WORKLOAD=<name>` again and fix the referenced config.
- OOM errors: reduce `mem_fraction_static` or `context_length` in `instances.json`.
- Port conflicts: change the instance `port` in `instances.json`.
- Chat or benchmark errors: pass `INSTANCE=<name>` from the active workload.
- MoE config edits disappear: persist them under `sglang-moe-configs/`, not inside a container.
