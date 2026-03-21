# Smelter

GPU inference service: SGLang in Docker serving one model at a time via Ollama-compatible and OpenAI-compatible APIs.

## Prerequisites

- NVIDIA GPU with drivers installed (`nvidia-smi` must work)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick Start

> **Note:** The first run downloads model weights into `models/`. Ensure you have enough disk space.

```bash
make pull                                                  # Pull the Docker image
make use MODEL=nemotron-cascade-2 HARDWARE=rtx-pro-6000   # Select model + hardware
make start                                                 # Start (waits for readiness)
make health                                                # Verify API endpoints
```

The service is available at `http://<server-ip>:11435` by default.

## Configuration

Two JSON files and one selector drive all runtime configuration:

- **`models.json`** — model definitions and shared settings (port, log level, timeouts)
- **`hardware.json`** — hardware profiles with GPU settings and per-model tuning
- **`.active`** — current model + hardware selection, written by `make use`

See [docs/configuration.md](docs/configuration.md) for details on adding models and hardware profiles.

## API

The service exposes Ollama-compatible and OpenAI-compatible (`/v1/*`) endpoints. Any client that speaks either protocol works out of the box.

## Make Targets

| Target                 | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `make use`             | Select active model + hardware profile (writes `.active`)       |
| `make start`           | Start the service (detached, waits for readiness)               |
| `make stop`            | Stop the service                                                |
| `make dev`             | Start in foreground with verbose logging                        |
| `make health`          | Test all API endpoints                                          |
| `make logs`            | Tail service logs                                               |
| `make chat`            | Interactive chat session                                        |
| `make download`        | Pre-download model weights                                      |
| `make pull`            | Pull/update the Docker image                                    |
| `make benchmark`       | Run the benchmark harness                                       |
| `make refresh-moe-configs` | Refresh MoE kernel configs with before/after benchmarks    |

## Troubleshooting

- **Service won't start**: Check logs with `make logs`. Ensure `nvidia-smi` works and that the model becomes ready before `STARTUP_TIMEOUT`.
- **OOM errors**: Reduce `context_length` or `mem_fraction_static` in the hardware profile's model entry.
- **Port conflict**: Change `port` in `models.json` `_shared` if 11435 is already in use.
- **Startup looks healthy but requests fail**: Run `make health` to distinguish container health from model readiness.
- **MoE config changes disappeared after a restart**: Save them under `sglang-moe-configs/`; edits inside the container are not persisted.
