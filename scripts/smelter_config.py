#!/usr/bin/env python3
"""Shared config loading and compose rendering helpers for Smelter."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
ACTIVE_FILE = PROJECT_DIR / ".active"
MODELS_JSON = PROJECT_DIR / "models.json"
HARDWARE_JSON = PROJECT_DIR / "hardware.json"
INSTANCES_JSON = PROJECT_DIR / "instances.json"
WORKLOADS_JSON = PROJECT_DIR / "workloads.json"
GENERATED_DIR = PROJECT_DIR / ".smelter"
GENERATED_COMPOSE = GENERATED_DIR / "compose.generated.yml"
DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.10.post1-cu130-runtime"


class ConfigError(RuntimeError):
    """Raised when the repo config is invalid."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"{label} not found: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{path.name} must contain a top-level JSON object")
    return payload


def _require_keys(label: str, payload: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ConfigError(f"{label} is missing required keys: {', '.join(missing)}")


def _ensure_string_list(label: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{label} must be a list of strings")
    return value


def _ensure_gpu_ids(label: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a non-empty list of GPU ids")
    if not all(isinstance(item, int) for item in value):
        raise ConfigError(f"{label} must contain integer GPU ids")
    if len(set(value)) != len(value):
        raise ConfigError(f"{label} must not contain duplicate GPU ids")
    if any(item < 0 for item in value):
        raise ConfigError(f"{label} must not contain negative GPU ids")
    return value


def _load_active_workload(require_active: bool) -> str | None:
    if not ACTIVE_FILE.exists():
        if require_active:
            raise ConfigError("no active workload. Run: make use WORKLOAD=<name>")
        return None
    active = ACTIVE_FILE.read_text(encoding="utf-8").strip()
    if not active:
        if require_active:
            raise ConfigError("active workload file is empty")
        return None
    return active


def _validate_models(models_raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = models_raw.get("_shared", {})
    if not isinstance(shared, dict):
        raise ConfigError("models.json _shared must be an object")

    models = {name: cfg for name, cfg in models_raw.items() if name != "_shared"}
    for model_name, cfg in models.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"models.json entry {model_name} must be an object")
        _require_keys(f"models.{model_name}", cfg, ["model_id"])
        _ensure_string_list(f"models.{model_name}.extra_args", cfg.get("extra_args", []))
    return shared, models


def _validate_hardware(hardware: dict[str, Any]) -> None:
    _require_keys(
        "hardware.json",
        hardware,
        ["description", "gpu_info", "shm_size", "gpu_count"],
    )
    if not isinstance(hardware["gpu_info"], dict):
        raise ConfigError("hardware.json gpu_info must be an object")
    if not isinstance(hardware["gpu_count"], int) or hardware["gpu_count"] <= 0:
        raise ConfigError("hardware.json gpu_count must be a positive integer")


def _validate_instances(
    instances: dict[str, Any],
    models: dict[str, Any],
    hardware: dict[str, Any],
) -> None:
    gpu_count = hardware["gpu_count"]
    for instance_name, cfg in instances.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"instances.{instance_name} must be an object")
        _require_keys(
            f"instances.{instance_name}",
            cfg,
            [
                "model",
                "port",
                "gpu_ids",
                "tp",
                "ep",
                "mem_fraction_static",
                "context_length",
                "attention_backend",
                "chunked_prefill_size",
                "num_continuous_decode_steps",
            ],
        )
        model_name = cfg["model"]
        if model_name not in models:
            raise ConfigError(f"instances.{instance_name} references unknown model {model_name}")

        port = cfg["port"]
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ConfigError(f"instances.{instance_name}.port must be an integer between 1 and 65535")

        gpu_ids = _ensure_gpu_ids(f"instances.{instance_name}.gpu_ids", cfg["gpu_ids"])
        if max(gpu_ids) >= gpu_count:
            raise ConfigError(
                f"instances.{instance_name}.gpu_ids contains a GPU id outside hardware.json gpu_count={gpu_count}"
            )

        tp = cfg["tp"]
        ep = cfg["ep"]
        if not isinstance(tp, int) or tp <= 0:
            raise ConfigError(f"instances.{instance_name}.tp must be a positive integer")
        if not isinstance(ep, int) or ep <= 0:
            raise ConfigError(f"instances.{instance_name}.ep must be a positive integer")
        if tp > len(gpu_ids):
            raise ConfigError(
                f"instances.{instance_name}.tp={tp} exceeds assigned GPU count {len(gpu_ids)}"
            )
        if ep > len(gpu_ids):
            raise ConfigError(
                f"instances.{instance_name}.ep={ep} exceeds assigned GPU count {len(gpu_ids)}"
            )

        _ensure_string_list(f"instances.{instance_name}.extra_args", cfg.get("extra_args", []))


def _validate_workloads(workloads: dict[str, Any], instances: dict[str, Any]) -> None:
    for workload_name, instance_names in workloads.items():
        if not isinstance(instance_names, list) or not instance_names:
            raise ConfigError(f"workloads.{workload_name} must be a non-empty list")
        if not all(isinstance(name, str) for name in instance_names):
            raise ConfigError(f"workloads.{workload_name} must contain instance names")
        if len(set(instance_names)) != len(instance_names):
            raise ConfigError(f"workloads.{workload_name} must not repeat the same instance")

        seen_ports: set[int] = set()
        seen_gpus: set[int] = set()
        for instance_name in instance_names:
            if instance_name not in instances:
                raise ConfigError(
                    f"workloads.{workload_name} references unknown instance {instance_name}"
                )
            instance_cfg = instances[instance_name]
            port = instance_cfg["port"]
            if port in seen_ports:
                raise ConfigError(f"workloads.{workload_name} reuses port {port}")
            seen_ports.add(port)

            gpu_ids = set(instance_cfg["gpu_ids"])
            overlap = seen_gpus & gpu_ids
            if overlap:
                overlap_text = ", ".join(str(value) for value in sorted(overlap))
                raise ConfigError(
                    f"workloads.{workload_name} overlaps GPU assignments on GPU(s) {overlap_text}"
                )
            seen_gpus |= gpu_ids


def load_state(*, require_active: bool = True) -> dict[str, Any]:
    models_raw = _load_json(MODELS_JSON, "models.json")
    hardware = _load_json(HARDWARE_JSON, "hardware.json")
    instances = _load_json(INSTANCES_JSON, "instances.json")
    workloads = _load_json(WORKLOADS_JSON, "workloads.json")

    shared, models = _validate_models(models_raw)
    if not models:
        raise ConfigError("no models are configured in models.json")

    _validate_hardware(hardware)
    _validate_instances(instances, models, hardware)
    _validate_workloads(workloads, instances)

    active_workload = _load_active_workload(require_active)
    if active_workload is not None and active_workload not in workloads:
        if require_active:
            raise ConfigError(f"unknown active workload {active_workload}")
        active_workload = None

    return {
        "shared": shared,
        "models": models,
        "hardware": hardware,
        "instances": instances,
        "workloads": workloads,
        "active_workload": active_workload,
        "active_instances": workloads.get(active_workload, []) if active_workload else [],
    }


def resolve_instance_runtime(state: dict[str, Any], instance_name: str) -> dict[str, Any]:
    instances = state["instances"]
    if instance_name not in instances:
        raise ConfigError(f"unknown instance {instance_name}")
    if state["active_workload"] and instance_name not in state["active_instances"]:
        raise ConfigError(
            f"instance {instance_name} is not part of active workload {state['active_workload']}"
        )

    instance_cfg = instances[instance_name]
    model_name = instance_cfg["model"]
    model_cfg = state["models"][model_name]
    shared = state["shared"]
    hardware = state["hardware"]

    model_args = model_cfg.get("extra_args", [])
    instance_args = instance_cfg.get("extra_args", [])
    all_extra = model_args + instance_args

    return {
        "instance_name": instance_name,
        "model_name": model_name,
        "model_id": model_cfg["model_id"],
        "port": instance_cfg["port"],
        "gpu_ids": list(instance_cfg["gpu_ids"]),
        "mem_fraction_static": instance_cfg["mem_fraction_static"],
        "context_length": instance_cfg["context_length"],
        "tp": instance_cfg["tp"],
        "ep": instance_cfg["ep"],
        "attention_backend": instance_cfg["attention_backend"],
        "chunked_prefill_size": instance_cfg["chunked_prefill_size"],
        "num_continuous_decode_steps": instance_cfg["num_continuous_decode_steps"],
        "extra_launch_args": all_extra,
        "log_level": os.environ.get("SMELTER_LOG_LEVEL_OVERRIDE", shared.get("log_level", "warning")),
        "startup_timeout": shared.get("startup_timeout", 600),
        "docker_image": hardware.get("docker_image", DEFAULT_IMAGE),
        "shm_size": hardware["shm_size"],
        "gpu_info": hardware.get("gpu_info", {}),
    }


def render_compose(state: dict[str, Any]) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    hardware = state["hardware"]
    image = hardware.get("docker_image", DEFAULT_IMAGE)
    services: dict[str, Any] = {}

    instance_names = (
        list(state["active_instances"])
        if state.get("active_workload")
        else sorted(state["instances"])
    )

    for instance_name in instance_names:
        runtime = resolve_instance_runtime(
            {
                **state,
                "active_workload": None,
                "active_instances": list(state["instances"]),
            },
            instance_name,
        )
        gpu_ids = [str(value) for value in runtime["gpu_ids"]]
        command_items = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            runtime["model_id"],
            "--host",
            "0.0.0.0",
            "--port",
            str(runtime["port"]),
            "--mem-fraction-static",
            str(runtime["mem_fraction_static"]),
            "--context-length",
            str(runtime["context_length"]),
            "--tp",
            str(runtime["tp"]),
            "--ep",
            str(runtime["ep"]),
            "--trust-remote-code",
            "--attention-backend",
            runtime["attention_backend"],
            "--chunked-prefill-size",
            str(runtime["chunked_prefill_size"]),
            "--num-continuous-decode-steps",
            str(runtime["num_continuous_decode_steps"]),
            "--log-level",
            runtime["log_level"],
            *runtime["extra_launch_args"],
        ]

        services[instance_name] = {
            "image": image,
            "container_name": f"smelter-{instance_name}",
            "volumes": [
                f"{PROJECT_DIR / 'models'}:/root/.cache/huggingface",
                f"{PROJECT_DIR / 'sglang-moe-configs'}:/sglang-moe-configs:ro",
            ],
            "restart": "unless-stopped",
            "network_mode": "host",
            "environment": {
                "HF_TOKEN": "${HF_TOKEN:-}",
                "SGLANG_MOE_CONFIG_DIR": "/sglang-moe-configs",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                "CUDA_MODULE_LOADING": "LAZY",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "SGLANG_DISABLE_DEEP_GEMM": "1",
                "SGLANG_SET_CPU_AFFINITY": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                "SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
                "NVIDIA_VISIBLE_DEVICES": ",".join(gpu_ids),
            },
            "command": command_items,
            "ulimits": {"memlock": -1, "stack": 67108864},
            "ipc": "host",
            "shm_size": runtime["shm_size"],
            "healthcheck": {
                "test": ["CMD-SHELL", f"curl -f http://localhost:{runtime['port']}/health || exit 1"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "300s",
            },
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": gpu_ids,
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            },
        }

    payload = {"name": "smelter", "services": services}
    GENERATED_COMPOSE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return GENERATED_COMPOSE
