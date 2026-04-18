#!/usr/bin/env python3
"""Emit shell exports for the active Smelter workload or instance."""

from __future__ import annotations

import os
import shlex
import sys

from smelter_config import (
    DEFAULT_IMAGE,
    ConfigError,
    load_state,
    render_compose,
    resolve_instance_runtime,
)


def main() -> int:
    instance_name = os.environ.get("INSTANCE") or None
    try:
        state = load_state(require_active=True)
        compose_path = render_compose(state)
        shared = state["shared"]
        exports: dict[str, str] = {
            "ACTIVE_WORKLOAD": state["active_workload"],
            "ACTIVE_INSTANCES": " ".join(state["active_instances"]),
            "COMPOSE_FILE": str(compose_path),
            "DOCKER_IMAGE": state["hardware"].get("docker_image", DEFAULT_IMAGE),
            "LOG_LEVEL": str(shared.get("log_level", "warning")),
            "STARTUP_TIMEOUT": str(shared.get("startup_timeout", 600)),
        }
        if instance_name:
            runtime = resolve_instance_runtime(state, instance_name)
            exports.update(
                {
                    "INSTANCE_NAME": runtime["instance_name"],
                    "MODEL_NAME": runtime["model_name"],
                    "MODEL_ID": runtime["model_id"],
                    "PORT": str(runtime["port"]),
                    "GPU_IDS": ",".join(str(value) for value in runtime["gpu_ids"]),
                    "MEM_FRACTION_STATIC": str(runtime["mem_fraction_static"]),
                    "CONTEXT_LENGTH": str(runtime["context_length"]),
                    "TP": str(runtime["tp"]),
                    "EP": str(runtime["ep"]),
                    "ATTENTION_BACKEND": runtime["attention_backend"],
                    "CHUNKED_PREFILL_SIZE": str(runtime["chunked_prefill_size"]),
                    "NUM_CONTINUOUS_DECODE_STEPS": str(runtime["num_continuous_decode_steps"]),
                    "EXTRA_LAUNCH_ARGS": " ".join(runtime["extra_launch_args"]),
                }
            )
    except ConfigError as exc:
        print(f"echo {shlex.quote(f'Error: {exc}')} >&2; exit 1")
        return 0

    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")
    print('export HF_TOKEN="${HF_TOKEN:-}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
