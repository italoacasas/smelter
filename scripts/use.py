#!/usr/bin/env python3
"""Switch the active workload by writing to .active."""

from __future__ import annotations

import sys
from pathlib import Path

from smelter_config import ACTIVE_FILE, ConfigError, load_state, resolve_instance_runtime


def main() -> int:
    try:
        state = load_state(require_active=False)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    workloads = state["workloads"]
    hardware = state["hardware"]

    if len(sys.argv) < 2:
        print("Available workloads:")
        for workload_name, instance_names in workloads.items():
            print(f"  {workload_name}")
            for instance_name in instance_names:
                runtime = resolve_instance_runtime(
                    {
                        **state,
                        "active_workload": workload_name,
                        "active_instances": instance_names,
                    },
                    instance_name,
                )
                gpu_ids = ",".join(str(value) for value in runtime["gpu_ids"])
                print(
                    f"    {instance_name:28s} {runtime['model_id']} "
                    f"port={runtime['port']} gpus={gpu_ids}"
                )
        print(f"\nHardware: {hardware.get('description', 'unknown')}")
        print(f"Usage: {sys.argv[0]} <workload>")
        return 1

    workload_name = sys.argv[1]
    if workload_name not in workloads:
        print(f"Error: unknown workload '{workload_name}'", file=sys.stderr)
        print(f"Available: {', '.join(workloads.keys())}", file=sys.stderr)
        return 1

    ACTIVE_FILE.write_text(f"{workload_name}\n", encoding="utf-8")

    print(f"Active workload: {workload_name}")
    for instance_name in workloads[workload_name]:
        runtime = resolve_instance_runtime(
            {
                **state,
                "active_workload": workload_name,
                "active_instances": workloads[workload_name],
            },
            instance_name,
        )
        gpu_ids = ",".join(str(value) for value in runtime["gpu_ids"])
        print(f"  {instance_name}")
        print(f"    Model:   {runtime['model_id']}")
        print(f"    Port:    {runtime['port']}")
        print(f"    GPUs:    {gpu_ids}")
        print(f"    Context: {runtime['context_length']}")
        print(f"    MemFrac: {runtime['mem_fraction_static']}")

    print("\nRestart the service: make stop && make start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
