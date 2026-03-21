#!/usr/bin/env python3
"""Switch the active model and/or hardware profile by writing to .active."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    project_dir = Path(__file__).resolve().parent.parent
    models_path = project_dir / "models.json"
    hardware_path = project_dir / "hardware.json"
    active_path = project_dir / ".active"

    if not models_path.exists():
        print("Error: models.json not found.", file=sys.stderr)
        return 1
    if not hardware_path.exists():
        print("Error: hardware.json not found.", file=sys.stderr)
        return 1

    models = {k: v for k, v in json.loads(models_path.read_text(encoding="utf-8")).items() if k != "_shared"}
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))

    if len(sys.argv) < 3:
        print("Available models:")
        for name, cfg in models.items():
            print(f"  {name:30s} {cfg['model_id']}")
        print("\nAvailable hardware profiles:")
        for name, cfg in hardware.items():
            print(f"  {name:30s} {cfg.get('description', '')}")
        print(f"\nUsage: {sys.argv[0]} <model> <hardware>")
        return 1

    model_name = sys.argv[1]
    hw_name = sys.argv[2]

    if model_name not in models:
        print(f"Error: unknown model '{model_name}'", file=sys.stderr)
        print(f"Available: {', '.join(models.keys())}", file=sys.stderr)
        return 1

    if hw_name not in hardware:
        print(f"Error: unknown hardware profile '{hw_name}'", file=sys.stderr)
        print(f"Available: {', '.join(hardware.keys())}", file=sys.stderr)
        return 1

    hw_cfg = hardware[hw_name]
    model_cfg = models[model_name]
    hw_models = hw_cfg.get("models", {})

    if model_name not in hw_models:
        print(f"Warning: no tuning for model '{model_name}' on hardware '{hw_name}'", file=sys.stderr)
        print(f"Add a models.{model_name} entry to hardware.json[{hw_name}]", file=sys.stderr)
        return 1

    active_path.write_text(f"{model_name}\n{hw_name}\n", encoding="utf-8")

    gpu_info = hw_cfg.get("gpu_info", {})
    hw_model = hw_models[model_name]

    print(f"Active: {model_name} on {hw_name}")
    print(f"  Model:    {model_cfg['model_id']}")
    print(f"  GPU:      {gpu_info.get('name', 'unknown')} ({gpu_info.get('vram_gb', '?')} GB)")
    print(f"  Context:  {hw_model.get('context_length', '?')}")
    print(f"  MemFrac:  {hw_model.get('mem_fraction_static', '?')}")
    print(f"\nRestart the service: make stop && make start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
