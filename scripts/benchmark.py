#!/usr/bin/env python3
"""Run a fixed sglang.bench_serving benchmark for comparing config changes.

Reads config from models.json + .active, runs the benchmark in an ephemeral
Docker container, and saves timestamped JSONL results to benchmarks/results/.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

NUM_PROMPTS = 100
RANDOM_INPUT_LEN = 1024
RANDOM_OUTPUT_LEN = 1024
MAX_CONCURRENCY = 1


def load_config(project_dir: Path) -> dict[str, str]:
    models_path = project_dir / "models.json"
    active_file = project_dir / ".active"

    if not models_path.exists():
        raise SystemExit("Error: models.json not found.")
    if not active_file.exists():
        raise SystemExit("Error: no active config. Run: make use MODEL=<name> HARDWARE=<name>")

    lines = active_file.read_text(encoding="utf-8").strip().splitlines()
    active_model = lines[0]
    active_hardware = lines[1] if len(lines) > 1 else "unknown"
    models = json.loads(models_path.read_text(encoding="utf-8"))

    if active_model not in models:
        raise SystemExit(f"Error: unknown model '{active_model}' in .active")

    cfg = models[active_model]
    shared = models.get("_shared", {})

    return {
        "MODEL_ID": cfg["model_id"],
        "MODEL_NAME": active_model,
        "HARDWARE": active_hardware,
        "PORT": str(shared.get("port", 11435)),
    }


def get_sglang_image(compose_path: Path) -> str:
    for line in compose_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("image:"):
            return stripped.split(":", 1)[1].strip()
    raise SystemExit("Error: could not find image in docker-compose.yml.")


def update_latest_snapshot(
    project_dir: Path,
    result_path: Path,
    model_name: str,
    hardware: str,
    label: str,
    timestamp: str,
) -> None:
    """Update benchmarks/latest.json with a summary of the most recent run."""
    snapshot_path = project_dir / "benchmarks" / "latest.json"

    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        snapshot = {}

    raw = json.loads(result_path.read_text(encoding="utf-8").strip())

    key = f"{model_name}/{hardware}"
    snapshot[key] = {
        "model_id": raw.get("server_info", {}).get("model_path", ""),
        "timestamp": timestamp,
        "label": label,
        "result_file": str(result_path.relative_to(project_dir)),
        "completed": raw.get("completed"),
        "output_tok_per_s": round(raw.get("output_throughput", 0), 2),
        "mean_ttft_ms": round(raw.get("mean_ttft_ms", 0), 2),
        "mean_tpot_ms": round(raw.get("mean_tpot_ms", 0), 2),
        "mean_e2e_latency_ms": round(raw.get("mean_e2e_latency_ms", 0), 2),
        "p99_itl_ms": round(raw.get("p99_itl_ms", 0), 2),
    }

    snapshot_path.write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Snapshot: {snapshot_path.relative_to(project_dir)} [{key}]")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "benchmark"


def main(argv: list[str]) -> int:
    project_dir = Path(__file__).resolve().parent.parent
    compose_path = project_dir / "docker-compose.yml"
    config = load_config(project_dir)

    port = config["PORT"]
    model = config["MODEL_ID"]
    image = get_sglang_image(compose_path)

    output_dir = project_dir / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    label = argv[0] if argv else "manual"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    file_stem = f"{timestamp}-{slugify(label)}"

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_name = f"{file_stem}.jsonl"

        cmd = [
            "docker", "run", "--rm", "--network", "host",
            "-v", f"{tmpdir}:/results",
            image,
            "python3", "-m", "sglang.bench_serving",
            "--backend", "sglang",
            "--host", "127.0.0.1",
            "--port", port,
            "--model", model,
            "--dataset-name", "random",
            "--random-input-len", str(RANDOM_INPUT_LEN),
            "--random-output-len", str(RANDOM_OUTPUT_LEN),
            "--num-prompts", str(NUM_PROMPTS),
            "--max-concurrency", str(MAX_CONCURRENCY),
            "--output-file", f"/results/{jsonl_name}",
        ]

        print(f"Image:       {image}")
        print(f"Model:       {model}")
        print(f"Port:        {port}")
        print(f"Label:       {label}")
        print(f"Prompts:     {NUM_PROMPTS}")
        print(f"Concurrency: {MAX_CONCURRENCY}")
        print(f"Input/Output: {RANDOM_INPUT_LEN}/{RANDOM_OUTPUT_LEN} tokens")
        print()

        result = subprocess.run(cmd)

        src = Path(tmpdir) / jsonl_name
        if src.exists():
            dst = output_dir / jsonl_name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print()
            print(f"Saved: {dst.relative_to(project_dir)}")
            update_latest_snapshot(
                project_dir, dst,
                config["MODEL_NAME"], config["HARDWARE"],
                label, timestamp,
            )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
