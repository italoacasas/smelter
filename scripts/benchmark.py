#!/usr/bin/env python3
"""Run sglang.bench_serving against one active workload instance."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from smelter_config import ConfigError, load_state, resolve_instance_runtime

NUM_PROMPTS = int(os.environ.get("BENCH_NUM_PROMPTS", 8))
RANDOM_INPUT_LEN = int(os.environ.get("BENCH_INPUT_LEN", 1024))
RANDOM_OUTPUT_LEN = int(os.environ.get("BENCH_OUTPUT_LEN", 1024))
MAX_CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", 1))

def update_latest_snapshot(
    project_dir: Path,
    result_path: Path,
    instance_name: str,
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

    key = instance_name
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
    try:
        state = load_state(require_active=True)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    instance_name = os.environ.get("INSTANCE")
    if not instance_name:
        active = state["active_instances"]
        if len(active) == 1:
            instance_name = active[0]
        else:
            raise SystemExit(
                f"Error: active workload has {len(active)} instances ({', '.join(active)}). Pass INSTANCE=<name>."
            )

    try:
        config = resolve_instance_runtime(state, instance_name)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    port = str(config["port"])
    model = config["model_id"]
    image = config["docker_image"]

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
        print(f"Instance:    {instance_name}")
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
                instance_name,
                label, timestamp,
            )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
