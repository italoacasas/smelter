#!/usr/bin/env python3
"""Sweep per-instance GPU tuning values with the current quick benchmark."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from smelter_config import ConfigError, INSTANCES_JSON, load_state, resolve_instance_runtime


PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = PROJECT_DIR / "scripts" / "benchmark.py"
RESULTS_DIR = PROJECT_DIR / "benchmarks" / "results"

MEM_FRACTION_VALUES = [0.80, 0.85, 0.88, 0.90]
CONTEXT_LENGTH_VALUES = [16384, 24576, 32768]


def read_instances() -> dict[str, Any]:
    return json.loads(INSTANCES_JSON.read_text(encoding="utf-8"))


def write_instances(data: dict[str, Any]) -> None:
    INSTANCES_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def set_instance_tuning(
    data: dict[str, Any],
    instance_name: str,
    mem_fraction_static: float,
    context_length: int,
) -> dict[str, Any]:
    instance_cfg = data[instance_name]
    instance_cfg["mem_fraction_static"] = mem_fraction_static
    instance_cfg["context_length"] = context_length
    return data


def load_result(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"Error: benchmark result file is empty: {path}")
    return json.loads(lines[-1])


def parse_saved_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Saved: "):
            return PROJECT_DIR / line.removeprefix("Saved: ").strip()
    raise SystemExit("Error: could not determine benchmark output path from benchmark.py output.")


def run_benchmark(instance_name: str, label: str) -> tuple[int, Path | None]:
    env = os.environ.copy()
    env["INSTANCE"] = instance_name
    command = [sys.executable, str(BENCHMARK_SCRIPT), label]

    print(f"\n{'=' * 72}")
    print(f"Benchmark: {label}")
    print(f"{'=' * 72}")

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode, None
    return 0, parse_saved_path(result.stdout)


def build_report(instance_name: str, results: list[dict[str, Any]]) -> str:
    lines = [
        "# GPU Tuning Matrix",
        "",
        f"- Generated at: `{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}`",
        f"- Instance: `{instance_name}`",
        f"- Prompt count: `{os.environ.get('BENCH_NUM_PROMPTS', '8')}`",
        f"- Input length: `{os.environ.get('BENCH_INPUT_LEN', '1024')}`",
        f"- Output length: `{os.environ.get('BENCH_OUTPUT_LEN', '1024')}`",
        f"- Max concurrency: `{os.environ.get('BENCH_CONCURRENCY', '1')}`",
        "",
        "| MemFrac | Context | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2E (ms) | P99 ITL (ms) | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for entry in results:
        if entry["status"] == "FAIL":
            lines.append(
                f"| {entry['mem_fraction_static']:.2f} | {entry['context_length']} | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |"
            )
            continue

        metrics = entry["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{entry['mem_fraction_static']:.2f}",
                    str(entry["context_length"]),
                    f"{metrics.get('output_throughput', 0.0):.2f}",
                    f"{metrics.get('mean_ttft_ms', 0.0):.2f}",
                    f"{metrics.get('mean_tpot_ms', 0.0):.2f}",
                    f"{metrics.get('mean_e2e_latency_ms', 0.0):.2f}",
                    f"{metrics.get('p99_itl_ms', 0.0):.2f}",
                    "OK",
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    instance_name = os.environ.get("INSTANCE")
    if not instance_name:
        print("Error: gpu-tuning-matrix requires INSTANCE=<name> from the active workload.")
        return 1

    try:
        state = load_state(require_active=True)
        resolve_instance_runtime(state, instance_name)
    except ConfigError as exc:
        print(f"Error: {exc}")
        return 1

    if not INSTANCES_JSON.exists():
        print("Error: instances.json not found.")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    backup_path = INSTANCES_JSON.with_suffix(".json.phase3-backup")
    shutil.copy2(INSTANCES_JSON, backup_path)
    print(f"Backed up instances.json to {backup_path.name}")

    results: list[dict[str, Any]] = []
    total = len(MEM_FRACTION_VALUES) * len(CONTEXT_LENGTH_VALUES)
    current = 0

    try:
        for mem_fraction_static in MEM_FRACTION_VALUES:
            for context_length in CONTEXT_LENGTH_VALUES:
                current += 1
                label = (
                    f"tune-{instance_name}-"
                    f"mfs{mem_fraction_static:.2f}-ctx{context_length}"
                )

                print(f"\n{'#' * 72}")
                print(
                    f"# Config {current}/{total}: "
                    f"MEM_FRACTION_STATIC={mem_fraction_static:.2f} "
                    f"CONTEXT_LENGTH={context_length}"
                )
                print(f"{'#' * 72}")

                instance_data = read_instances()
                write_instances(
                    set_instance_tuning(
                        instance_data,
                        instance_name,
                        mem_fraction_static,
                        context_length,
                    )
                )

                returncode, result_path = run_benchmark(instance_name, label)
                if returncode != 0 or result_path is None:
                    results.append(
                        {
                            "mem_fraction_static": mem_fraction_static,
                            "context_length": context_length,
                            "status": "FAIL",
                        }
                    )
                    continue

                results.append(
                    {
                        "mem_fraction_static": mem_fraction_static,
                        "context_length": context_length,
                        "status": "OK",
                        "result_path": result_path,
                        "metrics": load_result(result_path),
                    }
                )
    finally:
        shutil.copy2(backup_path, INSTANCES_JSON)
        backup_path.unlink()
        print(f"\nRestored original instances.json from {backup_path.name}")

    report_path = RESULTS_DIR / (
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
        f"gpu-tuning-{instance_name}.md"
    )
    report_path.write_text(build_report(instance_name, results), encoding="utf-8")

    failures = [entry for entry in results if entry["status"] == "FAIL"]
    print(f"\nWrote comparison report: {report_path.relative_to(PROJECT_DIR)}")
    print(
        f"Completed {len(results) - len(failures)}/{len(results)} configurations successfully."
    )
    if failures:
        print("Failed configs:")
        for failure in failures:
            print(
                f"  MEM_FRACTION_STATIC={failure['mem_fraction_static']:.2f} "
                f"CONTEXT_LENGTH={failure['context_length']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
