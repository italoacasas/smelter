#!/usr/bin/env python3
"""Phase 3: GPU Tuning — run a benchmark matrix over MEM_FRACTION_STATIC and CONTEXT_LENGTH."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
HARDWARE_JSON = PROJECT_DIR / "hardware.json"
ACTIVE_FILE = PROJECT_DIR / ".active"
BENCHMARK_SCRIPT = PROJECT_DIR / "scripts" / "benchmark.py"
RESULTS_DIR = PROJECT_DIR / "benchmarks" / "results"

MEM_FRACTION_VALUES = ["0.80", "0.85", "0.88", "0.90"]
CONTEXT_LENGTH_VALUES = ["16384", "24576", "32768"]

WARM_RUNS = 3


def read_hardware() -> dict:
    return json.loads(HARDWARE_JSON.read_text(encoding="utf-8"))


def write_hardware(data: dict) -> None:
    HARDWARE_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def set_model_tuning(data: dict, hw_name: str, model_name: str,
                     mem_frac: str, ctx_len: str) -> dict:
    """Set mem_fraction_static and context_length for the active model in the hardware profile."""
    profile = data[hw_name]
    models = profile.setdefault("models", {})
    model_cfg = models.setdefault(model_name, {})
    model_cfg["mem_fraction_static"] = float(mem_frac)
    model_cfg["context_length"] = int(ctx_len)
    return data


def run_benchmark(label: str, cold_start: bool = True) -> int:
    cmd = [
        sys.executable, str(BENCHMARK_SCRIPT),
        "--label", label,
        "--warm-runs", str(WARM_RUNS),
    ]
    if cold_start:
        cmd.append("--cold-start")
    print(f"\n{'=' * 60}")
    print(f"Running benchmark: {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    return result.returncode


def find_result_files(label_slug: str) -> tuple[Path | None, Path | None]:
    """Find the most recent CSV and MD files matching the label slug."""
    csvs = sorted(RESULTS_DIR.glob(f"*-{label_slug}.csv"))
    mds = sorted(RESULTS_DIR.glob(f"*-{label_slug}.md"))
    return (csvs[-1] if csvs else None, mds[-1] if mds else None)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_comparison_table(all_results: list[dict[str, Any]]) -> str:
    lines = ["# Phase 3: GPU Tuning Matrix Results", ""]
    lines.append(f"- Generated at: `{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}`")
    lines.append(f"- Warm runs per config: {WARM_RUNS}")
    lines.append(f"- MEM_FRACTION_STATIC values: {', '.join(MEM_FRACTION_VALUES)}")
    lines.append(f"- CONTEXT_LENGTH values: {', '.join(CONTEXT_LENGTH_VALUES)}")
    lines.append("")

    # Cold start table
    cold_results = [r for r in all_results if r.get("cold_start")]
    if cold_results:
        lines.append("## Cold Start")
        lines.append("")
        headers = ["MEM_FRAC", "CTX_LEN", "Health (s)", "Model Ready (s)", "First Req (s)", "First Token (s)", "Throughput (tok/s)", "Status"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in cold_results:
            row = [
                r["mem_fraction"],
                r["context_length"],
                f"{r['health_ready_s']:.1f}" if r["health_ready_s"] is not None else "FAIL",
                f"{r['model_ready_s']:.1f}" if r["model_ready_s"] is not None else "FAIL",
                f"{r['first_req_s']:.1f}" if r["first_req_s"] is not None else "FAIL",
                f"{r['first_token_s']:.3f}" if r["first_token_s"] is not None else "FAIL",
                f"{r['throughput']:.1f}" if r["throughput"] is not None else "FAIL",
                r["status"],
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Warm averages table (one section per prompt case)
    warm_results = [r for r in all_results if not r.get("cold_start")]
    for case_name in ["short", "medium", "long"]:
        case_results = [r for r in warm_results if r["case"] == case_name]
        if not case_results:
            continue
        lines.append(f"## Warm Averages — {case_name}")
        lines.append("")
        headers = ["MEM_FRAC", "CTX_LEN", "Avg First Token (s)", "Avg Total (s)", "Avg Throughput (tok/s)", "Avg Completion Tokens", "Status"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for r in case_results:
            row = [
                r["mem_fraction"],
                r["context_length"],
                f"{r['avg_first_token_s']:.3f}" if r["avg_first_token_s"] is not None else "FAIL",
                f"{r['avg_total_s']:.3f}" if r["avg_total_s"] is not None else "FAIL",
                f"{r['avg_throughput']:.1f}" if r["avg_throughput"] is not None else "FAIL",
                f"{r['avg_completion_tokens']:.0f}" if r["avg_completion_tokens"] is not None else "FAIL",
                r["status"],
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_results(csv_path: Path, mem_frac: str, ctx_len: str) -> list[dict[str, Any]]:
    """Parse a benchmark CSV into structured results for the comparison."""
    rows = load_csv_rows(csv_path)
    results = []

    cold_rows = [r for r in rows if r.get("scenario") == "cold_start"]
    for r in cold_rows:
        results.append({
            "cold_start": True,
            "mem_fraction": mem_frac,
            "context_length": ctx_len,
            "case": r.get("case", "short"),
            "health_ready_s": float(r["container_health_ready_s"]) if r.get("container_health_ready_s") else None,
            "model_ready_s": float(r["model_ready_s"]) if r.get("model_ready_s") else None,
            "first_req_s": float(r["first_successful_request_s"]) if r.get("first_successful_request_s") else None,
            "first_token_s": float(r["first_token_latency_s"]) if r.get("first_token_latency_s") else None,
            "throughput": float(r["throughput_tokens_per_second"]) if r.get("throughput_tokens_per_second") else None,
            "status": "OK",
        })

    warm_rows = [r for r in rows if r.get("scenario") == "warm"]
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in warm_rows:
        grouped.setdefault(r["case"], []).append(r)

    for case_name, case_rows in grouped.items():
        count = len(case_rows)
        first_tokens = [float(r["first_token_latency_s"]) for r in case_rows if r.get("first_token_latency_s")]
        totals = [float(r["total_response_time_s"]) for r in case_rows if r.get("total_response_time_s")]
        throughputs = [float(r["throughput_tokens_per_second"]) for r in case_rows if r.get("throughput_tokens_per_second")]
        completions = [float(r["completion_tokens"]) for r in case_rows if r.get("completion_tokens")]

        results.append({
            "cold_start": False,
            "mem_fraction": mem_frac,
            "context_length": ctx_len,
            "case": case_name,
            "avg_first_token_s": sum(first_tokens) / len(first_tokens) if first_tokens else None,
            "avg_total_s": sum(totals) / len(totals) if totals else None,
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else None,
            "avg_completion_tokens": sum(completions) / len(completions) if completions else None,
            "status": "OK",
        })

    return results


def main() -> int:
    if not HARDWARE_JSON.exists():
        print("Error: hardware.json not found.")
        return 1
    if not ACTIVE_FILE.exists():
        print("Error: no active config. Run: make use MODEL=<name> HARDWARE=<name>")
        return 1

    lines = ACTIVE_FILE.read_text(encoding="utf-8").strip().splitlines()
    model_name = lines[0]
    hw_name = lines[1]

    hw_data = read_hardware()
    if hw_name not in hw_data:
        print(f"Error: unknown hardware profile '{hw_name}'")
        return 1

    # Back up original hardware.json
    backup_path = HARDWARE_JSON.with_suffix(".json.phase3-backup")
    shutil.copy2(HARDWARE_JSON, backup_path)
    print(f"Backed up hardware.json to {backup_path.name}")

    all_results: list[dict[str, Any]] = []
    failed_configs: list[str] = []
    total = len(MEM_FRACTION_VALUES) * len(CONTEXT_LENGTH_VALUES)
    current = 0

    try:
        for mem_frac in MEM_FRACTION_VALUES:
            for ctx_len in CONTEXT_LENGTH_VALUES:
                current += 1
                label = f"tune-mfs{mem_frac}-ctx{ctx_len}"
                slug = slugify(label)

                print(f"\n{'#' * 60}")
                print(f"# Config {current}/{total}: MEM_FRACTION_STATIC={mem_frac} CONTEXT_LENGTH={ctx_len}")
                print(f"{'#' * 60}")

                # Update hardware.json model_overrides for the active model
                hw_data = read_hardware()
                hw_data = set_model_tuning(hw_data, hw_name, model_name, mem_frac, ctx_len)
                write_hardware(hw_data)

                rc = run_benchmark(label, cold_start=True)

                if rc != 0:
                    print(f"\nBenchmark FAILED for MFS={mem_frac} CTX={ctx_len}")
                    failed_configs.append(f"MFS={mem_frac} CTX={ctx_len}")
                    all_results.append({
                        "cold_start": True,
                        "mem_fraction": mem_frac,
                        "context_length": ctx_len,
                        "case": "short",
                        "health_ready_s": None,
                        "model_ready_s": None,
                        "first_req_s": None,
                        "first_token_s": None,
                        "throughput": None,
                        "status": "FAIL",
                    })
                    continue

                csv_path, _ = find_result_files(slug)
                if csv_path:
                    results = parse_results(csv_path, mem_frac, ctx_len)
                    all_results.extend(results)
                else:
                    print(f"Warning: could not find result files for {slug}")

    finally:
        # Restore original hardware.json
        shutil.copy2(backup_path, HARDWARE_JSON)
        print(f"\nRestored original hardware.json from {backup_path.name}")
        backup_path.unlink()

    # Write comparison report
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = RESULTS_DIR / f"{timestamp}-phase3-gpu-tuning-matrix.md"
    report_path.write_text(build_comparison_table(all_results), encoding="utf-8")
    print(f"\nWrote comparison report: {report_path.relative_to(PROJECT_DIR)}")

    if failed_configs:
        print(f"\nFailed configs: {', '.join(failed_configs)}")

    # Summary
    print(f"\nCompleted {total - len(failed_configs)}/{total} configurations successfully.")
    print(f"Individual reports are in benchmarks/results/ with 'tune-' prefix.")
    print(f"Comparison report: {report_path.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
