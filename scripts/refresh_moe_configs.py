#!/usr/bin/env python3
"""Refresh persisted SGLang MoE config files with before/after benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = PROJECT_DIR / "scripts" / "benchmark.py"
START_SCRIPT = PROJECT_DIR / "scripts" / "start.sh"
RESULTS_DIR = PROJECT_DIR / "benchmarks" / "results"
UPSTREAM_TUNING_URL = (
    "https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
)
DEFAULT_LABEL = "moe-config-refresh"
DEFAULT_WARM_RUNS = 3
DEFAULT_COLD_CASE = "short"
FALLBACK_SOURCE_DEVICES = (
    "NVIDIA_L40S",
    "NVIDIA_H100_80GB_HBM3",
    "NVIDIA_B200",
)


@dataclass(frozen=True)
class ComposeMeta:
    image: str
    model_id: str
    tp_size: int
    ep_size: int
    hf_cache_source: Path
    moe_config_source: Path
    hf_token: str


@dataclass(frozen=True)
class BenchmarkArtifact:
    label: str
    csv_path: Path
    md_path: Path


def slugify(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-"
        for character in value.lower()
    ).strip("-") or "benchmark"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh repo-backed MoE Triton config files for the configured model "
            "and benchmark before/after performance."
        )
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="Base label used in generated benchmark and comparison report filenames.",
    )
    parser.add_argument(
        "--source-device",
        help=(
            "Upstream source profile to seed from, for example NVIDIA_L40S. "
            "Spaces are accepted and converted to underscores."
        ),
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=DEFAULT_WARM_RUNS,
        help="Warm benchmark runs per prompt case for both before and after measurements.",
    )
    parser.add_argument(
        "--cold-case",
        choices=["short", "medium", "long"],
        default=DEFAULT_COLD_CASE,
        help="Prompt case used for the cold-start request in both benchmark passes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available upstream source profiles for the configured model and exit.",
    )
    return parser.parse_args(argv)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = PROJECT_DIR,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def command_flag_value(command_items: list[str], flag: str, default: str) -> str:
    try:
        index = command_items.index(flag)
    except ValueError:
        return default

    if index + 1 >= len(command_items):
        return default
    return str(command_items[index + 1])


def resolve_compose_meta() -> ComposeMeta:
    compose_output = run_command(["docker", "compose", "config", "--format", "json"])
    config = json.loads(compose_output.stdout)
    service = config["services"]["sglang"]
    command = [str(item) for item in service.get("command", [])]
    environment = service.get("environment", {})

    hf_cache_source: Path | None = None
    moe_config_source: Path | None = None
    for volume in service.get("volumes", []):
        target = volume.get("target")
        source = volume.get("source")
        if not source:
            continue
        if target == "/root/.cache/huggingface":
            hf_cache_source = Path(source)
        elif target == "/sglang-moe-configs":
            moe_config_source = Path(source)

    return ComposeMeta(
        image=str(service["image"]),
        model_id=command_flag_value(
            command, "--model-path", "nvidia/Nemotron-Cascade-2-30B-A3B"
        ),
        tp_size=int(command_flag_value(command, "--tp", "1")),
        ep_size=int(command_flag_value(command, "--ep", "1")),
        hf_cache_source=hf_cache_source or (PROJECT_DIR / "models"),
        moe_config_source=moe_config_source or (PROJECT_DIR / "sglang-moe-configs"),
        hf_token=str(environment.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))),
    )


def sanitize_device_name(value: str) -> str:
    return value.replace(" ", "_").strip()


def get_host_device_name() -> str:
    result = run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], cwd=None
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        raise SystemExit("Error: no NVIDIA GPU name was returned by nvidia-smi.")
    return sanitize_device_name(names[0])


def split_filename_around_device(filename: str) -> tuple[str, str, str]:
    marker = ",device_name="
    if marker not in filename:
        raise ValueError(f"Could not parse device name from filename: {filename}")

    prefix, remainder = filename.split(marker, 1)
    prefix = f"{prefix}{marker}"
    suffix_markers = [",dtype=", ",block_shape=", ",per_channel_quant=True", ".json"]
    suffix_positions = [
        remainder.find(candidate)
        for candidate in suffix_markers
        if remainder.find(candidate) != -1
    ]
    if not suffix_positions:
        raise ValueError(f"Could not determine suffix for filename: {filename}")

    suffix_start = min(suffix_positions)
    return prefix, remainder[:suffix_start], remainder[suffix_start:]


def fetch_moe_metadata(
    compose_meta: ComposeMeta,
    *,
    host_device_name: str,
    requested_source_device: str | None,
) -> dict[str, Any]:
    requested_source_device = (
        sanitize_device_name(requested_source_device)
        if requested_source_device is not None
        else ""
    )

    script = r"""
import json
import os
import sys
from pathlib import Path

import triton

sys.path.insert(0, "/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton")

from common_utils import get_config_filename, get_model_config


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def split_filename_around_device(filename: str) -> tuple[str, str, str]:
    marker = ",device_name="
    if marker not in filename:
        raise ValueError(f"Could not parse device name from filename: {filename}")
    prefix, remainder = filename.split(marker, 1)
    prefix = f"{prefix}{marker}"
    suffix_markers = [",dtype=", ",block_shape=", ",per_channel_quant=True", ".json"]
    suffix_positions = [
        remainder.find(candidate)
        for candidate in suffix_markers
        if remainder.find(candidate) != -1
    ]
    if not suffix_positions:
        raise ValueError(f"Could not determine suffix for filename: {filename}")
    suffix_start = min(suffix_positions)
    return prefix, remainder[:suffix_start], remainder[suffix_start:]


model_id = os.environ["MODEL_ID"]
tp_size = int(os.environ["TP_SIZE"])
ep_size = int(os.environ["EP_SIZE"])
host_device = os.environ["HOST_DEVICE"]
requested_source_device = os.environ.get("SOURCE_DEVICE", "")

config = get_model_config(model_id, tp_size, ep_size)
target_primary_name = get_config_filename(
    config["num_experts"],
    config["shard_intermediate_size"],
    config["hidden_size"],
    config["topk"],
    config["dtype"],
    False,
    False,
    False,
    False,
    config["block_shape"],
)
target_down_name = f"{target_primary_name[:-5]}_down.json"
prefix, _, suffix = split_filename_around_device(target_primary_name)
shape_prefix = target_primary_name.split(",device_name=", 1)[0]
triton_version = triton.__version__
version_dir = f"triton_{triton_version.replace('.', '_')}"

config_root = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs"
)

candidates = []
for path in config_root.glob("triton_*/*.json"):
    name = path.name
    if name.endswith("_down.json"):
        continue
    if not name.startswith(prefix) or not name.endswith(suffix):
        continue

    version_dir_name = path.parent.name
    version = version_dir_name.replace("triton_", "").replace("_", ".")
    device = name[len(prefix) : len(name) - len(suffix)]
    down_path = path.with_name(f"{path.stem}_down.json")
    candidates.append(
        {
            "device": device,
            "version": version,
            "version_dir": version_dir_name,
            "primary_path": str(path),
            "down_path": str(down_path) if down_path.exists() else None,
            "has_dedicated_down": down_path.exists(),
        }
    )


def newest(matches):
    return max(matches, key=lambda item: version_tuple(item["version"]))


selected = None
if requested_source_device:
    matches = [item for item in candidates if item["device"] == requested_source_device]
    if matches:
        selected = newest(matches)
else:
    preferred_devices = []
    if host_device:
        preferred_devices.append(host_device)
    for device in ["NVIDIA_L40S", "NVIDIA_H100_80GB_HBM3", "NVIDIA_B200"]:
        if device not in preferred_devices:
            preferred_devices.append(device)

    for device in preferred_devices:
        matches = [item for item in candidates if item["device"] == device]
        if matches:
            selected = newest(matches)
            break

    if selected is None and candidates:
        selected = newest(candidates)

candidates.sort(
    key=lambda item: (version_tuple(item["version"]), item["device"]),
    reverse=True,
)

print(
    json.dumps(
        {
            "model_id": model_id,
            "tp_size": tp_size,
            "ep_size": ep_size,
            "architecture": config["architecture"],
            "num_experts": config["num_experts"],
            "shard_intermediate_size": config["shard_intermediate_size"],
            "shape_prefix": shape_prefix,
            "triton_version": triton_version,
            "version_dir": version_dir,
            "target_primary_name": target_primary_name,
            "target_down_name": target_down_name,
            "candidates": candidates,
            "selected": selected,
        }
    )
)
"""

    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{compose_meta.hf_cache_source}:/root/.cache/huggingface",
        "-e",
        f"MODEL_ID={compose_meta.model_id}",
        "-e",
        f"TP_SIZE={compose_meta.tp_size}",
        "-e",
        f"EP_SIZE={compose_meta.ep_size}",
        "-e",
        f"HOST_DEVICE={host_device_name}",
        "-e",
        f"SOURCE_DEVICE={requested_source_device}",
        "-e",
        f"HF_TOKEN={compose_meta.hf_token}",
        "-e",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "-e",
        "TRANSFORMERS_VERBOSITY=error",
        "--entrypoint",
        "python3",
        compose_meta.image,
        "-W",
        "ignore",
        "-c",
        script,
    ]
    result = run_command(command)
    payload = result.stdout.strip().splitlines()
    if not payload:
        raise SystemExit("Error: failed to resolve MoE config metadata from the SGLang image.")
    return json.loads(payload[-1])


def print_profile_listing(metadata: dict[str, Any], *, target_primary: Path, target_down: Path) -> None:
    print(f"Configured model: {metadata['model_id']}")
    print(f"Architecture: {metadata['architecture']}")
    print(f"TP/EP: {metadata['tp_size']}/{metadata['ep_size']}")
    print(f"Exact shape: {metadata['shape_prefix']}")
    print(f"Current Triton version: {metadata['triton_version']}")
    print(f"Target primary file: {target_primary.relative_to(PROJECT_DIR)}")
    print(f"Target down file: {target_down.relative_to(PROJECT_DIR)}")
    print("")

    candidates = metadata.get("candidates", [])
    if not candidates:
        print("No upstream source profiles were found for this exact filename shape.")
        print(f"Manual tuning may be required: {UPSTREAM_TUNING_URL}")
        return

    print("Available source profiles:")
    for candidate in candidates:
        down_suffix = " with dedicated _down" if candidate["has_dedicated_down"] else ""
        print(
            f"- {candidate['device']} (triton {candidate['version']}){down_suffix}: "
            f"{candidate['primary_path']}"
        )

    if metadata.get("selected"):
        selected = metadata["selected"]
        print("")
        print(
            "Default selection: "
            f"{selected['device']} from triton {selected['version']}"
        )


def run_benchmark(*, label: str, warm_runs: int, cold_case: str) -> BenchmarkArtifact:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--cold-start",
        "--warm-runs",
        str(warm_runs),
        "--cold-case",
        cold_case,
        "--label",
        label,
    ]
    try:
        result = run_command(command)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="", file=sys.stderr)
        raise
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    csv_path: Path | None = None
    md_path: Path | None = None
    for line in result.stdout.splitlines():
        if line.startswith("Wrote CSV report: "):
            csv_path = PROJECT_DIR / line.removeprefix("Wrote CSV report: ").strip()
        elif line.startswith("Wrote Markdown report: "):
            md_path = PROJECT_DIR / line.removeprefix("Wrote Markdown report: ").strip()

    if csv_path is None or md_path is None:
        raise SystemExit("Error: could not determine benchmark artifact paths from benchmark.py output.")

    return BenchmarkArtifact(label=label, csv_path=csv_path, md_path=md_path)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def summarize_cold(rows: list[dict[str, str]]) -> dict[str, float] | None:
    cold_rows = [row for row in rows if row["scenario"] == "cold_start"]
    if not cold_rows:
        return None
    row = cold_rows[0]
    return {
        "container_health_ready_s": float(row["container_health_ready_s"]),
        "model_ready_s": float(row["model_ready_s"]),
        "first_successful_request_s": float(row["first_successful_request_s"]),
        "first_token_latency_s": float(row["first_token_latency_s"]),
        "total_response_time_s": float(row["total_response_time_s"]),
        "throughput_tokens_per_second": float(row["throughput_tokens_per_second"] or 0.0),
    }


def summarize_warm(
    rows: list[dict[str, str]],
    *,
    skip_first_iteration: bool,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["scenario"] != "warm":
            continue
        iteration = int(row["iteration"])
        if skip_first_iteration and iteration < 2:
            continue
        grouped.setdefault(row["case"], []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for case_name in ["short", "medium", "long"]:
        case_rows = grouped.get(case_name, [])
        if not case_rows:
            continue
        count = len(case_rows)
        summary[case_name] = {
            "runs": float(count),
            "first_token_latency_s": sum(
                float(row["first_token_latency_s"]) for row in case_rows
            )
            / count,
            "total_response_time_s": sum(
                float(row["total_response_time_s"]) for row in case_rows
            )
            / count,
            "throughput_tokens_per_second": sum(
                float(row["throughput_tokens_per_second"] or 0.0)
                for row in case_rows
            )
            / count,
        }
    return summary


def format_value(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_table(headers: list[str], rows: list[list[float | int | str | None]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(format_value(value)))

    def render(row: list[float | int | str | None]) -> str:
        return "| " + " | ".join(
            format_value(value).ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def signed_delta(after: float, before: float) -> str:
    delta = after - before
    return f"{delta:+.3f}"


def write_comparison_report(
    *,
    label: str,
    before_artifact: BenchmarkArtifact,
    after_artifact: BenchmarkArtifact,
    metadata: dict[str, Any],
    target_primary: Path,
    target_down: Path,
) -> Path:
    before_rows = load_csv_rows(before_artifact.csv_path)
    after_rows = load_csv_rows(after_artifact.csv_path)

    before_cold = summarize_cold(before_rows)
    after_cold = summarize_cold(after_rows)
    before_warm = summarize_warm(before_rows, skip_first_iteration=False)
    after_warm = summarize_warm(after_rows, skip_first_iteration=False)
    before_steady = summarize_warm(before_rows, skip_first_iteration=True)
    after_steady = summarize_warm(after_rows, skip_first_iteration=True)

    generated_at = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = RESULTS_DIR / f"{generated_at}-{slugify(label)}-comparison.md"

    lines = [
        f"# MoE Refresh Comparison: {label}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Model: `{metadata['model_id']}`",
        f"- Architecture: `{metadata['architecture']}`",
        f"- TP/EP: `{metadata['tp_size']}` / `{metadata['ep_size']}`",
        f"- Exact shape: `{metadata['shape_prefix']}`",
        f"- Triton target dir: `{metadata['version_dir']}`",
        f"- Target primary file: `{target_primary.relative_to(PROJECT_DIR)}`",
        f"- Target down file: `{target_down.relative_to(PROJECT_DIR)}`",
        f"- Before benchmark: `{before_artifact.csv_path.relative_to(PROJECT_DIR)}`",
        f"- After benchmark: `{after_artifact.csv_path.relative_to(PROJECT_DIR)}`",
    ]

    selected = metadata.get("selected")
    if selected is not None:
        lines.append(
            f"- Source profile: `{selected['device']}` from `triton_{selected['version'].replace('.', '_')}`"
        )
        lines.append(f"- Source primary path: `{selected['primary_path']}`")
        if selected["down_path"]:
            lines.append(f"- Source down path: `{selected['down_path']}`")
        else:
            lines.append("- Source down path: mirrored from source primary")

    if before_cold and after_cold:
        lines.extend(
            [
                "",
                "## Cold Start",
                "",
                build_table(
                    [
                        "Metric",
                        "Before",
                        "After",
                        "Delta",
                    ],
                    [
                        [
                            "Health Ready (s)",
                            before_cold["container_health_ready_s"],
                            after_cold["container_health_ready_s"],
                            signed_delta(
                                after_cold["container_health_ready_s"],
                                before_cold["container_health_ready_s"],
                            ),
                        ],
                        [
                            "Model Ready (s)",
                            before_cold["model_ready_s"],
                            after_cold["model_ready_s"],
                            signed_delta(
                                after_cold["model_ready_s"],
                                before_cold["model_ready_s"],
                            ),
                        ],
                        [
                            "First Success (s)",
                            before_cold["first_successful_request_s"],
                            after_cold["first_successful_request_s"],
                            signed_delta(
                                after_cold["first_successful_request_s"],
                                before_cold["first_successful_request_s"],
                            ),
                        ],
                        [
                            "First Token (s)",
                            before_cold["first_token_latency_s"],
                            after_cold["first_token_latency_s"],
                            signed_delta(
                                after_cold["first_token_latency_s"],
                                before_cold["first_token_latency_s"],
                            ),
                        ],
                        [
                            "Total Time (s)",
                            before_cold["total_response_time_s"],
                            after_cold["total_response_time_s"],
                            signed_delta(
                                after_cold["total_response_time_s"],
                                before_cold["total_response_time_s"],
                            ),
                        ],
                        [
                            "Throughput (tok/s)",
                            before_cold["throughput_tokens_per_second"],
                            after_cold["throughput_tokens_per_second"],
                            signed_delta(
                                after_cold["throughput_tokens_per_second"],
                                before_cold["throughput_tokens_per_second"],
                            ),
                        ],
                    ],
                ),
            ]
        )

    def append_warm_section(
        title: str,
        before_summary: dict[str, dict[str, float]],
        after_summary: dict[str, dict[str, float]],
    ) -> None:
        if not before_summary or not after_summary:
            return
        rows: list[list[float | int | str | None]] = []
        for case_name in ["short", "medium", "long"]:
            if case_name not in before_summary or case_name not in after_summary:
                continue
            before_case = before_summary[case_name]
            after_case = after_summary[case_name]
            rows.append(
                [
                    case_name,
                    int(before_case["runs"]),
                    before_case["first_token_latency_s"],
                    after_case["first_token_latency_s"],
                    signed_delta(
                        after_case["first_token_latency_s"],
                        before_case["first_token_latency_s"],
                    ),
                    before_case["total_response_time_s"],
                    after_case["total_response_time_s"],
                    signed_delta(
                        after_case["total_response_time_s"],
                        before_case["total_response_time_s"],
                    ),
                    before_case["throughput_tokens_per_second"],
                    after_case["throughput_tokens_per_second"],
                    signed_delta(
                        after_case["throughput_tokens_per_second"],
                        before_case["throughput_tokens_per_second"],
                    ),
                ]
            )

        if not rows:
            return

        lines.extend(
            [
                "",
                f"## {title}",
                "",
                build_table(
                    [
                        "Case",
                        "Runs",
                        "Before First Token (s)",
                        "After First Token (s)",
                        "Delta",
                        "Before Total (s)",
                        "After Total (s)",
                        "Delta",
                        "Before Throughput",
                        "After Throughput",
                        "Delta",
                    ],
                    rows,
                ),
            ]
        )

    append_warm_section("Warm Averages", before_warm, after_warm)
    append_warm_section("Steady-State Warm Averages", before_steady, after_steady)

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def write_failure_report(
    *,
    label: str,
    before_artifact: BenchmarkArtifact,
    metadata: dict[str, Any],
    target_primary: Path,
    target_down: Path,
    error: subprocess.CalledProcessError,
    restore_returncode: int,
) -> Path:
    generated_at = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = RESULTS_DIR / f"{generated_at}-{slugify(label)}-failure.md"
    selected = metadata.get("selected")

    lines = [
        f"# MoE Refresh Failure: {label}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Model: `{metadata['model_id']}`",
        f"- Architecture: `{metadata['architecture']}`",
        f"- TP/EP: `{metadata['tp_size']}` / `{metadata['ep_size']}`",
        f"- Exact shape: `{metadata['shape_prefix']}`",
        f"- Triton target dir: `{metadata['version_dir']}`",
        f"- Target primary file: `{target_primary.relative_to(PROJECT_DIR)}`",
        f"- Target down file: `{target_down.relative_to(PROJECT_DIR)}`",
        f"- Baseline benchmark: `{before_artifact.csv_path.relative_to(PROJECT_DIR)}`",
        f"- Restore command exit code: `{restore_returncode}`",
    ]
    if selected is not None:
        lines.append(
            f"- Source profile: `{selected['device']}` from `triton_{selected['version'].replace('.', '_')}`"
        )
    lines.extend(
        [
            "",
            "## Failure",
            "",
            "```text",
            (error.stderr or error.stdout or str(error)).strip(),
            "```",
        ]
    )
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def backup_targets(target_primary: Path, target_down: Path, backup_dir: Path) -> dict[str, bool]:
    target_primary.parent.mkdir(parents=True, exist_ok=True)
    existence = {
        "primary": target_primary.exists(),
        "down": target_down.exists(),
    }
    if existence["primary"]:
        shutil.copy2(target_primary, backup_dir / "primary.json")
    if existence["down"]:
        shutil.copy2(target_down, backup_dir / "down.json")
    return existence


def restore_targets(
    *,
    target_primary: Path,
    target_down: Path,
    backup_dir: Path,
    existence: dict[str, bool],
) -> None:
    if existence["primary"]:
        shutil.copy2(backup_dir / "primary.json", target_primary)
    elif target_primary.exists():
        target_primary.unlink()

    if existence["down"]:
        shutil.copy2(backup_dir / "down.json", target_down)
    elif target_down.exists():
        target_down.unlink()


def refresh_targets(
    *,
    image: str,
    metadata: dict[str, Any],
    target_primary: Path,
    target_down: Path,
) -> None:
    selected = metadata.get("selected")
    if selected is None:
        raise SystemExit(
            "Error: no compatible upstream source profile was found for this MoE shape. "
            f"Manual tuning may be required: {UPSTREAM_TUNING_URL}"
        )

    target_primary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{target_primary.parent}:/output",
        "-e",
        f"SOURCE_PRIMARY={selected['primary_path']}",
        "-e",
        f"SOURCE_DOWN={selected['down_path'] or ''}",
        "-e",
        f"TARGET_PRIMARY={target_primary.name}",
        "-e",
        f"TARGET_DOWN={target_down.name}",
        "--entrypoint",
        "bash",
        image,
        "-lc",
        (
            'cp "$SOURCE_PRIMARY" "/output/$TARGET_PRIMARY" && '
            'if [ -n "$SOURCE_DOWN" ]; then '
            'cp "$SOURCE_DOWN" "/output/$TARGET_DOWN"; '
            'else '
            'cp "$SOURCE_PRIMARY" "/output/$TARGET_DOWN"; '
            "fi"
        ),
    ]
    result = run_command(command)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def ensure_positive_warm_runs(value: int) -> None:
    if value < 1:
        raise SystemExit("Error: --warm-runs must be at least 1.")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ensure_positive_warm_runs(args.warm_runs)

    compose_meta = resolve_compose_meta()
    host_device_name = get_host_device_name()
    metadata = fetch_moe_metadata(
        compose_meta,
        host_device_name=host_device_name,
        requested_source_device=args.source_device,
    )

    target_dir = compose_meta.moe_config_source / "configs" / metadata["version_dir"]
    target_primary = target_dir / metadata["target_primary_name"]
    target_down = target_dir / metadata["target_down_name"]

    if args.list:
        print_profile_listing(metadata, target_primary=target_primary, target_down=target_down)
        return 0

    selected = metadata.get("selected")
    if selected is None:
        print_profile_listing(metadata, target_primary=target_primary, target_down=target_down)
        raise SystemExit(
            "Error: refresh aborted because no upstream source profile matched the configured model."
        )

    print(f"Configured model: {metadata['model_id']}")
    print(f"Exact shape: {metadata['shape_prefix']}")
    print(f"Target primary file: {target_primary.relative_to(PROJECT_DIR)}")
    print(f"Target down file: {target_down.relative_to(PROJECT_DIR)}")
    print(
        "Selected source profile: "
        f"{selected['device']} from triton {selected['version']}"
    )
    print("")
    print("Running baseline benchmark before refresh...")
    before_artifact = run_benchmark(
        label=f"{args.label}-before",
        warm_runs=args.warm_runs,
        cold_case=args.cold_case,
    )

    with tempfile.TemporaryDirectory(prefix="refresh-moe-configs-") as backup_root:
        backup_dir = Path(backup_root)
        existence = backup_targets(target_primary, target_down, backup_dir)
        refresh_targets(
            image=compose_meta.image,
            metadata=metadata,
            target_primary=target_primary,
            target_down=target_down,
        )

        try:
            print("")
            print("Running post-refresh benchmark...")
            after_artifact = run_benchmark(
                label=f"{args.label}-after",
                warm_runs=args.warm_runs,
                cold_case=args.cold_case,
            )
        except subprocess.CalledProcessError as error:
            print("")
            print("Post-refresh benchmark failed. Restoring previous config files...")
            restore_targets(
                target_primary=target_primary,
                target_down=target_down,
                backup_dir=backup_dir,
                existence=existence,
            )
            start_result = run_command([str(START_SCRIPT)], check=False)
            if start_result.stdout:
                print(start_result.stdout, end="")
            if start_result.stderr:
                print(start_result.stderr, end="", file=sys.stderr)
            failure_report = write_failure_report(
                label=args.label,
                before_artifact=before_artifact,
                metadata=metadata,
                target_primary=target_primary,
                target_down=target_down,
                error=error,
                restore_returncode=start_result.returncode,
            )
            print(
                f"Failure report: {failure_report.relative_to(PROJECT_DIR)}"
            )
            raise SystemExit(error.returncode) from error

    comparison_path = write_comparison_report(
        label=args.label,
        before_artifact=before_artifact,
        after_artifact=after_artifact,
        metadata=metadata,
        target_primary=target_primary,
        target_down=target_down,
    )

    print("")
    print("Refresh complete.")
    print(f"Before benchmark: {before_artifact.csv_path.relative_to(PROJECT_DIR)}")
    print(f"After benchmark: {after_artifact.csv_path.relative_to(PROJECT_DIR)}")
    print(f"Comparison report: {comparison_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
