#!/usr/bin/env python3
"""Refresh persisted SGLang MoE config files with before/after benchmarks."""

from __future__ import annotations

import argparse
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

from smelter_config import ConfigError, load_state, resolve_instance_runtime


PROJECT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_SCRIPT = PROJECT_DIR / "scripts" / "benchmark.py"
START_SCRIPT = PROJECT_DIR / "scripts" / "start.sh"
RESULTS_DIR = PROJECT_DIR / "benchmarks" / "results"
UPSTREAM_TUNING_URL = (
    "https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
)
DEFAULT_LABEL = "moe-config-refresh"
FALLBACK_SOURCE_DEVICES = (
    "NVIDIA_L40S",
    "NVIDIA_H100_80GB_HBM3",
    "NVIDIA_B200",
)


@dataclass(frozen=True)
class RuntimeMeta:
    active_workload: str
    instance_name: str
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
    result_path: Path
    metrics: dict[str, Any]


def slugify(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-") or "benchmark"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh repo-backed MoE Triton config files for one target instance "
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
        "--list",
        action="store_true",
        help="List available upstream source profiles for the configured instance and exit.",
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


def resolve_runtime_meta() -> RuntimeMeta:
    try:
        state = load_state(require_active=True)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    instance_name = os.environ.get("INSTANCE")
    if not instance_name:
        if len(state["active_instances"]) == 1:
            instance_name = state["active_instances"][0]
        else:
            raise SystemExit(
                "Error: refresh-moe-configs requires INSTANCE=<name> when the active workload has multiple instances."
            )

    try:
        runtime = resolve_instance_runtime(state, instance_name)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    return RuntimeMeta(
        active_workload=state["active_workload"],
        instance_name=runtime["instance_name"],
        image=runtime["docker_image"],
        model_id=runtime["model_id"],
        tp_size=runtime["tp"],
        ep_size=runtime["ep"],
        hf_cache_source=PROJECT_DIR / "models",
        moe_config_source=PROJECT_DIR / "sglang-moe-configs",
        hf_token=os.environ.get("HF_TOKEN", ""),
    )


def sanitize_device_name(value: str) -> str:
    return value.replace(" ", "_").strip()


def get_host_device_name() -> str:
    result = run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        cwd=None,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        raise SystemExit("Error: no NVIDIA GPU name was returned by nvidia-smi.")
    return sanitize_device_name(names[0])


def fetch_moe_metadata(
    runtime_meta: RuntimeMeta,
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
        f"{runtime_meta.hf_cache_source}:/root/.cache/huggingface",
        "-e",
        f"MODEL_ID={runtime_meta.model_id}",
        "-e",
        f"TP_SIZE={runtime_meta.tp_size}",
        "-e",
        f"EP_SIZE={runtime_meta.ep_size}",
        "-e",
        f"HOST_DEVICE={host_device_name}",
        "-e",
        f"SOURCE_DEVICE={requested_source_device}",
        "-e",
        f"HF_TOKEN={runtime_meta.hf_token}",
        "-e",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "-e",
        "TRANSFORMERS_VERBOSITY=error",
        "--entrypoint",
        "python3",
        runtime_meta.image,
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


def load_benchmark_metrics(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"Error: benchmark result file is empty: {path}")
    return json.loads(lines[-1])


def parse_saved_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Saved: "):
            return PROJECT_DIR / line.removeprefix("Saved: ").strip()
    raise SystemExit("Error: could not determine benchmark artifact path from benchmark.py output.")


def run_benchmark(*, instance_name: str, label: str) -> BenchmarkArtifact:
    env = os.environ.copy()
    env["INSTANCE"] = instance_name
    command = [sys.executable, str(BENCHMARK_SCRIPT), label]

    try:
        result = run_command(command, env=env)
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

    result_path = parse_saved_path(result.stdout)
    return BenchmarkArtifact(
        label=label,
        result_path=result_path,
        metrics=load_benchmark_metrics(result_path),
    )


def signed_delta(after: float, before: float) -> str:
    return f"{after - before:+.2f}"


def write_comparison_report(
    *,
    label: str,
    runtime_meta: RuntimeMeta,
    metadata: dict[str, Any],
    before_artifact: BenchmarkArtifact,
    after_artifact: BenchmarkArtifact,
    target_primary: Path,
    target_down: Path,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = RESULTS_DIR / f"{generated_at}-{slugify(label)}-comparison.md"

    metric_rows = [
        (
            "Output throughput (tok/s)",
            before_artifact.metrics.get("output_throughput", 0.0),
            after_artifact.metrics.get("output_throughput", 0.0),
        ),
        (
            "Mean TTFT (ms)",
            before_artifact.metrics.get("mean_ttft_ms", 0.0),
            after_artifact.metrics.get("mean_ttft_ms", 0.0),
        ),
        (
            "Mean TPOT (ms)",
            before_artifact.metrics.get("mean_tpot_ms", 0.0),
            after_artifact.metrics.get("mean_tpot_ms", 0.0),
        ),
        (
            "Mean E2E latency (ms)",
            before_artifact.metrics.get("mean_e2e_latency_ms", 0.0),
            after_artifact.metrics.get("mean_e2e_latency_ms", 0.0),
        ),
        (
            "P99 ITL (ms)",
            before_artifact.metrics.get("p99_itl_ms", 0.0),
            after_artifact.metrics.get("p99_itl_ms", 0.0),
        ),
    ]

    lines = [
        f"# MoE Refresh Comparison: {label}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Workload: `{runtime_meta.active_workload}`",
        f"- Instance: `{runtime_meta.instance_name}`",
        f"- Model: `{metadata['model_id']}`",
        f"- Architecture: `{metadata['architecture']}`",
        f"- TP/EP: `{metadata['tp_size']}` / `{metadata['ep_size']}`",
        f"- Exact shape: `{metadata['shape_prefix']}`",
        f"- Triton target dir: `{metadata['version_dir']}`",
        f"- Target primary file: `{target_primary.relative_to(PROJECT_DIR)}`",
        f"- Target down file: `{target_down.relative_to(PROJECT_DIR)}`",
        f"- Before benchmark: `{before_artifact.result_path.relative_to(PROJECT_DIR)}`",
        f"- After benchmark: `{after_artifact.result_path.relative_to(PROJECT_DIR)}`",
        "",
        "| Metric | Before | After | Delta |",
        "| --- | --- | --- | --- |",
    ]

    selected = metadata.get("selected")
    if selected is not None:
        lines.insert(
            11,
            f"- Source profile: `{selected['device']}` from `triton_{selected['version'].replace('.', '_')}`",
        )

    for name, before, after in metric_rows:
        lines.append(
            f"| {name} | {before:.2f} | {after:.2f} | {signed_delta(after, before)} |"
        )

    lines.extend(
        [
            "",
            f"- Completed before: `{before_artifact.metrics.get('completed')}`",
            f"- Completed after: `{after_artifact.metrics.get('completed')}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_failure_report(
    *,
    label: str,
    runtime_meta: RuntimeMeta,
    metadata: dict[str, Any],
    before_artifact: BenchmarkArtifact,
    target_primary: Path,
    target_down: Path,
    error: subprocess.CalledProcessError,
    restore_returncode: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = RESULTS_DIR / f"{generated_at}-{slugify(label)}-failure.md"
    selected = metadata.get("selected")

    lines = [
        f"# MoE Refresh Failure: {label}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Workload: `{runtime_meta.active_workload}`",
        f"- Instance: `{runtime_meta.instance_name}`",
        f"- Model: `{metadata['model_id']}`",
        f"- Architecture: `{metadata['architecture']}`",
        f"- TP/EP: `{metadata['tp_size']}` / `{metadata['ep_size']}`",
        f"- Exact shape: `{metadata['shape_prefix']}`",
        f"- Triton target dir: `{metadata['version_dir']}`",
        f"- Target primary file: `{target_primary.relative_to(PROJECT_DIR)}`",
        f"- Target down file: `{target_down.relative_to(PROJECT_DIR)}`",
        f"- Baseline benchmark: `{before_artifact.result_path.relative_to(PROJECT_DIR)}`",
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


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    runtime_meta = resolve_runtime_meta()
    host_device_name = get_host_device_name()
    metadata = fetch_moe_metadata(
        runtime_meta,
        host_device_name=host_device_name,
        requested_source_device=args.source_device,
    )

    target_dir = runtime_meta.moe_config_source / "configs" / metadata["version_dir"]
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

    print(f"Configured workload: {runtime_meta.active_workload}")
    print(f"Configured instance: {runtime_meta.instance_name}")
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
        instance_name=runtime_meta.instance_name,
        label=f"{args.label}-before",
    )

    with tempfile.TemporaryDirectory(prefix="refresh-moe-configs-") as backup_root:
        backup_dir = Path(backup_root)
        existence = backup_targets(target_primary, target_down, backup_dir)
        refresh_targets(
            image=runtime_meta.image,
            metadata=metadata,
            target_primary=target_primary,
            target_down=target_down,
        )

        try:
            print("")
            print("Running post-refresh benchmark...")
            after_artifact = run_benchmark(
                instance_name=runtime_meta.instance_name,
                label=f"{args.label}-after",
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
                runtime_meta=runtime_meta,
                before_artifact=before_artifact,
                metadata=metadata,
                target_primary=target_primary,
                target_down=target_down,
                error=error,
                restore_returncode=start_result.returncode,
            )
            print(f"Failure report: {failure_report.relative_to(PROJECT_DIR)}")
            raise SystemExit(error.returncode) from error

    comparison_path = write_comparison_report(
        label=args.label,
        runtime_meta=runtime_meta,
        before_artifact=before_artifact,
        after_artifact=after_artifact,
        metadata=metadata,
        target_primary=target_primary,
        target_down=target_down,
    )

    print("")
    print("Refresh complete.")
    print(f"Before benchmark: {before_artifact.result_path.relative_to(PROJECT_DIR)}")
    print(f"After benchmark: {after_artifact.result_path.relative_to(PROJECT_DIR)}")
    print(f"Comparison report: {comparison_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
