## Repo-Backed MoE Configs

This directory stores Triton MoE kernel config files on the host so they survive container recreation.

- `configs/triton_3_5_1/E=128,N=1856,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json`
- `configs/triton_3_5_1/E=128,N=1856,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition_down.json`

Docker Compose mounts `./sglang-moe-configs` into the container and sets `SGLANG_MOE_CONFIG_DIR=/sglang-moe-configs`.

`make refresh-moe-configs` derives the exact MoE filename from the configured `MODEL_ID` plus the Compose `--tp` / `--ep` settings, benchmarks the current repo-backed files, refreshes them from the best available upstream seed profile in the current SGLang image, then benchmarks again and writes a comparison report under `benchmarks/results/`. If the post-refresh benchmark fails, it restores the previous repo-backed files and writes a failure report instead.

The current exact-shape map is seeded from SGLang's upstream `NVIDIA_L40S` config for the same `E=128,N=1856` shape because it booted cleanly on this host and removed the missing-config warnings. The `_down` file currently mirrors the same mapping because SGLang probes both filenames for this model.

Refresh these files from the current image with:

```bash
make refresh-moe-configs
```

List other exact-shape upstream seed profiles for the configured model with:

```bash
make refresh-moe-configs REFRESH_MOE_ARGS="--list"
```
