# Plan: Add Qwen3.6-35B-A3B

Source: https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6

## Model

- Primary: `Qwen/Qwen3.6-35B-A3B-FP8` (quantized, recommended)
- Alt: `Qwen/Qwen3.6-35B-A3B` (BF16)
- Native context: 262,144 tokens (extensible past 1M)
- Min recommended context: 128K (to preserve thinking capabilities)
- Modality: text, image, video

## Hardware fit (this host: 2× RTX Pro 6000 Blackwell, 96GB each)

| Hardware | Memory | BF16 TP | FP8 TP |
|----------|--------|---------|--------|
| H100     | 80GB   | 1       | 1      |
| H200     | 141GB  | 1       | 1      |
| B200     | 183GB  | 1       | 1      |

FP8 fits comfortably on 1×96GB Blackwell. BF16 also fits on 1 GPU. `tp=2` available if throughput > latency.

## Cookbook launch flags

```
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --mem-fraction-static 0.8 \
  --host 0.0.0.0 \
  --port 30000
```

Optional:
- `--mm-attention-backend fa3` (H100/H200) / `fa4` (B200/Blackwell)
- `--page-size 64` (required with V2 Mamba `extra_buffer`)
- env `SGLANG_USE_CUDA_IPC_TRANSPORT=1` — improves TTFT

## Required env vars (not per-instance in current schema)

- `SGLANG_ENABLE_SPEC_V2=1` — needed for `--mamba-scheduler-strategy extra_buffer` + `--page-size 64` + EAGLE combo
- `SGLANG_USE_CUDA_IPC_TRANSPORT=1` — optional TTFT improvement

`smelter_config.py:326-335` hard-codes the compose `environment` dict. No per-instance override exists today. To enable spec-v2, add these keys to that dict (safe globally — only activates when `--speculative-*` flags are present).

## Proposed config

### Option A — Full cookbook setup (requires smelter_config.py edit)

`models.json`
```json
"qwen36-35b-a3b-fp8": {
  "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
  "extra_args": [
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
    "--speculative-algorithm", "EAGLE",
    "--speculative-num-steps", "3",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4",
    "--mamba-scheduler-strategy", "extra_buffer",
    "--mm-attention-backend", "fa4",
    "--page-size", "64"
  ]
}
```

`instances.json`
```json
"qwen36": {
  "model": "qwen36-35b-a3b-fp8",
  "port": 11435,
  "gpu_ids": [0],
  "tp": 1,
  "ep": 1,
  "mem_fraction_static": 0.8,
  "context_length": 131072,
  "attention_backend": "flashinfer",
  "chunked_prefill_size": 16384,
  "num_continuous_decode_steps": 4
}
```

`workloads.json`
```json
{ "qwen36": ["qwen36"] }
```

`scripts/smelter_config.py` (env dict ~line 326)
```python
"SGLANG_ENABLE_SPEC_V2": "1",
"SGLANG_USE_CUDA_IPC_TRANSPORT": "1",
```

### Option B — Conservative (no code change)

Same as A but drop from `extra_args`:
- `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens`
- `--mamba-scheduler-strategy extra_buffer`
- `--page-size 64`

Keeps: `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `--mm-attention-backend fa4`.

## Open decisions

- **A vs B** — take the perf win (A) or avoid touching `smelter_config.py` (B)?
- **tp=1 vs tp=2** — tp=1 leaves GPU 1 free for a second instance; tp=2 maxes throughput for a single workload.
- **Context length** — 131072 (128K, balanced) vs 262144 (native) vs smaller for more KV room.
- **FP8 vs BF16** — FP8 matches cookbook primary; BF16 available if accuracy regression observed.

## Follow-up after chosen

1. Apply chosen option to `models.json`, `instances.json`, `workloads.json`.
2. If Option A: patch `scripts/smelter_config.py` env dict.
3. `make use WORKLOAD=qwen36`
4. `make download` (FP8 weights, ~35GB)
5. `make start`
6. `INSTANCE=qwen36 make health`
7. `INSTANCE=qwen36 make bench`
8. Update `docs/configuration.md` "Current Checked-In Workloads" section.
