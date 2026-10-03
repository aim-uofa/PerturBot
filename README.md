# openpi — portable PiperX π₀.₅

A focused training and inference release based on [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi).
It retains the JAX model implementation and adds a portable PiperX data pipeline, **single-node 8-GPU launcher**, local inference, and an optional WebSocket service.

[中文说明](README_zh.md) · [Training](docs/training.md) · [Inference](docs/inference.md) · [Checkpoints](docs/checkpoints.md)

## What is included

- Three RGB cameras and a 14-dimensional dual-arm/gripper interface.
- Manifest-driven conversion to the **pinned LeRobot v2.1** dataset format.
- π₀.₅ fine-tuning with global batch 256 and 8-way FSDP; one JAX process, **not `torchrun`**.
- CPU normalization over all selected frames, deterministic episode holdout, optional validation and local JSONL metrics.
- In-process inference with **no port required**, or an explicitly configurable loopback WebSocket server.
- A source audit and an inference-only checkpoint exporter.

No training data, private normalization statistics, fine-tuned weights, credentials, corporate endpoints, run logs, or original Git history are distributed. This is not an official Physical Intelligence release. Code availability does not imply permission to redistribute training data or weights.

## Requirements

- Linux x86-64; Python **3.11**; `git` and [uv](https://docs.astral.sh/uv/).
- For the documented full fine-tuning recipe: **one machine with 8 NVIDIA GPUs**, preferably A100-80GB / H100-80GB, and a CUDA 12-compatible driver. Eight small GPUs are not a guaranteed substitute.
- Sufficient CPU RAM and disk for datasets, pretrained weights, optimizer state and retained checkpoints. An uncompressed full training checkpoint can be tens of GB; reserve several hundred GB for a run.
- Public Internet access on first installation/asset download, or prepopulated caches. No company network or Google account is required for the public assets.

The lockfile pins JAX 0.5.3, Flax 0.10.2 and LeRobot to a specific public Git revision. Do not replace LeRobot with an arbitrary newer release.

## Quick start: data → statistics → 8-GPU training

Run all commands from the repository root.

```bash
# 1. Install the locked environment; skip unrelated large Git LFS assets.
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen

# 2. Keep this setting in every conversion/statistics/training shell.
export HF_LEROBOT_HOME="$PWD/data/lerobot"

# 3. Prepare your own episode manifest; see docs/training.md for the schema.
#    Skip conversion if local/piperx already exists in LeRobot v2.1 format.
uv run examples/piperx_real/convert_data.py \
  --manifest /path/to/episodes.json \
  --raw-root /path/to/raw_data \
  --repo-id local/piperx

# 4. Compute dataset-specific statistics. Do NOT reuse an unrelated dataset's stats.
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_piperx --repo-id local/piperx

# 5. Train: one host, eight visible GPUs, no cluster setup or rendezvous port.
bash scripts/train_8gpu.sh --exp-name piperx_run --data.repo-id local/piperx
```

The base weights are downloaded from `gs://openpi-assets/checkpoints/pi05_base/params` on first use. To prefetch, use `uv run scripts/prefetch_assets.py`. To use an existing local copy, append `--weight-loader.params-path /path/to/pi05_base/params` to training.

Default outputs:

```text
assets/pi05_piperx/piperx/norm_stats.json
checkpoints/pi05_piperx/piperx_run/
├── run_info.json
├── metrics.jsonl
└── 15000/
    ├── assets/piperx/norm_stats.json
    ├── params/                 # EMA weights used for inference
    └── train_state/            # training/optimizer state for resume
```

New checkpoints use **completed optimizer steps** (`15000` after 15000 updates). Older snapshots may use zero-based names such as `14999`; do not rename or reinterpret them blindly.

## Inference

Use the **step directory**, not `params/`:

```bash
# Synthetic API smoke test only. Does not operate any robot.
CUDA_VISIBLE_DEVICES=0 uv run examples/piperx_real/infer.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --dummy --prompt 'Place the object on the plate.'
```

For real observations, pass `--observation observation.npz` instead of `--dummy`. The NPZ contains `image`, `wrist_left_image`, `wrist_right_image` (RGB uint8 HWC) and `state` (float32, shape `(14,)`). Output is an absolute action chunk of shape `(10, 14)` in the demonstration's units. See the [input/action contract](docs/inference.md) before connecting hardware.

Optional remote interface:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --host 127.0.0.1 --port 8000

# In another shell:
uv run examples/piperx_real/client.py --host 127.0.0.1 --port 8000
```

The service has no authentication or TLS. It is local-only by default. Do not expose it directly to the Internet.

## Important compatibility notes

- The PiperX recipe keeps `action_horizon=10`, `discrete_state_input=False`, and the original state layout. The discrete-state setting differs from upstream π₀.₅ defaults; changing it is **not** a harmless deployment option.
- `pi05_piperx` trains **delta joints + absolute grippers** and converts predictions back to absolute joint targets during inference.
- `pi05_piperx_absolute` is for checkpoints trained on **absolute** actions. Choose the mode from the recorded training configuration, not today's edited configuration file.
- Weights and normalization statistics must come from the **same training run and action convention**.
- There is **no substantiated “best checkpoint” claim** in this source release. Training loss, validation action MSE, and real-robot success rate are different metrics; see [checkpoint selection](docs/checkpoints.md).

## Tests and release review

```bash
JAX_PLATFORMS=cpu uv run pytest

# Test real JAX sharding, validation, save/resume and parameter restoration using
# a small synthetic model on eight virtual CPU devices. This is NOT an 8-GPU benchmark.
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest tests/test_training_smoke.py -q

uv run python scripts/audit_release.py
bash scripts/train_8gpu.sh --dry-run --exp-name example
```

The default tests are offline/synthetic and do not download π₀.₅ weights or operate a robot. Some retained upstream tests outside the default test paths require large models or external datasets.

See [validation results and limitations](docs/validation.md). Before publishing, complete the [release checklist](docs/release_checklist.md). Preserve [LICENSE](LICENSE), [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt), and [NOTICE](NOTICE). Confirm the applicable model/data terms separately; this repository's code license does not automatically license a fine-tuned checkpoint or dataset.
