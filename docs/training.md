# Training PiperX π₀.₅

## 1. Reproducible environment

Use Linux, Python 3.11, a CUDA 12-compatible NVIDIA driver and the checked-in `uv.lock`:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
export HF_LEROBOT_HOME="$PWD/data/lerobot"
# Optional, to put public downloads on a large local disk:
export OPENPI_DATA_HOME="$HOME/.cache/openpi"
```

Do not copy another machine's `.venv` or install a different LeRobot revision. The main recipe is JAX. The retained upstream PyTorch implementation is not an interchangeable training backend for the commands in this guide; its installation/weight-conversion requirements differ.

The source snapshot is independent of any original Git remote/submodules. No internal proxy, service token or cluster job launcher is required.

## 2. Data contract

The pinned LeRobot version expects v2.1 metadata, not the newer v3 layout. A local dataset resolves as `$HF_LEROBOT_HOME/<repo-id>`; use `local/piperx` for the examples. You may use a Hugging Face dataset ID if you have access to a compatible dataset.

| Dataset column | Type/shape | Meaning |
| --- | --- | --- |
| `image` | RGB image, HWC | Head/scene camera |
| `wrist_left_image` | RGB image, HWC | Left wrist camera |
| `wrist_right_image` | RGB image, HWC | Right wrist camera |
| `state` | float32 `(14,)` | Left joints 0–5, right joints 6–11, left/right grippers 12–13 |
| `action` | float32 `(14,)` per frame | Absolute next-state target; loader constructs the action horizon |
| task metadata | string | Natural-language instruction; loaded through `task_index` |

Joint units are preserved from `joint_positions`; the converter does not infer radians/degrees or apply calibration. Grippers are read from `range_mm`. Training and deployment must use the same units, coordinate conventions and joint order.

### Raw episode conversion

Prepare your own manifest, for example:

```json
[
  {"episode": "task_01/episode_000", "prompt": "Place the object on the plate."},
  {"episode": "task_01/episode_001", "prompt": "Place the object on the plate."}
]
```

Each relative path is resolved inside `--raw-root`:

```text
raw_data/task_01/episode_000/
├── mixed_data.json
├── head_left/00000.jpg, 00001.jpg, ...
├── wrist_left/00000.jpg, 00001.jpg, ...
└── wrist_right/00000.jpg, 00001.jpg, ...
```

`mixed_data.json` contains a `motion` list. Each element is:

```json
{
  "frame_id": 0,
  "arm_left": {"joint_positions": [0, 0, 0, 0, 0, 0]},
  "arm_right": {"joint_positions": [0, 0, 0, 0, 0, 0]},
  "gripper_left": {"range_mm": 20},
  "gripper_right": {"range_mm": 20}
}
```

Provide at least two consecutive, time-ordered frames per episode and the true recording FPS. Zero values above are a schema illustration, not robot calibration or a command.

```bash
uv run examples/piperx_real/convert_data.py \
  --manifest /path/to/episodes.json --raw-root /path/to/raw_data \
  --repo-id local/piperx --fps 30
```

The converter resizes RGB images to 256×256, stores observation `t` with absolute state `t+1` as its action, and drops the last observation (it has no next-state target). Model transforms subsequently resize to 224×224. It rejects duplicate/path-traversing episodes, missing data and non-finite state values. It never overwrites an existing dataset or uploads it. After an interrupted conversion, choose a fresh output directory or manually inspect/recover the incomplete dataset; implicit deletion/resume is deliberately absent.

If using `--output-dir`, ensure the result is still discoverable at `$HF_LEROBOT_HOME/<repo-id>` during normalization/training. Prefer the default output location.

## 3. Normalization must match the run

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_piperx --repo-id local/piperx \
  --batch-size 32 --num-workers 4
```

This finite CPU pass includes the final partial batch and writes:

- `assets/pi05_piperx/piperx/norm_stats.json`: state/action mean, std and approximate quantiles.
- `assets/pi05_piperx/piperx/norm_stats_info.json`: dataset/split/seed/action-convention provenance.

For this π₀.₅ recipe, training uses quantile normalization. Joint deltas are computed **before** action normalization; grippers stay absolute. Never compute statistics with the absolute config and then train with the delta config.

`--max-frames 100` is only for testing the data path, not model training or model comparison. Full statistics must be recomputed without that flag. Training checks the generated provenance sidecar and rejects limited-frame statistics or a mismatched dataset/action convention/split. Legacy assets without a sidecar require manual verification. Replacing an existing stats file requires `--overwrite` explicitly. Do not use raw training loss to compare models trained with different normalization conventions.

### Optional held-out validation

Default validation is disabled to preserve the full-data recipe. To enable it, use **identical settings in both commands**:

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_piperx --repo-id local/piperx \
  --val-episodes-frac 0.05 --seed 42 --overwrite

bash scripts/train_8gpu.sh --exp-name piperx_with_val \
  --data.repo-id local/piperx --val-episodes-frac 0.05 --seed 42 \
  --val-interval 1000 --val-batches 4
```

The split is a deterministic seeded split of **whole episodes**. Normalization uses only the training subset. Validation uses EMA parameters, the same validation batches and fixed sampling noise; its MSE is measured on the **14 real normalized action dimensions**, not the 32 padded model dimensions. By default only a bounded set of full batches is evaluated, not every held-out frame. The validation subset must contain at least one global batch.

This is an episode holdout, not a task-/object-/scene-disjoint benchmark. `val/action_mse` is a proxy for prediction quality, not a measured robot success rate.

## 4. Single-node 8-GPU launch

```bash
bash scripts/train_8gpu.sh --exp-name piperx_run --data.repo-id local/piperx
```

The launcher:

1. Uses the caller's `CUDA_VISIBLE_DEVICES`, or defaults to `0,1,2,3,4,5,6,7`.
2. Requires exactly eight visible GPUs and fails rather than falling back to CPU.
3. Runs **one** JAX process. The model/data mesh provides parallelism; do not launch eight Python copies.
4. Uses the locked environment; does not set proxies, cluster interfaces or rendezvous ports.

| Setting | Default |
| --- | --- |
| Model | π₀.₅, action dimension 32 (14 real + padding) |
| Action horizon | 10 |
| Discrete state input | `False`, preserving this PiperX variant |
| Action convention | Delta joints / absolute grippers |
| Global batch | 256; 32 examples per device |
| FSDP devices | 8 |
| Steps | 15000 |
| Learning rate | Warmup 1000 to 5e-5; remains 5e-5 |
| Optimizer | AdamW, gradient clip 1.0 |
| EMA | 0.999 |
| Save / retain | Every 1000; keep latest plus multiples of 3000 |
| W&B | Disabled; local metrics always enabled |

Approximate epochs = `steps × global_batch / usable_training_frames`; adapt the duration to your dataset. This starting recipe is not claimed to be optimal for every dataset.

Examples of explicit overrides:

```bash
# Smaller global batch if memory is insufficient; adjust duration if matching epochs.
bash scripts/train_8gpu.sh --exp-name smaller_batch \
  --data.repo-id local/piperx --batch-size 64

# Existing public base weights and a larger checkpoint disk.
bash scripts/train_8gpu.sh --exp-name local_weights \
  --data.repo-id local/piperx \
  --weight-loader.params-path /path/to/pi05_base/params \
  --checkpoint-base-dir /path/to/checkpoints

# Resume the latest retained checkpoint of the same run.
bash scripts/train_8gpu.sh --exp-name piperx_run --data.repo-id local/piperx --resume
```

A direct `uv run scripts/train.py pi05_piperx ...` call may use fewer GPUs with compatible `--fsdp-devices` and `--batch-size`. That is not the validated 8-device launcher path, and full-model memory requirements still apply. FSDP reduces parameter/optimizer memory at communication cost; it does not make arbitrary 8×GPU hardware sufficient.

`--overwrite` explicitly deletes the chosen experiment directory; it is **never added by the launcher**. Use a new experiment name unless you intentionally want that deletion.

## 5. Logs, recovery and reproducibility

`metrics.jsonl` contains completed optimizer steps and metrics, even with no tracking account. `run_info.json` records the dataset, split indices, seed and key model/data settings. Optional W&B can be enabled with `--wandb-enabled` after you configure your own account.

Resume restores model, optimizer and EMA state; it does **not** restore the precise data-loader iterator position. It is not guaranteed to be bitwise identical to an uninterrupted run. Keep data, transforms, normalization, seed and model architecture unchanged when resuming.

Checkpoint `params/` contains EMA inference weights when EMA is enabled. Keep `train_state/` for further training. Use `scripts/export_checkpoint.py` only for an inference-only distribution, not a resumable run.

## Troubleshooting

- **No GPUs / count mismatch:** run `uv run scripts/check_environment.py --expected-gpus 8`; check driver installation and GPU visibility. The host must expose GPUs to the container if applicable.
- **Dataset not found:** set the same `HF_LEROBOT_HOME` in every shell, and verify `local/piperx/meta/info.json` reports LeRobot v2.1.
- **Missing stats:** recompute for the exact config, dataset and action convention. The constant asset ID `piperx` does not mean statistics are interchangeable across datasets.
- **Out of memory:** reduce global batch to a multiple of eight, keep FSDP=8, and use appropriate GPUs. Changing EMA or action horizon changes the recipe and must be documented.
- **Small validation dataset:** reduce batch size or increase held-out data; do not duplicate validation episodes silently to fabricate a benchmark.
- **Offline environment:** cache the locked packages, dataset, base weights and `gs://big_vision/paligemma_tokenizer.model` in advance. `scripts/prefetch_assets.py` uses public anonymous storage and prints the cache locations; no private mirror is configured.
