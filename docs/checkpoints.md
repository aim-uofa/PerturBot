# Checkpoint selection and distribution

## Do not equate “latest”, “lowest training loss” and “best”

A valid best-checkpoint claim requires the same task set, evaluation split, number of trials, action convention, normalization, sampling settings and evaluation metric. Prefer held-out **real-robot success rate**, with per-task trial counts and uncertainty. A training loss from a different dataset or normalization scale is not comparable.

This source release includes no private historical evaluation report or fine-tuned checkpoint. It makes no claim that a particular historical run is optimal. Checkpoint existence, a deployment-script reference, and an optimizer finishing its run are evidence of different things, not a substitute for measured success.

## Local validation-based selection

If validation is enabled, local `metrics.jsonl` records `val/action_mse`. To select the lowest-MSE checkpoint **still on disk**:

```bash
uv run python scripts/select_checkpoint.py \
  --run-dir checkpoints/pi05_piperx/piperx_with_val
```

The command also reports the best observed step and whether that checkpoint was retained. There is no fallback to training loss when validation metrics are missing. Validation MSE remains only a proxy; evaluate candidates on the actual deployment benchmark.

Default retention keeps the latest checkpoint plus multiples of 3000. An intermediate low-MSE checkpoint can be deleted. If every evaluated checkpoint must be retained, use matching save/validation intervals and, for example, `--keep-period 1000`; budget disk accordingly. Align saved and scored steps for meaningful selection.

New checkpoints use completed-update numbering. A historical `14999` directory may mean the result after 15000 updates. Compare recorded training state/metrics rather than changing a directory name to make it appear equivalent.

## Export only inference artifacts

First review the original run configuration. Choose `pi05_piperx` for delta joints or `pi05_piperx_absolute` for absolute actions. In particular, do not rely on a current source file that may have changed since the old run.

```bash
# Read-only plan: reports expected bytes and the selected normalization asset.
uv run python scripts/export_checkpoint.py \
  --source /path/to/run/15000 \
  --destination /path/to/distribution/piperx_checkpoint \
  --config pi05_piperx --dry-run

# After confirming model/data rights, copy to a fresh destination (no upload).
uv run python scripts/export_checkpoint.py \
  --source /path/to/run/15000 \
  --destination /path/to/distribution/piperx_checkpoint \
  --config pi05_piperx
```

The export contains:

```text
piperx_checkpoint/
├── params/                    # complete Orbax parameter directory
├── assets/piperx/norm_stats.json
├── release_metadata.json      # portable model/action contract, no original paths
└── README.md
```

The exporter:

- Retains the entire Orbax parameter tree, not selected shard files.
- Validates that a single normalization asset exists, or accepts `--asset-id` explicitly.
- Renames the asset namespace to the portable `piperx` ID.
- Excludes optimizer state, raw data, run logs, host metadata and original dataset/owner paths from its generated metadata.
- Refuses to overwrite a destination or export symlinked parameter trees.
- Copies locally; it does not upload or publish anything.

The exporter cannot determine the old action convention from tensors alone. Its `--config` declaration is supplied by you and must be checked against the original training record. Its default horizon is 10; non-default model variants require corresponding inference configuration changes, not just relabeling metadata. Exported weights are **not** resumable training checkpoints.

Before publishing any weights, verify checkpoint integrity with a restore/inference test, review the dataset and model licenses, document the evaluation protocol and results, and add a model card with limitations and safety information.
