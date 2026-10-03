# Dataset normalization assets

No private normalization statistics are bundled. Compute your own:

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py --config-name pi05_piperx --repo-id local/piperx
```

Output: `assets/pi05_piperx/piperx/norm_stats.json` and `norm_stats_info.json`.
Both the dataset identity and the absolute/delta convention matter. If using an episode holdout, compute statistics on the training split with the same fraction and seed used by training.

Generated assets are ignored by Git. Inference loads the matching statistics from the checkpoint, not from this workspace.
