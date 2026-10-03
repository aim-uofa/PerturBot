"""Check locally generated normalization provenance before creating a training run."""

import json
import logging
from pathlib import Path


def validate_norm_stats(config) -> None:
    if config.data.repo_id == "fake":
        return
    assets_dir = str(config.data.assets.assets_dir or config.assets_dirs)
    if "://" in assets_dir:
        return  # Remote upstream assets may not have this release's provenance sidecar.
    asset_id = config.data.assets.asset_id or config.data.repo_id
    if not asset_id:
        return
    info_path = Path(assets_dir) / asset_id / "norm_stats_info.json"
    if not info_path.exists():
        logging.warning("No normalization provenance sidecar found; verify dataset and action convention manually.")
        return
    info = json.loads(info_path.read_text())
    expected = {
        "repo_id": config.data.repo_id,
        "action_horizon": config.model.action_horizon,
        "extra_delta_transform": getattr(config.data, "extra_delta_transform", None),
        "val_episodes_frac": config.val_episodes_frac,
    }
    if config.val_episodes_frac > 0:
        expected["seed"] = config.seed
    for key, value in expected.items():
        if info.get(key) != value:
            raise ValueError(
                f"Normalization provenance mismatch for {key}: stats={info.get(key)!r}, training={value!r}. "
                "Recompute statistics with the same dataset, action convention and split settings."
            )
    if info.get("max_frames") is not None:
        raise ValueError(
            "Limited-frame statistics are for smoke tests. Recompute without --max-frames before training."
        )
