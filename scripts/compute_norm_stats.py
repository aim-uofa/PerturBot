"""Compute state/action statistics on CPU, including the final incomplete batch."""

import dataclasses
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class SelectStatsFields(transforms.DataTransformFn):
    def __call__(self, sample: dict) -> dict:
        return {key: np.asarray(sample[key]) for key in ("state", "actions")}


def collate_stats(items: list[dict]) -> dict:
    return {key: np.stack([item[key] for item in items]) for key in ("state", "actions")}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config,
    num_workers: int,
    max_frames: int | None = None,
    *,
    episodes: list[int] | None = None,
) -> tuple[torch.utils.data.DataLoader, int]:
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config, episodes=episodes)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, SelectStatsFields()],
    )
    if max_frames is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_frames, len(dataset))))
    if len(dataset) < 2:
        raise ValueError("At least two training frames are required to compute statistics.")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing.get_context("spawn") if num_workers else None,
        collate_fn=collate_stats,
    )
    return loader, len(dataset)


def main(
    config_name: str = "pi05_piperx",
    repo_id: str | None = None,
    assets_base_dir: Path = Path("assets"),
    asset_id: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: int | None = None,
    val_episodes_frac: float | None = None,
    seed: int | None = None,
    *,
    overwrite: bool = False,
):
    """Use the same config, split fraction and seed as training.

    --max-frames is a prefix-only smoke test, NOT release-quality statistics.
    This command does not require a GPU or download base model weights.
    """
    if batch_size <= 0 or num_workers < 0 or (max_frames is not None and max_frames < 2):
        raise ValueError("Require batch_size > 0, num_workers >= 0, and max_frames >= 2 (or omit it).")
    config = _config.get_config(config_name)
    assets = dataclasses.replace(config.data.assets, assets_dir=None, asset_id=asset_id or config.data.assets.asset_id)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, repo_id=repo_id or config.data.repo_id, assets=assets),
        assets_base_dir=str(assets_base_dir),
        val_episodes_frac=config.val_episodes_frac if val_episodes_frac is None else val_episodes_frac,
        seed=config.seed if seed is None else seed,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise ValueError("This release's normalization entry point supports LeRobot datasets, not RLDS.")
    if not data_config.asset_id:
        raise ValueError("An asset_id or repo_id is required.")
    output_dir = config.assets_dirs / data_config.asset_id
    if (output_dir / "norm_stats.json").exists() and not overwrite:
        raise FileExistsError(f"{output_dir / 'norm_stats.json'} exists; use --overwrite to replace it.")
    train_episodes, _ = _data_loader.compute_train_val_episodes(
        data_config.repo_id, config.val_episodes_frac, seed=config.seed
    )
    loader, num_frames = create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        batch_size,
        config.model,
        num_workers,
        max_frames,
        episodes=train_episodes,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, desc="Computing training-set statistics"):
        for key, running_stats in stats.items():
            running_stats.update(batch[key])
    normalize.save(output_dir, {key: stat.get_statistics() for key, stat in stats.items()})
    info = {
        "config_name": config_name,
        "repo_id": data_config.repo_id,
        "asset_id": data_config.asset_id,
        "num_frames": num_frames,
        "max_frames": max_frames,
        "val_episodes_frac": config.val_episodes_frac,
        "seed": config.seed,
        "action_horizon": config.model.action_horizon,
        "extra_delta_transform": getattr(config.data, "extra_delta_transform", None),
    }
    (output_dir / "norm_stats_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote statistics for {num_frames} frames to {output_dir}")
    if max_frames is not None:
        print("WARNING: limited-frame statistics are for smoke tests. Recompute without --max-frames for training.")


if __name__ == "__main__":
    tyro.cli(main)
