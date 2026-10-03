"""Shared, portable loading utilities for the PiperX inference examples."""

import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.policies import piperx_real_policy
from openpi.policies import policy_config
from openpi.training import config as _config


def resolve_config(config_name: str, checkpoint: str, asset_id: str | None = None) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    if not isinstance(config.data, _config.LeRobotPiperXRealDataConfig):
        raise ValueError("Choose a PiperX config: pi05_piperx or pi05_piperx_absolute.")
    root = Path(checkpoint).expanduser()
    if root.is_dir():
        metadata_path = root / "release_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            expected = {
                "pi05": config.model.pi05,
                "action_horizon": config.model.action_horizon,
                "discrete_state_input": config.model.discrete_state_input,
                "extra_delta_transform": config.data.extra_delta_transform,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Checkpoint {key}={metadata.get(key)!r} does not match config {value!r}.")
        # A legacy checkpoint may use a different asset id. Only infer it if unambiguous.
        if asset_id is None:
            candidates = sorted((root / "assets").glob("**/norm_stats.json"))
            if len(candidates) == 1:
                asset_id = candidates[0].parent.relative_to(root / "assets").as_posix()
            elif len(candidates) > 1:
                raise ValueError("Multiple normalization assets found; specify --asset-id explicitly.")
    if asset_id is not None:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, assets=dataclasses.replace(config.data.assets, asset_id=asset_id)),
        )
    return config


def load_policy(checkpoint: str, config_name: str = "pi05_piperx", asset_id: str | None = None, num_steps: int = 10):
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    config = resolve_config(config_name, checkpoint, asset_id)
    return policy_config.create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": num_steps})


def load_observation(path: Path | None, prompt: str) -> dict:
    """NPZ arrays: image, wrist_left_image, wrist_right_image, state. Never load pickle."""
    if path is None:
        observation = piperx_real_policy.make_piperx_real_example()
        observation.pop("actions", None)
    else:
        with np.load(path, allow_pickle=False) as data:
            observation = {
                f"observation/{key}": data[key] for key in ("image", "wrist_left_image", "wrist_right_image", "state")
            }
    observation["prompt"] = prompt
    return observation
