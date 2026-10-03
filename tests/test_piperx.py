import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.piperx_real import convert_data
from openpi import transforms
from openpi.models import model
from openpi.policies import piperx_inference
from openpi.policies import piperx_real_policy
from openpi.training import config
from openpi.training import data_loader
from openpi.training import provenance
from scripts import compute_norm_stats


def test_public_recipe():
    recipe = config.get_config("pi05_piperx")
    assert recipe.model.pi05
    assert recipe.model.action_horizon == 10
    assert recipe.model.discrete_state_input is False
    assert recipe.batch_size == 256
    assert recipe.fsdp_devices == 8
    assert recipe.wandb_enabled is False
    assert recipe.data.extra_delta_transform is True
    assert recipe.data.assets.asset_id == "piperx"
    assert recipe.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert config.get_config("pi05_piperx_absolute").data.extra_delta_transform is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"val_episodes_frac": -0.1},
        {"val_episodes_frac": 1.0},
        {"val_action_dim": 33},
        {"fsdp_devices": 0},
        {"num_workers": -1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError, match="must|positive|nonnegative"):
        dataclasses.replace(config.get_config("pi05_piperx"), **kwargs)


def test_delta_round_trip():
    state = np.arange(14, dtype=np.float32)
    actions = np.stack([state + 0.25, state + 0.5])
    mask = transforms.make_bool_mask(12, -2)
    delta = transforms.DeltaActions(mask)({"state": state.copy(), "actions": actions.copy()})
    np.testing.assert_allclose(delta["actions"][:, :12], [[0.25] * 12, [0.5] * 12])
    np.testing.assert_allclose(delta["actions"][:, 12:], actions[:, 12:])
    restored = transforms.AbsoluteActions(mask)(delta)
    np.testing.assert_allclose(restored["actions"], actions)


def test_observation_and_output_schema():
    example = piperx_real_policy.make_piperx_real_example()
    example["observation/wrist_left_image"] = np.zeros((3, 16, 20), dtype=np.float32)
    inputs = piperx_real_policy.PiperXRealInputs(model.ModelType.PI05)(example)
    assert inputs["state"].shape == (14,)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (16, 20, 3)
    assert inputs["image"]["left_wrist_0_rgb"].dtype == np.uint8
    actions = piperx_real_policy.PiperXRealOutputs()({"actions": np.zeros((10, 32))})["actions"]
    assert actions.shape == (10, 14)
    example["observation/state"] = np.full(14, np.nan)
    with pytest.raises(ValueError, match="finite"):
        piperx_real_policy.PiperXRealInputs(model.ModelType.PI05)(example)


@pytest.mark.parametrize("image", [np.zeros((10, 10)), np.ones((10, 10, 3)) * 255, np.zeros((10, 10, 4))])
def test_invalid_image(image):
    with pytest.raises(ValueError, match="Expected|Floating"):
        piperx_real_policy._parse_image(image)  # noqa: SLF001


def test_episode_split(monkeypatch):
    monkeypatch.setattr(
        data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda _: SimpleNamespace(total_episodes=20)
    )
    train, val = data_loader.compute_train_val_episodes("local/demo", 0.2, seed=42)
    assert len(train) == 16
    assert len(val) == 4
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(20))
    assert (train, val) == data_loader.compute_train_val_episodes("local/demo", 0.2, seed=42)
    assert (train, val) != data_loader.compute_train_val_episodes("local/demo", 0.2, seed=7)
    assert data_loader.compute_train_val_episodes("local/demo", 0) == (None, None)
    with pytest.raises(ValueError, match="val_frac"):
        data_loader.compute_train_val_episodes("local/demo", 1)


def test_norm_stats_keeps_tail_and_episode_filter(monkeypatch):
    dataset = [{"state": np.full(14, i), "actions": np.full((10, 14), i)} for i in range(5)]
    seen = []

    def make_dataset(*args, episodes=None):
        seen.append(episodes)
        return dataset

    monkeypatch.setattr(data_loader, "create_torch_dataset", make_dataset)
    loader, count = compute_norm_stats.create_torch_dataloader(
        config.DataConfig(repo_id="local/demo"), 10, 4, None, 0, episodes=[2, 3]
    )
    batches = list(loader)
    assert count == 5
    assert seen == [[2, 3]]
    assert [batch["state"].shape[0] for batch in batches] == [4, 1]
    np.testing.assert_array_equal(batches[-1]["state"], np.full((1, 14), 4))


def test_norm_stats_small_subset(monkeypatch):
    dataset = [{"state": np.full(14, i), "actions": np.full((10, 14), i)} for i in range(5)]
    monkeypatch.setattr(data_loader, "create_torch_dataset", lambda *args, **kwargs: dataset)
    loader, count = compute_norm_stats.create_torch_dataloader(
        config.DataConfig(repo_id="local/demo"), 10, 8, None, 0, 3
    )
    assert count == 3
    assert sum(batch["state"].shape[0] for batch in loader) == 3


def test_raw_state_order():
    state = convert_data.parse_state(
        {
            "arm_left": {"joint_positions": [1, 2, 3, 4, 5, 6]},
            "arm_right": {"joint_positions": [7, 8, 9, 10, 11, 12]},
            "gripper_left": {"range_mm": 13},
            "gripper_right": {"range_mm": 14},
        }
    )
    np.testing.assert_array_equal(state, np.arange(1, 15, dtype=np.float32))


def test_manifest_and_duplicate_protection(tmp_path):
    episode = tmp_path / "episode_000"
    episode.mkdir()
    (episode / "mixed_data.json").write_text('{"motion": []}')
    manifest = tmp_path / "manifest.json"
    entry = {"episode": "episode_000", "prompt": "pick up the object"}
    manifest.write_text(json.dumps([entry]))
    assert convert_data.load_manifest(manifest, tmp_path) == [(episode, entry["prompt"])]
    manifest.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="Duplicate"):
        convert_data.load_manifest(manifest, tmp_path)
    manifest.write_text(json.dumps([{**entry, "episode": "../outside"}]))
    with pytest.raises(ValueError, match="relative"):
        convert_data.load_manifest(manifest, tmp_path)


def test_legacy_asset_detection_and_contract(tmp_path):
    assets = tmp_path / "assets" / "legacy" / "dataset"
    assets.mkdir(parents=True)
    (assets / "norm_stats.json").write_text("{}")
    resolved = piperx_inference.resolve_config("pi05_piperx", str(tmp_path))
    assert resolved.data.assets.asset_id == "legacy/dataset"
    (tmp_path / "release_metadata.json").write_text(
        json.dumps(
            {
                "pi05": True,
                "action_horizon": 10,
                "discrete_state_input": False,
                "extra_delta_transform": False,
            }
        )
    )
    with pytest.raises(ValueError, match="extra_delta_transform"):
        piperx_inference.resolve_config("pi05_piperx", str(tmp_path))
    piperx_inference.resolve_config("pi05_piperx_absolute", str(tmp_path))


def test_norm_provenance_rejects_mismatches(tmp_path):
    recipe = dataclasses.replace(config.get_config("pi05_piperx"), assets_base_dir=str(tmp_path))
    output = recipe.assets_dirs / "piperx"
    output.mkdir(parents=True)
    info = {
        "repo_id": "local/piperx",
        "action_horizon": 10,
        "extra_delta_transform": True,
        "val_episodes_frac": 0.0,
        "max_frames": None,
    }
    path = output / "norm_stats_info.json"
    path.write_text(json.dumps(info))
    provenance.validate_norm_stats(recipe)
    with pytest.raises(ValueError, match="val_episodes_frac"):
        provenance.validate_norm_stats(dataclasses.replace(recipe, val_episodes_frac=0.05))
    path.write_text(json.dumps({**info, "extra_delta_transform": False}))
    with pytest.raises(ValueError, match="extra_delta_transform"):
        provenance.validate_norm_stats(recipe)
    path.write_text(json.dumps({**info, "max_frames": 100}))
    with pytest.raises(ValueError, match="Limited-frame"):
        provenance.validate_norm_stats(recipe)
