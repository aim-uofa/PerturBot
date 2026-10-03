"""Exercise conversion -> real LeRobot loading -> full normalization, using synthetic data only."""

import json

from lerobot.common.datasets import lerobot_dataset
import numpy as np
from PIL import Image

from examples.piperx_real import convert_data
from openpi import transforms
from openpi.shared import normalize
from openpi.training import config
from scripts import compute_norm_stats


def test_lerobot_conversion_and_normalization(tmp_path, monkeypatch):
    data_home = tmp_path / "datasets"
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", data_home)
    monkeypatch.setattr(convert_data, "HF_LEROBOT_HOME", data_home)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Normalization only uses data transforms; no tokenizer/model download in tests.
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    raw = tmp_path / "raw"
    entries = []
    for episode_index in range(2):
        episode = raw / f"episode_{episode_index}"
        episode.mkdir(parents=True)
        motion = []
        for index in range(4):
            motion.append(
                {
                    "frame_id": index,
                    "arm_left": {"joint_positions": [index + episode_index] * 6},
                    "arm_right": {"joint_positions": [index + episode_index + 1] * 6},
                    "gripper_left": {"range_mm": 10 + index},
                    "gripper_right": {"range_mm": 20 + index},
                }
            )
            for camera in convert_data.CAMERAS.values():
                directory = episode / camera
                directory.mkdir(exist_ok=True)
                Image.fromarray(np.full((16, 16, 3), index * 30, dtype=np.uint8)).save(directory / f"{index:05d}.jpg")
        (episode / "mixed_data.json").write_text(json.dumps({"motion": motion}))
        entries.append({"episode": episode.name, "prompt": "Pick up the object."})
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps(entries))
    convert_data.main(manifest, raw, repo_id="local/synthetic", image_size=16, image_writer_threads=0)
    metadata = lerobot_dataset.LeRobotDatasetMetadata("local/synthetic")
    assert metadata.total_episodes == 2
    assert metadata.total_frames == 6
    compute_norm_stats.main(repo_id="local/synthetic", assets_base_dir=tmp_path / "assets", batch_size=4, num_workers=0)
    output = tmp_path / "assets" / "pi05_piperx" / "piperx"
    stats = normalize.load(output)
    assert stats["state"].mean.shape == (14,)
    assert stats["actions"].mean.shape == (14,)
    assert np.isfinite(stats["actions"].q01).all()
    assert json.loads((output / "norm_stats_info.json").read_text())["num_frames"] == 6
