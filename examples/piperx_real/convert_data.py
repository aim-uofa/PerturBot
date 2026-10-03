"""Convert a portable episode manifest to the pinned LeRobot v2.1 image format."""

import itertools
import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro

CAMERAS = {
    "image": "head_left",
    "wrist_left_image": "wrist_left",
    "wrist_right_image": "wrist_right",
}


def parse_state(item: dict) -> np.ndarray:
    """Preserve source joint units and gripper range_mm; do not reorder grippers."""
    left = item["arm_left"]["joint_positions"]
    right = item["arm_right"]["joint_positions"]
    if len(left) != 6 or len(right) != 6:
        raise ValueError("Each arm must contain exactly six joint positions.")
    state = np.asarray(
        [*left, *right, item["gripper_left"]["range_mm"], item["gripper_right"]["range_mm"]], dtype=np.float32
    )
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("State must contain 14 finite numbers.")
    return state


def load_manifest(manifest: Path, raw_root: Path) -> list[tuple[Path, str]]:
    raw_root = raw_root.resolve()
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty list of {episode, prompt} objects.")
    episodes = []
    seen = set()
    for entry in entries:
        relative = Path(entry["episode"])
        path = (raw_root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(raw_root):
            raise ValueError("Episode paths must stay inside --raw-root and be relative.")
        if path in seen:
            raise ValueError(f"Duplicate episode: {relative}")
        prompt = entry["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt for {relative}")
        if not (path / "mixed_data.json").is_file():
            raise FileNotFoundError(path / "mixed_data.json")
        seen.add(path)
        episodes.append((path, prompt.strip()))
    return episodes


def read_episode(episode: Path) -> tuple[list[int], list[np.ndarray]]:
    motion = json.loads((episode / "mixed_data.json").read_text())["motion"]
    if not isinstance(motion, list) or len(motion) < 2:
        raise ValueError(f"Episode needs at least two motion frames: {episode}")
    frame_ids = [item["frame_id"] for item in motion]
    if any(not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 for frame in frame_ids):
        raise ValueError("frame_id must be a nonnegative integer.")
    if any(b != a + 1 for a, b in itertools.pairwise(frame_ids)):
        raise ValueError(f"Frames must be consecutive and ordered for the declared FPS: {episode}")
    return frame_ids, [parse_state(item) for item in motion]


def main(
    manifest: Path,
    raw_root: Path,
    repo_id: str = "local/piperx",
    output_dir: Path | None = None,
    fps: int = 30,
    image_size: int = 256,
    image_writer_threads: int = 4,
):
    """Write a new local dataset; never overwrite, delete, resume, or upload data implicitly.

    Default output: $HF_LEROBOT_HOME/<repo-id>. Manifest paths are relative to
    --raw-root; prompt strings are explicit, with no private task lookup tables.
    """
    if fps <= 0 or image_size <= 0 or image_writer_threads < 0:
        raise ValueError("fps and image_size must be positive; image_writer_threads must be nonnegative.")
    parts = repo_id.split("/")
    if len(parts) != 2 or any(part in ("", ".", "..") or "\\" in part for part in parts):
        raise ValueError("repo_id must be namespace/dataset, e.g. local/piperx.")
    episodes = load_manifest(manifest, raw_root)
    output_dir = output_dir or HF_LEROBOT_HOME / repo_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; choose a fresh output directory.")
    features = {
        key: {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]}
        for key in CAMERAS
    }
    features.update(
        {
            "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (14,), "names": ["action"]},
        }
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="piperx",
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_threads=image_writer_threads,
        image_writer_processes=0,
    )
    try:
        for episode, prompt in episodes:
            frame_ids, states = read_episode(episode)
            # Match the source pipeline: observation at t -> absolute state at t+1.
            # The last frame is a target only, not a padded training observation.
            for index, frame_id in enumerate(frame_ids[:-1]):
                frame = {"state": states[index], "action": states[index + 1], "task": prompt}
                for key, folder in CAMERAS.items():
                    with Image.open(episode / folder / f"{frame_id:05d}.jpg") as image:
                        resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                        frame[key] = np.asarray(resized, dtype=np.uint8)
                dataset.add_frame(frame)
            dataset.save_episode()
            print(f"Converted {episode.name}: {len(states) - 1} frames")
    finally:
        dataset.stop_image_writer()
    print(f"Saved {len(episodes)} episodes to {output_dir}. Nothing was uploaded.")


if __name__ == "__main__":
    tyro.cli(main)
