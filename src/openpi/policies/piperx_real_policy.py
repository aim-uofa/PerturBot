"""PiperX layout: [left joints (6), right joints (6), left gripper, right gripper]."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piperx_real_example() -> dict:
    """Creates a random input example matching the repacked keys."""
    return {
        "observation/state": np.random.rand(14).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "actions": np.random.rand(14).astype(np.float32),
        "prompt": "Place the eggplant on a plate.",
    }


def _parse_image(image) -> np.ndarray:
    """Accept RGB uint8 HWC or float [0, 1] CHW/HWC (e.g. LeRobot tensors)."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating-point images must be finite and in [0, 1].")
        image = (255 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("Images must be uint8 or floating point in [0, 1].")
    return image


@dataclasses.dataclass(frozen=True)
class PiperXRealInputs(transforms.DataTransformFn):
    """Map repacked PiperX sample -> OpenPi model inputs (train + infer)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist = _parse_image(data["observation/wrist_left_image"])
        right_wrist = _parse_image(data["observation/wrist_right_image"])

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("PiperX state must have shape (14,) and contain only finite values.")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Present during training; may be absent during pure inference.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperXRealOutputs(transforms.DataTransformFn):
    """Map model outputs -> PiperX action format (inference only)."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < self.action_dim or not np.isfinite(actions).all():
            raise ValueError("Policy returned invalid action shape or non-finite values.")
        return {"actions": actions[:, : self.action_dim]}
