"""Run inference locally; no server, network port, or robot controller required."""

from pathlib import Path

import numpy as np
import tyro

from openpi.policies import piperx_inference


def main(
    checkpoint: str,
    observation: Path | None = None,
    prompt: str = "Place the object on the plate.",
    config: str = "pi05_piperx",
    asset_id: str | None = None,
    num_steps: int = 10,
    output: Path | None = None,
    *,
    dummy: bool = False,
):
    """Provide --observation input.npz or explicitly opt into --dummy (never control hardware)."""
    if dummy == (observation is not None):
        raise ValueError("Provide exactly one of --observation or --dummy.")
    if output is not None and output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    obs = piperx_inference.load_observation(observation, prompt)
    policy = piperx_inference.load_policy(checkpoint, config, asset_id, num_steps)
    result = policy.infer(obs)
    actions = np.asarray(result["actions"])
    if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
        raise ValueError(f"Invalid policy output shape or values: {actions.shape}")
    print(f"actions: shape={actions.shape}, dtype={actions.dtype}")
    print("Outputs are absolute joint targets and gripper ranges, not motor commands.")
    if dummy:
        print("Synthetic observations: this is an API smoke test, not a robot evaluation.")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
        print(f"Saved {output}")


if __name__ == "__main__":
    tyro.cli(main)
