"""Fast accelerator preflight; never allocates a model or downloads weights."""

import sys

import jax
import tyro


def main(expected_gpus: int = 8):
    if expected_gpus < 0:
        raise ValueError("expected_gpus must be nonnegative.")
    devices = jax.devices()
    gpus = [device for device in devices if device.platform == "gpu"]
    print(f"Python {sys.version.split()[0]}; JAX {jax.__version__}; {len(gpus)} visible GPUs")
    if len(gpus) != expected_gpus:
        raise RuntimeError(
            f"Expected {expected_gpus} GPUs, found {len(gpus)}. Check the NVIDIA driver, CUDA-enabled JAX, "
            "and CUDA_VISIBLE_DEVICES. The 8-GPU launcher does not silently fall back to CPU."
        )
    for device in gpus:
        print(f"  device {device.id}: {device.device_kind}")


if __name__ == "__main__":
    tyro.cli(main)
