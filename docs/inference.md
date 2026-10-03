# Inference and the action contract

## Model loading

Use a step directory containing both `params/` and `assets/`. Passing only `params/` omits the normalization assets and is incorrect. `params/` already contains the inference EMA parameters saved by training.

```python
from openpi.policies import piperx_inference

policy = piperx_inference.load_policy(
    "checkpoints/pi05_piperx/piperx_run/15000",
    config_name="pi05_piperx",
    num_steps=10,
)
actions = policy.infer(observation)["actions"]
```

This local API needs no server, service registration, IP address or port. Use a suitable NVIDIA GPU for full π₀.₅ inference (the upstream estimate is more than 8 GB; actual usage depends on precision/workload). First use compiles the JAX graph and is slower than warmed-up calls.

## Observation dictionary

```python
import numpy as np

observation = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_left_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/wrist_right_image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(14, dtype=np.float32),
    "prompt": "Place the object on the plate.",
}
```

Replace these zero arrays with **synchronized real observations**. The example is not valid robot calibration. Images are **RGB**, not OpenCV BGR; convert BGR explicitly. The adapter accepts uint8 HWC, or finite float images in `[0,1]` in HWC/CHW form. It rejects invalid shapes, NaNs and out-of-range float images.

State/action ordering:

```text
0:6    left-arm joint positions
6:12   right-arm joint positions
12     left gripper range_mm
13     right gripper range_mm
```

Do not use the common but different `[left 6 + left gripper, right 6 + right gripper]` ordering. Preserve the units and calibration of the training demonstrations.

To use the CLI with recorded inputs:

```python
np.savez(
    "observation.npz",
    image=observation["observation/image"],
    wrist_left_image=observation["observation/wrist_left_image"],
    wrist_right_image=observation["observation/wrist_right_image"],
    state=observation["observation/state"],
)
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run examples/piperx_real/infer.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --observation observation.npz \
  --prompt 'Place the object on the plate.' --output actions.npy
```

The NPZ loader uses `allow_pickle=False`. Output files are not overwritten implicitly.

## Delta versus absolute is not optional guesswork

For `pi05_piperx`, training targets are:

```text
action_joint[t+k] - current_state_joint[t]   for dimensions 0:12
absolute gripper target[t+k]                for dimensions 12:14
```

The **current observation's state**, not each future state's predecessor, is the delta reference for the entire predicted chunk. Inference unnormalizes and adds the current joints back. The returned `(10,14)` actions are therefore **absolute targets**. Do not add the state again in your controller.

For `pi05_piperx_absolute`, the model directly predicts absolute targets. Use this only for checkpoints trained with the corresponding absolute-action transforms. Loading an absolute checkpoint with the delta config adds an erroneous state offset; changing normalization cannot fix that.

Other compatibility requirements: `action_horizon=10`, model action dimension 32, and **`discrete_state_input=False`** for this PiperX recipe. Although the π₀.₅ backbone supports other settings, deployment must match training.

Legacy step directories may have a different asset ID. If there is exactly one `assets/**/norm_stats.json`, the loading helper detects its relative ID. If there are several, specify `--asset-id`. An exported checkpoint includes `release_metadata.json`; incompatible model/action flags are rejected. For older checkpoints without metadata, the operator must verify the recorded training settings.

## Optional WebSocket service

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --host 127.0.0.1 --port 8000
```

`--host` and `--port` are ordinary user-controlled network settings; no internal IP discovery or cluster environment variables are consulted. Health endpoint: `http://127.0.0.1:8000/healthz`.

The lightweight client package can be installed separately on another machine with Python 3.9+ (Python 3.11 recommended):

```bash
pip install ./packages/openpi-client
python examples/piperx_real/client.py --host 127.0.0.1 --port 8000
```

The example client sends **synthetic** inputs and prints action shapes only. It does not drive hardware. To serve a remote machine, bind to a trusted interface explicitly, configure your firewall/tunnel, and point the client to that address. The protocol has **no authentication/TLS**; never expose it directly to the public Internet.

`--record-dir` enables local policy recording for debugging. Recordings may contain sensitive images and instructions and must not be committed or published without review.

## Robot safety

This release is an inference interface, **not** a production robot controller. A deployment integration must independently enforce joint/gripper limits, calibration, control frequency, synchronization, watchdogs, collision handling, an emergency stop and human supervision. Validate outputs offline first and use a safeguarded workspace before any physical tests. Neither synthetic inference nor low training loss is a real-robot validation result.
