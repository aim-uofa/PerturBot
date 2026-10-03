"""Minimal remote client. Synthetic inputs only; this script never drives a robot."""

import argparse

import numpy as np
from openpi_client import websocket_client_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Place the object on the plate.")
    args = parser.parse_args()
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_left_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_right_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(14, dtype=np.float32),
        "prompt": args.prompt,
    }
    actions = client.infer(obs)["actions"]
    print(f"Received actions: {actions.shape}. Synthetic smoke test only; do not execute these actions.")


if __name__ == "__main__":
    main()
