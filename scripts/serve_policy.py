"""Optional WebSocket inference. Local Python inference requires no port."""

import dataclasses
import logging
from pathlib import Path

import tyro

from openpi.policies import piperx_inference
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    # Step directory containing params/ and assets/, not the params directory itself.
    checkpoint: str
    config: str = "pi05_piperx"
    asset_id: str | None = None
    num_steps: int = 10
    # Loopback by default. Set explicitly to serve on a trusted network.
    host: str = "127.0.0.1"
    port: int = 8000
    record_dir: Path | None = None


def main(args: Args):
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535].")
    policy = piperx_inference.load_policy(args.checkpoint, args.config, args.asset_id, args.num_steps)
    metadata = policy.metadata
    if args.record_dir is not None:
        policy = _policy.PolicyRecorder(policy, str(args.record_dir))
    logging.info("Serving on %s:%d (no authentication; trusted networks only)", args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
