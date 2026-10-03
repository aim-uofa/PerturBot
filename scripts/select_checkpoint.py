"""Rank saved checkpoints by a held-out metric, never by training loss by default."""

import argparse
import json
import math
from pathlib import Path


def select_checkpoint(run_dir: Path, metric: str = "val/action_mse", mode: str = "min") -> dict:
    if mode not in ("min", "max"):
        raise ValueError("mode must be min or max.")
    metrics_path = run_dir / "metrics.jsonl"
    observations = {}
    for line in metrics_path.read_text().splitlines():
        record = json.loads(line)
        value = record.get(metric)
        if isinstance(value, int | float) and math.isfinite(value):
            observations[int(record["step"])] = float(value)
    if not observations:
        raise ValueError(f"No finite {metric!r} values found. Training loss is not a success-rate evaluation.")
    available = {step: value for step, value in observations.items() if (run_dir / str(step) / "params").is_dir()}
    if not available:
        raise ValueError("No scored checkpoint remains on disk. Check the checkpoint retention policy.")
    rank = min if mode == "min" else max
    step = rank(available, key=available.get)
    observed_step = rank(observations, key=observations.get)
    return {
        "checkpoint": str(run_dir / str(step)),
        "step": step,
        "metric": metric,
        "mode": mode,
        "value": available[step],
        "best_observed_step": observed_step,
        "best_observed_value": observations[observed_step],
        "best_observed_still_available": observed_step in available,
        "warning": "A held-out action metric is a proxy, not a real-robot success rate. Compare only the same protocol.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metric", default="val/action_mse")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    args = parser.parse_args()
    print(json.dumps(select_checkpoint(args.run_dir, args.metric, args.mode), indent=2))


if __name__ == "__main__":
    main()
