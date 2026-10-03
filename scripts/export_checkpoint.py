"""Export inference weights + one normalization asset, without optimizer or run logs.

This is a local copy only. Verify permission to distribute model weights and data
statistics before publishing the resulting directory. No upload is performed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


def plan_export(source: Path, asset_id: str | None = None) -> tuple[Path, dict]:
    source = source.resolve()
    if not (source / "params" / "_METADATA").is_file() or not (source / "params" / "manifest.ocdbt").is_file():
        raise ValueError("Expected a completed Orbax step directory with params/_METADATA and params/manifest.ocdbt.")
    if asset_id is None:
        candidates = sorted((source / "assets").glob("**/norm_stats.json"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one norm_stats.json, or specify --asset-id.")
        stats_path = candidates[0]
    else:
        stats_path = source / "assets" / asset_id / "norm_stats.json"
    if not stats_path.resolve().is_relative_to(source / "assets"):
        raise ValueError("The normalization asset must stay inside the source assets directory.")
    stats = json.loads(stats_path.read_text())["norm_stats"]
    clean_stats = {}
    for key in ("state", "actions"):
        clean_stats[key] = {}
        for field in ("mean", "std", "q01", "q99"):
            values = stats[key][field]
            if (
                not isinstance(values, list)
                or len(values) != 14
                or not all(isinstance(value, int | float) and math.isfinite(value) for value in values)
            ):
                raise ValueError(f"Expected 14 finite entries in {key}.{field}.")
            clean_stats[key][field] = values
    files = list((source / "params").rglob("*"))
    if any(path.is_symlink() for path in [source / "params", *files]):
        raise ValueError("Refusing to export symlinks inside params/.")
    return stats_path, {"norm_stats": clean_stats}


def export_checkpoint(
    source: Path,
    destination: Path,
    *,
    config: str = "pi05_piperx",
    asset_id: str | None = None,
    action_horizon: int = 10,
    dry_run: bool = False,
) -> dict:
    if config not in ("pi05_piperx", "pi05_piperx_absolute") or action_horizon <= 0:
        raise ValueError("Choose a supported PiperX config and a positive action_horizon.")
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise FileExistsError("Destination must be new and outside the source checkpoint.")
    stats_path, stats = plan_export(source, asset_id)
    files = [path for path in (source / "params").rglob("*") if path.is_file()]
    summary = {"parameter_files": len(files), "parameter_bytes": sum(path.stat().st_size for path in files)}
    if dry_run:
        return {**summary, "dry_run": True, "destination": str(destination), "normalization_asset": str(stats_path)}
    metadata = {
        "format_version": 1,
        "config": config,
        "pi05": True,
        "action_horizon": action_horizon,
        "model_action_dim": 32,
        "robot_action_dim": 14,
        "discrete_state_input": False,
        "extra_delta_transform": config == "pi05_piperx",
        "asset_id": "piperx",
        "source_step": int(source.name) if source.name.isdigit() else None,
        "warning": "Verify upstream model terms and rights to the fine-tuning data before distribution.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        shutil.copytree(source / "params", staging / "params")
        assets = staging / "assets" / "piperx"
        assets.mkdir(parents=True)
        stats_bytes = (json.dumps(stats, indent=2, allow_nan=False) + "\n").encode()
        (assets / "norm_stats.json").write_bytes(stats_bytes)
        metadata["norm_stats_sha256"] = hashlib.sha256(stats_bytes).hexdigest()
        (staging / "release_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (staging / "README.md").write_text(
            "# PiperX inference checkpoint\n\n"
            "Contains params/ and assets/piperx/norm_stats.json. Optimizer state, data and run logs are omitted.\n"
            "See release_metadata.json for the required model and action transforms.\n"
            "This directory is not a resumable training checkpoint. No benchmark claim is implied.\n"
        )
        staging.rename(destination)
    except BaseException:
        # Only our newly-created staging directory is removed; never touch the input.
        shutil.rmtree(staging)
        raise
    return {**summary, "dry_run": False, "destination": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config", choices=("pi05_piperx", "pi05_piperx_absolute"), default="pi05_piperx")
    parser.add_argument("--asset-id")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
