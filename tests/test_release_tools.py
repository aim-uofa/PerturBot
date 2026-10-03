import json
from pathlib import Path
import subprocess

import pytest

from scripts import audit_release
from scripts import export_checkpoint
from scripts import select_checkpoint


def make_checkpoint(root: Path):
    (root / "params").mkdir(parents=True)
    (root / "params" / "_METADATA").write_text("{}")
    (root / "params" / "manifest.ocdbt").write_bytes(b"test manifest")
    stats = {key: {field: [0.0] * 14 for field in ("mean", "std", "q01", "q99")} for key in ("state", "actions")}
    assets = root / "assets" / "legacy" / "dataset"
    assets.mkdir(parents=True)
    (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    (root / "train_state").mkdir()
    (root / "train_state" / "optimizer").write_text("private training state")
    (root / "run.log").write_text("private run log")


def test_checkpoint_export_is_inference_only(tmp_path):
    source, destination = tmp_path / "source" / "1000", tmp_path / "export"
    make_checkpoint(source)
    plan = export_checkpoint.export_checkpoint(source, destination, dry_run=True)
    assert plan["parameter_files"] == 2
    assert not destination.exists()
    export_checkpoint.export_checkpoint(source, destination)
    assert (destination / "params" / "manifest.ocdbt").is_file()
    assert (destination / "assets" / "piperx" / "norm_stats.json").is_file()
    assert not (destination / "train_state").exists()
    assert not (destination / "run.log").exists()
    text = (destination / "release_metadata.json").read_text()
    assert str(tmp_path) not in text
    assert "legacy" not in text
    assert json.loads(text)["source_step"] == 1000
    with pytest.raises(FileExistsError):
        export_checkpoint.export_checkpoint(source, destination)


def test_export_rejects_symlinks(tmp_path):
    source = tmp_path / "source"
    make_checkpoint(source)
    (source / "params" / "symlink").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        export_checkpoint.export_checkpoint(source, tmp_path / "export")


def test_checkpoint_selection_reports_pruned_best(tmp_path):
    for step in (2, 3):
        (tmp_path / str(step) / "params").mkdir(parents=True)
    records = [{"step": step, "val/action_mse": value} for step, value in ((1, 0.1), (2, 0.2), (3, 0.3))]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records))
    result = select_checkpoint.select_checkpoint(tmp_path)
    assert result["step"] == 2
    assert result["best_observed_step"] == 1
    assert result["best_observed_still_available"] is False


def test_no_validation_is_not_best_checkpoint(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 100, "loss": 0.001}\n')
    with pytest.raises(ValueError, match="Training loss"):
        select_checkpoint.select_checkpoint(tmp_path)


def test_audit_reports_location_not_contents(tmp_path):
    secret = "hf_" + "a" * 30
    (tmp_path / "bad.py").write_text(f'key = "{secret}"\n')
    _, findings = audit_release.audit(tmp_path)
    assert findings
    assert all(secret not in finding for finding in findings)


def test_launcher_dry_run():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/train_8gpu.sh", "--dry-run", "--exp-name", "test with spaces", "--batch-size", "64"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "scripts/train.py pi05_piperx" in result.stdout
    assert "--batch-size 64" in result.stdout
    assert "torchrun" not in result.stdout
