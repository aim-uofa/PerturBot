# Release checklist

## Source package

- [ ] Run `uv run python scripts/audit_release.py`; review every finding manually.
- [ ] Confirm no data, private norm stats, model files, credentials, private manifests, logs, recordings, machine-specific endpoints or original Git history are present.
- [ ] Inspect the actual tracked files/archive, not just `.gitignore`. Ignore rules cannot clean existing Git history.
- [ ] Keep `LICENSE`, `LICENSE_GEMMA.txt`, `NOTICE` and inherited copyright notices intact.
- [ ] Set the intended public repository URL and maintainer details after the hosting repository exists. No hosting repository is created by these scripts.
- [ ] Run the default CPU tests, syntax/lint checks, CLI help and source audit.

## Training/inference reproducibility

- [ ] Install in a fresh Linux/Python 3.11 environment from the lockfile.
- [ ] Verify a genuine 8-GPU model run, including checkpoint save/resume, on the documented hardware. Virtual CPU devices only verify sharding plumbing, not GPU memory/performance or full π₀.₅ convergence.
- [ ] Confirm dataset format/revision, image views, joint/gripper ordering and units, normalization provenance, action convention, seed and split.
- [ ] Verify actual-weight inference on a GPU with the matching checkpoint assets. A synthetic tiny-model test is not this check.
- [ ] Record hardware, software versions, training duration and memory usage without publishing private hostnames or credentials.

## Weights/data and claims

- [ ] Obtain the necessary organizational/data-owner approvals and review upstream model/data terms before public distribution.
- [ ] Publish only approved data and statistics; norm stats are derived data, not automatically public just because code is open source.
- [ ] Run the checkpoint export in `--dry-run` mode, then inspect the resulting inference-only package.
- [ ] Establish “best checkpoint” using a comparable held-out benchmark; document metrics, trials, uncertainty and selection criteria.
- [ ] Include a model card and robot-safety limitations. Avoid unsupported performance or safety claims.

The automated scanner is a conservative heuristic, not a complete secret detector or legal approval. Public distribution remains a separate explicit action.
