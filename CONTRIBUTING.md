# Contributing

Use Python 3.11 and the locked `uv` environment. Keep robot/data assumptions explicit.

Before submitting changes:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
JAX_PLATFORMS=cpu uv run pytest
uv run ruff check scripts examples/piperx_real tests src/openpi/policies/piperx_inference.py
uv run python scripts/audit_release.py
```

Do not commit data, model weights, norm statistics, credentials, machine-specific endpoints, run logs or private manifests. Document any action-layout or normalization change; old checkpoints must not silently acquire a new action convention. Keep the upstream license and notices intact.
