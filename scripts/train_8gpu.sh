#!/usr/bin/env bash
# Single host, single JAX process, eight GPUs. No torchrun or rendezvous port.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export GIT_LFS_SKIP_SMUDGE=1
CONFIG_NAME="${CONFIG_NAME:-pi05_piperx}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf '%s\n' \
        'Usage: bash scripts/train_8gpu.sh [--dry-run] --exp-name NAME [train.py overrides]' \
        'Example: bash scripts/train_8gpu.sh --exp-name first_run --data.repo-id local/piperx' \
        'Optional environment: CONFIG_NAME=pi05_piperx_absolute; CUDA_VISIBLE_DEVICES=...' \
        'Model/data paths, batch size, FSDP and logging are train.py command-line overrides.'
    exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
    shift
    printf 'CUDA_VISIBLE_DEVICES=%q JAX_PLATFORMS=cuda\n' "$CUDA_VISIBLE_DEVICES"
    printf 'uv run --frozen scripts/train.py %q ' "$CONFIG_NAME"
    printf '%q ' "$@"
    printf '\n'
    exit 0
fi
uv run --frozen scripts/check_environment.py --expected-gpus 8
exec uv run --frozen scripts/train.py "$CONFIG_NAME" "$@"
