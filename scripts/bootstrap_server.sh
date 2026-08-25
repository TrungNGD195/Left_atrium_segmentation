#!/usr/bin/env bash
# Create the reproducible GPU environment used by the training server.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$uv_bin" ]]; then
    echo "uv was not found at $uv_bin. Set UV_BIN or install uv first." >&2
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for persistent training sessions." >&2
    echo "Install it once with: sudo apt update && sudo apt install -y tmux" >&2
    exit 1
fi

cd "$repo_root"
bash scripts/sync_server.sh

"$uv_bin" venv .venv --python python3
"$uv_bin" pip install --python .venv/bin/python \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
"$uv_bin" pip install --python .venv/bin/python -r requirements.txt

mkdir -p logs results/vit_large/e1 results/vit_large/e2

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the project virtual environment.")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
