#!/usr/bin/env bash
# Copy this worktree to a rented GPU box and set it up (uv, the locked deps, a GPU check).
#   experiments/016-gpu/remote.sh HOST PORT [USER]
# Then e.g.: ssh -p PORT USER@HOST 'cd rl-lab && uv run experiments/016-gpu/bench_step.py'
# Credentials stay with ssh (your key); nothing here stores them.
set -euo pipefail
HOST=$1 PORT=$2 USER=${3:-root}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
rsync -az --delete -e "$SSH" \
  --exclude .git --exclude .venv --exclude __pycache__ --exclude node_modules \
  --filter 'P experiments/016-gpu/runs/***' --exclude site/dist --exclude site/.astro --exclude 'experiments/*/runs/*.zip' \
  "$ROOT/" "$USER@$HOST:rl-lab/"
$SSH "$USER@$HOST" 'set -e
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  cd rl-lab && uv sync --locked -q
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())"'
