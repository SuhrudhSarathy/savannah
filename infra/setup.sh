#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking uv..."
if ! command -v uv &>/dev/null; then
    echo "    uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "    uv found: $(uv --version)"
fi

echo "==> Installing project dependencies..."
uv sync

echo "==> Configuring W&B..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login "$WANDB_API_KEY"
else
    echo "    WANDB_API_KEY not set."
    echo "    Run: uv run wandb login <your-api-key>"
fi

echo ""
echo "Setup complete."
echo "  Train:      bash infra/run_train.sh [hydra overrides]"
echo "  Multi-run:  uv run python infra/multi_run.py configs/experiments.yaml"
