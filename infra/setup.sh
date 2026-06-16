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

echo "==> Installing FFmpeg system libraries (required by torchcodec, a lerobot dep)..."
# torchcodec ships without bundled FFmpeg — the shared libs must be present on
# the system at runtime.  Supported: FFmpeg 4–7.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -qq && sudo apt-get install -y ffmpeg
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "    Homebrew not found — install it from https://brew.sh then re-run."
        exit 1
    fi
else
    echo "    Unsupported OS: install FFmpeg 4–7 manually before continuing."
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

echo "==> Checking GCS credentials..."
if [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    echo "    GOOGLE_APPLICATION_CREDENTIALS set: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "    GOOGLE_APPLICATION_CREDENTIALS not set."
    echo "    Run 'gcloud auth application-default login' for local dev,"
    echo "    or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key path."
    echo "    (Only required if trainer.gcs_bucket is set.)"
fi

echo ""
echo "Setup complete."
echo "  Train:      bash infra/run_train.sh [hydra overrides]"
echo "  Multi-run:  uv run python infra/multi_run.py configs/experiments.yaml"
