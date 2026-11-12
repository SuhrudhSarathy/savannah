#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0

Installs uv and the FFmpeg system libraries (needed by torchcodec), syncs the
project with the simulator extra, and logs into W&B if WANDB_API_KEY is set.
torch comes from PyPI, whose Linux wheels bundle CUDA 13 (needs an NVIDIA
driver >= 580).
EOF
}

case "${1:-}" in
    "") ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
esac

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

echo "==> Installing project dependencies (with simulators)..."
uv sync --extra sim

# franka_sim isn't on PyPI (private repo): install it from a local checkout,
# after `uv sync` (an exact sync would remove it again).
echo "==> Installing franka_sim (optional, for franka_sim sim eval)..."
if [ -n "${FRANKA_SIM_PATH:-}" ]; then
    uv pip install -e "$FRANKA_SIM_PATH"
else
    echo "    FRANKA_SIM_PATH not set — skipping. franka_* tasks still train, but sim eval needs it:"
    echo "    git clone git@github.com:SuhrudhSarathy/franka_sim.git && FRANKA_SIM_PATH=\$PWD/franka_sim bash infra/setup.sh"
fi

echo "==> Configuring W&B..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run wandb login "$WANDB_API_KEY"
else
    echo "    WANDB_API_KEY not set."
    echo "    Run: uv run wandb login <your-api-key>"
fi

echo "==> Configuring Hugging Face Hub (checkpoints)..."
if [ -n "${HF_TOKEN:-}" ]; then
    uv run hf auth login --token "$HF_TOKEN"
else
    echo "    HF_TOKEN not set — checkpoints can't be pushed to the Hub."
    echo "    Run: uv run hf auth login   (or train with trainer.checkpoint.push_to_hub=false)"
fi

echo ""
echo "Setup complete."
echo "  Train:      bash infra/run_train.sh [hydra overrides]"
echo "  Multi-run:  uv run python infra/multi_run.py configs/experiments.yaml"
