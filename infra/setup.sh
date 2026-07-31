#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--cuda cu126|cu130]

  --cuda cu126|cu130   Pin torch/torchvision to this CUDA build (see the
                        cu126/cu130 extras in pyproject.toml). If omitted,
                        auto-detects from the driver's CUDA Version reported
                        by \`nvidia-smi\`; installs without a CUDA-specific
                        torch build if no NVIDIA GPU is found.

CUDA_EXTRA env var is equivalent to --cuda.
EOF
}

CUDA_EXTRA="${CUDA_EXTRA:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            CUDA_EXTRA="${2:-}"
            shift 2
            ;;
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
done

if [[ -n "$CUDA_EXTRA" && "$CUDA_EXTRA" != "cu126" && "$CUDA_EXTRA" != "cu130" ]]; then
    echo "Invalid --cuda value: '$CUDA_EXTRA' (expected cu126 or cu130)" >&2
    exit 1
fi

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

echo "==> Selecting CUDA build for torch..."
if [[ -z "$CUDA_EXTRA" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        driver_cuda="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' || true)"
        if [[ -n "$driver_cuda" ]]; then
            major="${driver_cuda%%.*}"
            minor="${driver_cuda#*.}"
            if (( major >= 13 )); then
                CUDA_EXTRA="cu130"
            elif (( major == 12 && minor >= 6 )); then
                CUDA_EXTRA="cu126"
            else
                echo "    Driver supports CUDA $driver_cuda, older than the cu126 build this repo pins." >&2
                echo "    Re-run with --cuda cu126, or add a matching [[tool.uv.index]] entry in pyproject.toml." >&2
                exit 1
            fi
            echo "    Detected driver CUDA $driver_cuda -> using extra '$CUDA_EXTRA'"
        else
            echo "    nvidia-smi found but its CUDA Version couldn't be parsed -- pass --cuda explicitly." >&2
            exit 1
        fi
    else
        echo "    No NVIDIA GPU detected (nvidia-smi not found) -- installing without a pinned CUDA torch build."
    fi
else
    echo "    Using requested extra '$CUDA_EXTRA'"
fi

echo "==> Installing project dependencies..."
if [[ -n "$CUDA_EXTRA" ]]; then
    uv sync --extra "$CUDA_EXTRA"
else
    uv sync
fi

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
