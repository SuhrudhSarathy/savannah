#!/usr/bin/env bash
# Runs a batch of training experiments and optionally destroys the vast.ai
# instance when done.
#
# Usage:
#   bash infra/run_multi_train.sh configs/experiments.yaml [--dry-run]
#
# Env vars:
#   AUTO_DESTROY=true        — destroy the instance after training
#   VAST_INSTANCE_ID=<id>    — numeric instance ID from vast.ai dashboard
#   VAST_API_KEY=<key>       — required only for auto-detect fallback
set -euo pipefail

echo "==> Starting training..."
TRAIN_EXIT=0
uv run python infra/multi_run.py "$@" || TRAIN_EXIT=$?

echo "==> Training finished (exit code $TRAIN_EXIT)."

if [ "${AUTO_DESTROY:-false}" = "true" ]; then
    INSTANCE_ID=""

    if [ -n "${VAST_INSTANCE_ID:-}" ]; then
        INSTANCE_ID="$VAST_INSTANCE_ID"
    elif command -v vastai &>/dev/null && [ -n "${VAST_API_KEY:-}" ]; then
        echo "==> Auto-detecting instance ID via vastai CLI..."
        INSTANCE_ID=$(vastai show instances --raw \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['id'])" \
            2>/dev/null || true)
    fi

    if [ -n "${INSTANCE_ID:-}" ]; then
        echo "==> Destroying instance $INSTANCE_ID..."
        vastai destroy instance "$INSTANCE_ID"
        echo "    Instance destroyed."
    else
        echo "==> AUTO_DESTROY=true but could not determine instance ID."
        echo "    Set VAST_INSTANCE_ID=<id> or VAST_API_KEY=<key> and retry manually."
    fi
fi

exit "$TRAIN_EXIT"
