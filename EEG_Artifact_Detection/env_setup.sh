#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-train}

rm -rf output
if [ "$MODE" = "train" ]; then
    rm -rf checkpoints
    echo "Removed checkpoints and output directories for training."
fi
rm -rf data/train data/test data/val

