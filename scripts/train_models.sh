#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
export PYTHONPATH="."
echo "➡️  Train RandomForest baseline"
python -m src.models.train_baseline --model rf
# echo "➡️  Train MLP baseline"
# python -m src.models.train_baseline --model mlp --epochs 10 --lr 0.001 --batch_size 512
echo "✅ Done. Artefakte liegen unter ./models/"
ls -lh models
