#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
export PYTHONPATH="."

TOK_PATH="data/processed/protein_embeddings/P35968_tokens.npy"  # VEGFR2
if [ ! -f "$TOK_PATH" ]; then
  echo "❌ $TOK_PATH fehlt. Bitte erst Embeddings erzeugen."
  exit 1
fi

echo "➡️  Trainiere Cross-Attention TDI (VEGFR2)…"
python -m src.models.tdi_cross_attn --prot_tokens "$TOK_PATH" --epochs 10 --lr 1e-3 --batch_size 128 --heads 4 --d_model 256

echo "✅ Fertig. Artefakte unter ./models/"
ls -lh models | sed -n '1,200p'
