#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
export PYTHONPATH="."
ASPIRIN="CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE="Cn1cnc2n(C)c(=O)n(C)c(=O)c12"
python -m src.models.predict --smiles "$ASPIRIN" "$CAFFEINE"
