#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
export PYTHONPATH="."
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  echo "⚠️  Bitte Seeds angeben. Beispiel:"
  echo "./scripts/optimize_quick.sh 'CC(=O)Oc1ccccc1C(=O)O' 'Cn1cnc2n(C)c(=O)n(C)c(=O)c12'"
  exit 1
fi
echo "➡️  RF-basiert optimieren…"
./scripts/optimize_compounds.py --seeds "${SEEDS[@]}" --mutations 40 --topn 30 --out_csv reports/opt_top.csv --out_sdf reports/opt_top.sdf
echo "➡️  TDI-basiert optimieren…"
./scripts/optimize_compounds_tdi.py --seeds "${SEEDS[@]}" --mutations 30 --topn 20 --out_csv reports/opt_tdi_top.csv --out_sdf reports/opt_tdi_top.sdf
echo "✅ Fertig. Siehe Ordner: reports/"
