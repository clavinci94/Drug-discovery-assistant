#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
python - << 'PY'
import json, os, pprint
p = "models/baseline_rf_metrics.json"
if not os.path.exists(p):
    print("Kein Report gefunden. Bitte erst trainieren (./scripts/train_models.sh)."); raise SystemExit(1)
with open(p) as f: m = json.load(f)
def pick(d, keys): return {k: d[k] for k in keys}
for split in ["train","test"]:
    s = m.get(split, {})
    if s:
        print(f"\n== {split.upper()} ==")
        print("AUC:", round(s["roc_auc"],3), "| PR-AUC:", round(s["pr_auc"],3), "| ACC:", round(s["accuracy"],3))
        # kleine Confusion-Matrix
        cm = s.get("confusion_matrix")
        if cm: print("Confusion Matrix:", cm)
PY
