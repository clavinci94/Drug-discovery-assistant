#!/usr/bin/env python
import os, sys, json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_curves(metrics_json, out_prefix):
    with open(metrics_json) as f:
        m = json.load(f)
    # Wir erwarten nur Testvorhersagen-probs nicht gespeichert -> wir approximieren Kurven aus Metriken? 
    # Fallback: Wir zeigen nur Schwellenanalyse aus y_true, y_prob wenn vorhanden; sonst überspringen.
    y_true = m.get("test",{}).get("y_true")
    y_prob = m.get("test",{}).get("y_prob")
    if not (y_true and y_prob):
        print("⚠️  Keine y_true/y_prob im Report. Wir zeichnen nur einen leichten Overview.")
        with open(f"{out_prefix}_summary.txt","w") as fsum:
            fsum.write(json.dumps(m, indent=2))
        return
    y = np.array(y_true); p = np.array(y_prob)
    fpr, tpr, thr = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    pr, rc, thr2 = precision_recall_curve(y, p); pr_auc = auc(rc, pr)
    plt.figure(); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC AUC={roc_auc:.3f}"); plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc.png", dpi=150)
    plt.figure(); plt.plot(rc,pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR AUC={pr_auc:.3f}"); plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr.png", dpi=150)
    # Schwellen-Grid
    best = max([(th,(p>=th).mean()) for th in np.linspace(0.1,0.9,17)], key=lambda x: x[1])
    with open(f"{out_prefix}_summary.txt","w") as fsum:
        fsum.write(f"ROC_AUC={roc_auc:.3f} PR_AUC={pr_auc:.3f}\nBestThrApprox={best[0]:.2f}\n")
    print(f"✅ ROC/PR gespeichert unter Prefix {out_prefix}")
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="pfad zu *_metrics.json")
    ap.add_argument("--out", default="reports/baseline")
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_curves(args.metrics, args.out)
