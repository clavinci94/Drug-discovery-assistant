#!/usr/bin/env python
import os, sys, json, argparse
# >>> Pfad-Fix: Projektwurzel zum PYTHONPATH hinzufügen
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features.protein.uniprot_features import fetch_uniprot_features_json, flatten_features

def build_report(accession: str, imp_path: str, top_k: int, out_csv: str, out_png: str):
    imp = np.load(imp_path)  # (L,)
    L = imp.shape[0]
    pos = np.arange(1, L+1, dtype=int)
    df = pd.DataFrame({"position": pos, "importance": imp})
    uj = fetch_uniprot_features_json(accession)
    feats = flatten_features(uj)
    fdf = pd.DataFrame(feats)

    def annotate(p):
        hits = fdf[(fdf["begin"]<=p) & (fdf["end"]>=p)]
        hits = hits.sort_values(by=["type"])
        labels = []
        for _, row in hits.iterrows():
            lab = row["type"]
            if row.get("description"):
                lab += f"({row['description']})"
            labels.append(lab)
        return "; ".join(labels[:4])
    df["features"] = [annotate(p) for p in df["position"].tolist()]

    top = df.sort_values("importance", ascending=False).head(top_k)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    top.to_csv(out_csv, index=False)

    plt.figure(figsize=(12,3))
    plt.plot(df["position"].values, df["importance"].values, linewidth=1)
    plt.scatter(top["position"].values, top["importance"].values)
    plt.title(f"Protein Importance along sequence – {accession}")
    plt.xlabel("Residue position"); plt.ylabel("Attention-based importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"✅ CSV: {out_csv}")
    print(f"✅ Plot: {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accession", default="P35968")
    ap.add_argument("--imp", default="models/tdi_cross_attn_protein_importance.npy")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out_csv", default="reports/vegfr2_importance_topk.csv")
    ap.add_argument("--out_png", default="reports/vegfr2_importance_plot.png")
    args = ap.parse_args()
    build_report(args.accession, args.imp, args.topk, args.out_csv, args.out_png)
