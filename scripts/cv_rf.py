#!/usr/bin/env python
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

DATA_CSV = "data/processed/dataset_molecules.csv"
FP_NPZ   = "data/processed/morgan_fp.npz"
OUT_JSON = "reports/rf_cv_metrics.json"
OUT_NPY  = "reports/rf_cv_probs.npy"

def load_dataset():
    df = pd.read_csv(DATA_CSV)
    Xfp = np.load(FP_NPZ)["X"].astype(np.float32)
    y   = df["label_active"].values.astype(int)
    # optionale Deskriptoren
    desc_cols = ["MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors","NumRotatableBonds","RingCount"]
    if all(c in df.columns for c in desc_cols):
        Xd = df[desc_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([Xfp, Xd])
    else:
        X = Xfp
    return shuffle(X, y, random_state=42)

def main():
    os.makedirs("reports", exist_ok=True)
    X, y = load_dataset()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros_like(y, dtype=np.float32)
    metrics = []
    for fold, (tr, te) in enumerate(kf.split(X,y), 1):
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:,1]
        probs[te] = p
        metrics.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[te], p)),
            "pr_auc": float(average_precision_score(y[te], p))
        })
        print(f"Fold {fold}: AUC={metrics[-1]['roc_auc']:.3f} PR-AUC={metrics[-1]['pr_auc']:.3f}")
    with open(OUT_JSON,"w") as f: json.dump({"folds":metrics}, f, indent=2)
    np.save(OUT_NPY, np.vstack([y, probs]).T)
    print(f"✅ saved {OUT_JSON} , {OUT_NPY}")

if __name__ == "__main__":
    main()
