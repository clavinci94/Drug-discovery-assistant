#!/usr/bin/env python
import os, json, numpy as np, pandas as pd, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

DATA_CSV = "data/processed/dataset_molecules.csv"
FP_NPZ   = "data/processed/morgan_fp.npz"
OUT_JSON = "reports/rf_hpo_best.json"

def load_dataset():
    df = pd.read_csv(DATA_CSV)
    Xfp = np.load(FP_NPZ)["X"].astype(np.float32)
    y   = df["label_active"].values.astype(int)
    desc_cols = ["MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors","NumRotatableBonds","RingCount"]
    if all(c in df.columns for c in desc_cols):
        Xd = df[desc_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([Xfp, Xd])
    else:
        X = Xfp
    return shuffle(X, y, random_state=42)

def objective(trial):
    X, y = load_dataset()
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 8, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt","log2", None]),
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": 42
    }
    aucs=[]
    for tr, te in kf.split(X,y):
        rf = RandomForestClassifier(**params)
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))

def main():
    os.makedirs("reports", exist_ok=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    with open(OUT_JSON,"w") as f: json.dump(study.best_trial.params, f, indent=2)
    print("✅ best params:", study.best_trial.params)

if __name__ == "__main__":
    main()
