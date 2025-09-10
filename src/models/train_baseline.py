import os, json
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from joblib import dump
import torch

from src.models.baselines import build_random_forest, RFConfig

DATA_CSV = "data/processed/dataset_molecules.csv"
FP_NPZ = "data/processed/morgan_fp.npz"
ARTIF_DIR = "models"
HPO_JSON = "reports/rf_hpo_best.json"

def load_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not (os.path.exists(DATA_CSV) and os.path.exists(FP_NPZ)):
        raise FileNotFoundError("Dataset fehlt. Bitte zuerst Phase-1-Preprocessing laufen lassen.")
    df = pd.read_csv(DATA_CSV)
    fps = np.load(FP_NPZ)["X"]
    y = df["label_active"].values.astype(int)
    desc_cols = ["MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors","NumRotatableBonds","RingCount"]
    if all(c in df.columns for c in desc_cols):
        X_desc = df[desc_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([fps, X_desc])
    else:
        X = fps
    return X.astype(np.float32), y, df

def evaluate_and_log(y_true, y_prob, y_pred, split: str):
    metrics = {
        "split": split,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist()
    }
    print(f"\n== {split} metrics ==")
    print(f"AUC: {metrics['roc_auc']:.3f} | PR-AUC: {metrics['pr_auc']:.3f} | ACC: {metrics['accuracy']:.3f}")
    return metrics

def train_rf():
    X, y, df = load_dataset()
    X, y = shuffle(X, y, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Default-Config
    rf_kwargs = dict(
        n_estimators=400, max_depth=None,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    )
    # HPO-Overrides, wenn vorhanden
    if os.path.exists(HPO_JSON):
        with open(HPO_JSON) as f:
            best = json.load(f)
        rf_kwargs.update(best)
        rf_kwargs["class_weight"] = "balanced_subsample"
        rf_kwargs["n_jobs"] = -1
        rf_kwargs["random_state"] = 42
        print("⚙️  Using HPO params:", rf_kwargs)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(Xtr, ytr)

    prob_tr = rf.predict_proba(Xtr)[:,1]; pred_tr = (prob_tr >= 0.5).astype(int)
    mtr = evaluate_and_log(ytr, prob_tr, pred_tr, "train")

    prob_te = rf.predict_proba(Xte)[:,1]; pred_te = (prob_te >= 0.5).astype(int)
    mte = evaluate_and_log(yte, prob_te, pred_te, "test")

    os.makedirs(ARTIF_DIR, exist_ok=True)
    dump(rf, os.path.join(ARTIF_DIR, "baseline_rf.joblib"))
    with open(os.path.join(ARTIF_DIR, "baseline_rf_metrics.json"), "w") as f:
        json.dump({"train": mtr, "test": mte, "rf_params": rf_kwargs}, f, indent=2)
    print(f"✅ Gespeichert: {os.path.join(ARTIF_DIR,'baseline_rf.joblib')}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rf","mlp"], default="rf")
    args = p.parse_args()
    if args.model == "rf":
        train_rf()
    else:
        print("MLP-Training aus diesem Script entfernt – RF bitte nutzen.")
