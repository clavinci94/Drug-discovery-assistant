import argparse, os, json
import numpy as np
from joblib import load
import pandas as pd

from src.features.molecular.featurization import morgan_fp, basic_descriptors

MODEL_PATH = "models/baseline_rf.joblib"

def featurize_smiles_list(smiles_list, n_bits=2048):
    fps = []
    descs = []
    for smi in smiles_list:
        fps.append(morgan_fp(smi, n_bits=n_bits))
        descs.append(basic_descriptors(smi))
    X_fp = np.vstack(fps).astype(np.float32)
    desc_df = pd.DataFrame(descs).fillna(0).values.astype(np.float32)
    X = np.hstack([X_fp, desc_df])
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", nargs="+", required=True, help="Liste von SMILES")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}. Bitte erst trainieren.")

    model = load(args.model)
    X = featurize_smiles_list(args.smiles)
    proba = model.predict_proba(X)[:,1]
    pred = (proba >= 0.5).astype(int)

    for smi, p, y in zip(args.smiles, proba, pred):
        print(f"{smi}\tprob_active={p:.3f}\tpred={y}")

if __name__ == "__main__":
    main()
