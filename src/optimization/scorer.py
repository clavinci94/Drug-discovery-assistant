import os, numpy as np, pandas as pd
from typing import List, Dict
from joblib import load
from src.features.molecular.featurization import morgan_fp, basic_descriptors
from src.optimization.objectives import summarize

RF_MODEL = "models/baseline_rf.joblib"

def rf_probs(smiles: List[str]) -> np.ndarray:
    if not os.path.exists(RF_MODEL):
        raise FileNotFoundError("RF-Modell fehlt. Bitte ./scripts/train_models.sh ausführen.")
    model = load(RF_MODEL)
    # FP + Deskriptoren
    fps = [morgan_fp(s, n_bits=2048) for s in smiles]
    import pandas as pd, numpy as np
    desc = pd.DataFrame([basic_descriptors(s) for s in smiles]).fillna(0).values.astype(np.float32)
    X = np.hstack([np.vstack(fps).astype(np.float32), desc])
    return model.predict_proba(X)[:,1]

def score_candidates(smiles: List[str]) -> pd.DataFrame:
    p = rf_probs(smiles)
    rows = [summarize(smi, pr) for smi, pr in zip(smiles, p)]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df
