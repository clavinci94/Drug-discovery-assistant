import os
import numpy as np
import torch
import pandas as pd
from typing import List

from src.features.molecular.featurization import morgan_fp
from src.models.tdi_cross_attn import CrossTDI
from src.optimization.objectives import summarize

N_BITS   = 2048
TDI_MODEL = "models/tdi_cross_attn.pt"
PROT_TOK  = "data/processed/protein_embeddings/P35968_tokens.npy"

def tdi_probs(smiles: List[str]) -> np.ndarray:
    if not os.path.exists(TDI_MODEL):
        raise FileNotFoundError(f"TDI-Model fehlt: {TDI_MODEL}")
    if not os.path.exists(PROT_TOK):
        raise FileNotFoundError(f"Protein-Token-Embedding fehlt: {PROT_TOK}")
    # Molekül-Features (nur Bits für CrossTDI)
    xb = np.vstack([morgan_fp(s, n_bits=N_BITS) for s in smiles]).astype(np.float32)
    prot = np.load(PROT_TOK).astype(np.float32)  # (L, Dp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossTDI(n_bits=N_BITS, d_prot=prot.shape[1], d_model=256, n_heads=4).to(device)
    state = torch.load(TDI_MODEL, map_location=device)
    model.load_state_dict(state)
    model.eval()

    xt = torch.tensor(xb, dtype=torch.float32, device=device)
    pt = torch.tensor(prot, dtype=torch.float32, device=device).unsqueeze(0).repeat(xt.size(0), 1, 1)

    with torch.no_grad():
        logits, _ = model(xt, pt)
        prob = torch.sigmoid(logits).cpu().numpy()
    return prob

def score_candidates_tdi(smiles: List[str]) -> pd.DataFrame:
    p = tdi_probs(smiles)
    rows = [summarize(smi, pr) for smi, pr in zip(smiles, p)]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df
