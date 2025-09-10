from typing import Tuple
import torch
import numpy as np
import esm  # from fair-esm

def load_esm2(model_name: str = "esm2_t6_8M_UR50D"):
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

@torch.no_grad()
def embed_sequence(seq: str, model_name: str = "esm2_t6_8M_UR50D") -> np.ndarray:
    model, alphabet, batch_converter = load_esm2(model_name)
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", seq)])
    out = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
    token_reps = out["representations"][model.num_layers][0]  # (L+2, D)
    token_reps = token_reps[1:-1]  # remove BOS/EOS
    return token_reps.mean(dim=0).cpu().numpy()  # pooled  (D,)

@torch.no_grad()
def embed_sequence_tokens(seq: str, model_name: str = "esm2_t6_8M_UR50D") -> np.ndarray:
    """Per-residue embeddings (L, D)."""
    model, alphabet, batch_converter = load_esm2(model_name)
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", seq)])
    out = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
    token_reps = out["representations"][model.num_layers][0]  # (L+2, D)
    token_reps = token_reps[1:-1]  # remove BOS/EOS
    return token_reps.cpu().numpy()  # (L, D)
