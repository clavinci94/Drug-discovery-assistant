import os
import pandas as pd
from typing import Optional
from tqdm import tqdm

from src.data.collectors.chembl_collector import ChEMBLCollector
from src.features.molecular.featurization import featurize_dataframe

def build_dataset_for_target(target_chembl_id: str,
                             limit: int = 500,
                             output_csv: str = "data/processed/dataset_molecules.csv",
                             n_bits: int = 2048):
    os.makedirs("data/processed", exist_ok=True)
    chembl = ChEMBLCollector()
    df = chembl.get_bioactivities(target_chembl_id, limit=limit)

    def make_label(row):
        try:
            v = float(row.get("standard_value", "nan"))
            t = (row.get("standard_type") or "").upper()
            if t in {"IC50","EC50","KI","KD"} and v > 0:
                return 1 if v < 1000 else 0
        except:
            pass
        return None

    df["label_active"] = [make_label(r) for _, r in df.iterrows()]
    df = df[df["label_active"].notna()].copy()
    df["label_active"] = df["label_active"].astype(int)

    feat_df, fps = featurize_dataframe(df, smiles_col="canonical_smiles", n_bits=n_bits)

    import numpy as np
    np.savez_compressed("data/processed/morgan_fp.npz", X=fps)
    feat_df.to_csv(output_csv, index=False)
    print(f"✅ Gespeichert: {output_csv}  |  morgan_fp.npz shape={fps.shape}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=str, required=True, help="Target CHEMBL ID, z.B. CHEMBL279")
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()
    build_dataset_for_target(args.target, limit=args.limit)
