from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs

def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.SanitizeMol(mol)
    return mol

def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)  # ExplicitBitVect
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def basic_descriptors(smiles: str) -> Dict[str, Any]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {k: np.nan for k in [
            "MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors","NumRotatableBonds","RingCount"
        ]}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
    }

def featurize_dataframe(df: pd.DataFrame, smiles_col: str = "canonical_smiles",
                        radius: int = 2, n_bits: int = 2048) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    df = df[df[smiles_col].notna()].drop_duplicates(subset=[smiles_col])
    desc_rows = []
    fps = []
    for smi in df[smiles_col]:
        desc_rows.append(basic_descriptors(smi))
        fps.append(morgan_fp(smi, radius=radius, n_bits=n_bits))
    desc_df = pd.DataFrame(desc_rows, index=df.index)
    X_fp = np.vstack(fps).astype(np.uint8)
    out_df = pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    return out_df, X_fp
