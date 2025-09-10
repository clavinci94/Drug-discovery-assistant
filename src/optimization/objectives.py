import numpy as np
from typing import Dict, Tuple, List
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors

def safe_mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m: Chem.SanitizeMol(m)
    return m

def qed_score(smiles: str) -> float:
    m=safe_mol(smiles)
    return float(QED.qed(m)) if m else 0.0

def lipinski_violations(smiles: str) -> int:
    m=safe_mol(smiles)
    if not m: return 4
    mw  = rdMolDescriptors.CalcExactMolWt(m)
    logp= Crippen.MolLogP(m)
    hbd = rdMolDescriptors.CalcNumHBD(m)
    hba = rdMolDescriptors.CalcNumHBA(m)
    viol = 0
    viol += int(mw  > 500)
    viol += int(logp> 5)
    viol += int(hbd > 5)
    viol += int(hba > 10)
    return viol

def desirability(prob_active: float, smiles: str, w_qed: float=0.5, pen_lip: float=0.2) -> float:
    """Aggregierter Score: Aktivität * (0.5+0.5*QED) - penalty(Lipinski)."""
    q = qed_score(smiles)
    v = lipinski_violations(smiles)
    return float(prob_active) * (0.5 + w_qed* q) - pen_lip * v

def summarize(smiles: str, prob_active: float) -> Dict:
    q = qed_score(smiles)
    v = lipinski_violations(smiles)
    return {
        "smiles": smiles,
        "prob_active": float(prob_active),
        "qed": q,
        "lipinski_violations": v,
        "score": desirability(prob_active, smiles)
    }
