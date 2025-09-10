import random
from typing import List, Set
from rdkit import Chem
from rdkit.Chem import BRICS

def brics_mutations(smiles: str, n_variants: int = 20) -> List[str]:
    m = Chem.MolFromSmiles(smiles)
    if not m: return []
    frags = list(BRICS.BRICSDecompose(m))
    if not frags: return []
    # einfache Rekombination von 2-3 zufälligen Fragmenten
    res: Set[str] = set()
    for _ in range(n_variants*3):
        parts = random.sample(frags, k=min(len(frags), random.choice([2,3])))
        try:
            new = BRICS.BRICSBuild(list(map(Chem.MolFromSmiles, parts)))
            for nm in new:
                smi = Chem.MolToSmiles(nm, isomericSmiles=True)
                if smi and smi != smiles:
                    res.add(smi)
                    if len(res) >= n_variants:
                        break
        except Exception:
            pass
        if len(res) >= n_variants:
            break
    return list(res)
