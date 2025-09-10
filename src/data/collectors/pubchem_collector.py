import requests, pandas as pd, time
from typing import List

class PubChemCollector:
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.session = requests.Session()

    def get_compound_properties(self, compound_names: List[str]) -> pd.DataFrame:
        rows = []
        for name in compound_names:
            try:
                url = f"{self.base_url}/compound/name/{name}/property/MolecularWeight,MolecularFormula,CanonicalSMILES,HeavyAtomCount,HBondDonorCount,HBondAcceptorCount/JSON"
                r = self.session.get(url, timeout=15)
                r.raise_for_status()
                p = r.json()['PropertyTable']['Properties'][0]
                rows.append({
                    'compound_name': name,
                    'cid': p.get('CID'),
                    'molecular_weight': p.get('MolecularWeight'),
                    'molecular_formula': p.get('MolecularFormula'),
                    'canonical_smiles': p.get('CanonicalSMILES'),
                    'heavy_atom_count': p.get('HeavyAtomCount'),
                    'hbd_count': p.get('HBondDonorCount'),
                    'hba_count': p.get('HBondAcceptorCount')
                })
                time.sleep(0.2)  # leichtes Rate-Limit
            except Exception as e:
                print(f"⚠️  Fehler bei {name}: {e}")
        return pd.DataFrame(rows)
