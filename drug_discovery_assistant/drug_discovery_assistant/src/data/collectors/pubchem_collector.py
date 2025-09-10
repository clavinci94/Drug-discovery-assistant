"""
PubChem Data Collector
"""

import requests
import pandas as pd
import time
from typing import List, Dict

class PubChemCollector:
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.session = requests.Session()
        
    def get_compound_properties(self, compound_names: List[str]) -> pd.DataFrame:
        """Holt molekulare Properties für Compound-Liste"""
        df_data = []
        
        for compound in compound_names:
            try:
                url = f"{self.base_url}/compound/name/{compound}/property/MolecularWeight,MolecularFormula,CanonicalSMILES,HeavyAtomCount,HBondDonorCount,HBondAcceptorCount/JSON"
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                properties = data['PropertyTable']['Properties'][0]
                
                df_data.append({
                    'compound_name': compound,
                    'cid': properties.get('CID'),
                    'molecular_weight': properties.get('MolecularWeight'),
                    'molecular_formula': properties.get('MolecularFormula'),
                    'canonical_smiles': properties.get('CanonicalSMILES'),
                    'heavy_atom_count': properties.get('HeavyAtomCount'),
                    'hbd_count': properties.get('HBondDonorCount'),
                    'hba_count': properties.get('HBondAcceptorCount')
                })
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Fehler bei {compound}: {e}")
                continue
        
        return pd.DataFrame(df_data)
