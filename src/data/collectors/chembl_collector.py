import requests, pandas as pd
from typing import Dict

class ChEMBLCollector:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DrugDiscoveryAssistant/1.0 (Educational)',
            'Accept': 'application/json'
        })

    def _get_json(self, url: str, params: Dict | None = None):
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        try:
            return r.json()
        except Exception as e:
            # Hilfreiche Fehlermeldung, falls doch XML/HTML kommt
            snippet = r.text[:200].replace("\n", " ")
            raise RuntimeError(f"Unexpected non-JSON from {url} → {snippet}") from e

    def get_target_info(self, target_id: str) -> Dict:
        # Einzel-Resource: JSON explizit anfordern
        data = self._get_json(f"{self.base_url}/target/{target_id}.json")
        # Manche Antworten packen das Objekt unter 'target'
        return data.get('target', data)

    def get_bioactivities(self, target_id: str, limit: int = 100) -> pd.DataFrame:
        # Listen-Endpoint direkt als JSON anfordern
        params = {'target_chembl_id': target_id, 'limit': limit}
        data = self._get_json(f"{self.base_url}/activity.json", params=params)
        activities = data.get('activities', [])
        rows = []
        for a in activities:
            rows.append({
                'molecule_chembl_id': a.get('molecule_chembl_id'),
                'canonical_smiles': a.get('canonical_smiles'),
                'standard_type': a.get('standard_type'),
                'standard_value': a.get('standard_value'),
                'standard_units': a.get('standard_units'),
                'target_chembl_id': a.get('target_chembl_id')
            })
        return pd.DataFrame(rows)
