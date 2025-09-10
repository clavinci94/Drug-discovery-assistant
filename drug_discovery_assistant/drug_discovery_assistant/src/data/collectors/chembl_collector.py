"""
ChEMBL Data Collector
"""

import requests
import pandas as pd
import time
import json
from typing import Dict, List, Optional

class ChEMBLCollector:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DrugDiscoveryAssistant/1.0 (Educational)'
        })
        
    def get_target_info(self, target_id: str) -> Dict:
        """Holt Target-Informationen"""
        url = f"{self.base_url}/target/{target_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_bioactivities(self, target_id: str, limit: int = 100) -> pd.DataFrame:
        """Sammelt Bioaktivitätsdaten für einen Target"""
        url = f"{self.base_url}/activity"
        params = {
            'target_chembl_id': target_id,
            'limit': limit,
            'format': 'json'
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        activities = data.get('activities', [])
        
        df_data = []
        for activity in activities:
            df_data.append({
                'molecule_chembl_id': activity.get('molecule_chembl_id'),
                'canonical_smiles': activity.get('canonical_smiles'),
                'standard_type': activity.get('standard_type'),
                'standard_value': activity.get('standard_value'),
                'standard_units': activity.get('standard_units'),
                'target_chembl_id': activity.get('target_chembl_id')
            })
        
        return pd.DataFrame(df_data)
