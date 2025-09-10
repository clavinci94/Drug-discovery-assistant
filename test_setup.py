import sys
from src.config.database import setup_database
from src.data.collectors.chembl_collector import ChEMBLCollector
from src.data.collectors.pubchem_collector import PubChemCollector

def test_complete_setup():
    print("🧬 Drug Discovery Assistant - Setup Test")
    print("="*50)
    print("\n1) Datenbank Setup…")
    setup_database()
    print("\n2) ChEMBL API Test…")
    try:
        chembl = ChEMBLCollector()
        info = chembl.get_target_info("CHEMBL279")
        print(f"✅ Target: {info.get('pref_name','Unknown')}")
        df = chembl.get_bioactivities("CHEMBL279", limit=10)
        print(f"✅ {len(df)} Bioactivities gesammelt")
    except Exception as e:
        print(f"❌ ChEMBL Fehler: {e}")
    print("\n3) PubChem API Test…")
    try:
        pubchem = PubChemCollector()
        props = pubchem.get_compound_properties(['aspirin','caffeine'])
        print(f"✅ {len(props)} Compound Properties gesammelt")
        if not props.empty:
            print(props[['compound_name','molecular_weight']].to_string(index=False))
    except Exception as e:
        print(f"❌ PubChem Fehler: {e}")
    print("\n🎉 Setup Test abgeschlossen!")
if __name__ == "__main__":
    test_complete_setup()
