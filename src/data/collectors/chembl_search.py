import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"User-Agent":"DrugDiscoveryAssistant/1.0","Accept":"application/json"}

def search_target(query: str) -> str | None:
    r = requests.get(f"{BASE}/target/search.json", params={"q": query, "limit": 10}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    items = r.json().get("targets", [])
    if not items: return None
    # bevorzugt Human & Single Protein
    items.sort(key=lambda t: (t.get("organism")!="Homo sapiens", t.get("target_type")!="SINGLE PROTEIN"))
    return items[0].get("target_chembl_id")

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "VEGFR2"
    print(search_target(q) or "")
