import requests

U_BASE = "https://rest.uniprot.org"

def fetch_uniprot_features_json(accession: str):
    url = f"{U_BASE}/uniprotkb/{accession}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_features(ujson):
    feats = []
    for f in ujson.get("features", []):
        typ = f.get("type")
        loc = f.get("location", {})
        beg = int(loc.get("start", {}).get("value", -1))
        end = int(loc.get("end", {}).get("value", -1))
        desc = f.get("description", "")
        feats.append({
            "type": typ,
            "begin": beg,
            "end": end,
            "description": desc
        })
    return feats

if __name__ == "__main__":
    import sys, json
    acc = sys.argv[1] if len(sys.argv) > 1 else "P35968"
    data = fetch_uniprot_features_json(acc)
    feats = flatten_features(data)
    print(json.dumps({"accession": acc, "features": feats}, indent=2))
