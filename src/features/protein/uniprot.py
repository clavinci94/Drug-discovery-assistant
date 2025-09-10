import requests

U_BASE = "https://rest.uniprot.org"

def fetch_uniprot_accession(query: str="KDR AND organism_id:9606") -> str | None:
    """
    Holt die beste UniProtKB Accession (Swiss-Prot bevorzugt) für die Query.
    Standard: Human KDR (VEGFR2).
    """
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,reviewed,protein_name,organism_name,length"
    }
    r = requests.get(f"{U_BASE}/uniprotkb/search", params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    # bevorzugt reviewed (Swiss-Prot)
    results.sort(key=lambda x: (not x.get("entryType","").lower()=="reviewed", x.get("primaryAccession","")))
    return results[0]["primaryAccession"]

def fetch_fasta(accession: str) -> str:
    r = requests.get(f"{U_BASE}/uniprotkb/{accession}.fasta", timeout=30)
    r.raise_for_status()
    return r.text

def fasta_to_seq(fasta_txt: str) -> str:
    lines = [ln.strip() for ln in fasta_txt.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines)
