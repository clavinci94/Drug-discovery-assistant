import os, numpy as np
from src.features.protein.uniprot import fetch_uniprot_accession, fetch_fasta, fasta_to_seq
from src.features.protein.embeddings import embed_sequence, embed_sequence_tokens

def embed_uniprot_and_save(query: str="KDR AND organism_id:9606",
                           out_dir: str="data/processed/protein_embeddings",
                           verbose: bool = False) -> tuple[str,str,str]:
    os.makedirs(out_dir, exist_ok=True)
    acc = fetch_uniprot_accession(query) or "P35968"
    seq = fasta_to_seq(fetch_fasta(acc))
    emb_pool = embed_sequence(seq)
    emb_tokens = embed_sequence_tokens(seq)
    p_pool = os.path.join(out_dir, f"{acc}_pool.npy")
    p_tok = os.path.join(out_dir, f"{acc}_tokens.npy")
    np.save(p_pool, emb_pool)
    np.save(p_tok, emb_tokens)
    if verbose:
        print(f"✅ saved pooled:  {p_pool}   dim={emb_pool.shape}")
        print(f"✅ saved tokens:  {p_tok}   shape={emb_tokens.shape}")
    return acc, p_pool, p_tok

if __name__ == "__main__":
    acc, p_pool, p_tok = embed_uniprot_and_save(verbose=True)
    print(acc + "|" + p_pool + "|" + p_tok)
