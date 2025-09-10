#!/usr/bin/env python
import os, sys, argparse, pandas as pd
from rdkit import Chem
from rdkit.Chem import SDWriter

# >>> Fix: Projektwurzel in sys.path aufnehmen
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.optimization.generator import brics_mutations
from src.optimization.scorer_tdi import score_candidates_tdi

def save_sdf(df: pd.DataFrame, out_sdf: str, smiles_col="smiles"):
    w = SDWriter(out_sdf)
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(r[smiles_col])
        if m:
            for k,v in r.items():
                if k!=smiles_col:
                    m.SetProp(str(k), str(v))
            w.write(m)
    w.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--mutations", type=int, default=30)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--out_csv", default="reports/opt_tdi_top.csv")
    ap.add_argument("--out_sdf", default="reports/opt_tdi_top.sdf")
    args = ap.parse_args()

    # 1) Kandidaten generieren (BRICS)
    cand=[]
    for s in args.seeds:
        from src.optimization.generator import brics_mutations
        cand += brics_mutations(s, n_variants=args.mutations)
    cand = list(dict.fromkeys([c for c in cand if c]))  # de-dupe

    # 2) Scoren via Cross-Attention TDI
    df = score_candidates_tdi(cand)

    # 3) Top-N speichern
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.head(args.topn).to_csv(args.out_csv, index=False)
    save_sdf(df.head(args.topn), args.out_sdf)
    print(f"✅ Saved: {args.out_csv}  |  {args.out_sdf}")
    print(df.head(args.topn).to_string(index=False))

if __name__ == "__main__":
    main()
