#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"
export PYTHONPATH="."

TARGET="${1:-CHEMBL279}"
LIMIT="${2:-600}"

echo "➡️  Phase 1: Build dataset for target ${TARGET} (limit=${LIMIT})"
python -m src.data.processors.prepare_dataset --target "${TARGET}" --limit "${LIMIT}"

echo "➡️  Protein-Embeddings (ESM-2) für VEGFR2/KDR (UniProt P35968)"
python - << 'PY'
from src.features.protein.embed_and_save import embed_uniprot_and_save
acc, p_pool, p_tok = embed_uniprot_and_save("accession:P35968", verbose=True)
print("ACC_PATHS:", acc, p_pool, p_tok)
PY

echo "✅ Phase 1 fertig."
