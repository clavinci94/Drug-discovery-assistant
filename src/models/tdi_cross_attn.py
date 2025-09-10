import os, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, confusion_matrix

DATA_CSV = "data/processed/dataset_molecules.csv"
FP_NPZ   = "data/processed/morgan_fp.npz"

class MolBitEmbed(nn.Module):
    """
    Wandelt binäre Morgan-FP (Nbits) in d_model um.
    Idee: alle aktiven Bits werden über eine Embedding-Matrix summiert (Bag-of-Bits).
    """
    def __init__(self, n_bits: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(n_bits, d_model)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x_bits):  # x_bits: (B, Nbits) uint8/float
        idx = x_bits.nonzero(as_tuple=False)  # (nnz, 2) -> [batch, bit]
        if idx.numel() == 0:
            return torch.zeros(x_bits.size(0), self.emb.embedding_dim, device=x_bits.device)
        b = idx[:,0]
        bit = idx[:,1]
        v = self.emb(bit)  # (nnz, d)
        out = torch.zeros(x_bits.size(0), v.size(1), device=x_bits.device)
        out.index_add_(0, b, v)  # sum over active bits per sample
        # Normierung (optional)
        counts = x_bits.sum(dim=1).clamp(min=1).unsqueeze(1)
        out = out / counts
        return out  # (B, d)

class CrossTDI(nn.Module):
    def __init__(self, n_bits: int, d_prot: int, d_model: int = 256, n_heads: int = 4, p_drop: float = 0.2):
        super().__init__()
        self.mol = MolBitEmbed(n_bits, d_model)
        self.proj_prot = nn.Linear(d_prot, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, 1)
        )
    def forward(self, x_bits, prot_tokens):
        # x_bits: (B, Nbits) ; prot_tokens: (B, L, Dp)
        q = self.mol(x_bits).unsqueeze(1)          # (B, 1, d)
        k = self.proj_prot(prot_tokens)            # (B, L, d)
        v = k
        attn_out, attn_w = self.attn(q, k, v, need_weights=True)   # (B, 1, d), (B, 1, L)
        fused = torch.cat([attn_out.squeeze(1), q.squeeze(1)], dim=-1)  # (B, 2d)
        logit = self.ffn(fused).squeeze(-1)        # (B,)
        return logit, attn_w  # attn weights für Explainability

def load_data() -> tuple[np.ndarray, np.ndarray, int]:
    if not (os.path.exists(DATA_CSV) and os.path.exists(FP_NPZ)):
        raise FileNotFoundError("Dataset fehlt. Bitte Preprocessing laufen lassen.")
    df  = pd.read_csv(DATA_CSV)
    Xfp = np.load(FP_NPZ)["X"].astype(np.float32)  # (N, 2048)
    y   = df["label_active"].values.astype(int)
    return Xfp, y, Xfp.shape[1]

def train_cross_tdi(prot_tokens_npy: str, epochs=8, lr=1e-3, batch_size=128, heads=4, d_model=256):
    Xfp, y, n_bits = load_data()
    prot_tok = np.load(prot_tokens_npy).astype(np.float32)  # (L, Dp)
    Dp = prot_tok.shape[1]
    Xfp, y = shuffle(Xfp, y, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(Xfp, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossTDI(n_bits=n_bits, d_prot=Dp, d_model=d_model, n_heads=heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def batches(X, y, bs):
        n = len(X)
        for i in range(0, n, bs):
            yield X[i:i+bs], y[i:i+bs]

    # Precompute prot tokens batch (same for all samples)
    # broadcast per batch in training loop
    for ep in range(1, epochs+1):
        model.train(); total=0.0
        for xb, yb in batches(Xtr, ytr, batch_size):
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            yb = torch.tensor(yb, dtype=torch.float32, device=device)
            prot = torch.tensor(prot_tok, dtype=torch.float32, device=device).unsqueeze(0).repeat(xb.size(0),1,1)
            opt.zero_grad()
            logits, _ = model(xb, prot)
            loss = bce(logits, yb)
            loss.backward(); opt.step()
            total += loss.item()*len(xb)
        print(f"Epoch {ep}/{epochs} - loss: {total/len(Xtr):.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        prot  = torch.tensor(prot_tok, dtype=torch.float32, device=device).unsqueeze(0).repeat(Xte_t.size(0),1,1)
        logits, attn_w = model(Xte_t, prot)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred = (prob>=0.5).astype(int)
    m = {
        "roc_auc": float(roc_auc_score(yte, prob)),
        "pr_auc": float(average_precision_score(yte, prob)),
        "accuracy": float(accuracy_score(yte, pred)),
        "report": classification_report(yte, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(yte, pred).tolist()
    }
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tdi_cross_attn.pt")
    with open("models/tdi_cross_attn_metrics.json","w") as f: json.dump(m, f, indent=2)
    # Speichere mittlere Attention über Test-Set (Token-Importance fürs Protein)
    attn_mean = attn_w.squeeze(1).mean(dim=0).cpu().numpy()  # (L,)
    np.save("models/tdi_cross_attn_protein_importance.npy", attn_mean)
    print("✅ saved: models/tdi_cross_attn.pt , tdi_cross_attn_metrics.json , tdi_cross_attn_protein_importance.npy")
    print(f"AUC={m['roc_auc']:.3f} | PR-AUC={m['pr_auc']:.3f} | ACC={m['accuracy']:.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prot_tokens", required=True, help="Pfad zu *_tokens.npy (L, D)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    args = p.parse_args()
    train_cross_tdi(args.prot_tokens, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    heads=args.heads, d_model=args.d_model)
