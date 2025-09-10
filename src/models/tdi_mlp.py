import os, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report, confusion_matrix
from src.models.baselines import MLP

DATA_CSV = "data/processed/dataset_molecules.csv"
FP_NPZ = "data/processed/morgan_fp.npz"

def load_mol_data():
    if not (os.path.exists(DATA_CSV) and os.path.exists(FP_NPZ)):
        raise FileNotFoundError("Dataset fehlt. Bitte erst Preprocessing laufen lassen.")
    df = pd.read_csv(DATA_CSV)
    X_fp = np.load(FP_NPZ)["X"].astype(np.float32)
    y = df["label_active"].values.astype(int)
    # optional: einfache Deskriptoren anhängen
    desc_cols = ["MolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors","NumRotatableBonds","RingCount"]
    if all(c in df.columns for c in desc_cols):
        X_desc = df[desc_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([X_fp, X_desc])
    else:
        X = X_fp
    return X, y

def load_protein_emb(npy_path: str):
    emb = np.load(npy_path).astype(np.float32)
    return emb

def train_tdi(protein_emb_path: str, epochs: int=10, lr: float=1e-3, batch_size: int=256):
    X_mol, y = load_mol_data()
    prot = load_protein_emb(protein_emb_path)  # (D,)
    # broadcast protein embedding
    prot_tile = np.repeat(prot[None, :], X_mol.shape[0], axis=0)
    X = np.hstack([X_mol, prot_tile]).astype(np.float32)

    X, y = shuffle(X, y, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X.shape[1], hidden=512, p_drop=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    def batches(X, y, bs):
        n = len(X)
        for i in range(0, n, bs):
            yield X[i:i+bs], y[i:i+bs]

    for ep in range(1, epochs+1):
        model.train(); total=0.0
        for xb, yb in batches(Xtr, ytr, batch_size):
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            yb = torch.tensor(yb, dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = bce(model(xb).squeeze(-1), yb)
            loss.backward(); opt.step()
            total += loss.item()*len(xb)
        print(f"Epoch {ep}/{epochs} - loss: {total/len(Xtr):.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        logits = torch.tensor(Xte, dtype=torch.float32, device=device)
        logits = model(logits).squeeze(-1).cpu()
        prob = torch.sigmoid(logits).numpy()
        pred = (prob>=0.5).astype(int)

    m = {
        "roc_auc": float(roc_auc_score(yte, prob)),
        "pr_auc": float(average_precision_score(yte, prob)),
        "accuracy": float(accuracy_score(yte, pred)),
        "report": classification_report(yte, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(yte, pred).tolist()
    }
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tdi_mlp.pt")
    with open("models/tdi_mlp_metrics.json","w") as f: json.dump(m, f, indent=2)
    print("✅ TDI-MLP gespeichert: models/tdi_mlp.pt")
    print(f"AUC={m['roc_auc']:.3f} | PR-AUC={m['pr_auc']:.3f} | ACC={m['accuracy']:.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--protein_emb", required=True, help="Pfad zu *.npy Protein-Embedding")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    train_tdi(args.protein_emb, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
