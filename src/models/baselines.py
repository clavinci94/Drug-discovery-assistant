from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

# --- Random Forest (sklearn) ---
from sklearn.ensemble import RandomForestClassifier

# --- MLP (PyTorch) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class RFConfig:
    n_estimators: int = 400
    max_depth: Optional[int] = None
    class_weight: Optional[str] = "balanced_subsample"
    n_jobs: int = -1
    random_state: int = 42

def build_random_forest(cfg: RFConfig | None = None) -> RandomForestClassifier:
    cfg = cfg or RFConfig()
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        class_weight=cfg.class_weight,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state
    )

# ---- Simple MLP with MC-Dropout for Uncertainty ----
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, p_drop: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.do1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.do2 = nn.Dropout(p_drop)
        self.out = nn.Linear(hidden//2, 1)

    def forward(self, x, mc_dropout: bool = False):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.do1.p, training=mc_dropout)  # MC dropout toggle
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.do2.p, training=mc_dropout)
        x = self.out(x)
        return x

@torch.no_grad()
def mc_dropout_predict(model: MLP, x: torch.Tensor, T: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mean probability and epistemic uncertainty (std) via MC dropout.
    """
    probs = []
    for _ in range(T):
        logits = model(x, mc_dropout=True).squeeze(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.stack(probs, axis=0)  # (T, N)
    return probs.mean(axis=0), probs.std(axis=0)
