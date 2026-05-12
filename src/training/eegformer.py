from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..embedding.eegformer import EEGFormer


def train_eegformer(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    emb_dim: int = 128,
    num_classes: int | None = None,
    dropout: float = 0.25,
    device: torch.device | None = None,
) -> EEGFormer:
    if num_classes is None:
        num_classes = int(np.max(y)) + 1 if len(y) else 0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EEGFormer(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        emb_dim=emb_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def extract_embeddings(
    model: EEGFormer,
    X: np.ndarray,
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).unsqueeze(1)
            xb = xb.to(device)
            _, emb = model(xb, return_emb=True)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)
