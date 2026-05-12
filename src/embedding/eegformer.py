from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.attn = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)


class EEGFormer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        emb_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)
        self.conv4 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.attn = AttentionPooling(16)
        self.proj = nn.Linear(16, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_emb: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        pooled = self.attn(x)
        emb = torch.relu(self.proj(pooled))
        logits = self.classifier(emb)
        return (logits, emb) if return_emb else logits
