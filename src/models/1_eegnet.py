from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 24,
        n_samples: int = 512,
        num_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        pool1: int = 4,
        pool2: int = 4,
        dropout_eeg: float = 0.5,
        dropout_classifier: float = 0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes

        temporal_padding = kern_length // 2
        separable_padding = 16 // 2

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kern_length),
                padding=(0, temporal_padding),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout2d(dropout_eeg),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, separable_padding),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout2d(dropout_eeg),
        )

        feature_dim = self._infer_feature_dim()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_classifier),
            nn.Linear(feature_dim, num_classes),
        )

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.n_samples)
            features = self._forward_conv_features(x)
            return int(features.flatten(start_dim=1).shape[1])

    def _forward_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, C, T), got {tuple(x.shape)}.")

        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv_features(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits