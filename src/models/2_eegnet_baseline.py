from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetBackbone(nn.Module):
    """
    EEGNet baseline

        X_s -> phi(X_s) -> A_s -> agg(A_s) -> b_s

    Inputs:
        x:    (B, C, T)
        mask: (B, T), True = valid signal, False = padding

    Outputs:
        logits_subject: (B, L)
        logits_time:    (B, T', L)

    Aggregation: 0.5 * mean_t(A_s) + 0.5 * max_t(A_s)
    
    Return:
        logits_subject: (B, L). direct to CrossEntropyLoss.
        logits_time:    (B, T', L). corresponds to A_s.
    """

    def __init__(
        self,
        n_channels: int = 24,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        temporal_kernel: int = 63,
        separable_kernel: int = 15,
        pool1: int = 8,
        pool2: int = 8,
        dropout: float = 0.5,
        meanmax_alpha: float = 0.5,
    ):

        super().__init__()

        if not 0.0 <= meanmax_alpha <= 1.0:
            raise ValueError(
                f"meanmax_alpha must be in [0, 1], got {meanmax_alpha}"
            )

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.F1 = F1
        self.D = D
        self.F2 = F2

        self.pool1 = pool1
        self.pool2 = pool2
        self.total_pool = pool1 * pool2

        self.meanmax_alpha = float(meanmax_alpha)

        # ------------------------------------------------------------------
        # Temporal convolution
        # Input:  (B, 1, C, T)
        # Output: (B, F1, C, T)
        # ------------------------------------------------------------------
        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        # ------------------------------------------------------------------
        # Spatial depthwise convolution
        # Input:  (B, F1, C, T)
        # Output: (B, F1 * D, 1, T / pool1)
        # ------------------------------------------------------------------
        self.spatial_block = nn.Sequential(
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
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Separable temporal convolution
        # Input:  (B, F1 * D, 1, T / pool1)
        # Output: (B, F2, 1, T')
        # ------------------------------------------------------------------
        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
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
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Fully-convolutional temporal classifier
        # Input:  (B, F2, 1, T')
        # Output: (B, L, 1, T')
        # ------------------------------------------------------------------
        self.classifier = nn.Conv2d(
            in_channels=F2,
            out_channels=n_classes,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        x:
            Tensor EEG con forma (B, C, T).

        mask:
            Máscara temporal con forma (B, T).
            True indica señal real.
            False indica padding.

        Returns
        -------
        logits_subject:
            Tensor con forma (B, L). Se pasa directamente a CrossEntropyLoss.

        logits_time:
            Tensor con forma (B, T', L). Corresponde a A_s.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (B, C, T), got {tuple(x.shape)}"
            )

        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} EEG channels, got {x.shape[1]}"
            )

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(
                    f"Expected mask with shape (B, T), got {tuple(mask.shape)}"
                )

            if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[2]:
                raise ValueError(
                    "Mask shape must match batch and time dimensions. "
                    f"Got x={tuple(x.shape)}, mask={tuple(mask.shape)}"
                )

            mask = mask.bool()

        # (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)

        z = self.temporal_block(x)
        z = self.spatial_block(z)
        z = self.separable_block(z)

        # (B, F2, 1, T') -> (B, L, 1, T')
        logits = self.classifier(z)

        # (B, L, 1, T') -> (B, L, T')
        logits = logits.squeeze(2)

        # (B, L, T') -> (B, T', L)
        logits_time = logits.transpose(1, 2)

        logits_subject = self.aggregate_meanmax(
            logits_time=logits_time,
            mask=mask,
        )

        return logits_subject, logits_time

    def aggregate_meanmax(
        self,
        logits_time: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Agregación final del baseline:

            b_s = (1 - alpha) * mean_t(A_s) + alpha * max_t(A_s)

        con alpha = 0.5 por defecto.
        """

        if logits_time.ndim != 3:
            raise ValueError(
                f"Expected logits_time with shape (B, T', L), "
                f"got {tuple(logits_time.shape)}"
            )

        B, T_prime, _ = logits_time.shape

        if mask is None:
            mask_down = torch.ones(
                B,
                T_prime,
                dtype=torch.bool,
                device=logits_time.device,
            )
        else:
            mask_down = self.downsample_mask(mask=mask, target_len=T_prime)

        mean_logits = self._masked_mean(logits_time, mask_down)
        max_logits = self._masked_max(logits_time, mask_down)

        alpha = self.meanmax_alpha

        return (1.0 - alpha) * mean_logits + alpha * max_logits

    @staticmethod
    def _masked_mean(
        logits_time: torch.Tensor,
        mask_down: torch.Tensor,
    ):
        logits_masked = logits_time.masked_fill(
            ~mask_down.unsqueeze(-1),
            0.0,
        )

        denom = mask_down.sum(dim=1).clamp_min(1).unsqueeze(-1)

        return logits_masked.sum(dim=1) / denom

    @staticmethod
    def _masked_max(
        logits_time: torch.Tensor,
        mask_down: torch.Tensor,
    ):
        logits_masked = logits_time.masked_fill(
            ~mask_down.unsqueeze(-1),
            -torch.inf,
        )

        values = logits_masked.max(dim=1).values

        # Fallback seguro en caso extremo de que un sujeto quedara sin
        # muestras válidas tras la máscara.
        valid_subject = mask_down.any(dim=1).unsqueeze(-1)
        values = torch.where(valid_subject, values, torch.zeros_like(values))

        return values

    @staticmethod
    def downsample_mask(
        mask: torch.Tensor,
        target_len: int,
    ):
        """
        Convierte una máscara temporal de longitud T a longitud T'.
        """

        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask with shape (B, T), got {tuple(mask.shape)}"
            )

        mask_down = F.interpolate(
            mask.float().unsqueeze(1),
            size=target_len,
            mode="nearest",
        )

        return mask_down.squeeze(1).bool()


if __name__ == "__main__":
    B = 4
    C = 24
    T = 38400
    L = 3

    x = torch.randn(B, C, T)
    mask = torch.ones(B, T, dtype=torch.bool)

    model = EEGNetBackbone(
        n_channels=C,
        n_classes=L,
        F1=8,
        D=2,
        F2=16,
        pool1=8,
        pool2=8,
        meanmax_alpha=0.5,
    )

    logits_subject, logits_time = model(x, mask=mask)

    print("Input:", tuple(x.shape))
    print("Logits time:", tuple(logits_time.shape))
    print("Logits subject:", tuple(logits_subject.shape))
