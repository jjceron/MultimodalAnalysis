from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """ EEGNet baseline

        X_s -> phi(X_s) -> A_s -> agg(A_s) -> b_s

    Inputs:

    Outputs:

    Return:

    """
    def __init__(
        self,
        # Input parameters
        n_channels: int = 24, 
        n_classes: int = 3,
        # EEGNet parameters
        F1: int =8,
        D: int = 2,
        F2: int = 16,
        temporal_kern: int = 63,
        separable_kern: int = 15,
        pool1: int = 8,
        pool2: int = 8,
        dropout: float = 0.5,
        meanmax_alpha: float = 0.5,
    ) -> None:
        
        super().__init__()

        if not 0.0 <= meanmax_alpha <= 1.0:
            raise ValueError(f"meanmax_alpha must be in [0, 1], got {meanmax_alpha}")
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.temporal_kern = temporal_kern
        self.separable_kern = separable_kern
        self.pool1 = pool1
        self.pool2 = pool2
        self.dropout = float(dropout)
        self.meanmax_alpha = float(meanmax_alpha)
        self.total_pool = pool1 * pool2

        ### Temporal convolution
        # Input: (B, 1, C, T)
        # Output: (B, F1, C, T)
        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, temporal_kern),
                padding=(0, temporal_kern // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1), 
        )

        ### Spatial depthwise convolution
        # Input: (B, F1, C, T)
        # Output: (B, F1 * D, 1, T / pool1)
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
            nn.Dropout2d(dropout),
        )

        ### Separable convolution
        # Input: (B, F1 * D, 1, T / pool1)
        # Output: (B, F2, 1, T') 
        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, separable_kern),
                padding=(0, separable_kern // 2),
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
            nn.Dropout2d(dropout),
        )

        ### Fully-convolutional temporal classifier
        # Input: (B, F2, 1, T')
        # Output: 
        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=F2,
                out_channels=n_classes,
                kernel_size=(1, 1),
                bias=True,
            )
        )

    def forward(
        self, 
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        alpha: float | None = None,

    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass of EEGNet

        Args:
            x: (B, C, T)
            mask: (B, T)

        Returns:    
            logits_time: (B, T', L)
            logits_subj: (B, L)
        """

        if x.ndim != 3:
               raise ValueError(
                f"Expected input shape (B, C, T), got {tuple(x.shape)}."
            )
            
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected input with {self.n_channels} channels, got {x.shape[1]}."
            )
            
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(
                    f"Expected mask shape (B, T), got {tuple(mask.shape)}."
                )
            if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[2]:
                raise ValueError(
                    f"Mask shape must match batch size and time dimension of x. Expected (B, T), got x={tuple(x.shape)}, mask={tuple(mask.shape)}."
                )
            
        ### Forward pass through convolutional blocks
        x = x.unsqueeze(1)  # (B, 1, C, T)

        z = self.temporal_block(x)  # (B, F1, C, T)
        z = self.spatial_block(z)  # (B, F1 * D, 1, T / pool1)
        z = self.separable_block(z)  # (B, F2, 1, T')

        logits = self.classifier(z)  # (B, L, 1, T')
        logits = logits.squeeze(2)  # (B, L, T')

        logits_time = logits.permute(0, 2, 1)  # (B, T', L)

        logits_subj = self.agg_meanmax(
            logits_time=logits_time,
            mask=mask, 
            alpha=alpha,
        )

        return logits_subj, logits_time
    
    def agg_meanmax(
        self,
        logits_time: torch.Tensor,
        mask: torch.Tensor | None = None,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Aggregate temporal logits using mean-max pooling.

            b_s = (1 - alpha) * mean_t(A_s) + alpha * max_t(A_s)

        Args:
            logits_time: (B, T', L)
            mask: (B, T)
            alpha: weight for max pooling. If None, uses self.meanmax_alpha.

        Returns:
            logits_subj: (B, L)
        """

        if logits_time.ndim != 3:
            raise ValueError(
                f"Expected logits_time with shape (B, T', L), got {tuple(logits_time.shape)}."
            )

        if alpha is None:
            alpha = self.meanmax_alpha

        alpha = float(alpha)

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        B, T_prime, L = logits_time.shape

        if mask is None:
            mask_down = torch.ones(
                B,
                T_prime,
                dtype=torch.bool,
                device=logits_time.device,
            )
        else:
            mask_down = self._downsample_mask(mask=mask, target_length=T_prime)

        mean_logits = self._masked_mean(logits_time, mask_down)
        max_logits = self._masked_max(logits_time, mask_down)

        return (1.0 - alpha) * mean_logits + alpha * max_logits
    
    @staticmethod
    def _downsample_mask(
        mask: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """ Downsample binary mask to match target 
        length using nearest neighbor sampling.
        """
    
        if mask.ndim != 2:
            raise ValueError(
                f"Expected mask with shape (B, T), got {tuple(mask.shape)}."
            )
        
        mask_down = F.interpolate(
            mask.unsqueeze(1).float(),  # (B, 1, T)
            size=target_length,
            mode='nearest',
        )

        return mask_down.squeeze(1).bool()  # (B, T')
    
    @staticmethod
    def _masked_mean(
        logits_time: torch.Tensor,
        mask_down: torch.Tensor,
    ) -> torch.Tensor: 
        """ Compute masked mean over time dimension """

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
    ) -> torch.Tensor:
        """ Compute masked max over time dimension """

        logits_masked = logits_time.masked_fill(
            ~mask_down.unsqueeze(-1),
            -torch.inf,
        )
        
        values = logits_masked.max(dim=1).values

        # Set mean to zero for subjects with no valid time points
        valid_subj = mask_down.any(dim=1).unsqueeze(-1)
        values = torch.where(valid_subj, values, torch.zeros_like(values))

        return values
    
if __name__ == "__main__":
    model = EEGNet()
    x = torch.randn(4, 24, 1280)  # (B, C, T)
    mask = torch.ones(4, 1280).bool()  # (B, T)
    logits_subj, logits_time = model(x, mask=mask, alpha=0.5)
    print("Logits time shape:", logits_time.shape)  # (B, T', L)
    print("Logits subj shape:", logits_subj.shape)  # (B, L)
    