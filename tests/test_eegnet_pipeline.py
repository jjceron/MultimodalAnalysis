from __future__ import annotations

import torch
import torch.nn as nn

from src.models.eegnet import EEGNet
from src.data.build_eeg_ct import EEGDataset, create_kfold_dataloaders

def expected_t_prime(T: int, pool1: int, pool2: int) -> int:
    """
    Para AvgPool2d(kernel_size=(1, pool), stride=pool),
    la longitud temporal se reduce por división entera.
    """
    return (T // pool1) // pool2


def trace_eegnet_shapes(
    model: EEGNet,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    alpha: float | None = 0.5,
):
    """
    Ejecuta el modelo bloque por bloque para verificar dimensiones internas.

    No modifica EEGNet. Solo usa sus módulos ya definidos:
    temporal_block, spatial_block, separable_block, classifier y agg_meanmax.
    """

    model.eval()

    shapes = {}

    with torch.no_grad():
        # Entrada original
        shapes["x"] = tuple(x.shape)

        # (B, C, T) -> (B, 1, C, T)
        x_unsqueezed = x.unsqueeze(1)
        shapes["unsqueeze"] = tuple(x_unsqueezed.shape)

        # Temporal block
        z_temporal = model.temporal_block(x_unsqueezed)
        shapes["temporal"] = tuple(z_temporal.shape)

        # Spatial block
        z_spatial = model.spatial_block(z_temporal)
        shapes["spatial"] = tuple(z_spatial.shape)

        # Separable block
        z_separable = model.separable_block(z_spatial)
        shapes["separable"] = tuple(z_separable.shape)

        # Classifier
        logits = model.classifier(z_separable)
        shapes["classifier"] = tuple(logits.shape)

        # (B, L, 1, T') -> (B, L, T')
        logits_squeezed = logits.squeeze(2)
        shapes["logits_squeezed"] = tuple(logits_squeezed.shape)

        # (B, L, T') -> (B, T', L)
        logits_time = logits_squeezed.permute(0, 2, 1)
        shapes["logits_time"] = tuple(logits_time.shape)

        # Aggregation
        logits_subj = model.agg_meanmax(
            logits_time=logits_time,
            mask=mask,
            alpha=alpha,
        )
        shapes["logits_subj"] = tuple(logits_subj.shape)

        if mask is not None:
            mask_down = model._downsample_mask(
                mask=mask,
                target_length=logits_time.shape[1],
            )
            shapes["mask"] = tuple(mask.shape)
            shapes["mask_down"] = tuple(mask_down.shape)

    return shapes, logits_subj, logits_time


def assert_eegnet_shapes(
    model: EEGNet,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    logits_subj: torch.Tensor,
    logits_time: torch.Tensor,
    shapes: dict[str, tuple[int, ...]],
):
    B, C, T = x.shape
    L = model.n_classes
    T_prime = expected_t_prime(T, model.pool1, model.pool2)

    assert shapes["x"] == (B, C, T)
    assert shapes["unsqueeze"] == (B, 1, C, T)

    assert shapes["temporal"] == (
        B,
        model.F1,
        C,
        T,
    )

    assert shapes["spatial"] == (
        B,
        model.F1 * model.D,
        1,
        T // model.pool1,
    )

    assert shapes["separable"] == (
        B,
        model.F2,
        1,
        T_prime,
    )

    assert shapes["classifier"] == (
        B,
        L,
        1,
        T_prime,
    )

    assert shapes["logits_time"] == (
        B,
        T_prime,
        L,
    )

    assert shapes["logits_subj"] == (
        B,
        L,
    )

    assert tuple(logits_time.shape) == (B, T_prime, L)
    assert tuple(logits_subj.shape) == (B, L)

    if mask is not None:
        assert shapes["mask"] == (B, T)
        assert shapes["mask_down"] == (B, T_prime)


def test_eegnet_synthetic_shapes():
    """
    Test aislado de arquitectura.
    No carga datos reales.
    """

    B = 4
    C = 24
    T = 1280
    L = 3

    model = EEGNet(
        n_channels=C,
        n_classes=L,
        F1=8,
        D=2,
        F2=16,
        pool1=8,
        pool2=8,
        meanmax_alpha=0.5,
    )

    x = torch.randn(B, C, T)
    mask = torch.ones(B, T, dtype=torch.bool)

    shapes, logits_subj, logits_time = trace_eegnet_shapes(
        model=model,
        x=x,
        mask=mask,
        alpha=0.5,
    )

    print("\nSynthetic EEGNet shape trace")
    for name, shape in shapes.items():
        print(f"{name:20s}: {shape}")

    assert_eegnet_shapes(
        model=model,
        x=x,
        mask=mask,
        logits_subj=logits_subj,
        logits_time=logits_time,
        shapes=shapes,
    )

    # También comprobamos que la salida sirve para CrossEntropyLoss.
    y = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits_subj, y)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_eegnet_with_real_dataloader():
    """
    Test de integración:
    EEGDataset -> DataLoader -> batch["X"], batch["mask"] -> EEGNet.
    """

    dataset = EEGDataset(condition="closed")

    folds = create_kfold_dataloaders(
        dataset,
        k=5,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    train_loader, val_loader, test_loader = folds[1]

    batch = next(iter(train_loader))

    x = batch["X"]
    y = batch["y"]
    mask = batch["mask"]

    B, C, T = x.shape
    L = 3

    model = EEGNet(
        n_channels=C,
        n_classes=L,
        F1=8,
        D=2,
        F2=16,
        pool1=8,
        pool2=8,
        meanmax_alpha=0.5,
    )

    shapes, logits_subj, logits_time = trace_eegnet_shapes(
        model=model,
        x=x,
        mask=mask,
        alpha=0.5,
    )

    print("\nReal dataloader EEGNet shape trace")
    print(f"subject_id: {batch['subject_id']}")
    print(f"lengths:    {batch['lengths'].tolist()}")
    print(f"y:          {y.tolist()}")

    for name, shape in shapes.items():
        print(f"{name:20s}: {shape}")

    assert_eegnet_shapes(
        model=model,
        x=x,
        mask=mask,
        logits_subj=logits_subj,
        logits_time=logits_time,
        shapes=shapes,
    )

    loss = nn.CrossEntropyLoss()(logits_subj, y)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print(f"loss: {float(loss):.6f}")


if __name__ == "__main__":
    test_eegnet_synthetic_shapes()
    test_eegnet_with_real_dataloader()
    print("\nAll EEGNet pipeline shape tests passed.")