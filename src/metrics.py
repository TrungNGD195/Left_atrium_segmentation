"""
metrics.py
----------
Các hàm đánh giá và loss function dùng chung cho tất cả thí nghiệm E0–E4.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Dice Coefficient (F1-score cho phân vùng ảnh).
    Args:
        pred:   tensor nhị phân (B, 1, H, W)
        target: tensor nhị phân (B, 1, H, W)
    Returns:
        dice: float trong [0, 1]
    """
    pred_flat   = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    return ((2.0 * intersection + smooth) /
            (pred_flat.sum() + target_flat.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    IoU / Jaccard Index.
    Args:
        pred:   tensor nhị phân (B, 1, H, W)
        target: tensor nhị phân (B, 1, H, W)
    Returns:
        iou: float trong [0, 1]
    """
    pred_flat   = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


class DiceBCELoss(nn.Module):
    """
    Kết hợp BCE Loss và Dice Loss.

    Loss = bce_weight * BCE + dice_weight * (1 - Dice)

    Lý do dùng kết hợp:
      - BCE: ổn định gradient, tối ưu từng pixel
      - Dice: không bị ảnh hưởng bởi class imbalance (tâm nhĩ ~5% ảnh)
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_per_sample(pred: torch.Tensor, target: torch.Tensor,
                    smooth: float = 1e-6) -> torch.Tensor:
    """Dice for each item in a binary-mask batch, rather than pooled pixels."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    dims = tuple(range(1, pred.ndim))
    pred, target = pred.float(), target.float()
    intersection = (pred * target).sum(dim=dims)
    return (2.0 * intersection + smooth) / (
        pred.sum(dim=dims) + target.sum(dim=dims) + smooth
    )


def iou_per_sample(pred: torch.Tensor, target: torch.Tensor,
                   smooth: float = 1e-6) -> torch.Tensor:
    """IoU for each item in a binary-mask batch."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    dims = tuple(range(1, pred.ndim))
    pred, target = pred.float(), target.float()
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    return (intersection + smooth) / (union + smooth)


def hd95(prediction: np.ndarray, target: np.ndarray,
         spacing: Sequence[float] | None = None) -> float:
    """Symmetric 95th-percentile Hausdorff distance for one binary 3D volume."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("nan")
    structure = ndimage.generate_binary_structure(prediction.ndim, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(
        prediction, structure=structure, border_value=0
    )
    target_surface = target ^ ndimage.binary_erosion(
        target, structure=structure, border_value=0
    )
    sampling = tuple(spacing) if spacing is not None else None
    p_to_t = ndimage.distance_transform_edt(~target_surface, sampling=sampling)[pred_surface]
    t_to_p = ndimage.distance_transform_edt(~pred_surface, sampling=sampling)[target_surface]
    return float(np.percentile(np.concatenate((p_to_t, t_to_p)), 95))


def volume_metrics(prediction: np.ndarray, target: np.ndarray,
                   spacing: Sequence[float] | None = None) -> dict[str, float]:
    """Dice, IoU and HD95 after reconstructing slices into a (z, y, x) volume."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    intersection = float(np.logical_and(prediction, target).sum())
    pred_total, target_total = float(prediction.sum()), float(target.sum())
    return {
        "dice_3d": (2.0 * intersection + 1e-6) / (pred_total + target_total + 1e-6),
        "iou_3d": (intersection + 1e-6) / (pred_total + target_total - intersection + 1e-6),
        "hd95": hd95(prediction, target, spacing),
    }
