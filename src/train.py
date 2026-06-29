"""
train.py
--------
Script huấn luyện mô hình DINOv2 + Decoder cho phân vùng tâm nhĩ trái.

Cấu hình theo bài báo:
  - Loss:       BCEWithLogitsLoss
  - Optimizer:  Adam
  - LR:         0.001
  - Epochs:     35 (tối đa)
  - Early Stop: Dựa trên val Dice score

Cách chạy:
    python train.py
    python train.py --model vit_base --epochs 50 --batch_size 16
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from model import DINOv2Segmenter


# ────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────
def dice_score(pred, target, smooth=1e-6):
    """
    Tính Dice Score.
    Args:
        pred:   tensor nhị phân (B, 1, H, W)
        target: tensor nhị phân (B, 1, H, W)
    Returns:
        dice: float
    """
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """
    Tính IoU (Jaccard Index).
    """
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Huấn luyện 1 epoch."""
    model.train()
    # Đảm bảo encoder luôn ở chế độ eval (frozen BatchNorm, Dropout)
    model.encoder.backbone.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        logits = model(images)  # (B, 1, 448, 448)
        loss = criterion(logits, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            d = dice_score(preds, masks)
            j = iou_score(preds, masks)

        total_loss += loss.item()
        total_dice += d
        total_iou += j
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    return avg_loss, avg_dice, avg_iou


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Đánh giá trên tập validation."""
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for images, masks, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        d = dice_score(preds, masks)
        j = iou_score(preds, masks)

        total_loss += loss.item()
        total_dice += d
        total_iou += j
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    return avg_loss, avg_dice, avg_iou


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main(args):
    print("=" * 60)
    print("HUẤN LUYỆN DINOv2 LEFT ATRIUM SEGMENTATION")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # DataLoaders
    print(f"\nTải dữ liệu (batch_size={args.batch_size})...")
    loaders = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if "train" not in loaders or "val" not in loaders:
        print("LỖI: Thiếu dữ liệu train hoặc val. Hãy chạy prepare_data.py trước.")
        return

    # Model
    print(f"\nKhởi tạo mô hình DINOv2 ({args.model})...")
    model = DINOv2Segmenter(model_name=args.model).to(device)
    print(f"  Tổng tham số:       {model.num_total_params():>12,}")
    print(f"  Tham số huấn luyện: {model.num_trainable_params():>12,}")

    # Loss, Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=args.lr)

    # Thư mục lưu kết quả
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    # Training
    best_val_dice = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }

    print(f"\nBắt đầu huấn luyện ({args.epochs} epochs, patience={args.patience})...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss, val_dice, val_iou = validate(
            model, loaders["val"], criterion, device
        )

        elapsed = time.time() - t0

        # Ghi lịch sử
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        # In kết quả epoch
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| T Loss: {train_loss:.4f} Dice: {train_dice:.4f} IoU: {train_iou:.4f} "
            f"| V Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} "
            f"| {elapsed:.1f}s"
        )

        # Early stopping & checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "args": vars(args),
            }, best_ckpt_path)
            print(f"  → Lưu checkpoint tốt nhất (val_dice={val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⚠ Early stopping: val_dice không cải thiện "
                      f"sau {args.patience} epoch liên tiếp.")
                break

    # Lưu lịch sử huấn luyện
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT!")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint:    {best_ckpt_path}")
    print(f"  History:       {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv2 LA Segmenter")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"],
                        help="Phiên bản DINOv2 ViT")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Thư mục gốc chứa dữ liệu")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Thư mục lưu kết quả")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Số epoch tối đa")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Tốc độ học (learning rate)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Số epoch chờ trước khi early stopping")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Số worker cho DataLoader")
    args = parser.parse_args()
    main(args)
