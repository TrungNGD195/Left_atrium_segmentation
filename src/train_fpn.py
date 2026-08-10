"""
train_fpn.py
------------
Huấn luyện Mục 3: FPN Multi-scale Decoder để cải thiện chi tiết đường viền.

Vấn đề:
  DINOv2 chia ảnh 448×448 thành các patch 14×14.
  Decoder gốc chỉ dùng đặc trưng từ lớp cuối → mất chi tiết cạnh,
  dự đoán đường viền tâm nhĩ bị vuông vức/răng cưa.

Giải pháp FPN:
  - Lấy đặc trưng từ 4 lớp Transformer cuối của DINOv2 (lớp 9, 10, 11, 12)
  - Mỗi lớp nông hơn còn giữ thông tin cạnh và texture cục bộ
  - FPN lateral connection + top-down merge: ngữ cảnh sâu lan truyền về lớp nông
  - Concat 4 feature maps → upsample → predict mask mượt và chính xác hơn

Cách chạy:
    python src/train_fpn.py --data_root /content/data --save_dir results
    python src/train_fpn.py --n_levels 4 --fpn_lr 1e-3 --save_dir results
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("src"))

from model import DINOv2FPNSegmenter
from dataset import get_dataloaders
from train import dice_score, iou_score
from train_fast import DiceBCELoss


# ────────────────────────────────────────────────────────────
# Training & Validation (với AMP FP16 để tăng tốc)
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Huấn luyện 1 epoch với Mixed Precision (AMP FP16)."""
    model.train()

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            d = dice_score(preds, masks)
            j = iou_score(preds, masks)

        total_loss += loss.item()
        total_dice += d
        total_iou  += j
        n += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Đánh giá trên tập validation với Mixed Precision."""
    model.eval()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0

    for images, masks, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds, masks)
        n += 1

    return total_loss / n, total_dice / n, total_iou / n


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main(args):
    print("=" * 60)
    print(f"MUC 3: FPN MULTI-SCALE DECODER (n_levels={args.n_levels})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTai du lieu (batch_size={args.batch_size})...")
    loaders = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode_2_5d=False,
    )

    if "train" not in loaders or "val" not in loaders:
        print("LOI: Thieu du lieu. Hay kiem tra data_root.")
        return

    print(f"\nKhoi tao mo hinh DINOv2 FPN (n_levels={args.n_levels})...")
    model = DINOv2FPNSegmenter(
        model_name=args.model,
        n_levels=args.n_levels,
    ).to(device)

    total   = sum(p.numel() for p in model.parameters())
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tong tham so:        {total:>12,}")
    print(f"  Tham so trainable:   {train_n:>12,}  ({train_n/total*100:.1f}%)")
    print(f"  (Encoder DINOv2 van frozen, chi train FPN Decoder)")

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"\nCau hinh optimizer:")
    print(f"  lr = {args.lr:.0e}, loss = BCE + Dice (50/50)")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = f"best_decoder_vit_small_bce_dice_fpn.pth"
    best_ckpt_path = os.path.join(args.save_dir, ckpt_name)
    print(f"  Checkpoint: {best_ckpt_path}")

    best_val_dice    = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss":   [], "val_dice":   [], "val_iou":   [],
    }

    print(f"\nBat dau huan luyen ({args.epochs} epochs, patience={args.patience})...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device
        )
        val_loss, val_dice, val_iou = validate(
            model, loaders["val"], criterion, device
        )

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| T Loss:{train_loss:.4f} Dice:{train_dice:.4f} IoU:{train_iou:.4f} "
            f"| V Loss:{val_loss:.4f} Dice:{val_dice:.4f} IoU:{val_iou:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice    = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou":  val_iou,
                "n_levels": args.n_levels,
                "model_name": args.model,
            }, best_ckpt_path)
            print(f"  -> Luu checkpoint tot nhat (val_dice={val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping sau {args.patience} epoch khong cai thien.")
                break

    history_path = os.path.join(args.save_dir, "training_history_fpn.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"HOAN TAT!")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint:    {best_ckpt_path}")
    print(f"  History:       {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPN Multi-scale Decoder training")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (AMP FP16 giup chay nhanh tren GPU T4)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate cho FPN Decoder")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_levels", type=int, default=4,
                        help="So luong block Transformer lay feature (2-6)")
    args = parser.parse_args()
    main(args)
