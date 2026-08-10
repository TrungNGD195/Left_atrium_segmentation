"""
train_lora.py
-------------
Huấn luyện Mục 2: LoRA Fine-tuning encoder DINOv2 để giải quyết Domain Shift.

Vấn đề:
  DINOv2 được pre-train trên ảnh tự nhiên (ImageNet). Khi áp dụng cho ảnh MRI,
  đặc trưng bị domain shift nghiêm trọng → Đóng băng encoder không tối ưu.

Giải pháp LoRA:
  - Unfreeze 2–4 block Transformer cuối cùng của DINOv2
  - Chèn ma trận LoRA (rank=4) vào các attention QKV/proj projection
  - Dùng differential learning rate: lr_encoder (1e-5) << lr_decoder (1e-3)
  - Chỉ cần train ~0.5% tham số so với full fine-tune

Cách chạy:
    python src/train_lora.py --data_root /content/data --save_dir results
    python src/train_lora.py --lora_blocks 4 --lora_rank 8 --save_dir results
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

from model import DINOv2Segmenter
from dataset import get_dataloaders
from train import dice_score, iou_score
from train_fast import DiceBCELoss
from lora import inject_lora_into_backbone, get_lora_parameters


# ────────────────────────────────────────────────────────────
# Training loop với PyTorch Mixed Precision (AMP FP16 siêu tốc)
# ────────────────────────────────────────────────────────────
def train_one_epoch_lora(model, loader, criterion, optimizer, scaler, device):
    """
    Huấn luyện 1 epoch với LoRA kết hợp Mixed Precision (AMP).
    Tăng tốc độ gấp 3 lần trên GPU T4 của Colab!
    """
    model.train()
    # Giữ encoder.backbone ở eval() để BatchNorm không bị nhiễu
    model.encoder.backbone.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Forward
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        # Scaled Backward (chống underflow gradient trong FP16)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            d = dice_score(preds, masks)
            j = iou_score(preds, masks)

        total_loss += loss.item()
        total_dice += d
        total_iou += j
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


@torch.no_grad()
def validate_lora(model, loader, criterion, device):
    """Đánh giá trên tập validation với Mixed Precision."""
    model.eval()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0

    for images, masks, _ in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou += iou_score(preds, masks)
        n += 1

    return total_loss / n, total_dice / n, total_iou / n


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main(args):
    print("=" * 60)
    print(f"MUC 2: LoRA FINE-TUNING ENCODER (DINOv2 + LoRA rank={args.lora_rank})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # DataLoaders (dùng ảnh gốc, không phải precomputed features)
    print(f"\nTai du lieu goc (batch_size={args.batch_size})...")
    loaders = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode_2_5d=False,
    )

    if "train" not in loaders or "val" not in loaders:
        print("LOI: Thieu du lieu. Hay kiem tra data_root.")
        return

    # Khởi tạo mô hình DINOv2 đầy đủ (encoder + decoder)
    print(f"\nKhoi tao mo hinh {args.model}...")
    model = DINOv2Segmenter(model_name=args.model).to(device)

    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tham so trainable truoc LoRA: {params_before:,}")

    # Inject LoRA vào N block cuối của encoder
    print(f"\nInjecting LoRA vao {args.lora_blocks} block cuoi (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    num_lora_params = inject_lora_into_backbone(
        backbone=model.encoder.backbone,
        num_last_blocks=args.lora_blocks,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    # QUAN TRỌNG: Phải chuyển model sang device lại vì LoRALinear vừa khởi tạo trên CPU
    model = model.to(device)
    
    print(f"  Tham so LoRA moi them: {num_lora_params:,}")

    # Tham số decoder (vẫn trainable như cũ)
    decoder_params = list(model.decoder.parameters())
    # Tham số LoRA trong encoder (mới, lr nhỏ hơn)
    lora_params = get_lora_parameters(model)

    params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tham so trainable sau LoRA:  {params_after:,}")
    print(f"  -> LoRA chi them {params_after / sum(p.numel() for p in model.parameters()) * 100:.2f}% tong so tham so!")

    # Differential learning rate: LoRA encoder << Decoder
    print(f"\nCau hinh Optimizer (Differential LR):")
    print(f"  - Decoder lr = {args.lr_decoder:.0e}")
    print(f"  - LoRA encoder lr = {args.lr_lora:.0e} ({args.lr_lora/args.lr_decoder:.0f}x nho hon)")
    optimizer = torch.optim.Adam([
        {"params": decoder_params, "lr": args.lr_decoder},
        {"params": lora_params,    "lr": args.lr_lora},
    ])

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    print(f"  Ham loss: BCE + Dice Loss (50%/50%)")

    # Checkpoint path
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = f"best_decoder_vit_small_bce_dice_lora.pth"
    best_ckpt_path = os.path.join(args.save_dir, ckpt_name)
    print(f"  Checkpoint se luu tai: {best_ckpt_path}")

    # Mixed Precision Scaler (AMP FP16 tang toc 3x)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val_dice = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }

    print(f"\nBat dau huan luyen ({args.epochs} epochs, patience={args.patience})...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch_lora(
            model, loaders["train"], criterion, optimizer, scaler, device
        )
        val_loss, val_dice, val_iou = validate_lora(
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
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "lora_blocks": args.lora_blocks,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "model_name": args.model,
            }, best_ckpt_path)
            print(f"  -> Luu checkpoint tot nhat (val_dice={val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping sau {args.patience} epoch khong cai thien.")
                break

    # Lưu history (riêng cho LoRA để không ghi đè lịch sử Mục 1)
    history_path = os.path.join(args.save_dir, "training_history_lora.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"HOAN TAT!")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint:    {best_ckpt_path}")
    print(f"  History:       {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning DINOv2 for LA Segmentation")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (mac dinh 32 chay rat nhanh tren GPU T4 voi AMP FP16)")
    parser.add_argument("--lr_decoder", type=float, default=1e-3,
                        help="Learning rate cho decoder (lon)")
    parser.add_argument("--lr_lora", type=float, default=1e-5,
                        help="Learning rate cho LoRA encoder (rat nho, tranh forget)")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 de tranh multiprocessing xung dot tren Colab")
    parser.add_argument("--lora_blocks", type=int, default=2,
                        help="So block cuoi cua DINOv2 se duoc ap dung LoRA (khuyen nghi: 2-4)")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Hang cua ma tran LoRA (khuyen nghi: 4 hoac 8)")
    parser.add_argument("--lora_alpha", type=float, default=4.0,
                        help="He so scale LoRA (thuong bang rank)")
    args = parser.parse_args()
    main(args)
