"""
train_partial_ft.py
-------------------
E3 – Partial Fine-tuning: Chỉ unfreeze N block Transformer cuối của DINOv2.

Khác với LoRA (E4):
  - LoRA: Giữ nguyên W_frozen, thêm ma trận A, B nhỏ vào
  - Partial FT: Unfreeze trực tiếp trọng số gốc của 4 block cuối

Chiến lược:
  - Block 0...(N-n-1): Frozen hoàn toàn
  - Block (N-n)...(N-1): Unfreeze trực tiếp, lr = 1e-5
  - Decoder: lr = 1e-3

Cách chạy:
    python src/train_partial_ft.py --data_root /content/data --seed 42 --save_dir results
    python src/train_partial_ft.py --unfreeze_blocks 4 --seed 2026
"""

import os, sys, argparse, json, time
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("src"))

from model import DINOv2Segmenter
from dataset import get_dataloaders
from train import dice_score, iou_score
from train_fast import DiceBCELoss


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unfreeze_last_blocks(model: DINOv2Segmenter, n_blocks: int) -> int:
    """
    Đóng băng toàn bộ encoder, sau đó unfreeze N block cuối.

    Args:
        model: DINOv2Segmenter
        n_blocks: số block Transformer cuối cần unfreeze

    Returns:
        Số tham số trainable trong encoder (không tính decoder)
    """
    # Bước 1: Freeze toàn bộ encoder trước
    for param in model.encoder.parameters():
        param.requires_grad = False

    backbone = model.encoder.backbone
    total_blocks = len(backbone.blocks)
    unfreeze_from = total_blocks - n_blocks

    print(f"  Tổng số blocks ViT: {total_blocks}")
    print(f"  Frozen: block 0 → {unfreeze_from - 1}")
    print(f"  Unfrozen: block {unfreeze_from} → {total_blocks - 1}  ({n_blocks} blocks)")

    # Bước 2: Unfreeze N block cuối (trực tiếp trọng số gốc)
    encoder_trainable = 0
    for i, block in enumerate(backbone.blocks):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True
                encoder_trainable += param.numel()

    # Cũng unfreeze norm cuối của ViT (layer norm sau blocks)
    if hasattr(backbone, "norm"):
        for param in backbone.norm.parameters():
            param.requires_grad = True
            encoder_trainable += param.numel()

    return encoder_trainable


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    # Giữ các block frozen ở eval mode (để BatchNorm/Dropout không bị ảnh hưởng)
    model.encoder.backbone.eval()
    # Chỉ set train cho các block được unfreeze
    backbone = model.encoder.backbone
    total_blocks = len(backbone.blocks)
    unfreeze_from = total_blocks - model._n_unfreeze
    for i, block in enumerate(backbone.blocks):
        if i >= unfreeze_from:
            block.train()

    total_loss = total_dice = total_iou = 0.0
    n = 0
    for images, masks, _ in tqdm(loader, desc="  Train", leave=False):
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
            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou  += iou_score(preds, masks)
        n += 1
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    n = 0
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


def main(args):
    print("=" * 60)
    print(f"E3 – PARTIAL FINE-TUNING ({args.unfreeze_blocks} blocks) | seed={args.seed}")
    print("=" * 60)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dữ liệu ──────────────────────────────────────────────
    loaders = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Mô hình ──────────────────────────────────────────────
    model = DINOv2Segmenter(model_name=args.model).to(device)
    model._n_unfreeze = args.unfreeze_blocks  # lưu lại để dùng trong train loop

    encoder_trainable = unfreeze_last_blocks(model, args.unfreeze_blocks)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:>12,}")
    print(f"Trainable params: {trainable_params:>12,}  ({trainable_params/total_params*100:.1f}%)")
    print(f"  → Encoder (unfrozen blocks): {encoder_trainable:,}")
    print(f"  → Decoder:                   {sum(p.numel() for p in model.decoder.parameters()):,}")

    # ── Optimizer với differential LR ────────────────────────
    # Lấy tham số unfrozen của encoder
    unfrozen_encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": unfrozen_encoder_params,    "lr": args.lr_encoder},
        {"params": model.decoder.parameters(), "lr": args.lr_decoder},
    ])
    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.save_dir,
        f"best_partial_ft_seed{args.seed}.pth"
    )

    best_val_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_dice": [], "val_loss": [], "val_dice": []}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice, _ = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device)
        val_loss, val_dice, val_iou = validate(
            model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"T Loss:{train_loss:.4f} Dice:{train_dice:.4f} | "
              f"V Loss:{val_loss:.4f} Dice:{val_dice:.4f} | {elapsed:.1f}s")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": "partial_ft",
                "unfreeze_blocks": args.unfreeze_blocks,
                "seed": args.seed,
            }, ckpt_path)
            print(f"  -> Saved best (val_dice={val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    total_train_time = time.time() - train_start

    # ── Đo VRAM peak ─────────────────────────────────────────
    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # ── Đo inference time ─────────────────────────────────────
    model.eval()
    dummy = torch.randn(1, 3, 448, 448).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.time() - t_inf) / 50 * 1000

    # ── Lưu efficiency metrics ────────────────────────────────
    efficiency = {
        "method": "E3_PartialFT",
        "unfreeze_blocks": args.unfreeze_blocks,
        "seed": args.seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / total_params * 100, 2),
        "best_val_dice": round(best_val_dice, 4),
        "vram_gb": round(vram_gb, 3),
        "total_train_time_s": round(total_train_time, 1),
        "inference_ms_per_image": round(inference_ms, 2),
        "checkpoint": ckpt_path,
    }
    eff_path = os.path.join(args.save_dir, f"efficiency_partial_ft_seed{args.seed}.json")
    with open(eff_path, "w") as f:
        json.dump(efficiency, f, indent=2)

    hist_path = os.path.join(args.save_dir, f"history_partial_ft_seed{args.seed}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HOAN TAT E3 (seed={args.seed})")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  VRAM peak:     {vram_gb:.2f} GB")
    print(f"  Train time:    {total_train_time/60:.1f} min")
    print(f"  Inference:     {inference_ms:.1f} ms/image")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3: Partial Fine-tuning")
    parser.add_argument("--model",            type=str,   default="vit_small")
    parser.add_argument("--data_root",        type=str,   default="data")
    parser.add_argument("--save_dir",         type=str,   default="results")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=24)
    parser.add_argument("--lr_encoder",       type=float, default=1e-5)
    parser.add_argument("--lr_decoder",       type=float, default=1e-3)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--num_workers",      type=int,   default=0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--unfreeze_blocks",  type=int,   default=4,
                        help="Số block Transformer cuối cần unfreeze (mặc định: 4)")
    args = parser.parse_args()
    main(args)
