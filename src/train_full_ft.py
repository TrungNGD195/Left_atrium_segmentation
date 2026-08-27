"""
train_full_ft.py
----------------
E2 – Full Fine-tuning: Unfreeze toàn bộ DINOv2 encoder.

Chiến lược:
  - Encoder DINOv2: lr = 1e-5  (nhỏ để không phá vỡ pretrained weights)
  - Decoder:        lr = 1e-3  (lớn hơn vì train từ đầu)
  - AMP FP16: ON   (bắt buộc vì full encoder rất nặng VRAM)

Cách chạy:
    python src/train_full_ft.py --data_root /content/data --seed 42 --save_dir results
"""

import os, sys, argparse, json, time
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("src"))

from model import DINOv2Segmenter
from dataset import get_dataloaders
from metrics import dice_score, iou_score, DiceBCELoss
from checkpointing import atomic_json_dump, atomic_torch_save, capture_rng_state, restore_rng_state


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
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
    print(f"E2 – FULL FINE-TUNING | seed={args.seed}")
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

    # Unfreeze TOÀN BỘ encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:>12,}")
    print(f"Trainable params: {trainable_params:>12,}  ({trainable_params/total_params*100:.1f}%)")

    # ── Optimizer với differential LR ────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": model.decoder.parameters(), "lr": args.lr_decoder},
    ])
    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    best_path = os.path.join(checkpoint_dir, "best.pth")
    last_path = os.path.join(checkpoint_dir, "last.pth")
    final_path = os.path.join(checkpoint_dir, "final.pth")
    history_path = os.path.join(args.save_dir, "training_history.json")

    best_val_dice = -1.0
    best_val_iou = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }
    start_epoch = 1

    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        best_val_dice = resume_checkpoint["best_val_dice"]
        best_val_iou = resume_checkpoint["best_val_iou"]
        patience_counter = resume_checkpoint["patience_counter"]
        history = resume_checkpoint["history"]
        restore_rng_state(resume_checkpoint.get("rng_state"))
        start_epoch = resume_checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} ({args.resume})")

    # Đo VRAM trước khi train
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()

    checkpoint = None
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice, train_iou = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device)
        val_loss, val_dice, val_iou = validate(
            model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"T Loss:{train_loss:.4f} Dice:{train_dice:.4f} IoU:{train_iou:.4f} | "
              f"V Loss:{val_loss:.4f} Dice:{val_dice:.4f} IoU:{val_iou:.4f} | {elapsed:.1f}s")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            is_best = False

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "patience_counter": patience_counter,
            "history": history,
            "rng_state": capture_rng_state(),
            "config": "E2_FullFT",
            "args": vars(args),
            "seed": args.seed,
        }
        atomic_torch_save(checkpoint, last_path)
        atomic_json_dump(history, history_path)
        if epoch % args.checkpoint_every == 0:
            atomic_torch_save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))
        if is_best:
            atomic_torch_save(checkpoint, best_path)
            print(f"  -> Saved best (val_dice={val_dice:.4f})")
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if checkpoint is not None:
        atomic_torch_save(checkpoint, final_path)

    total_train_time = time.time() - train_start

    # ── Đo VRAM peak ─────────────────────────────────────────
    vram_gb = 0.0
    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1e9

    # ── Đo inference time (trung bình trên val set) ───────────
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
    inference_ms = (time.time() - t_inf) / 50 * 1000  # ms per image

    # ── Lưu efficiency metrics ─────────────────────────────────
    efficiency = {
        "method": "E2_FullFT",
        "seed": args.seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / total_params * 100, 2),
        "best_val_dice": round(best_val_dice, 4),
        "best_val_iou": round(best_val_iou, 4),
        "vram_gb": round(vram_gb, 3),
        "total_train_time_s": round(total_train_time, 1),
        "inference_ms_per_image": round(inference_ms, 2),
        "checkpoint": best_path,
        "last_checkpoint": last_path,
    }
    eff_path = os.path.join(args.save_dir, f"efficiency_full_ft_seed{args.seed}.json")
    with open(eff_path, "w") as f:
        json.dump(efficiency, f, indent=2)

    hist_path = os.path.join(args.save_dir, f"history_full_ft_seed{args.seed}.json")
    atomic_json_dump(history, hist_path)

    print(f"\n{'='*60}")
    print(f"HOAN TAT E2 (seed={args.seed})")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  VRAM peak:     {vram_gb:.2f} GB")
    print(f"  Train time:    {total_train_time/60:.1f} min")
    print(f"  Inference:     {inference_ms:.1f} ms/image")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2: Full Fine-tuning")
    parser.add_argument("--model",       type=str,   default="vit_large")
    parser.add_argument("--data_root",   type=str,   default="data/processed")
    parser.add_argument("--save_dir",    type=str,   default="results")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=16,
                        help="Full FT cần VRAM nhiều hơn, giảm batch_size nếu OOM")
    parser.add_argument("--lr_encoder",  type=float, default=1e-5)
    parser.add_argument("--lr_decoder",  type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to a resumable last/epoch checkpoint")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                        help="Keep an additional full snapshot every N epochs")
    args = parser.parse_args()
    main(args)
