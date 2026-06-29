"""
train_fast.py
-------------
Huan luyen NHANH: chi train decoder tren features da duoc pre-compute.
Bo qua hoan toan viec chay DINOv2 encoder moi epoch.

Yeu cau: Chay extract_features.py truoc.

Cach chay:
    python train_fast.py
    python train_fast.py --epochs 35 --batch_size 32 --lr 0.001
"""

import os
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SegmentationDecoder
from train import dice_score, iou_score


# ────────────────────────────────────────────────────────────
# Dataset cho pre-computed features
# ────────────────────────────────────────────────────────────
class PrecomputedDataset(Dataset):
    """
    Dataset doc truc tiep features (.pt) da duoc trich xuat boi DINOv2.
    Khong can chay encoder nua -> rat nhanh.
    """

    def __init__(self, data_dir):
        self.feat_dir = os.path.join(data_dir, "features")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.filenames = sorted([
            f for f in os.listdir(self.feat_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        feat = torch.load(os.path.join(self.feat_dir, fname), weights_only=True)
        mask = torch.load(os.path.join(self.mask_dir, fname), weights_only=True)
        return feat, mask, fname


def get_fast_dataloaders(data_root="data", batch_size=32, num_workers=2):
    """Tao DataLoader tu pre-computed features."""
    precomp_dir = os.path.join(data_root, "precomputed")
    loaders = {}

    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        split_dir = os.path.join(precomp_dir, split)
        if not os.path.exists(os.path.join(split_dir, "features")):
            print(f"  [CANH BAO] {split_dir}/features khong ton tai. Bo qua.")
            continue

        dataset = PrecomputedDataset(split_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        loaders[split] = loader
        print(f"  [{split}] {len(dataset)} samples, {len(loader)} batches")

    return loaders


# ────────────────────────────────────────────────────────────
# Training loop (chi decoder)
# ────────────────────────────────────────────────────────────
def train_one_epoch(decoder, loader, criterion, optimizer, device):
    decoder.train()
    total_loss, total_dice, total_iou, num_batches = 0, 0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for features, masks, _ in pbar:
        features = features.to(device)
        masks = masks.to(device)

        logits = decoder(features)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
def validate(decoder, loader, criterion, device):
    decoder.eval()
    total_loss, total_dice, total_iou, num_batches = 0, 0, 0, 0

    for features, masks, _ in tqdm(loader, desc="  Val  ", leave=False):
        features = features.to(device)
        masks = masks.to(device)

        logits = decoder(features)
        loss = criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).float()

        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou += iou_score(preds, masks)
        num_batches += 1

    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
# Embed dim cho cac model DINOv2
EMBED_DIMS = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}


def main(args):
    print("=" * 60)
    print("HUAN LUYEN NHANH (PRE-COMPUTED FEATURES)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # DataLoaders
    print(f"\nTai pre-computed features (batch_size={args.batch_size})...")
    loaders = get_fast_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if "train" not in loaders or "val" not in loaders:
        print("LOI: Thieu du lieu. Hay chay extract_features.py truoc.")
        return

    # Chi tao decoder (khong can encoder)
    embed_dim = EMBED_DIMS[args.model]
    print(f"\nTao Decoder (embed_dim={embed_dim})...")
    decoder = SegmentationDecoder(in_channels=embed_dim, target_size=448).to(device)
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Tham so decoder: {num_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, "best_decoder.pth")

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

        train_loss, train_dice, train_iou = train_one_epoch(
            decoder, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_dice, val_iou = validate(
            decoder, loaders["val"], criterion, device
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
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "model_name": args.model,
                "embed_dim": embed_dim,
            }, best_ckpt_path)
            print(f"  -> Luu checkpoint tot nhat (val_dice={val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping sau {args.patience} epoch khong cai thien.")
                break

    # Luu history
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"HOAN TAT!")
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint:    {best_ckpt_path}")
    print(f"  History:       {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Train (precomputed)")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"],
                        help="Phien ban DINOv2 (de xac dinh embed_dim)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
