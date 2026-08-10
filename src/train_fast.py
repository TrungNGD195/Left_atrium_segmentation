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
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("src"))

from model import SegmentationDecoder
from train import dice_score, iou_score


# ────────────────────────────────────────────────────────────
# Ham Loss: BCE + Dice Loss (Giai quyet mat can bang du lieu)
# ────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """
    Ket hop BCE Loss + Dice Loss.
    - BCE Loss: Toi uu tung pixel (nhan biet diem pixel dung/sai).
    - Dice Loss: Toi uu dien tich chong lap giua prediction va ground truth.
      Khong bi anh huong boi mat can bang (class imbalance) nen xac dinh
      chinh xac vung nho hon nhieu so voi chi dung BCE.

    Cong thuc: L = bce_weight * L_bce + dice_weight * L_dice
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Tinh BCE Loss
        bce_loss = self.bce(logits, targets)

        # Tinh Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ────────────────────────────────────────────────────────────
# Dataset cho pre-computed features (Doc truc tiep tu SSD)
# ────────────────────────────────────────────────────────────
class PrecomputedDataset(Dataset):
    """
    Dataset doc truc tiep features (.pt) da duoc trich xuat boi DINOv2.
    Doc truc tiep tu SSD (/content/data) cuc ky nhanh va on dinh tuyet doi.
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


def get_fast_dataloaders(data_root="data", batch_size=32, num_workers=2, model_name="vit_small", device=None, mode_2_5d=False):
    """Tao DataLoader tu pre-computed features (tu dong trich xuat neu chua co)."""
    # Luu features vao thu muc rieng cho moi che do: precomputed hoac precomputed_2_5d
    subfolder = "precomputed_2_5d" if mode_2_5d else "precomputed"
    precomp_dir = os.path.join(data_root, subfolder)

    # Kiem tra xem da co du lieu precomputed chua
    train_feat_dir = os.path.join(precomp_dir, "train", "features")
    needs_extract = not os.path.exists(train_feat_dir) or len([f for f in os.listdir(train_feat_dir) if f.endswith(".pt")]) == 0

    if needs_extract:
        mode_label = "2.5D" if mode_2_5d else "2D"
        print(f"\n[*] Du lieu precomputed {mode_label} chua co hoac bi trong!")
        print("[*] Dang tu dong chay trich xuat features (chi mat ~45s tren GPU)...")
        from extract_features import extract_and_save
        from model import DINOv2Encoder
        from dataset import get_dataloaders as get_raw_loaders

        dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = DINOv2Encoder(model_name=model_name).to(dev)
        # Truyen mode_2_5d vao de doc anh theo che do phu hop
        raw_loaders = get_raw_loaders(data_root=data_root, batch_size=16, num_workers=0, mode_2_5d=mode_2_5d)

        for split_name, raw_loader in raw_loaders.items():
            out_dir = os.path.join(precomp_dir, split_name)
            count = extract_and_save(encoder, raw_loader, out_dir, dev)
            print(f"  -> Da trich xuat {count} samples cho tap [{split_name}]")
        print("[*] Trich xuat features hoan tat! Tiep tuc huan luyen...\n")

    loaders = {}
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        split_dir = os.path.join(precomp_dir, split)
        if not os.path.exists(os.path.join(split_dir, "features")):
            print(f"  [CANH BAO] {split_dir}/features khong ton tai. Bo qua.")
            continue

        dataset = PrecomputedDataset(split_dir)
        if len(dataset) == 0:
            print(f"  [CANH BAO] {split_dir} co 0 samples. Bo qua.")
            continue

        # Voi In-Memory Dataset, num_workers=0 la lua chon toi uu va an toan nhat tren Colab
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        loaders[split] = loader
        print(f"  [{split}] {len(dataset)} samples, {len(loader)} batches", flush=True)

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
    mode_label = "2.5D" if args.mode_2_5d else "2D"
    print("=" * 60)
    print(f"HUAN LUYEN NHANH (PRE-COMPUTED FEATURES) - Che do {mode_label}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # DataLoaders
    print(f"\nTai pre-computed features [{mode_label}] (batch_size={args.batch_size})...")
    loaders = get_fast_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name=args.model,
        device=device,
        mode_2_5d=args.mode_2_5d,
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

    if args.loss == "bce_dice":
        criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
        print(f"  Ham loss: BCE + Dice Loss (bce=0.5, dice=0.5)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"  Ham loss: BCE Loss (baseline)")
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    # Ten checkpoint phan biet ca loss va che do 2D/2.5D
    suffix = f"_{args.loss}"
    if args.mode_2_5d:
        suffix += "_2_5d"
    ckpt_name = f"best_decoder{suffix}.pth"
    best_ckpt_path = os.path.join(args.save_dir, ckpt_name)

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
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "bce_dice"],
                        help="Ham loss: bce (baseline) hoac bce_dice (cai tien)")
    parser.add_argument("--mode_2_5d", action="store_true", default=False,
                        help="Bat che do dau vao 2.5D: ghep [z-1, z, z+1] vao 3 kenh RGB")
    args = parser.parse_args()
    main(args)
