"""
extract_features.py
-------------------
Chay DINOv2 encoder 1 lan duy nhat tren toan bo du lieu,
luu feature maps ra dia de tang toc huan luyen.

Cach chay:
    python extract_features.py
    python extract_features.py --model vit_small --batch_size 4
"""

import os
import argparse
import torch
from tqdm import tqdm

from dataset import get_dataloaders
from model import DINOv2Encoder


def extract_and_save(encoder, loader, output_dir, device):
    """
    Chay encoder tren toan bo DataLoader, luu features va masks.
    
    Luu thanh cac file .pt:
      - features/<filename>.pt  -> tensor (embed_dim, 32, 32)
      - masks/<filename>.pt     -> tensor (1, 448, 448)
    """
    feat_dir = os.path.join(output_dir, "features")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    encoder.eval()
    count = 0

    with torch.no_grad():
        for images, masks, fnames in tqdm(loader, desc=f"  {os.path.basename(output_dir)}"):
            images = images.to(device)
            
            # Chay encoder
            features = encoder(images)  # (B, embed_dim, 32, 32)
            
            # Luu tung sample
            for i in range(images.size(0)):
                name = fnames[i].replace(".png", "")
                torch.save(features[i].cpu(), os.path.join(feat_dir, f"{name}.pt"))
                torch.save(masks[i].cpu(), os.path.join(mask_dir, f"{name}.pt"))
                count += 1

    return count


def main(args):
    print("=" * 60)
    print("TRICH XUAT FEATURES DINOV2 (CHI CHAY 1 LAN)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Tai encoder
    print(f"\nTai DINOv2 Encoder ({args.model})...")
    encoder = DINOv2Encoder(model_name=args.model).to(device)
    print(f"  Embed dim: {encoder.embed_dim}")

    # DataLoaders
    print(f"\nTai du lieu (batch_size={args.batch_size})...")
    loaders = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Trich xuat cho tung tap
    output_base = os.path.join(args.data_root, "precomputed")
    os.makedirs(output_base, exist_ok=True)

    total = 0
    for split_name, loader in loaders.items():
        print(f"\n--- Trich xuat tap {split_name} ---")
        output_dir = os.path.join(output_base, split_name)
        count = extract_and_save(encoder, loader, output_dir, device)
        total += count
        print(f"  -> Luu {count} samples vao {output_dir}")

    print(f"\n{'=' * 60}")
    print(f"HOAN TAT! Tong: {total} samples")
    print(f"Thu muc: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv2 features")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
