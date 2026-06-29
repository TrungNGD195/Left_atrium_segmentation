"""
evaluate.py
-----------
Đánh giá mô hình trên tập test và trực quan hoá kết quả phân vùng.

Chức năng:
  1. Tải checkpoint tốt nhất
  2. Chạy inference trên tập test
  3. Tính Dice Score và IoU cho từng sample và trung bình
  4. Tạo hình ảnh so sánh: Ground Truth (đỏ) vs Prediction (xanh)
  5. Vẽ biểu đồ loss và metrics từ training history

Cách chạy:
    python evaluate.py
    python evaluate.py --model vit_base --checkpoint results/best_model.pth
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Không cần GUI
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import DINOv2Segmenter
from train import dice_score, iou_score


# ────────────────────────────────────────────────────────────
# Denormalize ảnh để hiển thị
# ────────────────────────────────────────────────────────────
def denormalize(img_tensor):
    """
    Chuyển tensor đã chuẩn hoá ImageNet về [0, 1] để hiển thị.
    Args:
        img_tensor: (3, H, W)
    Returns:
        numpy array (H, W, 3) trong [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ────────────────────────────────────────────────────────────
# Trực quan hoá
# ────────────────────────────────────────────────────────────
def visualize_prediction(image, gt_mask, pred_mask, fname, dice, iou, save_path):
    """
    Tạo hình ảnh so sánh giống Figure 3 trong bài báo.
    Đỏ = Ground Truth, Xanh = Prediction.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Ảnh gốc
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI Input", fontsize=12)
    axes[0].axis("off")

    # 2. Ground Truth
    axes[1].imshow(image, cmap="gray")
    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[gt_mask > 0] = [1, 0, 0, 0.4]  # Đỏ, alpha=0.4
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth (đỏ)", fontsize=12)
    axes[1].axis("off")

    # 3. Prediction
    axes[2].imshow(image, cmap="gray")
    pred_overlay = np.zeros((*pred_mask.shape, 4))
    pred_overlay[pred_mask > 0] = [0, 1, 0, 0.4]  # Xanh, alpha=0.4
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction (xanh)", fontsize=12)
    axes[2].axis("off")

    # 4. So sánh chồng lớp
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(gt_overlay)
    axes[3].imshow(pred_overlay)
    axes[3].set_title(f"Overlap | Dice={dice:.4f} IoU={iou:.4f}", fontsize=12)
    axes[3].axis("off")

    plt.suptitle(fname, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history_path, save_dir):
    """Vẽ biểu đồ loss và metrics từ training history."""
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss (BCEWithLogits)", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Val")
    axes[1].set_title("Dice Score", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(epochs, history["train_iou"], "b-", label="Train")
    axes[2].plot(epochs, history["val_iou"], "r-", label="Val")
    axes[2].set_title("IoU (Jaccard)", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Biểu đồ huấn luyện: {save_path}")


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, loader, device, vis_dir, max_vis=20):
    """
    Đánh giá mô hình trên toàn bộ tập dữ liệu.

    Args:
        model: mô hình đã tải checkpoint
        loader: DataLoader (test)
        device: cuda/cpu
        vis_dir: thư mục lưu hình ảnh trực quan
        max_vis: số lượng hình ảnh trực quan tối đa
    Returns:
        all_dice: list Dice score cho từng sample
        all_iou: list IoU cho từng sample
    """
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)

    all_dice = []
    all_iou = []
    vis_count = 0

    for images, masks, fnames in tqdm(loader, desc="  Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()

        # Tính metrics cho từng sample trong batch
        for i in range(images.size(0)):
            pred_i = preds[i]    # (1, H, W)
            mask_i = masks[i]    # (1, H, W)

            d = dice_score(pred_i, mask_i)
            j = iou_score(pred_i, mask_i)
            all_dice.append(d)
            all_iou.append(j)

            # Trực quan hoá một số sample
            if vis_count < max_vis:
                img_np = denormalize(images[i])
                # Chuyển sang grayscale để hiển thị
                img_gray = np.mean(img_np, axis=2)
                gt_np = mask_i.squeeze().cpu().numpy()
                pred_np = pred_i.squeeze().cpu().numpy()

                save_path = os.path.join(vis_dir, f"vis_{vis_count:03d}_{fnames[i]}")
                visualize_prediction(
                    img_gray, gt_np, pred_np, fnames[i], d, j, save_path
                )
                vis_count += 1

    return all_dice, all_iou


def main(args):
    print("=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH DINOv2 LEFT ATRIUM SEGMENTATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Tải mô hình
    print(f"\nKhởi tạo mô hình ({args.model})...")
    model = DINOv2Segmenter(model_name=args.model).to(device)

    # Tải checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Tải checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "decoder_state_dict" in ckpt:
            model.decoder.load_state_dict(ckpt["decoder_state_dict"])
            print("  -> Đã tải decoder_state_dict vào bộ giải mã (decoder)")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print("  -> Đã tải model_state_dict vào toàn bộ mô hình")
        else:
            raise KeyError("Không tìm thấy decoder_state_dict hoặc model_state_dict trong checkpoint.")
        print(f"  Epoch:    {ckpt.get('epoch', '?')}")
        print(f"  Val Dice: {ckpt.get('val_dice', '?')}")
    else:
        print(f"CẢNH BÁO: Không tìm thấy checkpoint tại {args.checkpoint}")
        print("Sử dụng mô hình chưa huấn luyện (chỉ để test).")

    # DataLoader
    print(f"\nTải dữ liệu test...")
    loaders = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if "test" not in loaders:
        print("LỖI: Không tìm thấy dữ liệu test.")
        return

    # Đánh giá
    vis_dir = os.path.join(args.save_dir, "visualizations")
    all_dice, all_iou = evaluate_model(
        model, loaders["test"], device, vis_dir, max_vis=args.max_vis
    )

    # Kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ TRÊN TẬP TEST:")
    print(f"  Số lượng samples: {len(all_dice)}")
    print(f"  Dice Score:")
    print(f"    Mean:   {np.mean(all_dice):.4f}")
    print(f"    Std:    {np.std(all_dice):.4f}")
    print(f"    Min:    {np.min(all_dice):.4f}")
    print(f"    Max:    {np.max(all_dice):.4f}")
    print(f"  IoU (Jaccard):")
    print(f"    Mean:   {np.mean(all_iou):.4f}")
    print(f"    Std:    {np.std(all_iou):.4f}")
    print(f"    Min:    {np.min(all_iou):.4f}")
    print(f"    Max:    {np.max(all_iou):.4f}")

    # Lưu kết quả ra JSON
    results = {
        "num_samples": len(all_dice),
        "dice_mean": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "dice_min": float(np.min(all_dice)),
        "dice_max": float(np.max(all_dice)),
        "iou_mean": float(np.mean(all_iou)),
        "iou_std": float(np.std(all_iou)),
        "iou_min": float(np.min(all_iou)),
        "iou_max": float(np.max(all_iou)),
        "per_sample_dice": all_dice,
        "per_sample_iou": all_iou,
    }
    results_path = os.path.join(args.save_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Kết quả chi tiết: {results_path}")

    # Vẽ biểu đồ training history (nếu có)
    history_path = os.path.join(args.save_dir, "training_history.json")
    if os.path.exists(history_path):
        print("\nVẽ biểu đồ training...")
        plot_training_history(history_path, args.save_dir)

    print(f"  Hình ảnh trực quan: {vis_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 LA Segmenter")
    parser.add_argument("--model", type=str, default="vit_small",
                        choices=["vit_small", "vit_base", "vit_large"],
                        help="Phiên bản DINOv2 ViT")
    parser.add_argument("--checkpoint", type=str, default="results/best_model.pth",
                        help="Đường dẫn checkpoint")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Thư mục gốc dữ liệu")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Thư mục lưu kết quả")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Kích thước batch")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Số worker cho DataLoader")
    parser.add_argument("--max_vis", type=int, default=20,
                        help="Số hình ảnh trực quan tối đa")
    args = parser.parse_args()
    main(args)
