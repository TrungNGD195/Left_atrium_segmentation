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
    python evaluate.py --model vit_large --checkpoint results/best_model.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Không cần GUI
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from scipy import ndimage

# Đảm bảo luôn tìm thấy các module trong src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import DINOv2Segmenter
from metrics import dice_score, iou_score


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
# ────────────────────────────────────────────────────────────
# Post-processing: Largest Connected Component (LCC)
# ────────────────────────────────────────────────────────────
def apply_lcc(pred_np):
    """
    Giữ lại duy nhất thành phần liên thông lớn nhất trong mặt nạ nhị phân.
    Loại bỏ các đốm nhiễu nhỏ rải rác nằm ngoài vùng tâm nhĩ trái.

    Args:
        pred_np: numpy array (H, W) nhị phân {0, 1}
    Returns:
        lcc_mask: numpy array (H, W) nhị phân chỉ giữ thành phần lớn nhất
    """
    if pred_np.sum() == 0:
        return pred_np  # Không có vùng dự đoán → trả về nguyên

    labeled, num_features = ndimage.label(pred_np)
    if num_features == 0:
        return pred_np

    # Tìm thành phần lớn nhất (bỏ qua nhãn 0 là nền)
    component_sizes = ndimage.sum(pred_np, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1
    lcc_mask = (labeled == largest_label).astype(np.float32)
    return lcc_mask


# ────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, loader, device, vis_dir, max_vis=20, use_lcc=False):
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

    lcc_label = " +LCC" if use_lcc else ""   # Nhãn hiển thị trên thanh tiến trình
    all_dice = []
    all_iou  = []
    sample_metrics = []
    records = []

    for images, masks, fnames in tqdm(loader, desc=f"  Evaluating{lcc_label}"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()

        # Tính metrics cho từng sample trong batch
        for i in range(images.size(0)):
            pred_i = preds[i]    # (1, H, W)
            mask_i = masks[i]    # (1, H, W)

            pred_np = pred_i.squeeze().cpu().numpy()

            # --- Áp dụng hậu xử lý LCC (nếu được bật) ---
            if use_lcc:
                pred_np = apply_lcc(pred_np)
                pred_i = torch.from_numpy(pred_np).unsqueeze(0).to(device)

            d = float(dice_score(pred_i, mask_i))
            j = float(iou_score(pred_i, mask_i))
            all_dice.append(d)
            all_iou.append(j)
            sample_metrics.append({"filename": fnames[i], "dice": d, "iou": j})

            img_np = denormalize(images[i])
            img_gray = np.mean(img_np, axis=2)
            gt_np = mask_i.squeeze().cpu().numpy()

            records.append({
                "img_gray": img_gray,
                "gt_np": gt_np,
                "pred_np": pred_np,
                "fname": fnames[i],
                "dice": d,
                "iou": j
            })

    # Lưu trực quan hóa thông minh: Top đẹp nhất (Dice cao nhất) + Đại diện
    if records and max_vis > 0:
        # Sắp xếp theo Dice từ cao xuống thấp
        sorted_records = sorted(records, key=lambda x: x["dice"], reverse=True)
        
        # Chọn top đẹp nhất (chiếm 60% số ảnh lưu), trung bình (20%) và biên (20%)
        n_best = min(int(max_vis * 0.6), len(sorted_records))
        n_med = min(int(max_vis * 0.2), len(sorted_records) - n_best)
        
        selected = sorted_records[:n_best]
        if n_med > 0:
            mid_idx = len(sorted_records) // 2
            selected.extend(sorted_records[mid_idx : mid_idx + n_med])
        
        # Lưu các ảnh đã chọn
        for idx, item in enumerate(selected):
            save_name = f"vis_{idx+1:02d}_dice_{item['dice']:.4f}_{item['fname']}"
            save_path = os.path.join(vis_dir, save_name)
            visualize_prediction(
                item["img_gray"], item["gt_np"], item["pred_np"], item["fname"], item["dice"], item["iou"], save_path
            )

    return all_dice, all_iou, sample_metrics


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
    use_lcc = args.use_lcc
    all_dice, all_iou, sample_metrics = evaluate_model(
        model, loaders["test"], device, vis_dir, max_vis=args.max_vis, use_lcc=use_lcc
    )

    # Kết quả
    print("\n" + "=" * 60)
    print(f"KẾT QUẢ TRÊN TẬP TEST:")
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
    ckpt_basename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    lcc_tag = "_lcc" if args.use_lcc else ""
    results = {
        "checkpoint": args.checkpoint,
        "num_samples": len(all_dice),
        "use_lcc": args.use_lcc,
        "metric_protocol": {
            "dice": "Dice coefficient; higher is better",
            "jaccard_iou": "Jaccard/Intersection over Union; higher is better",
            "summary": "mean and population standard deviation over test slices",
        },
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
        "per_sample_metrics": sample_metrics,
    }
    results_path = os.path.join(args.save_dir, f"test_results_{ckpt_basename}{lcc_tag}.json")
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
    parser.add_argument("--model", type=str, default="vit_large",
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
    parser.add_argument("--use_lcc", action="store_true", default=False,
                        help="Bật hậu xử lý Largest Connected Component")
    args = parser.parse_args()
    main(args)
