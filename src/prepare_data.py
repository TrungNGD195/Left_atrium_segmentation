"""
prepare_data.py
---------------
Chuyển đổi dữ liệu MRI 3D (NIfTI) thành các lát cắt 2D (PNG)
và chia thành tập Train / Val / Test theo tỷ lệ 70/10/20.

Cách chạy:
    python prepare_data.py
"""

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


# ────────────────────────────────────────────────────────────
# Cấu hình đường dẫn
# ────────────────────────────────────────────────────────────
RAW_DIR = os.path.join("data", "raw_3d")
IMAGES_DIR = os.path.join(RAW_DIR, "imagesTr")
LABELS_DIR = os.path.join(RAW_DIR, "labelsTr")

OUTPUT_BASE = "data"
SPLITS = {
    "train_2d": None,  # sẽ được gán danh sách subject sau
    "val_2d": None,
    "test_2d": None,
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42

# Kích thước đầu ra cho DINOv2 (patch_size=14, 448/14=32 patches)
TARGET_SIZE = (448, 448)


def get_subject_ids():
    """Lấy danh sách ID bệnh nhân từ thư mục imagesTr."""
    files = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.endswith(".nii.gz") and not f.startswith("._")
    ])
    # Trích xuất ID, ví dụ: "la_003.nii.gz" -> "la_003"
    subject_ids = [f.replace(".nii.gz", "") for f in files]
    return subject_ids


def split_subjects(subject_ids):
    """Chia danh sách bệnh nhân thành train/val/test."""
    # Bước 1: Tách test trước (20%)
    train_val_ids, test_ids = train_test_split(
        subject_ids,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
    )
    # Bước 2: Từ phần còn lại, tách val (10% tổng = 10/80 = 12.5% phần còn lại)
    val_relative_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_relative_ratio,
        random_state=RANDOM_SEED,
    )
    return train_ids, val_ids, test_ids


def normalize_slice(slice_2d):
    """Chuẩn hoá intensity về [0, 255] (uint8)."""
    s = slice_2d.astype(np.float64)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 0:
        s = (s - s_min) / (s_max - s_min) * 255.0
    else:
        s = np.zeros_like(s)
    return s.astype(np.uint8)


def process_subject(subject_id, output_dir):
    """
    Đọc 1 khối 3D, cắt thành các lát cắt 2D theo trục Z,
    lưu ảnh (image) và nhãn (mask) dưới dạng PNG.
    Chỉ lưu các lát cắt chứa ít nhất 1 pixel nhãn dương (có tâm nhĩ trái).
    """
    img_path = os.path.join(IMAGES_DIR, f"{subject_id}.nii.gz")
    lbl_path = os.path.join(LABELS_DIR, f"{subject_id}.nii.gz")

    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    img_data = img_nii.get_fdata()  # shape: (H, W, D)
    lbl_data = lbl_nii.get_fdata()

    # Tạo thư mục con cho image và mask
    img_out = os.path.join(output_dir, "images")
    msk_out = os.path.join(output_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    num_slices = img_data.shape[2]  # chiều sâu (D)
    saved_count = 0

    for z in range(num_slices):
        img_slice = img_data[:, :, z]
        lbl_slice = lbl_data[:, :, z]

        # Chỉ giữ lại các slice có chứa vùng tâm nhĩ trái
        if lbl_slice.max() < 1:
            continue

        # Chuẩn hoá ảnh về [0, 255]
        img_norm = normalize_slice(img_slice)

        # Chuyển nhãn về nhị phân (0 hoặc 255)
        lbl_binary = (lbl_slice > 0).astype(np.uint8) * 255

        # Lưu dưới dạng PNG
        fname = f"{subject_id}_slice{z:04d}.png"
        Image.fromarray(img_norm).save(os.path.join(img_out, fname))
        Image.fromarray(lbl_binary).save(os.path.join(msk_out, fname))
        saved_count += 1

    return saved_count


def main():
    print("=" * 60)
    print("CHUẨN BỊ DỮ LIỆU CHO DINOv2 LEFT ATRIUM SEGMENTATION")
    print("=" * 60)

    # 1. Lấy danh sách bệnh nhân
    subject_ids = get_subject_ids()
    print(f"\nTổng số bệnh nhân (subjects): {len(subject_ids)}")
    print(f"Danh sách: {subject_ids}")

    # 2. Chia train/val/test
    train_ids, val_ids, test_ids = split_subjects(subject_ids)
    print(f"\nPhân chia dữ liệu:")
    print(f"  Train ({len(train_ids)}): {train_ids}")
    print(f"  Val   ({len(val_ids)}): {val_ids}")
    print(f"  Test  ({len(test_ids)}): {test_ids}")

    splits = {
        "train_2d": train_ids,
        "val_2d": val_ids,
        "test_2d": test_ids,
    }

    # 3. Xử lý từng split
    total_stats = {}
    for split_name, ids in splits.items():
        output_dir = os.path.join(OUTPUT_BASE, split_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- Xử lý tập {split_name} ---")
        total_slices = 0
        for sid in tqdm(ids, desc=split_name):
            count = process_subject(sid, output_dir)
            total_slices += count

        total_stats[split_name] = total_slices
        print(f"  → Tổng số lát cắt 2D (có nhãn): {total_slices}")

    # 4. Tổng kết
    print("\n" + "=" * 60)
    print("TỔNG KẾT:")
    for split_name, count in total_stats.items():
        print(f"  {split_name}: {count} slices")
    print(f"  Tổng cộng: {sum(total_stats.values())} slices")
    print("=" * 60)
    print("Hoàn tất! Dữ liệu đã sẵn sàng.")


if __name__ == "__main__":
    main()
