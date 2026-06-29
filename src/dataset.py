"""
dataset.py
----------
PyTorch Dataset cho dữ liệu lát cắt 2D đã được chuẩn bị.
Xử lý tiền xử lý theo yêu cầu của DINOv2:
  - Chuyển ảnh Grayscale → RGB (3 kênh)
  - Resize về 448×448
  - Chuẩn hoá theo ImageNet mean/std
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ────────────────────────────────────────────────────────────
# Cấu hình
# ────────────────────────────────────────────────────────────
IMG_SIZE = 448  # DINOv2 patch_size=14 → 448/14 = 32 patches mỗi chiều

# Chuẩn hoá theo ImageNet (DINOv2 được huấn luyện trên ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LeftAtriumDataset(Dataset):
    """
    Dataset cho phân vùng tâm nhĩ trái.

    Mỗi sample gồm:
        - image: tensor (3, 448, 448) đã chuẩn hoá
        - mask:  tensor (1, 448, 448) nhị phân {0, 1}
    """

    def __init__(self, data_dir, augment=False):
        """
        Args:
            data_dir: đường dẫn tới thư mục (vd: 'data/train_2d')
                      bên trong phải có 'images/' và 'masks/'
            augment: có áp dụng data augmentation hay không
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        self.augment = augment

        # Lấy danh sách file, sắp xếp để đảm bảo thứ tự nhất quán
        self.filenames = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png")
        ])

        # Transform cho ảnh đầu vào (image)
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # → [0, 1], shape (3, H, W)
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Transform cho mask (dùng NEAREST để không tạo giá trị trung gian)
        self.msk_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # → [0, 1], shape (1, H, W)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Đọc ảnh grayscale và chuyển sang RGB (3 kênh giống nhau)
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")

        # Đọc mask (grayscale, 0 hoặc 255)
        msk = Image.open(os.path.join(self.msk_dir, fname)).convert("L")

        # Áp dụng transform
        img = self.img_transform(img)
        msk = self.msk_transform(msk)   # giá trị sẽ là 0.0 hoặc 1.0

        return img, msk, fname


def get_dataloaders(data_root="data", batch_size=32, num_workers=4):
    """
    Tạo DataLoader cho train, val, test.

    Returns:
        dict với key 'train', 'val', 'test' → DataLoader
    """
    loaders = {}

    for split, shuffle, augment in [
        ("train_2d", True, True),
        ("val_2d", False, False),
        ("test_2d", False, False),
    ]:
        data_dir = os.path.join(data_root, split)
        if not os.path.exists(os.path.join(data_dir, "images")):
            print(f"[CẢNH BÁO] Thư mục {data_dir}/images không tồn tại. Bỏ qua.")
            continue

        dataset = LeftAtriumDataset(data_dir, augment=augment)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train_2d"),  # drop_last chỉ cho train
        )
        # Tên key ngắn gọn: "train_2d" → "train"
        key = split.replace("_2d", "")
        loaders[key] = loader
        print(f"  [{key}] {len(dataset)} samples, {len(loader)} batches (batch_size={batch_size})")

    return loaders


if __name__ == "__main__":
    # Chạy thử để kiểm tra
    print("Kiểm tra DataLoader...")
    loaders = get_dataloaders(batch_size=4, num_workers=0)

    for key, loader in loaders.items():
        batch = next(iter(loader))
        imgs, msks, fnames = batch
        print(f"\n  {key}:")
        print(f"    images shape: {imgs.shape}, dtype: {imgs.dtype}")
        print(f"    masks  shape: {msks.shape}, dtype: {msks.dtype}")
        print(f"    masks  unique: {torch.unique(msks)}")
        print(f"    filenames: {fnames[:2]}")
