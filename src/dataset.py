"""
dataset.py
----------
PyTorch Dataset cho dữ liệu lát cắt 2D đã được chuẩn bị.
Xử lý tiền xử lý theo yêu cầu của DINOv2:
  - Chế độ 2D: Chuyển ảnh Grayscale → RGB (3 kênh giống nhau)
  - Chế độ 2.5D: Ghép 3 lát cắt liên tiếp [z-1, z, z+1] vào 3 kênh RGB
  - Resize về 448x448
  - Chuẩn hoá theo ImageNet mean/std
"""

import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ────────────────────────────────────────────────────────────
# Cấu hình
# ────────────────────────────────────────────────────────────
IMG_SIZE = 448  # DINOv2 patch_size=14 → 448/14 = 32 patches moi chieu

# Chuan hoa theo ImageNet (DINOv2 duoc huan luyen tren ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _parse_slice_info(fname):
    """
    Phan tich ten file de lay ten benh nhan va so thu tu lat cat.
    Vi du: 'la_003_slice0045.png' → ('la_003_slice', 45, '.png')
    """
    m = re.match(r"^(.+_slice)(\d+)(\..+)$", fname)
    if m:
        prefix = m.group(1)    # 'la_003_slice'
        num = int(m.group(2))  # 45
        ext = m.group(3)       # '.png'
        return prefix, num, ext
    return None, None, None


class LeftAtriumDataset(Dataset):
    """
    Dataset cho phan vung tam nhi trai.

    Che do 2D (mode_2_5d=False):
        - image: tensor (3, 448, 448) - Lat cat z nhan ban 3 kenh
    Che do 2.5D (mode_2_5d=True):
        - image: tensor (3, 448, 448) - [z-1, z, z+1] theo 3 kenh RGB
        → Mo hinh nhin thay ngu canh khong gian 3D!
    - mask:  tensor (1, 448, 448) nhi phan {0, 1}
    """

    def __init__(self, data_dir, augment=False, mode_2_5d=False):
        """
        Args:
            data_dir: duong dan toi thu muc (vd: 'data/train_2d')
                      ben trong phai co 'images/' va 'masks/'
            augment: co ap dung data augmentation hay khong
            mode_2_5d: neu True, ghep [z-1, z, z+1] vao 3 kenh RGB
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.msk_dir = os.path.join(data_dir, "masks")
        self.augment = augment
        self.mode_2_5d = mode_2_5d

        # Chi lay nhung file ton tai o ca 2 thu muc images va masks
        img_files = set(os.listdir(self.img_dir))
        msk_files = set(os.listdir(self.msk_dir)) if os.path.exists(self.msk_dir) else set()
        common_files = img_files.intersection(msk_files)
        self.filenames = sorted(list(common_files if len(common_files) > 0 else img_files))

        # Lookup nhanh: ten file co ton tai trong thu muc images khong (cho 2.5D)
        self._img_file_set = img_files

        # Transform resize rieng (dung cho tung kenh grayscale trong 2.5D)
        self.resize_pil = transforms.Resize(
            (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR
        )

        # Transform day du cho anh RGB (dung trong che do 2D)
        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Transform cho mask (dung NEAREST de khong tao gia tri trung gian)
        self.msk_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # → [0, 1], shape (1, H, W)
        ])

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.filenames)

    def _load_gray_resized(self, fname):
        """Doc anh grayscale, resize va tra ve tensor (H, W) voi gia tri [0, 1]."""
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("L")
        img = self.resize_pil(img)
        return transforms.ToTensor()(img).squeeze(0)  # (H, W)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        if self.mode_2_5d:
            # ── Che do 2.5D: ghep [z-1, z, z+1] vao 3 kenh RGB ──
            prefix, z, ext = _parse_slice_info(fname)

            if prefix is not None:
                # Ten file lat cat truoc va sau
                fname_prev = f"{prefix}{z - 1:04d}{ext}"
                fname_next = f"{prefix}{z + 1:04d}{ext}"

                # Neu lat bien khong ton tai, lay lat hien tai thay the (padding)
                slice_curr = self._load_gray_resized(fname)
                slice_prev = self._load_gray_resized(fname_prev) if fname_prev in self._img_file_set else slice_curr
                slice_next = self._load_gray_resized(fname_next) if fname_next in self._img_file_set else slice_curr

                # Stack theo 3 kenh: (3, H, W)
                img = torch.stack([slice_prev, slice_curr, slice_next], dim=0)
            else:
                # Fallback: khong phan tich duoc ten file, dung che do 2D
                raw = Image.open(os.path.join(self.img_dir, fname)).convert("L")
                raw = self.resize_pil(raw)
                t = transforms.ToTensor()(raw).squeeze(0)
                img = torch.stack([t, t, t], dim=0)

            # Chuan hoa theo ImageNet
            img = self.normalize(img)

        else:
            # ── Che do 2D: nhan ban grayscale thanh RGB ──
            img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
            img = self.img_transform(img)

        # Doc mask
        msk = Image.open(os.path.join(self.msk_dir, fname)).convert("L")
        msk = self.msk_transform(msk)  # (1, H, W), gia tri 0.0 hoac 1.0

        return img, msk, fname


def get_dataloaders(data_root="data", batch_size=32, num_workers=4, mode_2_5d=False):
    """
    Tao DataLoader cho train, val, test.

    Args:
        mode_2_5d: neu True, dung che do dau vao 2.5D (ghep 3 lat cat lien tiep)
    Returns:
        dict voi key 'train', 'val', 'test' → DataLoader
    """
    loaders = {}

    for split, shuffle, augment in [
        ("train_2d", True, True),
        ("val_2d", False, False),
        ("test_2d", False, False),
    ]:
        data_dir = os.path.join(data_root, split)
        if not os.path.exists(os.path.join(data_dir, "images")):
            print(f"  [CANH BAO] Thu muc {data_dir}/images khong ton tai. Bo qua.")
            continue

        dataset = LeftAtriumDataset(data_dir, augment=augment, mode_2_5d=mode_2_5d)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train_2d"),
        )
        key = split.replace("_2d", "")
        mode_label = "2.5D" if mode_2_5d else "2D"
        loaders[key] = loader
        print(f"  [{key}][{mode_label}] {len(dataset)} samples, {len(loader)} batches (batch_size={batch_size})")

    return loaders


if __name__ == "__main__":
    print("Kiem tra DataLoader 2D...")
    loaders_2d = get_dataloaders(batch_size=4, num_workers=0, mode_2_5d=False)
    for key, loader in loaders_2d.items():
        imgs, msks, fnames = next(iter(loader))
        print(f"  [{key}] images: {imgs.shape}, masks: {msks.shape}")

    print("\nKiem tra DataLoader 2.5D...")
    loaders_25d = get_dataloaders(batch_size=4, num_workers=0, mode_2_5d=True)
    for key, loader in loaders_25d.items():
        imgs, msks, fnames = next(iter(loader))
        print(f"  [{key}] images: {imgs.shape}, masks: {msks.shape}")
