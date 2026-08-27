"""Dataset loader for the patient-level, all-slice processed MRI dataset."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMG_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeftAtriumDataset(Dataset):
    """One split of axial slices; empty-mask slices are deliberately retained."""

    def __init__(self, split_dir: str | Path) -> None:
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/ and masks/ under {self.split_dir.resolve()}."
            )

        image_names = {path.name for path in self.images_dir.glob("*.png")}
        mask_names = {path.name for path in self.masks_dir.glob("*.png")}
        if not image_names:
            raise RuntimeError(f"No PNG slices found in {self.images_dir}.")
        if image_names != mask_names:
            raise RuntimeError(
                "Image and mask names differ. "
                f"Missing masks: {sorted(image_names - mask_names)[:5]}; "
                f"missing images: {sorted(mask_names - image_names)[:5]}."
            )
        self.filenames = sorted(image_names)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[index]
        image = Image.open(self.images_dir / filename).convert("L").convert("RGB")
        mask = Image.open(self.masks_dir / filename).convert("L")
        return self.image_transform(image), self.mask_transform(mask), filename


def get_dataloaders(
    data_root: str | Path = "data/processed",
    batch_size: int = 16,
    num_workers: int = 4,
    splits: Iterable[str] = ("train", "val"),
) -> dict[str, DataLoader]:
    """Create only the requested loaders; training never needs the test split."""
    root = Path(data_root)
    requested = tuple(splits)
    unknown = set(requested) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")

    loaders: dict[str, DataLoader] = {}
    for split in requested:
        dataset = LeftAtriumDataset(root / split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
        print(f"  [{split}] {len(dataset)} slices, {len(loaders[split])} batches")
    return loaders
