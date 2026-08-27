"""Build the reproducible patient-level dataset required by E0--E4.

Only labelled NIfTI volumes in imagesTr/ and labelsTr/ are split.  Every
axial slice is exported, including slices whose left-atrium mask is empty, so
test predictions can be reconstructed into complete 3D volumes.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


def nifti_id(path: Path) -> str:
    """Return the patient ID while ignoring macOS AppleDouble metadata."""
    if path.name.startswith("._"):
        return ""
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.suffix == ".nii":
        return path.stem
    return ""


def indexed_volumes(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory.resolve()}")
    volumes: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        patient_id = nifti_id(path)
        if patient_id:
            volumes[patient_id] = path
    return volumes


def labelled_patients(raw_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images = indexed_volumes(raw_dir / "imagesTr")
    labels = indexed_volumes(raw_dir / "labelsTr")
    if not images:
        raise RuntimeError(f"No labelled MRI volumes found in {raw_dir / 'imagesTr'}.")
    if images.keys() != labels.keys():
        only_images = sorted(images.keys() - labels.keys())
        only_labels = sorted(labels.keys() - images.keys())
        raise RuntimeError(
            "imagesTr and labelsTr do not contain the same patient IDs. "
            f"Only images: {only_images[:5]}; only labels: {only_labels[:5]}."
        )
    return images, labels


def split_patients(patient_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Create an exact 70/10/20 split when integer counts allow it."""
    count = len(patient_ids)
    if count < 3:
        raise ValueError("At least three labelled patients are required.")

    test_count = max(1, round(count * SPLIT_RATIOS["test"]))
    val_count = max(1, round(count * SPLIT_RATIOS["val"]))
    train_count = count - val_count - test_count
    if train_count < 1:
        raise ValueError(f"Cannot create a non-empty train split from {count} patients.")

    shuffled = patient_ids.copy()
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count :]),
    }


def normalize_volume(image: np.ndarray) -> np.ndarray:
    """Robustly scale one MRI volume to uint8 using its non-zero intensities."""
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Volume has no finite intensity values.")
    foreground = finite[np.abs(finite) > 1e-8]
    reference = foreground if foreground.size else finite
    low, high = np.percentile(reference, (1.0, 99.0))
    if high <= low:
        low, high = float(reference.min()), float(reference.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    return np.rint(np.clip((image - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def export_patient(
    patient_id: str,
    image_path: Path,
    label_path: Path,
    destination: Path,
) -> dict[str, object]:
    """Export all axial image/mask slice pairs for one patient."""
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)
    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError(
            f"{patient_id}: image {image.shape} and label {label.shape} must be matching 3D arrays."
        )

    image_u8 = normalize_volume(image)
    mask_u8 = (label > 0).astype(np.uint8) * 255
    image_dir = destination / "images"
    mask_dir = destination / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for z_index in range(image.shape[2]):
        filename = f"{patient_id}_slice{z_index:04d}.png"
        Image.fromarray(image_u8[:, :, z_index]).save(image_dir / filename)
        Image.fromarray(mask_u8[:, :, z_index]).save(mask_dir / filename)

    height, width, depth = image.shape
    spacing_x, spacing_y, spacing_z = image_nii.header.get_zooms()[:3]
    return {
        "original_shape_hwd": [int(height), int(width), int(depth)],
        "exported_slices": int(depth),
        # Stacked output volumes are ordered (z, h, w); h/w are resized to 448 at load time.
        "resampled_spacing_zhw_mm": [
            float(spacing_z),
            float(spacing_x * height / 448.0),
            float(spacing_y * width / 448.0),
        ],
    }


def remove_if_requested(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path.resolve()} already exists. Use --overwrite to regenerate this derived data."
            )
        shutil.rmtree(path)


def main(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    splits_dir = Path(args.splits_dir)
    images, labels = labelled_patients(raw_dir)
    splits = split_patients(sorted(images), args.seed)

    remove_if_requested(processed_dir, args.overwrite)
    remove_if_requested(splits_dir, args.overwrite)
    processed_dir.mkdir(parents=True, exist_ok=False)
    splits_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "seed": args.seed,
        "split_ratio": SPLIT_RATIOS,
        "input_size": [448, 448],
        "labelled_patient_count": len(images),
        "unlabelled_imagesTs_count": len(indexed_volumes(raw_dir / "imagesTs"))
        if (raw_dir / "imagesTs").is_dir()
        else 0,
        "patients": {},
    }

    for split, patient_ids in splits.items():
        (splits_dir / f"{split}_patients.txt").write_text(
            "\n".join(patient_ids) + "\n", encoding="utf-8"
        )
        print(f"{split}: {len(patient_ids)} patients")
        for patient_id in tqdm(patient_ids, desc=f"Export {split}"):
            metadata = export_patient(
                patient_id,
                images[patient_id],
                labels[patient_id],
                processed_dir / split,
            )
            manifest["patients"][patient_id] = {"split": split, **metadata}

    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Completed. Processed slices: {processed_dir.resolve()}")
    print(f"Fixed patient lists: {splits_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create patient-level 70/10/20 splits and export all MRI slices."
    )
    parser.add_argument("--raw-dir", default="data/raw/Task02_Heart")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and regenerate only the derived processed/ and splits/ directories.",
    )
    main(parser.parse_args())
