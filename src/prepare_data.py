"""Create reproducible, patient-level 2D data for left-atrium segmentation.

Supported inputs are either a labelled NIfTI ``imagesTr/`` + ``labelsTr/``
layout or the 2018 UTAH MICCAI NRRD layout. All labelled patients are split
once at the patient level, then every axial slice (including empty masks) is
exported as a matching PNG image/mask pair. Keeping empty slices is required
to reconstruct and evaluate complete 3D volumes later.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
UTAH_REQUIRED_FILES = ("lgemri.nrrd", "laendo.nrrd")


@dataclass(frozen=True)
class PatientVolume:
    """One labelled volume and enough provenance to audit the split."""

    image_path: Path
    label_path: Path
    source_group: str


def nifti_id(path: Path) -> str:
    """Return a patient ID while ignoring macOS AppleDouble metadata."""
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
        if path.is_file():
            patient_id = nifti_id(path)
            if patient_id:
                volumes[patient_id] = path
    return volumes


def nifti_patients(raw_dir: Path) -> dict[str, PatientVolume]:
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
    return {
        patient_id: PatientVolume(images[patient_id], labels[patient_id], "imagesTr")
        for patient_id in images
    }


def utah_patients(raw_dir: Path) -> dict[str, PatientVolume]:
    """Read all labelled UTAH patients from both official source folders."""
    patients: dict[str, PatientVolume] = {}
    for source_group in ("Training Set", "Testing Set"):
        source_dir = raw_dir / source_group
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing UTAH directory: {source_dir.resolve()}")
        for patient_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            missing = [name for name in UTAH_REQUIRED_FILES if not (patient_dir / name).is_file()]
            if missing:
                raise RuntimeError(f"{patient_dir}: missing required files {missing}.")
            patient_id = patient_dir.name
            if patient_id in patients:
                raise RuntimeError(
                    f"Duplicate patient ID {patient_id!r} occurs in both UTAH source folders."
                )
            patients[patient_id] = PatientVolume(
                patient_dir / "lgemri.nrrd",
                patient_dir / "laendo.nrrd",
                source_group,
            )
    if not patients:
        raise RuntimeError(f"No labelled UTAH patient folders found in {raw_dir.resolve()}.")
    return patients


def detect_format(raw_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (raw_dir / "Training Set").is_dir() or (raw_dir / "Testing Set").is_dir():
        return "utah2018"
    return "nifti"


def labelled_patients(raw_dir: Path, dataset_format: str) -> tuple[str, dict[str, PatientVolume]]:
    detected_format = detect_format(raw_dir, dataset_format)
    if detected_format == "nifti":
        return detected_format, nifti_patients(raw_dir)
    if detected_format == "utah2018":
        return detected_format, utah_patients(raw_dir)
    raise ValueError(f"Unsupported dataset format: {detected_format}")


def split_patients(patient_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Create a deterministic 70/10/20 split with every patient in one split."""
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


def nrrd_volume(path: Path) -> tuple[np.ndarray, list[float]]:
    """Load one NRRD volume in its stored x/y/z order and obtain voxel spacing."""
    try:
        import nrrd
    except ImportError as error:
        raise RuntimeError(
            "UTAH NRRD input needs pynrrd. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    volume, header = nrrd.read(str(path), index_order="F")
    directions = header.get("space directions")
    if directions is not None and len(directions) >= 3:
        spacing = [float(np.linalg.norm(direction)) for direction in directions[:3]]
    else:
        spacing = [float(value) for value in header.get("spacings", (1.0, 1.0, 1.0))[:3]]
    if len(spacing) != 3 or any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError(f"{path}: invalid NRRD voxel spacing {spacing}.")
    return np.asarray(volume), spacing


def load_patient_volume(patient: PatientVolume, dataset_format: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if dataset_format == "nifti":
        image_nii = nib.load(patient.image_path)
        label_nii = nib.load(patient.label_path)
        return (
            image_nii.get_fdata(dtype=np.float32),
            label_nii.get_fdata(dtype=np.float32),
            [float(value) for value in image_nii.header.get_zooms()[:3]],
        )
    image, image_spacing = nrrd_volume(patient.image_path)
    label, label_spacing = nrrd_volume(patient.label_path)
    if not np.allclose(image_spacing, label_spacing, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"{patient.image_path.parent}: image spacing {image_spacing} and label spacing "
            f"{label_spacing} differ."
        )
    return image.astype(np.float32, copy=False), label.astype(np.float32, copy=False), image_spacing


def export_patient(
    patient_id: str,
    patient: PatientVolume,
    dataset_format: str,
    destination: Path,
) -> dict[str, object]:
    """Export all axial image/mask slice pairs for one patient."""
    image, label, spacing_xyz = load_patient_volume(patient, dataset_format)
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
    spacing_x, spacing_y, spacing_z = spacing_xyz
    return {
        "source_group": patient.source_group,
        "image_path": str(patient.image_path),
        "label_path": str(patient.label_path),
        "original_shape_hwd": [int(height), int(width), int(depth)],
        "original_spacing_xyz_mm": [spacing_x, spacing_y, spacing_z],
        "exported_slices": int(depth),
        "resampled_spacing_zhw_mm": [
            spacing_z,
            spacing_x * height / 448.0,
            spacing_y * width / 448.0,
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
    dataset_format, patients = labelled_patients(raw_dir, args.dataset_format)
    splits = split_patients(sorted(patients), args.seed)

    print(f"Dataset format: {dataset_format}")
    print(f"Labelled patients: {len(patients)}")
    for split, patient_ids in splits.items():
        print(f"{split}: {len(patient_ids)} patients")
    if args.dry_run:
        print("Dry run completed; no files were written or removed.")
        return

    remove_if_requested(processed_dir, args.overwrite)
    remove_if_requested(splits_dir, args.overwrite)
    processed_dir.mkdir(parents=True, exist_ok=False)
    splits_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, object] = {
        "dataset_format": dataset_format,
        "raw_dir": str(raw_dir),
        "label_definition": "laendo.nrrd > 0" if dataset_format == "utah2018" else "label > 0",
        "seed": args.seed,
        "split_ratio": SPLIT_RATIOS,
        "split_counts": {split: len(ids) for split, ids in splits.items()},
        "input_size": [448, 448],
        "labelled_patient_count": len(patients),
        "patients": {},
    }

    for split, patient_ids in splits.items():
        (splits_dir / f"{split}_patients.txt").write_text(
            "\n".join(patient_ids) + "\n", encoding="utf-8"
        )
        for patient_id in tqdm(patient_ids, desc=f"Export {split}"):
            metadata = export_patient(
                patient_id,
                patients[patient_id],
                dataset_format,
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
    parser.add_argument("--raw-dir", default="data")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "nifti", "utah2018"),
        default="auto",
        help="Input layout; auto detects the UTAH 2018 folders.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and regenerate only the derived processed/ and splits/ directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and report deterministic split counts without writing files.",
    )
    main(parser.parse_args())
