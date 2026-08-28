#!/usr/bin/env python3
"""Build a self-contained E0--E2 handoff report without retraining.

This script is intentionally separate from training: it loads immutable best
checkpoints, reconstructs each test patient's 3D volume from ordered 2D
slices, and writes only to an external report directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


PHASES = (
    ("e0", "Frozen + BCE"),
    ("e1", "Frozen + BCE+Dice"),
    ("e2", "Full FT"),
)
FILENAME_PATTERN = re.compile(r"^(?P<patient>.+)_slice(?P<slice>\d+)\.png$")
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
MASK_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ]
)


def csv_write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def normalize_legacy_slice(image: np.ndarray) -> np.ndarray:
    """Match the per-slice min/max normalization used in commit fa2970b."""
    image = np.asarray(image, dtype=np.float64)
    low, high = image.min(), image.max()
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - low) / (high - low) * 255.0).astype(np.uint8)


def patient_slices(data_root: Path, split: str) -> dict[str, list[tuple[int, str]]]:
    image_dir = data_root / f"{split}_2d" / "images"
    grouped: dict[str, list[tuple[int, str]]] = {}
    for image_path in image_dir.glob("*.png"):
        match = FILENAME_PATTERN.match(image_path.name)
        if match is None:
            raise ValueError(f"Unexpected slice filename: {image_path.name}")
        grouped.setdefault(match["patient"], []).append((int(match["slice"]), image_path.name))
    if not grouped:
        raise FileNotFoundError(f"No PNG slices found in {image_dir}")
    for values in grouped.values():
        values.sort()
    return grouped


def write_patient_ids(data_root: Path, report_dir: Path) -> dict[str, list[str]]:
    ids_by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        grouped = patient_slices(data_root, split)
        patient_ids = sorted(grouped)
        ids_by_split[split] = patient_ids
        target = report_dir / "patient_ids" / f"{split}_patient_ids.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(patient_ids) + "\n", encoding="utf-8")
    return ids_by_split


def raw_spacing(raw_dir: Path, patient_id: str) -> tuple[float, float, float]:
    raw_label = raw_dir / "labelsTr" / f"{patient_id}.nii.gz"
    if not raw_label.is_file():
        raise FileNotFoundError(f"Missing raw label for spacing: {raw_label}")
    nii = nib.load(raw_label)
    height, width, _ = nii.shape[:3]
    spacing_x, spacing_y, spacing_z = nii.header.get_zooms()[:3]
    # Predictions and masks are both resized to 448×448 before metric calculation.
    return float(spacing_z), float(spacing_x * height / 448), float(spacing_y * width / 448)


def load_model(repo_root: Path, checkpoint_path: Path, device: torch.device):
    from model import DINOv2Segmenter

    model = DINOv2Segmenter(model_name="vit_large").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_phase(
    repo_root: Path,
    data_root: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    phase: str,
    method: str,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    from metrics import volume_metrics
    # The original run's PNG folders contain only the labelled ROI range.  A
    # valid 3D metric must include every axial slice, so rebuild input tensors
    # in memory from the original test NIfTI volumes with the same transforms.
    test_patients = sorted(patient_slices(data_root, "test"))
    model = load_model(repo_root, checkpoint_path, device)

    rows: list[dict[str, object]] = []
    for patient_id in test_patients:
        image_path = raw_dir / "imagesTr" / f"{patient_id}.nii.gz"
        label_path = raw_dir / "labelsTr" / f"{patient_id}.nii.gz"
        image_volume = nib.load(image_path).get_fdata(dtype=np.float32)
        label_volume = nib.load(label_path).get_fdata(dtype=np.float32)
        if image_volume.ndim != 3 or image_volume.shape != label_volume.shape:
            raise ValueError(f"{patient_id}: incompatible NIfTI shapes {image_volume.shape} / {label_volume.shape}")

        all_predictions: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        for start in range(0, image_volume.shape[2], batch_size):
            end = min(start + batch_size, image_volume.shape[2])
            image_batch = []
            target_batch = []
            for z_index in range(start, end):
                image_u8 = normalize_legacy_slice(image_volume[:, :, z_index])
                target_u8 = ((label_volume[:, :, z_index] > 0).astype(np.uint8) * 255)
                image_batch.append(IMAGE_TRANSFORM(Image.fromarray(image_u8).convert("RGB")))
                target_batch.append(MASK_TRANSFORM(Image.fromarray(target_u8).convert("L")))
            logits = model(torch.stack(image_batch).to(device, non_blocking=True))
            all_predictions.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(bool)[:, 0])
            all_targets.extend(torch.stack(target_batch).numpy().astype(bool)[:, 0])

        prediction_volume = np.stack(all_predictions)
        target_volume = np.stack(all_targets)
        spacing_zhw = raw_spacing(raw_dir, patient_id)
        metrics = volume_metrics(prediction_volume, target_volume, spacing=spacing_zhw)
        rows.append(
            {
                "phase": phase.upper(),
                "method": method,
                "patient_id": patient_id,
                "num_slices": image_volume.shape[2],
                "dice_3d": metrics["dice_3d"],
                "iou_3d": metrics["iou_3d"],
                "hd95_mm": metrics["hd95"],
                "spacing_z_mm": spacing_zhw[0],
                "spacing_y_mm": spacing_zhw[1],
                "spacing_x_mm": spacing_zhw[2],
                "checkpoint": str(checkpoint_path),
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def summary_row(
    phase: str,
    method: str,
    patient_rows: list[dict[str, object]],
    efficiency: dict[str, object],
) -> dict[str, object]:
    def stats(field: str) -> tuple[float, float, float, float]:
        values = np.asarray([float(row[field]) for row in patient_rows], dtype=float)
        return float(np.nanmean(values)), float(np.nanstd(values)), float(np.nanmin(values)), float(np.nanmax(values))

    dice_mean, dice_std, dice_min, dice_max = stats("dice_3d")
    iou_mean, iou_std, iou_min, iou_max = stats("iou_3d")
    hd95_mean, hd95_std, hd95_min, hd95_max = stats("hd95_mm")
    train_seconds = float(efficiency["total_train_time_s"])
    return {
        "phase": phase.upper(),
        "method": method,
        "test_patients": len(patient_rows),
        "dice_3d_mean": dice_mean,
        "dice_3d_sd": dice_std,
        "dice_3d_min": dice_min,
        "dice_3d_max": dice_max,
        "iou_3d_mean": iou_mean,
        "iou_3d_sd": iou_std,
        "iou_3d_min": iou_min,
        "iou_3d_max": iou_max,
        "hd95_mm_mean": hd95_mean,
        "hd95_mm_sd": hd95_std,
        "hd95_mm_min": hd95_min,
        "hd95_mm_max": hd95_max,
        "trainable_params": int(efficiency["trainable_params"]),
        "total_params": int(efficiency["total_params"]),
        "vram_gb": float(efficiency["vram_gb"]),
        "train_time_s": train_seconds,
        "train_time_min": train_seconds / 60.0,
        "inference_ms_per_image": float(efficiency["inference_ms_per_image"]),
        "best_validation_dice": float(efficiency["best_val_dice"]),
        "best_checkpoint": str(efficiency["checkpoint"]),
    }


def write_summary_markdown(path: Path, rows: list[dict[str, object]], commit: str) -> None:
    lines = [
        "# E0–E2 Delivery Results",
        "",
        f"- Training code commit: `{commit}`",
        "- Test protocol: reconstruct each patient from all ordered axial 2D slices.",
        "- Dice, IoU and HD95 are reported as mean ± population SD across test patients.",
        "- HD95 uses millimetres after accounting for the 448×448 in-plane resize.",
        "",
        "| Method | 3D Dice | 3D IoU | HD95 (mm) | Trainable Params | VRAM (GB) | Train Time (min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {dice_3d_mean:.4f} ± {dice_3d_sd:.4f} | "
            "{iou_3d_mean:.4f} ± {iou_3d_sd:.4f} | "
            "{hd95_mm_mean:.2f} ± {hd95_mm_sd:.2f} | "
            "{trainable_params:,} | {vram_gb:.3f} | {train_time_min:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Only E0–E2 are in scope; Partial FT and LoRA were not run.",
            "Per-patient values are in `metrics/patient_metrics.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_artifacts(repo_root: Path, report_dir: Path, commit: str, patient_ids: dict[str, list[str]]) -> None:
    subprocess.run(
        ["git", "-C", str(repo_root), "archive", "--format=tar.gz", "-o", str(report_dir / f"code_{commit[:8]}.tar.gz"), commit],
        check=True,
    )
    copy_file(repo_root / "logs" / "la-e0-e2-full.log", report_dir / "training_logs" / "la-e0-e2-full.log")
    for phase, _ in PHASES:
        phase_dir = repo_root / "results" / "vit_large" / phase
        copy_file(phase_dir / "checkpoints" / "best.pth", report_dir / "best_checkpoints" / f"{phase}_best.pth")
        visualization_dir = phase_dir / "visualizations"
        shutil.copytree(visualization_dir, report_dir / "predictions" / phase)
    provenance = [
        "# Delivery provenance",
        "",
        f"- Training source commit: `{commit}`",
        "- Checkpoints are copies of the immutable E0–E2 `best.pth` files.",
        "- No raw MRI data is included in this handoff folder.",
        "- Actual patient split used for the full experiment:",
    ]
    for split, ids in patient_ids.items():
        provenance.append(f"  - {split}: {', '.join(ids)}")
    (report_dir / "PROVENANCE.md").write_text("\n".join(provenance) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create E0–E2 delivery artifacts without retraining.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    report_dir = args.report_dir.resolve()
    data_root = (args.data_root or repo_root / "data").resolve()
    raw_dir = (args.raw_dir or data_root / "raw_3d").resolve()
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty report directory: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root / "src"))

    patient_ids = write_patient_ids(data_root, report_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_patient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    efficiency_rows: list[dict[str, object]] = []

    for phase, method in PHASES:
        phase_dir = repo_root / "results" / "vit_large" / phase
        checkpoint_path = phase_dir / "checkpoints" / "best.pth"
        efficiency_paths = sorted(phase_dir.glob("efficiency*.json"))
        if not checkpoint_path.is_file() or len(efficiency_paths) != 1:
            raise FileNotFoundError(f"Missing full-run checkpoint or efficiency JSON for {phase}")
        patient_rows = evaluate_phase(
            repo_root, data_root, raw_dir, checkpoint_path, phase, method, device, args.batch_size
        )
        all_patient_rows.extend(patient_rows)
        efficiency = json.loads(efficiency_paths[0].read_text(encoding="utf-8"))
        summary_rows.append(summary_row(phase, method, patient_rows, efficiency))
        efficiency_rows.append({"phase": phase.upper(), "method": method, **efficiency})

    patient_fields = list(all_patient_rows[0])
    efficiency_fields = list(efficiency_rows[0])
    summary_fields = list(summary_rows[0])
    csv_write(report_dir / "metrics" / "patient_metrics.csv", all_patient_rows, patient_fields)
    csv_write(report_dir / "metrics" / "efficiency_metrics.csv", efficiency_rows, efficiency_fields)
    csv_write(report_dir / "metrics" / "results_summary.csv", summary_rows, summary_fields)
    write_summary_markdown(report_dir / "RESULTS_SUMMARY.md", summary_rows, args.commit)
    copy_artifacts(repo_root, report_dir, args.commit, patient_ids)
    print(f"Delivery report written to: {report_dir}")


if __name__ == "__main__":
    main()
