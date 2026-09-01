#!/usr/bin/env python3
"""Create a reproducible UTAH patient split by linking existing PNG slices.

This never modifies the source processed dataset. It produces a staging tree
whose train_2d/val_2d/test_2d folders contain symlinks to source PNG files,
then exposes the layout expected by ``src/dataset.py`` through processed/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def split_patients(patient_ids: list[str], seed: int) -> dict[str, list[str]]:
    count = len(patient_ids)
    val_count = round(count * 0.10)
    test_count = round(count * 0.20)
    shuffled = patient_ids.copy()
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[: count - val_count - test_count]),
        "val": sorted(shuffled[count - val_count - test_count : count - test_count]),
        "test": sorted(shuffled[count - test_count :]),
    }


def link_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    os.symlink(source, destination)


def main(args: argparse.Namespace) -> None:
    source = args.source_processed.resolve()
    images_root, masks_root = source / "images", source / "masks"
    if not images_root.is_dir() or not masks_root.is_dir():
        raise FileNotFoundError("Expected source_processed/images and source_processed/masks.")

    image_patients = {path.name for path in images_root.iterdir() if path.is_dir()}
    mask_patients = {path.name for path in masks_root.iterdir() if path.is_dir()}
    if image_patients != mask_patients or len(image_patients) != 154:
        raise RuntimeError("Expected exactly 154 matched UTAH patient directories.")
    splits = split_patients(sorted(image_patients), args.seed)
    if {name: len(ids) for name, ids in splits.items()} != {"train": 108, "val": 15, "test": 31}:
        raise RuntimeError("Unexpected patient split counts.")

    output = args.output_root.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite only for this staging tree.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest: dict[str, object] = {
        "dataset": "2018 UTAH MICCAI",
        "split_seed": args.seed,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "source_processed": str(source),
        "patients": splits,
        "slice_counts": {},
    }
    splits_dir = output / "splits"
    splits_dir.mkdir()
    for split, patient_ids in splits.items():
        (splits_dir / f"{split}_patients.txt").write_text("\n".join(patient_ids) + "\n", encoding="utf-8")
        images_out, masks_out = output / f"{split}_2d" / "images", output / f"{split}_2d" / "masks"
        images_out.mkdir(parents=True)
        masks_out.mkdir(parents=True)
        slice_count = 0
        for patient_id in patient_ids:
            images = sorted((images_root / patient_id).glob("*.png"))
            masks = sorted((masks_root / patient_id).glob("*.png"))
            if not images or [path.name for path in images] != [path.name for path in masks]:
                raise RuntimeError(f"Mismatched image/mask slices for {patient_id}.")
            for slice_index, (image, mask) in enumerate(zip(images, masks, strict=True)):
                # Standardize the name independently of the source layout so
                # every flattened filename remains unambiguous and the 3D
                # evaluator can reconstruct patient volumes deterministically.
                filename = f"{patient_id}_slice{slice_index:03d}.png"
                link_file(image, images_out / filename)
                link_file(mask, masks_out / filename)
                slice_count += 1
        manifest["slice_counts"][split] = slice_count

    processed = output / "processed"
    processed.mkdir()
    for split in splits:
        os.symlink(f"../{split}_2d", processed / split)
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build linked 70/10/20 UTAH split staging tree.")
    parser.add_argument("--source-processed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
