"""Update the experiment-report branch after a completed E0/E1/E2 phase."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def latest_json(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No {pattern} found in {directory}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("E0", "E1", "E2"), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    test = json.loads(latest_json(args.results_dir, "test_results_*.json").read_text(encoding="utf-8"))
    efficiency = json.loads(latest_json(args.results_dir, "efficiency_*.json").read_text(encoding="utf-8"))
    volume = test["volume_3d"]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    elapsed_minutes = efficiency["total_train_time_s"] / 60

    section = f"""## {args.phase} — completed {timestamp}

| Field | Value |
|---|---:|
| Code commit | {args.commit} |
| Seed | {efficiency["seed"]} |
| Batch size | {efficiency.get("batch_size", "not recorded")} |
| DataLoader workers | {efficiency.get("num_workers", "not recorded")} |
| torch.compile | {efficiency.get("torch_compile", "not recorded")} |
| Test samples | {test["num_samples"]} |
| Test patients | {volume["num_patients"]} |
| 3D Dice (mean Â± SD) | {volume["dice_3d_mean"]:.4f} Â± {volume["dice_3d_std"]:.4f} |
| 3D IoU (mean Â± SD) | {volume["iou_3d_mean"]:.4f} Â± {volume["iou_3d_std"]:.4f} |
| HD95 mm (mean Â± SD) | {volume["hd95_mean"]:.4f} Â± {volume["hd95_std"]:.4f} |
| Dice (mean ± SD) | {test["dice_mean"]:.4f} ± {test["dice_std"]:.4f} |
| Jaccard / IoU (mean ± SD) | {test["iou_mean"]:.4f} ± {test["iou_std"]:.4f} |
| Dice range | {test["dice_min"]:.4f} – {test["dice_max"]:.4f} |
| IoU range | {test["iou_min"]:.4f} – {test["iou_max"]:.4f} |
| Best validation Dice / IoU | {efficiency["best_val_dice"]:.4f} / {efficiency["best_val_iou"]:.4f} |
| Peak VRAM | {efficiency["vram_gb"]:.3f} GB |
| Train time | {elapsed_minutes:.1f} min |
| Inference | {efficiency["inference_ms_per_image"]:.2f} ms/slice |

Artifacts remain on the training server at {args.results_dir}. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not args.report.exists():
        args.report.write_text(
            "# DINOv2 ViT-Large Experiment Progress\n\n"
            "Test metrics follow arXiv:2411.09598: Dice and Jaccard/IoU, mean ± SD.\n\n",
            encoding="utf-8",
        )
    with args.report.open("a", encoding="utf-8") as handle:
        handle.write(section)


if __name__ == "__main__":
    main()
