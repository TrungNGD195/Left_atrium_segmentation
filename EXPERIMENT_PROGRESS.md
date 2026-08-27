# DINOv2 ViT-Large Experiment Progress

This file is a template on main. Completed phase reports are generated on the experiment-reports branch by scripts/update_experiment_report.py.

Protocol follows arXiv:2411.09598: report test Dice and Jaccard/IoU as mean ± standard deviation. Checkpoints, raw data and logs remain on the training server.
## E0 — completed 2026-08-27 06:57 UTC

| Field | Value |
|---|---:|
| Code commit | fa2970b893d1d381c6b2aa76089de26e71dc5acd |
| Test samples | 263 |
| Dice (mean ± SD) | 0.5882 ± 0.3488 |
| Jaccard / IoU (mean ± SD) | 0.4924 ± 0.3094 |
| Dice range | 0.0000 – 0.9344 |
| IoU range | 0.0000 – 0.8769 |
| Best validation Dice / IoU | 0.6025 / 0.4997 |
| Peak VRAM | 1.985 GB |
| Train time | 0.4 min |
| Inference | 40.80 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/smoke/e0. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E1 — completed 2026-08-27 06:58 UTC

| Field | Value |
|---|---:|
| Code commit | fa2970b893d1d381c6b2aa76089de26e71dc5acd |
| Test samples | 263 |
| Dice (mean ± SD) | 0.6280 ± 0.2819 |
| Jaccard / IoU (mean ± SD) | 0.5103 ± 0.2592 |
| Dice range | 0.0000 – 0.9263 |
| IoU range | 0.0000 – 0.8628 |
| Best validation Dice / IoU | 0.6593 / 0.5244 |
| Peak VRAM | 1.985 GB |
| Train time | 0.3 min |
| Inference | 36.04 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/smoke/e1. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E2 — completed 2026-08-27 06:59 UTC

| Field | Value |
|---|---:|
| Code commit | fa2970b893d1d381c6b2aa76089de26e71dc5acd |
| Test samples | 263 |
| Dice (mean ± SD) | 0.6841 ± 0.2732 |
| Jaccard / IoU (mean ± SD) | 0.5733 ± 0.2611 |
| Dice range | 0.0000 – 0.9546 |
| IoU range | 0.0000 – 0.9131 |
| Best validation Dice / IoU | 0.6775 / 0.5634 |
| Peak VRAM | 6.130 GB |
| Train time | 1.2 min |
| Inference | 36.18 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/smoke/e2. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E0 — completed 2026-08-27 07:23 UTC

| Field | Value |
|---|---:|
| Code commit | fa2970b893d1d381c6b2aa76089de26e71dc5acd |
| Test samples | 263 |
| Dice (mean ± SD) | 0.7761 ± 0.2314 |
| Jaccard / IoU (mean ± SD) | 0.6798 ± 0.2456 |
| Dice range | 0.0000 – 0.9760 |
| IoU range | 0.0000 – 0.9531 |
| Best validation Dice / IoU | 0.7910 / 0.6743 |
| Peak VRAM | 1.985 GB |
| Train time | 9.4 min |
| Inference | 36.38 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e0. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

