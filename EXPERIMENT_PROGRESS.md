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

## E1 — completed 2026-08-27 07:37 UTC

| Field | Value |
|---|---:|
| Code commit | fa2970b893d1d381c6b2aa76089de26e71dc5acd |
| Test samples | 263 |
| Dice (mean ± SD) | 0.7917 ± 0.2012 |
| Jaccard / IoU (mean ± SD) | 0.6909 ± 0.2189 |
| Dice range | 0.0000 – 0.9710 |
| IoU range | 0.0000 – 0.9436 |
| Best validation Dice / IoU | 0.8031 / 0.6885 |
| Peak VRAM | 1.985 GB |
| Train time | 13.1 min |
| Inference | 61.25 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e1. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E2 — completed 2026-08-27 08:23 UTC

| Field | Value |
|---|---:|
| Code commit | fa2970b893d1d381c6b2aa76089de26e71dc5acd |
| Test samples | 263 |
| Dice (mean ± SD) | 0.7931 ± 0.2289 |
| Jaccard / IoU (mean ± SD) | 0.7043 ± 0.2531 |
| Dice range | 0.0000 – 0.9785 |
| IoU range | 0.0000 – 0.9579 |
| Best validation Dice / IoU | 0.8132 / 0.7178 |
| Peak VRAM | 6.130 GB |
| Train time | 44.1 min |
| Inference | 79.99 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e2. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E0 — completed 2026-09-01 03:39 UTC

| Field | Value |
|---|---:|
| Code commit | c2d685cd96a032b82adcd317e96e336da393534c |
| Seed | 42 |
| Test samples | 2728 |
| Test patients | 31 |
| 3D Dice (mean Â± SD) | 0.8779 Â± 0.0349 |
| 3D IoU (mean Â± SD) | 0.7840 Â± 0.0542 |
| HD95 mm (mean Â± SD) | 6.6848 Â± 5.1614 |
| Dice (mean ± SD) | 0.8097 ± 0.2766 |
| Jaccard / IoU (mean ± SD) | 0.7460 ± 0.2889 |
| Dice range | 0.0000 – 1.0000 |
| IoU range | 0.0000 – 1.0000 |
| Best validation Dice / IoU | 0.7692 / 0.6915 |
| Peak VRAM | 1.985 GB |
| Train time | 84.3 min |
| Inference | 68.35 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e0/seed_42. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E1 — completed 2026-09-01 05:20 UTC

| Field | Value |
|---|---:|
| Code commit | c2d685cd96a032b82adcd317e96e336da393534c |
| Seed | 42 |
| Test samples | 2728 |
| Test patients | 31 |
| 3D Dice (mean Â± SD) | 0.8422 Â± 0.0384 |
| 3D IoU (mean Â± SD) | 0.7292 Â± 0.0556 |
| HD95 mm (mean Â± SD) | 8.3310 Â± 3.4516 |
| Dice (mean ± SD) | 0.7601 ± 0.2996 |
| Jaccard / IoU (mean ± SD) | 0.6850 ± 0.3042 |
| Dice range | 0.0000 – 1.0000 |
| IoU range | 0.0000 – 1.0000 |
| Best validation Dice / IoU | 0.6795 / 0.5910 |
| Peak VRAM | 1.985 GB |
| Train time | 98.4 min |
| Inference | 36.30 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e1/seed_42. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E2 — completed 2026-09-01 09:50 UTC

| Field | Value |
|---|---:|
| Code commit | c2d685cd96a032b82adcd317e96e336da393534c |
| Seed | 42 |
| Test samples | 2728 |
| Test patients | 31 |
| 3D Dice (mean Â± SD) | 0.8932 Â± 0.0268 |
| 3D IoU (mean Â± SD) | 0.8081 Â± 0.0433 |
| HD95 mm (mean Â± SD) | 11.3152 Â± 21.0404 |
| Dice (mean ± SD) | 0.8344 ± 0.2605 |
| Jaccard / IoU (mean ± SD) | 0.7752 ± 0.2708 |
| Dice range | 0.0000 – 1.0000 |
| IoU range | 0.0000 – 1.0000 |
| Best validation Dice / IoU | 0.8396 / 0.7784 |
| Peak VRAM | 6.130 GB |
| Train time | 266.7 min |
| Inference | 35.95 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e2/seed_42. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

## E0 — completed 2026-09-01 10:25 UTC

| Field | Value |
|---|---:|
| Code commit | c2d685cd96a032b82adcd317e96e336da393534c |
| Seed | 2026 |
| Test samples | 2728 |
| Test patients | 31 |
| 3D Dice (mean Â± SD) | 0.8777 Â± 0.0306 |
| 3D IoU (mean Â± SD) | 0.7833 Â± 0.0484 |
| HD95 mm (mean Â± SD) | 6.3305 Â± 3.9484 |
| Dice (mean ± SD) | 0.8175 ± 0.2656 |
| Jaccard / IoU (mean ± SD) | 0.7530 ± 0.2804 |
| Dice range | 0.0000 – 1.0000 |
| IoU range | 0.0000 – 1.0000 |
| Best validation Dice / IoU | 0.7802 / 0.7023 |
| Peak VRAM | 1.985 GB |
| Train time | 33.1 min |
| Inference | 36.33 ms/slice |

Artifacts remain on the training server at /mnt/data/users/trungptit/Left_atrium_segmentation/results/vit_large/e0/seed_2026. The DINOv2 paper reports Dice and Jaccard/IoU as mean ± SD; results are not directly comparable unless data split and protocol match.

