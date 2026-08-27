# 🫀 Left Atrium Segmentation with DINOv2 ViT-Large

This repository trains a DINOv2 ViT-Large segmentation model for left-atrium MRI slices. The current, reproducible experiment scope is **E0 → E1 → E2**. GitHub `origin/main` is the only source of code; MRI data, checkpoints, logs and visualizations remain on the training server.

The paper [DINOv2-based Left Atrium Segmentation](docs/2411.09598v1.pdf) is a protocol reference. Its reported numbers are **not** results of this repository and must not be copied into project reports.

## Experiment protocol

| Phase | Encoder | Loss / method | Batch size | Purpose |
|---|---|---|---:|---|
| E0 | Frozen DINOv2 ViT-Large | BCE | 8 | Baseline |
| E1 | Frozen DINOv2 ViT-Large | BCE + Dice | 8 | Address class imbalance |
| E2 | Fully trainable DINOv2 ViT-Large | BCE + Dice | 1 | Full fine-tuning |

Each full phase runs for at most 50 epochs with early stopping patience 10. AMP is enabled. Each epoch replaces `last.pth`; a validation-Dice improvement creates `best.pth`; checkpoints are also saved every 5 epochs and at completion as `final.pth`. Restarting the runner resumes an unfinished phase from `last.pth`.

Official phase reports live on the [`experiment-reports`](../../tree/experiment-reports) branch. The `EXPERIMENT_PROGRESS.md` on `main` is intentionally only a template, so source code stays separate from server artifacts.

## Metrics

For the test set, report Dice and Jaccard/IoU as **mean ± population standard deviation over slices**, accompanied by minimum and maximum. The JSON evaluator records one Dice/IoU pair per slice. A comparison with the paper is valid only when dataset, patient split and evaluation protocol match.

## Repository structure

```text
Left_atrium_segmentation/
├── docs/
│   ├── 2411.09598v1.pdf
│   └── MÔ TẢ THỰC NGHIỆM - DINOv2 FOR LEFT ATRIUM SEGMENTATION.pdf
├── notebooks/
│   ├── colab_train.ipynb
│   └── eda.ipynb
├── scripts/
│   ├── bootstrap_server.sh           # Create/verify the GPU environment
│   ├── sync_server.sh                # Fast-forward code from origin/main
│   ├── start_experiments_tmux.sh     # Start E0→E2 via tmux or screen
│   ├── run_experiments.sh            # Sequential/resumable runner
│   └── update_experiment_report.py   # Write the report branch
├── src/
│   ├── model.py                      # DINOv2 ViT-Large segmenter
│   ├── dataset.py                    # 2D all-slice MRI loader
│   ├── prepare_data.py               # NIfTI → fixed patient-level splits
│   ├── train_baseline.py             # E0/E1
│   ├── train_full_ft.py              # E2
│   ├── evaluate.py                   # Test metrics and visualizations
│   ├── checkpointing.py              # Atomic/resumable checkpoints
│   └── metrics.py
├── EXPERIMENT_PROGRESS.md            # Template; reports are on experiment-reports
├── requirements.txt
└── README.md
```

The server-only directories, all excluded from Git, are:

```text
data/raw/Task02_Heart/                # Original NIfTI data
data/processed/{train,val,test}/      # PNG images and masks
data/splits/                          # Fixed patient IDs
results/vit_large/{e0,e1,e2}/         # Full checkpoints and evaluation artifacts
results/vit_large/smoke/{e0,e1,e2}/   # One-epoch smoke artifacts
logs/                                 # Persistent runner logs
```

## Data preparation

Only labelled `imagesTr`/`labelsTr` volumes are used. The seed-42 patient split is 14/2/4 for train/validation/test. Every axial slice, including an empty-mask slice, is retained. `imagesTs` is unlabelled and is not used for metrics.

Run once on the server:

```bash
.venv/bin/python src/prepare_data.py \
  --raw-dir data/raw/Task02_Heart \
  --processed-dir data/processed \
  --splits-dir data/splits \
  --seed 42
```

Use `--overwrite` only when deliberately regenerating the derived `processed/` and `splits/` directories.

## Code synchronization

Local is the only place to edit source code. Before pushing a change:

```bash
git fetch origin
git rebase origin/main
git push origin main
```

On the server, update only a clean worktree:

```bash
cd /mnt/data/users/trungptit/Left_atrium_segmentation
bash scripts/sync_server.sh
```

Do not edit source code or push from the server `main` worktree. The report worktree is the sole server worktree allowed to push, and it pushes only Markdown report metadata to `experiment-reports`.

## Server setup

Run once:

```bash
cd /mnt/data/users/trungptit/Left_atrium_segmentation
bash scripts/bootstrap_server.sh
```

The script creates `.venv`, installs the CUDA-compatible dependencies, verifies CUDA/GPU access, and prepares `logs/` and `results/vit_large/{e0,e1,e2}`. `tmux` is preferred; if it is unavailable, the runner automatically uses `screen`.

## Run and monitor experiments

Run the one-epoch smoke sequence first:

```bash
bash scripts/start_experiments_tmux.sh smoke
```

After smoke passes, run the full sequence:

```bash
bash scripts/start_experiments_tmux.sh full
```

The runner executes E0, evaluates and reports it, then continues with E1 and E2. If the session is interrupted, start the same command again; completed phases are skipped and an unfinished phase resumes from `last.pth`.

For a tmux session:

```bash
tmux ls
tmux attach -t la-e0-e2-full
# Detach: Ctrl+B, then D
```

For a screen session:

```bash
screen -ls
screen -r la-e0-e2-full
# Detach: Ctrl+A, then D
```

Follow the active log without attaching:

```bash
tail -f logs/la-e0-e2-full.log
```

## Evaluate an existing checkpoint

```bash
.venv/bin/python src/evaluate.py \
  --model vit_large \
  --checkpoint results/vit_large/e2/checkpoints/best.pth \
  --data_root data/processed \
  --save_dir results/vit_large/e2
```

ViT-Small checkpoints are incompatible with ViT-Large and must never be used for E0, E1 or E2.
