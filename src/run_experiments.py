"""
run_experiments.py
------------------
Master script chạy TẤT CẢ 5 thí nghiệm (E0–E4) × 3 seeds (42, 2026, 3407).

Thứ tự:
  E0: Frozen encoder + BCE Loss
  E1: Frozen encoder + BCE+Dice Loss
  E2: Full Fine-tuning
  E3: Partial Fine-tuning (4 blocks cuối)
  E4: LoRA Fine-tuning

Sau khi hoàn tất sẽ tổng hợp bảng kết quả Mean ± SD tự động.

Cách chạy:
    python src/run_experiments.py --data_root /content/data --save_dir results
    python src/run_experiments.py --skip E2 E3  # Bỏ qua E2, E3
    python src/run_experiments.py --only E4     # Chỉ chạy E4
"""

import os
import sys
import json
import argparse
import subprocess
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("src"))


SEEDS = [42, 2026, 3407]

# Mapping experiment ID → script và tham số
EXPERIMENTS = {
    "E0": {
        "name": "Frozen + BCE",
        "script": "src/train_fast.py",
        "extra_args": ["--loss", "bce"],
        "ckpt_pattern": "best_frozen_bce_seed{seed}.pth",
        "eff_pattern": "efficiency_frozen_bce_seed{seed}.json",
    },
    "E1": {
        "name": "Frozen + BCE+Dice",
        "script": "src/train_fast.py",
        "extra_args": ["--loss", "bce_dice"],
        "ckpt_pattern": "best_frozen_bcedice_seed{seed}.pth",
        "eff_pattern": "efficiency_frozen_bcedice_seed{seed}.json",
    },
    "E2": {
        "name": "Full Fine-tuning",
        "script": "src/train_full_ft.py",
        "extra_args": [],
        "ckpt_pattern": "best_full_ft_seed{seed}.pth",
        "eff_pattern": "efficiency_full_ft_seed{seed}.json",
    },
    "E3": {
        "name": "Partial Fine-tuning (4 blocks)",
        "script": "src/train_partial_ft.py",
        "extra_args": ["--unfreeze_blocks", "4"],
        "ckpt_pattern": "best_partial_ft_seed{seed}.pth",
        "eff_pattern": "efficiency_partial_ft_seed{seed}.json",
    },
    "E4": {
        "name": "LoRA",
        "script": "src/train_lora.py",
        "extra_args": ["--lora_rank", "4", "--lora_blocks", "2"],
        "ckpt_pattern": "best_lora_seed{seed}.pth",
        "eff_pattern": "efficiency_lora_seed{seed}.json",
    },
}


def run_one(exp_id: str, seed: int, args) -> dict:
    """Chạy một thí nghiệm với một seed. Trả về efficiency dict."""
    cfg = EXPERIMENTS[exp_id]
    print(f"\n{'─'*60}")
    print(f"  Chạy {exp_id} ({cfg['name']}) | seed={seed}")
    print(f"{'─'*60}")

    cmd = [
        sys.executable, cfg["script"],
        "--data_root", args.data_root,
        "--save_dir", args.save_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--patience", str(args.patience),
        "--num_workers", str(args.num_workers),
        "--seed", str(seed),
    ] + cfg["extra_args"]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [LỖI] {exp_id} seed={seed} thất bại! (code={result.returncode})")
        return {"error": True, "exp_id": exp_id, "seed": seed}

    # Đọc efficiency JSON đã được script lưu lại
    eff_path = os.path.join(
        args.save_dir,
        cfg["eff_pattern"].format(seed=seed)
    )
    if os.path.exists(eff_path):
        with open(eff_path) as f:
            return json.load(f)
    else:
        return {"exp_id": exp_id, "seed": seed, "elapsed_s": round(elapsed, 1)}


def aggregate_results(all_results: list[dict], save_dir: str):
    """
    Tổng hợp kết quả: tính Mean ± SD của val_dice theo từng method.
    Lưu ra summary_results.json và in bảng tổng hợp.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            key = r.get("method", "unknown")
            grouped[key].append(r)

    print("\n" + "=" * 70)
    print(f"{'Method':<22} {'Dice Mean':>10} {'Dice SD':>9} "
          f"{'Trainable':>12} {'VRAM GB':>8} {'Time min':>9}")
    print("=" * 70)

    summary = {}
    for method, records in grouped.items():
        dices  = [r["best_val_dice"] for r in records if "best_val_dice" in r]
        vrams  = [r.get("vram_gb", 0) for r in records]
        times  = [r.get("total_train_time_s", 0) for r in records]
        tparams = records[0].get("trainable_params", 0)

        mean_d = np.mean(dices)
        std_d  = np.std(dices)

        summary[method] = {
            "seeds": [r["seed"] for r in records],
            "val_dice_per_seed": dices,
            "mean_val_dice": round(mean_d, 4),
            "std_val_dice":  round(std_d, 4),
            "trainable_params": tparams,
            "trainable_ratio": records[0].get("trainable_ratio", 0),
            "mean_vram_gb": round(np.mean(vrams), 3),
            "mean_train_time_min": round(np.mean(times) / 60, 1),
        }

        print(f"  {method:<20} {mean_d:>10.4f} ±{std_d:>7.4f} "
              f"{tparams:>12,} {np.mean(vrams):>8.2f} {np.mean(times)/60:>9.1f}")

    print("=" * 70)

    summary_path = os.path.join(save_dir, "summary_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nĐã lưu: {summary_path}")

    return summary


def main(args):
    print("=" * 70)
    print("  CHẠY TOÀN BỘ THÍ NGHIỆM E0–E4 × 3 SEEDS")
    print(f"  data_root: {args.data_root}")
    print(f"  save_dir:  {args.save_dir}")
    print(f"  epochs:    {args.epochs}, patience: {args.patience}")
    print("=" * 70)

    os.makedirs(args.save_dir, exist_ok=True)

    # Xác định list experiments cần chạy
    if args.only:
        exp_ids = [e.upper() for e in args.only]
    elif args.skip:
        skip = [e.upper() for e in args.skip]
        exp_ids = [e for e in EXPERIMENTS if e not in skip]
    else:
        exp_ids = list(EXPERIMENTS.keys())

    print(f"\nThí nghiệm sẽ chạy: {exp_ids}")
    print(f"Seeds: {SEEDS}")
    total_runs = len(exp_ids) * len(SEEDS)
    print(f"Tổng số lần chạy: {total_runs}\n")

    all_results = []
    run_log = []

    for exp_id in exp_ids:
        for seed in SEEDS:
            result = run_one(exp_id, seed, args)
            all_results.append(result)
            run_log.append({
                "exp_id": exp_id,
                "seed": seed,
                "status": "error" if "error" in result else "done",
            })

            # Lưu log ngay sau mỗi run (để không mất kết quả nếu crash)
            log_path = os.path.join(args.save_dir, "run_log.json")
            with open(log_path, "w") as f:
                json.dump(run_log, f, indent=2)

    # Tổng hợp kết quả
    aggregate_results(all_results, args.save_dir)

    print("\n✓ Tất cả thí nghiệm hoàn tất!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master runner: E0–E4 × 3 seeds")
    parser.add_argument("--data_root",   type=str,   default="data")
    parser.add_argument("--save_dir",    type=str,   default="results")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=24)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--only",  nargs="+", default=None,
                        help="Chỉ chạy các experiment này. VD: --only E0 E1 E4")
    parser.add_argument("--skip",  nargs="+", default=None,
                        help="Bỏ qua các experiment này. VD: --skip E2 E3")
    args = parser.parse_args()
    main(args)
