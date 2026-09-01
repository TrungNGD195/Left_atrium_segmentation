#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    echo "Usage: bash scripts/run_experiments.sh [smoke|full]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reports_root="${REPORTS_WORKTREE:-$(dirname "$repo_root")/Left_atrium_reports}"
python_bin="$repo_root/.venv/bin/python"
epochs=1
result_prefix="$repo_root/results/vit_large/smoke"
if [[ "$mode" == "full" ]]; then
    epochs=50
    result_prefix="$repo_root/results/vit_large"
fi

cd "$repo_root"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Refusing to run: source worktree is dirty." >&2
    exit 1
fi
bash scripts/sync_server.sh
bash scripts/setup_reports_worktree.sh

seeds=(42 2026 3407)
# Batch size is not prescribed by the experiment specification. Batch 4 and
# torch.compile are used only after their dedicated smoke test on the RTX 4090.
# E2_BATCH_SIZE=1 and E2_COMPILE=0 remain OOM-safe fallbacks.
e2_batch_size="${E2_BATCH_SIZE:-4}"
if ! [[ "$e2_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "E2_BATCH_SIZE must be a positive integer; got '$e2_batch_size'." >&2
    exit 2
fi
e2_compile="${E2_COMPILE:-1}"
e2_compile_args=()
case "$e2_compile" in
    1|true|TRUE|yes|YES) e2_compile_args=(--compile --compile_mode reduce-overhead) ;;
    0|false|FALSE|no|NO) ;;
    *) echo "E2_COMPILE must be 0 or 1; got '$e2_compile'." >&2; exit 2 ;;
esac
echo "E2 configuration: batch_size=$e2_batch_size, compile=$e2_compile, AMP=on, encoder_lr=1e-5, decoder_lr=1e-3"

run_phase() {
    local phase="$1" seed="$2" result_dir="$3"
    shift 3
    local -a command=("$@")
    local resume=()
    local last="$result_dir/checkpoints/last.pth"
    local final="$result_dir/checkpoints/final.pth"
    local best="$result_dir/checkpoints/best.pth"

    if [[ -f "$final" ]]; then
        echo "$phase already completed; skipping."
        return
    fi
    mkdir -p "$result_dir"
    if [[ -f "$last" ]]; then
        resume=(--resume "$last")
    fi

    echo "=== Starting $phase seed=$seed (mode=$mode) ==="
    "$python_bin" "${command[@]}" --seed "$seed" --epochs "$epochs" --data_root data/processed --save_dir "$result_dir" --num_workers 4 "${resume[@]}"
    "$python_bin" src/evaluate.py --model vit_large --checkpoint "$best" --data_root data/processed --save_dir "$result_dir" --num_workers 4 --spacing-zhw 1.0 1.2857142857 1.2857142857
    "$python_bin" scripts/update_experiment_report.py --phase "$phase" --results-dir "$result_dir" --report "$reports_root/EXPERIMENT_PROGRESS.md" --commit "$(git rev-parse HEAD)" --data-manifest data/data_manifest.json
    git -C "$reports_root" add EXPERIMENT_PROGRESS.md
    if ! git -C "$reports_root" diff --cached --quiet; then
        git -C "$reports_root" commit -m "report: $phase $mode results"
        GIT_SSH_COMMAND="ssh -i ${GITHUB_REPORTS_KEY:-$HOME/.ssh/github_experiment_reports} -o IdentitiesOnly=yes" git -C "$reports_root" push --set-upstream origin experiment-reports
    fi
}

for seed in "${seeds[@]}"; do
    run_phase E0 "$seed" "$result_prefix/e0/seed_$seed" src/train_baseline.py --model vit_large --loss bce --batch_size 8
    run_phase E1 "$seed" "$result_prefix/e1/seed_$seed" src/train_baseline.py --model vit_large --loss bce_dice --batch_size 8
    run_phase E2 "$seed" "$result_prefix/e2/seed_$seed" src/train_full_ft.py --model vit_large --batch_size "$e2_batch_size" "${e2_compile_args[@]}"
done
