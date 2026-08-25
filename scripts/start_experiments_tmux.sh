#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session="la-e0-e2-$mode"
log_path="$repo_root/logs/$session.log"

command -v tmux >/dev/null 2>&1 || {
    echo "tmux is not installed. Run: sudo apt update && sudo apt install -y tmux" >&2
    exit 1
}
tmux has-session -t "$session" 2>/dev/null && {
    echo "Session already exists: $session" >&2
    exit 1
}

mkdir -p "$repo_root/logs"
tmux new-session -d -s "$session" "cd '$repo_root' && bash scripts/run_experiments.sh '$mode' > '$log_path' 2>&1"
echo "Started $session. Attach with: tmux attach -t $session"
