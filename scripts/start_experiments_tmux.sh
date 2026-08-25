#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session="la-e0-e2-$mode"
log_path="$repo_root/logs/$session.log"

mkdir -p "$repo_root/logs"
if command -v tmux >/dev/null 2>&1; then
    tmux has-session -t "$session" 2>/dev/null && {
        echo "Session already exists: $session" >&2
        exit 1
    }
    tmux new-session -d -s "$session" "cd '$repo_root' && bash scripts/run_experiments.sh '$mode' > '$log_path' 2>&1"
    echo "Started $session. Attach with: tmux attach -t $session"
elif command -v screen >/dev/null 2>&1; then
    screen -ls | grep -q "[.]$session[[:space:]]" && {
        echo "Session already exists: $session" >&2
        exit 1
    }
    screen -dmS "$session" bash -lc "cd '$repo_root' && bash scripts/run_experiments.sh '$mode' > '$log_path' 2>&1"
    echo "Started $session using screen. Attach with: screen -r $session"
else
    echo "Neither tmux nor screen is installed." >&2
    exit 1
fi
