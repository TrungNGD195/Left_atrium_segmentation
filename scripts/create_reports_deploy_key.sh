#!/usr/bin/env bash
set -euo pipefail

key_path="${GITHUB_REPORTS_KEY:-$HOME/.ssh/github_experiment_reports}"
mkdir -p "$(dirname "$key_path")"
chmod 700 "$(dirname "$key_path")"

if [[ ! -f "$key_path" ]]; then
    ssh-keygen -t ed25519 -N "" -f "$key_path" -C "left-atrium-experiment-reports"
fi
chmod 600 "$key_path"

echo "Add this public key to GitHub Deploy keys for TrungNGD195/Left_atrium_segmentation with Write access:"
cat "${key_path}.pub"
