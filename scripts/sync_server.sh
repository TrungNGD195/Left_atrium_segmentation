#!/usr/bin/env bash
# Synchronize the server working tree with the canonical GitHub branch.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Refusing to sync: the server worktree has uncommitted code changes." >&2
    echo "Commit or discard them on the development machine, then retry." >&2
    exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "Missing GitHub remote 'origin'." >&2
    exit 1
fi

git fetch --prune origin
git switch main
git pull --ff-only origin main
