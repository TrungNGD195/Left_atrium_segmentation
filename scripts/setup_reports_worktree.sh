#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reports_root="${REPORTS_WORKTREE:-$(dirname "$repo_root")/Left_atrium_reports}"
deploy_key="${GITHUB_REPORTS_KEY:-$HOME/.ssh/github_experiment_reports}"
github_ssh_url="git@github.com:TrungNGD195/Left_atrium_segmentation.git"

if [[ ! -f "$deploy_key" ]]; then
    echo "Missing deploy key: $deploy_key. Run scripts/create_reports_deploy_key.sh first." >&2
    exit 1
fi

export GIT_SSH_COMMAND="ssh -i $deploy_key -o IdentitiesOnly=yes"
cd "$repo_root"
git fetch --prune origin

# A linked worktree has a .git *file*, not a directory.
if [[ ! -e "$reports_root/.git" ]]; then
    git worktree add -B experiment-reports "$reports_root" origin/main
fi

# Keep fetch/pull on the read-only HTTPS remote used by the source worktree.
# A push URL is shared across linked worktrees and is used only for reports.
git -C "$reports_root" config remote.origin.pushurl "$github_ssh_url"
git -C "$reports_root" fetch --prune origin
git -C "$reports_root" switch experiment-reports
