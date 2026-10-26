#!/usr/bin/env bash
# Send your local code to the instance: the repo including .git (so runs record the commit),
# without runs/ and caches. Your Mac stays the one place you edit code.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
ssh cse291a 'mkdir -p ~/cse291/agentserve'
rsync -az --delete \
  --exclude 'runs/' --exclude 'runs_aws/' --exclude '__pycache__/' --exclude '.DS_Store' \
  "$repo/" cse291a:~/cse291/agentserve/
echo "pushed $(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet || echo ' (+ uncommitted changes)') to cse291a:~/cse291/agentserve"
