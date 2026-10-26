#!/usr/bin/env bash
# Copy the instance's results into runs_aws/ (kept apart from the Mac's runs/).
# Safe to run any time, also while a sweep runs; it only adds and updates files.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$repo/runs_aws"
rsync -az cse291a:~/cse291/agentserve/runs/ "$repo/runs_aws/"
echo "results in $repo/runs_aws/"
