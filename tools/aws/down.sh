#!/usr/bin/env bash
# Copy results back, then stop the instance. Refuses to stop while a sweep is running
# (pass --force to stop anyway). The EBS disk keeps everything; only compute billing stops.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
id="${CSE291A_INSTANCE_ID:?set CSE291A_INSTANCE_ID}"
region="${AWS_REGION:-us-west-2}"

if [ "${1:-}" != "--force" ] && ssh -o ConnectTimeout=10 cse291a 'pgrep -f "agentserve\.(sweep|runner)" >/dev/null'; then
  echo "a sweep is still running on the instance; not stopping (use --force to override)" >&2
  echo "it will stop itself 30 min after the sweep finishes (idle shutdown)" >&2
  exit 1
fi
"$here/pull.sh" || echo "warning: pull failed; results are still on the instance's disk" >&2
echo "stopping $id ..."
aws ec2 stop-instances --region "$region" --instance-ids "$id" >/dev/null
aws ec2 wait instance-stopped --region "$region" --instance-ids "$id"
echo "stopped"
