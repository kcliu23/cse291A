#!/usr/bin/env bash
# Mount the instance's local NVMe SSD (instance store) at /mnt/nvme for parked KV contexts.
# The root EBS volume is small and network-attached; the instance store is a local SSD, like
# an edge device's disk. It is EPHEMERAL: wiped when the instance stops, so run this after
# every start. Only parked contexts go there (deleted after each run); results stay in runs/.
#   bash tools/aws/mount_nvme.sh
#   export AGENTSERVE_SLOT_DIR=/mnt/nvme/slots     # then run the probe or a sweep
set -euo pipefail
dev=$(lsblk -dpno NAME,MOUNTPOINT | awk '$2=="" {print $1}' | while read d; do
        [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] && [ "$(lsblk -no TYPE "$d" | head -1)" = disk ] \
          && ! lsblk -no NAME "$d" | tail -n +2 | grep -q . && echo "$d"; done | head -1)
if mountpoint -q /mnt/nvme; then echo "/mnt/nvme already mounted"; else
  [ -n "$dev" ] || { echo "no unmounted, unpartitioned disk found (lsblk)"; exit 1; }
  echo "formatting $dev (instance store) and mounting at /mnt/nvme"
  sudo mkfs.xfs -f -q "$dev"
  sudo mkdir -p /mnt/nvme && sudo mount -o noatime "$dev" /mnt/nvme
fi
sudo mkdir -p /mnt/nvme/slots && sudo chown -R "$USER" /mnt/nvme
df -h /mnt/nvme
grep -q AGENTSERVE_SLOT_DIR ~/.bashrc || echo 'mountpoint -q /mnt/nvme && export AGENTSERVE_SLOT_DIR=/mnt/nvme/slots' >> ~/.bashrc
echo "set: export AGENTSERVE_SLOT_DIR=/mnt/nvme/slots"
