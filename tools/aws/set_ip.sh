#!/usr/bin/env bash
# Point the "cse291a" SSH host at a new public IP (called by up.sh, or by hand after a console start).
# Writes ~/.ssh/config.d/cse291a-host; the static settings live in ~/.ssh/config.d/cse291a.
set -euo pipefail
ip="${1:?usage: set_ip.sh <public-ip>}"
mkdir -p ~/.ssh/config.d && chmod 700 ~/.ssh ~/.ssh/config.d
printf 'Host cse291a\n  HostName %s\n' "$ip" > ~/.ssh/config.d/cse291a-host
echo "cse291a -> $ip"
