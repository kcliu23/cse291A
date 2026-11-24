#!/usr/bin/env bash
# Experiments added after the Phase 1 feedback. Resumable: rerun the same command after any interruption.
#   bash tools/run_feedback_sweeps.sh delay      # tool-delay sweep (54 runs); Mac and AWS
#   bash tools/run_feedback_sweeps.sh gpu        # Phase 1 slot sweep, cache test and 8-bit quality check on the GPU (AWS)
#   bash tools/run_feedback_sweeps.sh all        # gpu, then delay (AWS)
set -uo pipefail
cd "$(dirname "$0")/.."
what="${1:-delay}"

gpu() {
  for c in slots cache cache_off; do
    python -m agentserve.sweep configs/sweep_$c.json && python -m agentserve.analyze runs/$c
  done
  [ -f runs/kvq/summary.json ] || python tools/kv_quality.py
}

delay() {
  for d in 0 2 5 10 20 40; do
    python -m agentserve.sweep configs/sweep_delay$d.json && python -m agentserve.analyze runs/delay/d$d
  done
}

case "$what" in
  gpu) gpu ;;
  delay) delay ;;
  all) gpu; delay ;;
  *) echo "usage: $0 delay|gpu|all"; exit 1 ;;
esac
echo "[done] $what"
