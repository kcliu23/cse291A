#!/usr/bin/env bash
# Run by cron as root every 5 minutes (installed by bootstrap.sh).
# Stops the instance after IDLE_MIN minutes with no SSH connection, no sweep and no llama-server,
# so a forgotten instance cannot spend the daily budget. Shutdown behavior must be "Stop".
# A sweep left running in tmux keeps the instance up; it stops IDLE_MIN after the sweep ends.
IDLE_MIN="${IDLE_MIN:-30}"
STAMP="${STAMP:-/run/cse291a-last-active}"   # /run is cleared at boot: every boot starts the clock fresh
SHUTDOWN="${SHUTDOWN:-/sbin/shutdown -h now}"   # overridable for testing

busy() {
  pgrep -f 'agentserve\.(sweep|runner)' >/dev/null && return 0
  pgrep -x llama-server >/dev/null && return 0
  command -v ss >/dev/null || return 0      # cannot see SSH sessions: never stop blind
  ss -Htn state established '( sport = :22 )' | grep -q . && return 0   # any SSH / VS Code session
  return 1
}

now=$(date +%s)
if busy || [ ! -f "$STAMP" ]; then echo "$now" > "$STAMP"; exit 0; fi
idle_s=$(( now - $(cat "$STAMP") ))
if [ "$idle_s" -ge $(( IDLE_MIN * 60 )) ]; then
  logger -t cse291a "idle for $((idle_s / 60)) min, shutting down"
  $SHUTDOWN
fi
