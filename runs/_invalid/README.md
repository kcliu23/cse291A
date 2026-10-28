# Invalid runs (kept for the record)

`cache_stray_server/` (18 runs) and `cache_off_stray_server/` (3 runs), 2026-10-22.

A leftover llama-server (1 slot, n_ctx 8192) was already listening on port 8080. Each run's
own server failed to bind ("couldn't bind HTTP server socket" in server.log) and exited, but
the leftover server answered /health, so every request was served by it. The configured
--cache-ram / -np settings were never applied; these measurements describe the wrong server.

Detected because run_config.json reports n_slots = 1 for runs started with -np 4. Fixed in
commit "Refuse to run when the port is taken; verify the answering server is ours": the sweep
now refuses to start if the port is in use, and checks that the answering server is alive and
reports the expected slot count. The cache sweeps were re-run after the fix.

## `slots_orphaned/` (1 incomplete run)

Root cause of the above: the memory sweep (`sweep_slots.json`) was started at 10:01 local, and
its first server (-c 8192 -np 1) took port 8080. That sweep's Python process stopped at about
10:02 (system.jsonl ends then) without stopping its server, which kept running and then served
all cache-sweep requests. This run never finished (no DONE) and also received the cache
sweeps' traffic, so it is invalid. Sweeps must run one at a time on one port.
