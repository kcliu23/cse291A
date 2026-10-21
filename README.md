# agentserve — CSE 291A Option 1, Phase 1

Baseline for memory-aware agent serving on `llama-server`: a FIFO scheduler with fixed concurrency K, a seeded agent workload generator, full per-request / per-task / system instrumentation, a resumable sweep driver, and analysis scripts.

## Layout

| Path | What it is |
| --- | --- |
| `agentserve/workload.py` | Seeded generator: short/long context, short/long output, multi-step tasks with tool delays, Poisson arrivals |
| `agentserve/runner.py` | Async runner with pluggable admission policy. Phase 1 = `FIFOAdmission(K)` |
| `agentserve/client.py` | Streaming `/completion` client; fixed-length outputs (`ignore_eos` + `n_predict`) |
| `agentserve/monitor.py` | Samples `/metrics`, `/slots`, server RSS, `nvidia-smi` every 0.5 s |
| `agentserve/server.py` | Starts a fresh `llama-server` per run, measures cold start (launch → healthy) |
| `agentserve/sweep.py` | Grid sweep, resumable (`DONE` markers), crashes logged not dropped |
| `agentserve/analyze.py` | `runs.csv`, `summary.csv` (mean ± std over seeds), figures |
| `tools/mock_server.py` | Fake llama-server for pipeline testing only (its numbers are not measurements) |
| `configs/` | Workload mix and sweep definitions |

## Setup

```bash
pip install -r requirements.txt
```

On the instance, llama.cpp and the model come from the setup guide (pinned commit, `sha256sum` of the GGUF). Paths in `configs/sweep_*.json` assume `~/llama.cpp/build/bin/llama-server` and `~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

## Running on an Apple Silicon Mac

Phase 1 can run entirely on a Mac; the configs work unchanged once llama.cpp and the model are at the default paths.

```bash
# llama.cpp, pinned (Metal is on by default on macOS)
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
git checkout <TAG> && cmake -B build && cmake --build build --config Release -j
git rev-parse HEAD   # record in the report

# model
python3 -m venv ~/venvs/agentserve && source ~/venvs/agentserve/bin/activate
pip install -r requirements.txt "huggingface_hub[cli]"
mkdir -p ~/models && hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models
shasum -a 256 ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Memory on a Mac is unified, so `nvidia-smi` doesn't exist and RSS misses Metal buffers. The pipeline records instead:

- **Allocated budget** from `server.log`: `model_buffer_mib`, `kv_buffer_mib`, `compute_buffer_mib`. The KV cache is allocated once from `-c`, so this is the memory budget you are varying.
- **Physical footprint** (`footprint_mb`, sampled every ~2 s via macOS `footprint`), which includes Metal buffers. Best effort: blank if the tool's output format differs.

For clean numbers: plug in, close other apps, run `caffeinate -i` alongside sweeps, and note the chip, RAM and macOS version in the report.

## Run order

```bash
# 0. Pipeline test, no model needed (~3 min, laptop is fine)
python -m agentserve.sweep configs/sweep_mock.json && python -m agentserve.analyze runs/mock

# 1. Pilot on the instance (~10 min). Check every run says N/N ok, then look at
#    runs/pilot/analysis/runs.csv: latency of K=1 tells you the service rate.
python -m agentserve.sweep configs/sweep_pilot.json && python -m agentserve.analyze runs/pilot

# 2. Set "rates" in sweep_ksweep.json so the top rate overloads K=1, then:
python -m agentserve.sweep configs/sweep_ksweep.json --dry-run    # count runs
python -m agentserve.sweep configs/sweep_ksweep.json --max-runs 15  # one day's budget
python -m agentserve.analyze runs/ksweep

# 3. Memory budget (-c x -np) and prefix reuse
python -m agentserve.sweep configs/sweep_slots.json && python -m agentserve.analyze runs/slots
python -m agentserve.sweep configs/sweep_cache.json && python -m agentserve.analyze runs/cache
```

Run sweeps inside `tmux`. Re-running a sweep skips finished runs, so stopping at the daily cap and resuming tomorrow is safe.

## What gets recorded

Each run directory holds `workload.json` (the exact tasks), `run_config.json` (all parameters, workload hash, `/props`, cold-start time, versions), `server.log`, `requests.jsonl`, `tasks.jsonl` and `system.jsonl`.

| Level | Fields |
| --- | --- |
| Request | ready, send, first token, end; prompt tokens evaluated / cached; tokens generated; slot used; server prefill and decode timings |
| Task | arrival, admission, end; steps done; status `ok` / `failed` / `timeout` with the error text |
| System | busy slots, deferred requests, all `llamacpp:*` metrics, server RSS, GPU memory |

`analyze.py` derives: completed and failed tasks (by reason, e.g. `context_exceeded`), p50/p95/p99 task latency, queueing time, tasks/s and generated tokens/s, TTFT, cache hit ratio, per-class p95 latency (fairness for long tasks), peak memory, and **idle-hold fraction**: the share of admitted task time spent waiting on tools while holding an admission slot.

## Controls built in

- **Fixed output length.** Every step generates exactly `n_predict` tokens, so no policy can win by shortening outputs.
- **Fresh server per run.** No KV cache carries over between runs. Model load is excluded from task latency and reported separately as `server_startup_s` (cold start); 2 warm-up requests precede each measured run.
- **Dev vs. eval workloads.** Phase 1 uses seeds 0–99 only. Seeds 1000+ are reserved for the Phase 2 evaluation.
- **Nothing dropped.** Failed and timed-out tasks stay in `tasks.jsonl`; crashed runs get a `CRASHED` file, appear in `sweep_log.jsonl` and in `runs.csv` with `status=crashed`.
- **Repeats.** Each config runs over 3 workload seeds; summaries report mean ± std.

## Things to verify in the pilot

1. **Per-slot context.** `curl localhost:8080/slots | jq '.[].n_ctx'` should show `ctx / np`. If your llama.cpp build shares one unified KV pool across slots (`--kv-unified`), long tasks will not fail at `ctx / np`, which changes the bottleneck story. Note which mode you ran.
2. **`tokens_cached` is populated** in `requests.jsonl` for steps after the first. If it's always null, check the final stream chunk format of your build.
3. **`/metrics` names.** Recent builds may not export `kv_cache_usage_ratio`; `system.jsonl` keeps every `llamacpp:*` metric the build exposes.

## Phase 2 hook

Admission policies live in `runner.py`. Add a class with `acquire(task)` / `release(task)` and register it in `make_admission`; the sweep's `"policies"` list selects it.
