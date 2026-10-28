# agentserve: Memory Bottlenecks of Agent Serving on an Edge LLM Server

CSE 291A (Fall 2026), Option 1: Memory-Aware Agent Serving. Phase 1 code and data.

LLM agents call the model, wait for a tool, and call the model again with a longer conversation. This repository
measures how a single edge LLM server ([llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, Llama-3.2-3B)
should admit such agents under a fixed memory budget. It contains a seeded agent workload generator, an instrumented
async runner with a FIFO scheduler of fixed concurrency `K`, a resumable sweep driver, analysis scripts, every raw
measurement behind the Phase 1 report, and the report itself ([`paper/paper.pdf`](paper/paper.pdf)).

## Phase 1 results at a glance

| Finding | Evidence | Data |
| --- | --- | --- |
| The best fixed `K` equals the server's slot count (4) on both machines. Under overload it cuts p95 latency 2.0–2.4× vs. `K = 1` on the Mac and 3.4–4.0× on AWS. | 90 runs: `K ∈ {1,2,4,8,16}` × 3 rates × 3 seeds × 2 machines | `runs/ksweep`, `runs_aws/ksweep` |
| Above 4 slots, agents evict each other's cached context: prompt-cache reuse falls from 65% to 17–19%, p95 rises 205 → 410 s on the Mac and 149 → 203 s on AWS (0.2 tasks/s). | Per-request prefill time, reuse, decode speed, TTFT | same |
| The penalty scales with recomputation cost: AWS processes prompts ~4× faster, so the same lost reuse costs 0.6 s per request there vs. 2.4 s on the Mac. | Micro-benchmarks + sweep | `runs/micro`, `runs_aws/micro` |
| Causal test: retaining evicted context removes the penalty. A 1 GB host cache cuts p95 at `K = 8` from 314 to 199 s within budget; 8 GB matches the best `K` but needs 11.8 GB. | 21 runs | `runs/cache`, `runs/cache_off` |
| An 8-bit KV cache fits 8 slots in the memory of 4: +42% throughput at `K = 8`, outputs identical to f16 on 50 of 50 scored prompts. | 33 runs + 3 × 70 quality prompts | `runs/slots`, `runs/kvq` |
| Phase 2 feasibility: saving and restoring a slot's KV state takes 0.2–0.5 s vs. 2.6–9.7 s to recompute 1,000–3,500 tokens. | Probe | `runs/slot_probe` |

Every number in the report is recomputed from these folders by [`paper/make_figures.py`](paper/make_figures.py),
which writes them to [`paper/numbers.json`](paper/numbers.json).

## Repository layout

| Path | Contents |
| --- | --- |
| `agentserve/workload.py` | Seeded workload generator: task classes, multi-step agents with growing context and tool delays, Poisson arrivals |
| `agentserve/runner.py` | Async runner. Admission policies are pluggable; Phase 1 uses `FIFOAdmission(K)` |
| `agentserve/client.py` | Streaming `/completion` client with fixed-length outputs (`ignore_eos` + `n_predict`) |
| `agentserve/monitor.py` | Samples `/metrics`, `/slots`, server RSS, macOS `footprint` and `nvidia-smi` every 0.5 s |
| `agentserve/server.py` | Starts a fresh `llama-server` per run, refuses a busy port, verifies the answering server is its own |
| `agentserve/sweep.py` | Grid sweep driver, resumable (`DONE` markers); crashes are logged, never dropped |
| `agentserve/analyze.py` | Per-run metrics (`runs.csv`), mean ± std over seeds (`summary.csv`), figures |
| `configs/` | Workload mix and one JSON file per sweep |
| `tools/mock_server.py` | Fake `llama-server` for testing the pipeline without a model (its numbers are not measurements) |
| `tools/kv_quality.py` | 8-bit vs. f16 KV output-quality check |
| `tools/slot_probe.py` | Phase 2 feasibility probe: slot save/restore vs. recompute |
| `tools/aws/` | Scripts for the AWS replica: start/stop, SSH, code sync, bootstrap, idle auto-stop (see its README) |
| `paper/` | Report source (`paper.tex`, `refs.bib`), figures, `make_figures.py`, `numbers.json`, `paper.pdf` |
| `runs/` | All Mac measurements (below) |
| `runs_aws/` | AWS measurements, same layout |

## Data

### Experiments

| Experiment | Config | Runs | Folder | In the report |
| --- | --- | --- | --- | --- |
| Micro-benchmarks (`llama-bench`, `llama-batched-bench`) | commands below | – | `runs/micro`, `runs_aws/micro` | Sec. III |
| Pilot (calibrates arrival rates) | `sweep_pilot.json` | 2 + 2 | `runs/pilot`, `runs_aws/pilot` | – |
| FIFO `K` sweep | `sweep_ksweep.json` | 45 + 45 | `runs/ksweep`, `runs_aws/ksweep` | Fig. 1, Table I |
| Host-RAM context retention | `sweep_cache.json` | 18 | `runs/cache` | Fig. 2 |
| Prompt caching off | `sweep_cache_off.json` | 3 | `runs/cache_off` | Fig. 2 |
| Memory split (`-c` × `-np`, 8-bit KV) | `sweep_slots.json` | 33 | `runs/slots` | Sec. V |
| 8-bit KV output quality | `tools/kv_quality.py` | 3 servers × 70 prompts | `runs/kvq` | Sec. V |
| Slot save/restore probe | `tools/slot_probe.py` | 3 sizes | `runs/slot_probe` | Sec. VI |
| Excluded runs (see below) | – | 22 | `runs/_invalid` | – |

All sweeps use workload seeds 0–2 (development seeds 0–99). Seeds 1000+ are reserved for the Phase 2 evaluation.

### Files in each run

`runs/<sweep>/<server config>/<policy>_k<K>/rate<rate>/seed<seed>/` holds:

| File | Contents |
| --- | --- |
| `workload.json` | The exact tasks of this run (with the generator's SHA-256 hash) |
| `run_config.json` | All parameters, server command, `/props`, cold-start time, software versions |
| `requests.jsonl` | One line per model call: send / first-token / end times, prompt tokens evaluated and served from cache, tokens generated, slot, server prefill and decode timings |
| `tasks.jsonl` | One line per task: arrival, admission and end times, steps done, status (`ok` / `failed` / `timeout`) and error text |
| `system.jsonl` | 0.5 s samples: busy slots, deferred requests, all `llamacpp:*` metrics, server RSS, macOS footprint or GPU memory |
| `server.log` | Full `llama-server` log, including model / KV / compute buffer sizes |
| `DONE` | Written only when the run completed |

`runs/<sweep>/analysis/` holds `runs.csv` (one row per run), `summary.csv` (mean ± std over seeds) and figures.
`sweep_log.jsonl` in each sweep folder records every run start, finish and crash.

### Metrics

- **p95 task latency**: arrival to completion of a task, including queueing.
- **Throughput**: completed tasks per second of makespan.
- **Prompt-cache reuse**: prompt tokens served from a slot's cache / all prompt tokens (`timings.cache_n`).
- **Idle-hold fraction**: share of admitted task time spent waiting on a tool while holding an admission.
- **Failures** by cause: `context_exceeded` (task outgrew its slot) or `timeout` (600 s task limit).
- **Memory**: allocated model / KV / compute buffers from `server.log`, plus peak footprint (Mac) or `nvidia-smi` (AWS).

### Excluded runs

`runs/_invalid/` keeps 22 runs that measured the wrong server: a leftover `llama-server` from an interrupted sweep held
port 8080, so later runs' requests went to it. Its [README](runs/_invalid/README.md) explains the detection and the
fix (`server.py` now refuses a busy port and checks the answering server's slot count). The affected sweeps were rerun.
The saved KV files of the slot probe (`runs/slot_probe/slots/*.bin`, 715 MB) are not committed; rerun the probe to
recreate them.

## Reproducing

### Software and hardware

| | Pinned version |
| --- | --- |
| llama.cpp | build b11803, commit `a37cc46104eb50900d5ab4a9ba02a0abdac039db` |
| Model | `bartowski/Llama-3.2-3B-Instruct-GGUF`, file `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, SHA-256 `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| Python | 3.11+ with `requirements.txt` |
| Mac | Apple M2 Mac mini, 24 GB, macOS 26, Metal backend |
| AWS | g6f.2xlarge (NVIDIA L4-6Q, 6 GiB), Amazon Linux 2023, CUDA |

Configs use paths relative to the repository, so the expected layout is:

```
<root>/
  agentserve/                 this repository
  llama.cpp/build/bin/        llama-server, llama-bench, llama-batched-bench
  models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
  .venv/
```

### Setup (macOS; Metal is enabled by default)

```bash
cd <root>
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a37cc46104eb50900d5ab4a9ba02a0abdac039db
cmake -B build && cmake --build build --config Release -j --target llama-server llama-bench llama-batched-bench
cd ..

python3 -m venv .venv && source .venv/bin/activate
pip install -r agentserve/requirements.txt "huggingface_hub[cli]"
hf download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
shasum -a 256 models/Llama-3.2-3B-Instruct-Q4_K_M.gguf     # must match the hash above
```

On Linux with an NVIDIA GPU, `tools/aws/bootstrap.sh` does all of this (CUDA build, venv, verified model download).

### Run

All commands run from `agentserve/`, one sweep at a time (each uses port 8080). Inside `tmux` is recommended; on a Mac,
prefix long runs with `caffeinate -i`. Sweeps resume where they stopped.

```bash
# 0. Pipeline test without a model (~3 min)
python -m agentserve.sweep configs/sweep_mock.json && python -m agentserve.analyze runs/mock

# 1. Micro-benchmarks
B=../llama.cpp/build/bin; M=../models/Llama-3.2-3B-Instruct-Q4_K_M.gguf; mkdir -p runs/micro
$B/llama-bench -m $M -ngl 99 -p 256,1024,2048,4096 -n 128 -r 5 -o csv > runs/micro/llama_bench.csv
$B/llama-batched-bench -m $M -ngl 99 -c 16384 -npp 512,2048 -ntg 128 -npl 1,2,4,8 > runs/micro/batched_bench.txt 2>&1

# 2. Pilot, then the sweeps (the K sweep takes several hours on the M2)
python -m agentserve.sweep configs/sweep_pilot.json     && python -m agentserve.analyze runs/pilot
python -m agentserve.sweep configs/sweep_ksweep.json    && python -m agentserve.analyze runs/ksweep
python -m agentserve.sweep configs/sweep_cache.json     && python -m agentserve.analyze runs/cache
python -m agentserve.sweep configs/sweep_cache_off.json && python -m agentserve.analyze runs/cache_off
python -m agentserve.sweep configs/sweep_slots.json     && python -m agentserve.analyze runs/slots

# 3. 8-bit KV quality check (~15 min) and slot save/restore probe (~2 min)
python tools/kv_quality.py
python tools/slot_probe.py

# 4. Report figures and numbers (reads runs/ and runs_aws/)
python paper/make_figures.py
cd paper && latexmk -pdf paper.tex
```

`python -m agentserve.sweep <config> --dry-run` lists a sweep's runs; `--max-runs N` stops after N (useful under a
daily cloud budget). For the AWS replica, see [`tools/aws/README.md`](tools/aws/README.md); results are copied back
into `runs_aws/`.

## Experimental controls

- **Fixed output length.** Every step generates exactly its `n_predict` tokens, so no policy can win by shortening answers.
- **Fresh server per run** with two warm-up requests; cold start is reported separately and excluded from latency.
- **Fixed memory budget.** llama.cpp's host-RAM prompt cache is disabled (`--cache-ram 0`) except where it is the
  variable under study, so memory equals the model, KV and compute allocation.
- **Three seeds per configuration**, reported as mean ± std; failed and timed-out tasks stay in the data.
- **Server identity check.** A run refuses to start if its port is busy and verifies the answering server's slot count.

## Phase 2

Admission policies live in `agentserve/runner.py`: add a class with `acquire(task)` / `release(task)`, register it in
`make_admission`, and list it under `"policies"` in a sweep config. Phase 2 will add memory-bounded context retention
(keep, park on SSD via `llama-server` slot save/restore, or drop a waiting agent's context), evaluated on seeds 1000+.
The plan and timeline are in the report's Section VI and appendix.
