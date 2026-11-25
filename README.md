# agentserve: Memory-Aware Agent Serving on an Edge LLM Server

CSE 291A (Fall 2026), Option 1: Memory-Aware Agent Serving. Phase 1 and Phase 2 code and data.

LLM agents call the model, wait for a tool, and call the model again with a longer conversation. On one edge server
([llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, Llama-3.2-3B) with a fixed memory budget, admitting
few agents leaves the accelerator idle during tool waits, and admitting many makes them overwrite each other's cached
context. **Phase 1** measured this trade-off with a FIFO scheduler of fixed concurrency `K`. **Phase 2** adds a slot
manager that gives each agent slot affinity and **parks a waiting agent's KV cache on the local SSD** when its slot is
needed, then restores it when the agent returns.

- Final report: [`paper/final_report.pdf`](paper/final_report.pdf) (Phase 1 + Phase 2)
- Phase 1 report: [`paper/paper.pdf`](paper/paper.pdf)

## Phase 2 results at a glance

Held-out workload seeds 1000–1004, never used for development. 200 runs: 2 machines × 2 workloads × 10 configurations ×
5 seeds. Percentages compare configurations **on the same seed**.

| Finding | Evidence | Data |
| --- | --- | --- |
| With long tool calls (5–30 s), parking at `K = 8` lowers p95 task latency by 8–27% on the Mac and 18–36% on AWS, and raises throughput by up to 27% / 30%, against the strongest baseline (slot affinity at `K = 4`) on seeds 1001–1004. On seed 1000, the lightest, they tie. | Fig. 3, Table II of the final report | `runs/final_longtool`, `runs_aws/final_longtool` |
| Same `K = 8`, with vs. without parking: parking lowers p95 by 19–46% on the Mac and 1–10% on AWS (seeds 1001–1004). The Mac gains more because it recomputes context 4× more slowly. | same | same |
| With short tool calls (0.5–4 s), parking is within ±8% of affinity at `K = 4` on every seed, but 39–65% (Mac) and 9–25% (AWS) better than admitting `K = 8` without parking. | same | `runs*/final_mix` |
| Short tasks gain most: with long tool calls, chat p95 falls from 110 to 25 s (Mac) and 86 to 14 s (AWS) vs. FIFO `K = 4`. | per-class p95 | `runs*/final_longtool` |
| Parking is cheap: a save or restore takes a median of 34–80 ms; restore + resume is 10–28× faster than recomputing on the Mac and 5–16× on AWS, also when read from the SSD with the file cache dropped. Saves and restores use ≤ 2.4% of run time; server memory is unchanged (0.6–1.1 GB parked at peak). | Table III | `runs*/slot_probe`, `runs*/slot_probe_cold`, `context.jsonl` of every parking run |
| llama-server's own slot choice evicts waiting agents even when a free slot exists (79–85 of 225 returning steps lose their context with long tool calls at `K = 4`). Our affinity rule removes all of these and alone lowers p95 by 14–30% on the Mac. | Sec. V-D | `runs*/final_*/…/fifo_k4` vs. `slots_k4` |
| Parking needs tool time: on seeds 1001, 1002, 1004 with mean tool-call time swept 0–40 s, parking `K = 8` is 8–23% worse than affinity `K = 4` at 0 s, breaks even at 2–5 s, and at 40 s lowers p95 by 32–45% (Mac) / 32–47% (AWS) and raises throughput by 38% / 41%. | Fig. 5, Sec. V-B | `runs/delay`, `runs_aws/delay` |
| Negative results: a cost-aware variant (`park_cost`) behaved like plain `park`; restoring early could save at most 0.15 s per task; 8-bit KV with 8 slots plus parking moved latency from long to short tasks without raising throughput. | Sec. V-E | `runs/phase2_longtool`, `runs*/phase2_q8` |

Every number in the final report is recomputed from these folders by
[`paper/make_final_figures.py`](paper/make_final_figures.py), which writes them to
[`paper/numbers_final.json`](paper/numbers_final.json).

## How the Phase 2 policy works

All Phase 2 logic is client-side, in [`agentserve/context.py`](agentserve/context.py); llama.cpp is unmodified and only
needs `--slot-save-path`.

1. **Admission** stays FIFO with a limit `K`, which may now exceed the server's 4 slots.
2. **Slot choice.** Every request is sent with an explicit `id_slot`. An agent whose context is resident in an idle slot
   always gets that slot (affinity). Free slots go first to returning agents, then to new tasks. An empty slot is used
   if one exists; only when every idle slot holds a waiting agent's context is one evicted (least recently used).
3. **Parking.** Before evicting, the manager saves the slot's KV state (`POST /slots/{id}?action=save`) to a file. When
   the agent returns, it restores the file into a free slot (`action=restore`) and resumes with its whole history
   reused. Saving happens only when a slot is actually needed; each agent has at most one parked copy, deleted on
   restore or when the task ends.

Policies, selected by the `"policies"` list of a sweep config:

| Policy | Slot choice | Evicted context | Used in |
| --- | --- | --- | --- |
| `fifo` | llama-server decides | lost (recomputed) | Phase 1 baseline |
| `slots` | our affinity rule | dropped (recomputed) | "affinity" baseline |
| `park` | our affinity rule | **parked on disk, restored** | Phase 2 policy |
| `park_cost` | affinity; evicts the agent expected back last | parked, or dropped if < 600 tokens | development only (behaved like `park`) |

`slots` and `park` behave identically when no eviction is needed (e.g. at `K = 4` with 4 slots).

## Repository layout

| Path | Contents |
| --- | --- |
| `agentserve/workload.py` | Seeded workload generator: task classes, multi-step agents with growing context and tool delays, Poisson arrivals |
| `agentserve/runner.py` | Async runner: FIFO admission with limit `K`; uses the slot manager for Phase 2 policies |
| `agentserve/context.py` | **Phase 2 slot manager**: affinity, returning-agent priority, park / restore / drop, event log |
| `agentserve/client.py` | Streaming `/completion` client with fixed-length outputs (`ignore_eos` + `n_predict`) |
| `agentserve/monitor.py` | Samples `/metrics`, `/slots`, server RSS, macOS `footprint` and `nvidia-smi` every 0.5 s |
| `agentserve/server.py` | Starts a fresh `llama-server` per run, refuses a busy port, verifies the answering server is its own |
| `agentserve/sweep.py` | Grid sweep driver, resumable (`DONE` markers); creates and cleans the per-run parking folder |
| `agentserve/analyze.py` | Per-run metrics (`runs.csv`), mean ± std over seeds (`summary.csv`), figures |
| `configs/` | Workload mixes and one JSON file per sweep |
| `tools/mock_server.py` | Fake `llama-server`, including slot save/restore, for testing without a model (its numbers are not measurements) |
| `tools/slot_probe.py` | Save/restore vs. recompute probe; `--cold` drops the OS file cache before each restore |
| `tools/kv_quality.py` | 8-bit vs. f16 KV output-quality check |
| `tools/aws/` | AWS replica: start/stop, SSH, code sync, bootstrap, idle auto-stop, `mount_nvme.sh` for the local SSD |
| `paper/` | Final report (`final_report.tex/.pdf`, `make_final_figures.py`, `numbers_final.json`, `make_delay_figure.py`, `numbers_delay.json`) and Phase 1 report (`paper.tex/.pdf`, `make_figures.py`, `numbers.json`) |
| `runs/` | All Mac measurements |
| `runs_aws/` | All AWS measurements, same layout |

## Data

### Phase 2 experiments

| Experiment | Config | Runs per machine | Folder | In the final report |
| --- | --- | --- | --- | --- |
| Pipeline test (mock server) | `sweep_mock_phase2.json` | 4 | not committed | – |
| Smoke test, 4 policies | `sweep_phase2_smoke.json` | 4 | `runs*/phase2_smoke` | – |
| Development: 4 policies × 2 rates, seeds 0–2 (AWS) | `sweep_phase2_dev.json` | 24 | `runs_aws/phase2_dev` | – |
| Development: long tool calls, seeds 0–2 | `sweep_phase2_longtool.json` | 18 | `runs*/phase2_longtool` | Sec. V-E |
| Development: parking + 8-bit KV, 8 slots | `sweep_phase2_q8.json` | 12 | `runs*/phase2_q8` | Sec. V-E |
| **Final: long tool calls, seeds 1000–1004** | `sweep_final_longtool.json` + `_q8` | 45 + 5 | `runs*/final_longtool`, `runs*/final_longtool_q8` | Fig. 3, Tables II–III |
| **Final: short tool calls, seeds 1000–1004** | `sweep_final_mix.json` + `_q8` | 45 + 5 | `runs*/final_mix`, `runs*/final_mix_q8` | Fig. 3, Tables II–III |
| Tool-delay sweep: mean tool time 0, 2, 5, 10, 20, 40 s × FIFO K4 / affinity K4 / park K8, seeds 1001, 1002, 1004 | `sweep_delay{0,2,5,10,20,40}.json` (run all with `tools/run_feedback_sweeps.sh delay`) | 54 | `runs*/delay/d*` | Fig. 5, Sec. V-B |
| Save/restore probe (file cache / SSD) | `tools/slot_probe.py [--cold]` | 3 sizes | `runs*/slot_probe`, `runs*/slot_probe_cold` | Table III |

Workloads: `workload_mix.json` (Phase 1 mix, tool calls 0.5–4 s) and `workload_longtool.json` (same mix, tool calls
5–15 s for short agents and 10–30 s for long agents). The final sweeps use 0.1 tasks/s (long tool calls) and
0.2 tasks/s (short tool calls), so that both overload FIFO at `K = 4`.

### Phase 1 experiments

| Experiment | Config | Runs | Folder |
| --- | --- | --- | --- |
| Micro-benchmarks (`llama-bench`, `llama-batched-bench`) | commands below | – | `runs/micro`, `runs_aws/micro` |
| Pilot (calibrates arrival rates) | `sweep_pilot.json` | 2 + 2 | `runs/pilot`, `runs_aws/pilot` |
| FIFO `K` sweep | `sweep_ksweep.json` | 45 + 45 | `runs/ksweep`, `runs_aws/ksweep` |
| Host-RAM context retention | `sweep_cache.json` | 18 | `runs/cache` |
| Prompt caching off | `sweep_cache_off.json` | 3 | `runs/cache_off` |
| Memory split (`-c` × `-np`, 8-bit KV) | `sweep_slots.json` | 33 | `runs/slots` |
| 8-bit KV output quality | `tools/kv_quality.py` | 3 servers × 70 prompts | `runs/kvq` |
| GPU replicas of the retention, caching-off, memory-split and 8-bit quality experiments (Phase 1 feedback) | same configs, `tools/run_feedback_sweeps.sh gpu` | 18 + 3 + 33 + 210 prompts | `runs_aws/{cache,cache_off,slots,kvq}` |

### Files in each run

`runs/<sweep>/<server config>/<policy>_k<K>/rate<rate>/seed<seed>/` holds:

| File | Contents |
| --- | --- |
| `workload.json` | The exact tasks of this run (with the generator's SHA-256 hash) |
| `run_config.json` | All parameters, server command, `/props`, cold-start time, software versions; for Phase 2 policies also the policy and the peak parked bytes |
| `requests.jsonl` | One line per model call: send / first-token / end times, prompt tokens evaluated and served from cache, tokens generated, slot, server timings; for Phase 2 policies also `ctx_action` (`hit` / `restore` / `cold`), `slot_wait_ms` and the slot chosen by the manager |
| `tasks.jsonl` | One line per task: arrival, admission and end times, steps done, status (`ok` / `failed` / `timeout`) and error text |
| `context.jsonl` | Phase 2 policies only: every save, restore and drop, with slot, task, duration, bytes and tokens |
| `system.jsonl` | 0.5 s samples: busy slots, deferred requests, all `llamacpp:*` metrics, server RSS, macOS footprint or GPU memory |
| `server.log` | Full `llama-server` log, including model / KV / compute buffer sizes and slot selection |
| `DONE` | Written only when the run completed |

`runs/<sweep>/analysis/` holds `runs.csv` (one row per run), `summary.csv` (mean ± std over seeds) and figures, created by
`python -m agentserve.analyze runs/<sweep>`. `sweep_log.jsonl` in each sweep folder records every run start, finish and
crash.

### Metrics

- **p95 task latency**: arrival to completion of a task, including queueing and tool waits.
- **Throughput**: completed tasks divided by the time from the first arrival to the last completion.
- **Prompt-cache reuse**: prompt tokens served from a slot's cache / all prompt tokens (`timings.cache_n`).
- **Per-class p95**: the same latency for chat, RAG, short-agent and long-agent tasks.
- **Parking** (Phase 2): saves, restores and drops per run, their durations, peak bytes parked, and slot wait p95.
- **Failures** by cause: `context_exceeded` (task outgrew its slot) or `timeout` (task limit).
- **Memory**: allocated model / KV / compute buffers from `server.log`, plus peak footprint (Mac) or `nvidia-smi` (AWS).

### Not committed

Parked KV files are deleted after every run. The probe's saved states (`runs/slot_probe/slots/*.bin`) are not
committed; rerun the probe to recreate them. Runs disturbed by a leftover server during Phase 1 were excluded and rerun
(see the report's Experience section).

## Reproducing

### Software and hardware

| | Pinned version |
| --- | --- |
| llama.cpp | build b11803, commit `a37cc46104eb50900d5ab4a9ba02a0abdac039db` |
| Model | `bartowski/Llama-3.2-3B-Instruct-GGUF`, file `Llama-3.2-3B-Instruct-Q4_K_M.gguf`, SHA-256 `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| Python | 3.11+ with `requirements.txt` |
| Mac | Apple M2 Mac mini, 24 GB, macOS 26, Metal backend |
| AWS | g6f.2xlarge (NVIDIA L4-6Q, 6 GiB), Amazon Linux 2023, CUDA; parked contexts on the instance-store NVMe |

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
On AWS, run `bash tools/aws/mount_nvme.sh` after every instance start so parked contexts go to the local NVMe SSD
(`AGENTSERVE_SLOT_DIR`); by default they go to a `slots/` folder inside each run.

### Run Phase 2

All commands run from `agentserve/`, one sweep at a time (each uses port 8080). Inside `tmux` is recommended; on a Mac,
prefix long runs with `caffeinate -i`. Sweeps resume where they stopped.

```bash
# 0. Pipeline test without a model, all four policies (~6 min)
python -m agentserve.sweep configs/sweep_mock_phase2.json && python -m agentserve.analyze runs/mock_phase2

# 1. Save/restore probe, from the file cache and from the SSD (~2 min each; --cold needs sudo)
python tools/slot_probe.py
sudo -v && python tools/slot_probe.py --cold --out runs/slot_probe_cold

# 2. Final evaluation on held-out seeds 1000-1004 (~5 h per workload on the M2)
for c in final_longtool final_longtool_q8 final_mix final_mix_q8; do
  python -m agentserve.sweep configs/sweep_$c.json && python -m agentserve.analyze runs/$c
done

# 3. Final report figures and numbers (reads runs/ and runs_aws/; analyze both first)
python paper/make_final_figures.py
cd paper && latexmk -pdf final_report.tex
```

Development sweeps (`sweep_phase2_smoke.json`, `sweep_phase2_dev.json`, `sweep_phase2_longtool.json`,
`sweep_phase2_q8.json`) run the same way. `python -m agentserve.sweep <config> --dry-run` lists a sweep's runs;
`--max-runs N` stops after N (useful under a daily cloud budget). For the AWS replica, see
[`tools/aws/README.md`](tools/aws/README.md); results are copied back into `runs_aws/`.

### Run Phase 1

```bash
python -m agentserve.sweep configs/sweep_mock.json && python -m agentserve.analyze runs/mock     # pipeline test

B=../llama.cpp/build/bin; M=../models/Llama-3.2-3B-Instruct-Q4_K_M.gguf; mkdir -p runs/micro
$B/llama-bench -m $M -ngl 99 -p 256,1024,2048,4096 -n 128 -r 5 -o csv > runs/micro/llama_bench.csv
$B/llama-batched-bench -m $M -ngl 99 -c 16384 -npp 512,2048 -ntg 128 -npl 1,2,4,8 > runs/micro/batched_bench.txt 2>&1

for c in pilot ksweep cache cache_off slots; do
  python -m agentserve.sweep configs/sweep_$c.json && python -m agentserve.analyze runs/$c
done
python tools/kv_quality.py
python paper/make_figures.py && (cd paper && latexmk -pdf paper.tex)
```

## Experimental controls

- **Fixed output length.** Every step generates exactly its `n_predict` tokens, so no policy can win by shortening answers.
- **Fresh server per run** with two warm-up requests; cold start is reported separately and excluded from latency.
- **Fixed memory budget.** llama.cpp's host-RAM prompt cache is disabled (`--cache-ram 0`) except where it is the
  variable under study, so server memory equals the model, KV and compute allocation; parked contexts live on disk.
- **Held-out evaluation.** Development used seeds 0–99 (0–2 in practice); the final evaluation uses seeds 1000–1004 only.
- **Paired comparisons.** Each configuration runs once per seed; seeds differ strongly in load, so results are compared
  on the same seed and every seed is reported.
- **Nothing dropped.** Failed and timed-out tasks stay in `tasks.jsonl`; crashed runs get a `CRASHED` file and a line in
  `sweep_log.jsonl`.
- **Server identity check.** A run refuses to start if its port is busy and verifies the answering server's slot count.

## Limitations

The workload is synthetic (filler text, fixed output lengths, chosen tool delays). One 3B model on llama.cpp's fixed
slots is evaluated; paged servers such as vLLM behave differently. Parked files pass through the OS file cache; the
`--cold` probe shows parking stays far cheaper than recomputation when they are read from the SSD.
