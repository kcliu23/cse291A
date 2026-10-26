"""Agent workload runner with a pluggable admission policy.

Phase 1 baseline: FIFO admission with fixed concurrency K. A task is admitted
when fewer than K tasks are active, in arrival order, and it holds its
admission slot for its whole lifetime, including tool waits between steps.

Outputs (in --out):
    run_config.json   all parameters, workload hash, server /props, versions
    requests.jsonl    one line per model request (step)
    tasks.jsonl       one line per task, including failed and timed-out tasks
    system.jsonl      system samples (see monitor.py)

Usage:
    python -m agentserve.runner --server http://127.0.0.1:8080 \
        --workload workloads/dev.json --k 4 --out runs/test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import time
from collections import deque
from pathlib import Path

import aiohttp

from .client import CompletionError, complete
from .monitor import Monitor


# ---------------------------------------------------------------- admission policies
class FIFOAdmission:
    """Admit tasks in arrival order, at most k active at a time."""

    name = "fifo"

    def __init__(self, k: int):
        self.k = k
        self.active = 0
        self.waiters: deque[asyncio.Future] = deque()

    async def acquire(self, task: dict) -> None:
        if self.active < self.k and not self.waiters:
            self.active += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut in self.waiters:
                self.waiters.remove(fut)
            elif fut.done() and not fut.cancelled():
                self.release(task)  # granted just before cancellation: hand it on
            raise

    def release(self, task: dict) -> None:
        while self.waiters:
            fut = self.waiters.popleft()
            if not fut.done():
                fut.set_result(None)  # transfer the slot; active count unchanged
                return
        self.active -= 1


class SlotAssigner:
    """Optional slot pinning: each admitted task is pinned to the least-loaded server slot."""

    def __init__(self, n_slots: int):
        self.load = [0] * n_slots

    def assign(self) -> int:
        i = min(range(len(self.load)), key=lambda j: self.load[j])
        self.load[i] += 1
        return i

    def free(self, i: int) -> None:
        self.load[i] -= 1


# ---------------------------------------------------------------- runner
class Runner:
    def __init__(self, base_url: str, workload: dict, admission, out_dir: Path, *,
                 task_timeout_s: float, request_timeout_s: float, slot_policy: str,
                 n_slots: int | None, cache_prompt: bool, seed: int):
        self.base_url = base_url
        self.wl = workload
        self.adm = admission
        self.out = out_dir
        self.task_timeout_s = task_timeout_s
        self.request_timeout_s = request_timeout_s
        self.slot_policy = slot_policy
        self.slots = SlotAssigner(n_slots) if slot_policy == "pin" and n_slots else None
        self.cache_prompt = cache_prompt
        self.seed = seed
        self.req_records: list[dict] = []
        self.task_records: list[dict] = []
        self.t0 = 0.0

    def rel(self, t: float | None) -> float | None:
        return None if t is None else round(t - self.t0, 6)

    async def run_task(self, s: aiohttp.ClientSession, task: dict) -> None:
        delay = self.t0 + task["arrival_s"] - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        rec = {"task_id": task["id"], "cls": task["cls"], "n_steps": len(task["steps"]),
               "t_arrival": task["arrival_s"], "t_admit": None, "t_end": None,
               "steps_done": 0, "status": "ok", "error": None, "id_slot_pinned": None}
        state = {"admitted": False, "slot": None}
        try:
            await asyncio.wait_for(self._task_body(s, task, rec, state), timeout=self.task_timeout_s)
        except (TimeoutError, asyncio.TimeoutError):  # distinct before Python 3.11
            rec["status"] = "timeout"
            rec["error"] = f"task exceeded {self.task_timeout_s}s"
        except CompletionError as e:
            rec["status"] = "failed"
            rec["error"] = f"{e.kind}: {e.message}"
        finally:
            if state["admitted"]:
                self.adm.release(task)
            if state["slot"] is not None:
                self.slots.free(state["slot"])
            rec["t_end"] = self.rel(time.monotonic())
            self.task_records.append(rec)

    async def _task_body(self, s, task, rec, state) -> None:
        await self.adm.acquire(task)
        state["admitted"] = True
        rec["t_admit"] = self.rel(time.monotonic())
        if self.slots is not None:
            state["slot"] = self.slots.assign()
            rec["id_slot_pinned"] = state["slot"]

        history = self.wl["shared_prefix"] + task["context"]
        t_ready = time.monotonic()
        for i, step in enumerate(task["steps"]):
            prompt = history + step["input"]
            req = {"task_id": task["id"], "cls": task["cls"], "step": i,
                   "t_ready": self.rel(t_ready), "prompt_chars": len(prompt),
                   "n_predict": step["n_predict"], "status": "ok", "error": None}
            try:
                r = await complete(s, self.base_url, prompt, step["n_predict"],
                                   id_slot=state["slot"], cache_prompt=self.cache_prompt,
                                   seed=self.seed, timeout_s=self.request_timeout_s)
            except CompletionError as e:
                req.update(status="failed", error=f"{e.kind}: {e.message}",
                           t_end=self.rel(time.monotonic()))
                self.req_records.append(req)
                raise
            except asyncio.CancelledError:
                req.update(status="cancelled", t_end=self.rel(time.monotonic()))
                self.req_records.append(req)
                raise
            content = r.pop("content")
            for k in ("t_send", "t_first", "t_end"):
                r[k] = self.rel(r[k])
            req.update(r)
            self.req_records.append(req)
            rec["steps_done"] = i + 1
            history = prompt + content
            if step["tool_delay_s"] > 0:
                await asyncio.sleep(step["tool_delay_s"])
            history += step["tool_output"]
            t_ready = time.monotonic()

    async def warmup(self, s: aiohttp.ClientSession, n: int) -> None:
        for i in range(n):
            await complete(s, self.base_url, f"Warmup request {i}: hello", 8,
                           cache_prompt=False, timeout_s=120)

    async def run(self, warmup: int, server_pid: int | None, monitor_interval_s: float) -> None:
        # force_close: one fresh connection per request. llama-server closes a connection right
        # after a streamed reply; reusing it (keep-alive) races with that close and fails with
        # "Server disconnected" on back-to-back requests.
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        async with aiohttp.ClientSession(connector=connector, read_bufsize=2**22) as s:
            if warmup:
                await self.warmup(s, warmup)
            self.t0 = time.monotonic()
            stop = asyncio.Event()
            mon = Monitor(self.base_url, self.out / "system.jsonl", self.t0,
                          interval_s=monitor_interval_s, server_pid=server_pid)
            mon_task = asyncio.create_task(mon.run(stop))
            await asyncio.gather(*(self.run_task(s, t) for t in self.wl["tasks"]))
            stop.set()
            await mon_task
        self._write()

    def _write(self) -> None:
        with open(self.out / "requests.jsonl", "w") as f:
            for r in sorted(self.req_records, key=lambda r: (r["task_id"], r["step"])):
                f.write(json.dumps(r) + "\n")
        with open(self.out / "tasks.jsonl", "w") as f:
            for r in sorted(self.task_records, key=lambda r: r["task_id"]):
                f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- helpers
def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def _get_json(base_url: str, path: str) -> dict | list | None:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(base_url + path) as r:
                return await r.json() if r.status == 200 else None
    except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError, ValueError):
        return None


def make_admission(policy: str, k: int):
    if policy == "fifo":
        return FIFOAdmission(k)
    raise ValueError(f"unknown policy {policy!r}")


async def run_once(*, server: str, workload_path: str, out: str, policy: str = "fifo", k: int = 4,
                   task_timeout_s: float = 300, request_timeout_s: float = 300,
                   slot_policy: str = "auto", cache_prompt: bool = True, seed: int = 0,
                   warmup: int = 2, server_pid: int | None = None,
                   monitor_interval_s: float = 0.5, extra_meta: dict | None = None) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wl = json.loads(Path(workload_path).read_text())
    props = await _get_json(server, "/props")
    slots = await _get_json(server, "/slots")
    n_slots = len(slots) if isinstance(slots, list) else (props or {}).get("total_slots")
    if slot_policy == "pin" and not n_slots:
        raise SystemExit("slot_policy=pin needs the slot count from /slots or /props")

    cfg = {
        "policy": policy, "k": k, "server": server, "workload_path": str(workload_path),
        "workload_meta": wl["meta"], "task_timeout_s": task_timeout_s,
        "request_timeout_s": request_timeout_s, "slot_policy": slot_policy,
        "cache_prompt": cache_prompt, "seed": seed, "warmup": warmup, "n_slots": n_slots,
        "server_props": props, "python": platform.python_version(), "platform": platform.platform(),
        "agentserve_commit": _git_commit(Path(__file__).resolve().parent),
        "started_unix": time.time(), **(extra_meta or {}),
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    runner = Runner(server, wl, make_admission(policy, k), out_dir,
                    task_timeout_s=task_timeout_s, request_timeout_s=request_timeout_s,
                    slot_policy=slot_policy, n_slots=n_slots, cache_prompt=cache_prompt, seed=seed)
    t = time.monotonic()
    await runner.run(warmup, server_pid, monitor_interval_s)
    cfg["wall_s"] = round(time.monotonic() - t, 3)
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "DONE").write_text("")
    n_ok = sum(1 for r in runner.task_records if r["status"] == "ok")
    print(f"[run] {out_dir}: {n_ok}/{len(runner.task_records)} tasks ok, wall {cfg['wall_s']:.1f}s")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="fifo")
    ap.add_argument("--k", type=int, default=4, help="max concurrently admitted tasks")
    ap.add_argument("--task-timeout", type=float, default=300)
    ap.add_argument("--request-timeout", type=float, default=300)
    ap.add_argument("--slot-policy", choices=["auto", "pin"], default="auto",
                    help="auto: server picks the slot; pin: each task keeps one slot")
    ap.add_argument("--no-cache-prompt", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--server-pid", type=int, default=None, help="llama-server PID for RSS sampling")
    ap.add_argument("--monitor-interval", type=float, default=0.5)
    a = ap.parse_args()
    asyncio.run(run_once(server=a.server, workload_path=a.workload, out=a.out, policy=a.policy, k=a.k,
                         task_timeout_s=a.task_timeout, request_timeout_s=a.request_timeout,
                         slot_policy=a.slot_policy, cache_prompt=not a.no_cache_prompt,
                         warmup=a.warmup, server_pid=a.server_pid,
                         monitor_interval_s=a.monitor_interval))


if __name__ == "__main__":
    main()
