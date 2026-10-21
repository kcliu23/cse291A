"""Resumable experiment sweep.

For every combination of server config x policy x K x arrival rate x seed:
generate the workload, start a fresh server (no KV-cache carry-over between
runs), run the workload, stop the server. Finished runs contain a DONE file and
are skipped on the next invocation, so a sweep can stop at the daily budget
and resume the next day. Runs that crash are logged to sweep_log.jsonl, not
silently dropped.

Usage:
    python -m agentserve.sweep configs/sweep_phase1.json            # run everything left
    python -m agentserve.sweep configs/sweep_phase1.json --dry-run  # list pending runs
    python -m agentserve.sweep configs/sweep_phase1.json --max-runs 10
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import traceback
from pathlib import Path

from .runner import run_once
from .server import Server
from .workload import generate


def run_dir(root: Path, srv: dict, policy: str, k: int, rate: float, seed: int, slot_policy: str) -> Path:
    tag = f"c{srv['ctx']}_np{srv['parallel']}"
    pol = policy if slot_policy == "auto" else f"{policy}-{slot_policy}"
    return root / tag / f"{pol}_k{k}" / f"rate{rate}" / f"seed{seed}"


def plan(cfg: dict) -> list[dict]:
    root = Path(cfg["out_root"])
    runs = []
    for srv_over, policy, k, rate, seed, slot_policy in itertools.product(
            cfg["server_grid"], cfg.get("policies", ["fifo"]), cfg["k"], cfg["rates"],
            cfg["seeds"], cfg.get("slot_policies", ["auto"])):
        srv = {**cfg["server"], **srv_over}
        d = run_dir(root, srv, policy, k, rate, seed, slot_policy)
        runs.append({"dir": d, "server": srv, "policy": policy, "k": k, "rate": rate,
                     "seed": seed, "slot_policy": slot_policy, "done": (d / "DONE").exists()})
    return runs


def execute(cfg: dict, r: dict, log_f) -> None:
    d: Path = r["dir"]
    d.mkdir(parents=True, exist_ok=True)
    wl_cfg = json.loads(Path(cfg["workload_config"]).read_text())
    wl = generate(wl_cfg, r["seed"], r["rate"], cfg["n_tasks"])
    wl_path = d / "workload.json"
    wl_path.write_text(json.dumps(wl))
    t = time.time()
    status, err = "ok", None
    try:
        with Server(r["server"], d / "server.log") as srv:
            asyncio.run(run_once(
                server=srv.url, workload_path=str(wl_path), out=str(d), policy=r["policy"], k=r["k"],
                task_timeout_s=cfg.get("task_timeout_s", 300),
                request_timeout_s=cfg.get("request_timeout_s", 300),
                slot_policy=r["slot_policy"], cache_prompt=cfg.get("cache_prompt", True),
                warmup=cfg.get("warmup", 2), server_pid=srv.pid,
                monitor_interval_s=cfg.get("monitor_interval_s", 0.5),
                extra_meta={"server_config": r["server"], "server_startup_s": srv.startup_s,
                            "rate": r["rate"], "workload_seed": r["seed"]}))
    except Exception as e:  # record and continue
        status, err = "crashed", "".join(traceback.format_exception_only(e)).strip()
        (d / "CRASHED").write_text(traceback.format_exc())
        print(f"[sweep] CRASHED {d}: {err}")
    log_f.write(json.dumps({"dir": str(d), "status": status, "error": err,
                            "start_unix": t, "elapsed_s": round(time.time() - t, 2)}) + "\n")
    log_f.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    runs = plan(cfg)
    pending = [r for r in runs if not r["done"]]
    print(f"[sweep] {len(runs)} runs total, {len(runs) - len(pending)} done, {len(pending)} pending")
    if a.dry_run:
        for r in pending:
            print("  ", r["dir"])
        return
    root = Path(cfg["out_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_config.json").write_text(json.dumps(cfg, indent=2))
    with open(root / "sweep_log.jsonl", "a") as log_f:
        for i, r in enumerate(pending[: a.max_runs]):
            print(f"[sweep] ({i + 1}/{min(len(pending), a.max_runs or len(pending))}) {r['dir']}")
            execute(cfg, r, log_f)


if __name__ == "__main__":
    main()
