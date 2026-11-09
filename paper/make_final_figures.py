"""Figures and numbers for the final report (Phase 2), from the held-out evaluation runs.

Usage (repo root, after `python -m agentserve.analyze` on each runs*/final_* folder):
    python paper/make_final_figures.py                 # Mac runs in runs/, AWS runs in runs_aws/
    python paper/make_final_figures.py --root /path
Writes paper/fig_final.pdf and paper/numbers_final.json (every number the report quotes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
CELLS = [("mac", "longtool", "M2 Mac, long tool calls"), ("aws", "longtool", "AWS L4, long tool calls"),
         ("mac", "mix", "M2 Mac, short tool calls"), ("aws", "mix", "AWS L4, short tool calls")]
CONFIGS = [("base fifo K4", "FIFO\nK=4"), ("base slots K4", "Affinity\nK=4"), ("kvq8 fifo K8", "8-bit\nK=8"),
           ("base fifo K8", "FIFO\nK=8"), ("base park K8", "Park\nK=8")]
BASE = "base fifo K4"


def load(root: Path, mach: str, wl: str) -> pd.DataFrame:
    d = root / ("runs" if mach == "mac" else "runs_aws")
    df = pd.concat([pd.read_csv(d / f"final_{wl}" / "analysis" / "runs.csv"),
                    pd.read_csv(d / f"final_{wl}_q8" / "analysis" / "runs.csv")])
    df = df[df.status == "done"].copy()
    df["cfg"] = df.variant.fillna("base") + " " + df.policy + " K" + df.k.astype(int).astype(str)
    return df


def overhead(run_dirs: list[str]) -> dict:
    save, restore, duty, n_save = [], [], [], []
    for r in run_dirs:
        r = Path(r)
        ev = [json.loads(x) for x in open(r / "context.jsonl")] if (r / "context.jsonl").exists() else []
        req = [json.loads(x) for x in open(r / "requests.jsonl")]
        span = max(q["t_end"] for q in req if q.get("t_end") is not None) - min(q["t_send"] for q in req if q.get("t_send") is not None)
        s = [e["ms"] for e in ev if e["action"] == "save"]
        rs = [e["ms"] for e in ev if e["action"] == "restore"]
        save += s
        restore += rs
        n_save.append(len(s))
        duty.append((sum(s) + sum(rs)) / 1000 / span)
    q = lambda v, p: float(np.percentile(v, p)) if v else float("nan")  # noqa: E731
    return {"save_ms_median": q(save, 50), "save_ms_p95": q(save, 95), "restore_ms_median": q(restore, 50),
            "restore_ms_p95": q(restore, 95), "duty_mean": float(np.mean(duty)), "duty_max": float(np.max(duty)),
            "saves_per_run": float(np.mean(n_save))}


def affinity(run_dirs: list[str]) -> dict:
    """For runs where the server picks slots: how often a returning agent's context was gone, and
    whether a free slot existed when its slot was taken."""
    gone = same = evict = evict_free = 0
    for r in run_dirs:
        reqs = [json.loads(x) for x in open(Path(r) / "requests.jsonl")]
        reqs = [q for q in reqs if q["status"] == "ok" and q.get("id_slot") is not None]
        last = {}
        for q in reqs:
            last[q["task_id"]] = max(last.get(q["task_id"], 0), q["step"])
        ev = sorted([(q["t_send"], 0, q) for q in reqs] + [(q["t_end"], 1, q) for q in reqs],
                    key=lambda x: (x[0], -x[1]))
        owner, busy, done = {}, set(), set()
        for _, kind, q in ev:
            s = q["id_slot"]
            if kind == 1:
                busy.discard(s)
                if q["step"] == last[q["task_id"]]:
                    done.add(q["task_id"])
                continue
            prev = owner.get(s)
            if prev is not None and prev != q["task_id"] and prev not in done:
                evict += 1
                n_slots = max(4, max(owner) + 1)
                free = [x for x in range(n_slots) if x != s and x not in busy and (owner.get(x) is None or owner[x] in done)]
                evict_free += bool(free)
            if q["step"] > 0:
                mine = [x for x, o in owner.items() if o == q["task_id"]]
                if mine and mine[0] == s:
                    same += 1
                else:
                    gone += 1
            for x, o in list(owner.items()):
                if o == q["task_id"] and x != s:
                    owner[x] = None
            owner[s] = q["task_id"]
            busy.add(s)
    return {"returning_steps": same + gone, "context_gone": gone, "evictions": evict,
            "evictions_with_free_slot": evict_free}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE.parent)
    a = ap.parse_args()
    nums: dict = {"cells": {}}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.1), sharey="row")
    for j, (mach, wl, title) in enumerate(CELLS):
        df = load(a.root, mach, wl)
        p95 = df.pivot_table(index="seed", columns="cfg", values="lat_p95")
        tput = df.pivot_table(index="seed", columns="cfg", values="throughput_tasks_s")
        cell = {"p95": p95.round(2).to_dict(), "tput": tput.round(4).to_dict()}
        for col in ("lat_p95_chat", "lat_p95_agent_long", "cache_hit_ratio", "peak_parked_mb",
                    "peak_footprint_mb", "peak_gpu_mem_mb"):
            if col in df:
                cell[col] = df.groupby("cfg")[col].mean().round(3).dropna().to_dict()
        park_runs = df[df.cfg.isin(["base park K8", "base park K16"])].run.tolist()
        cell["overhead_park"] = overhead(park_runs)
        cell["affinity_fifo_k4"] = affinity(df[df.cfg == BASE].run.tolist())
        cell["affinity_slots_k4"] = affinity(df[df.cfg == "base slots K4"].run.tolist())
        nums["cells"][f"{mach}_{wl}"] = cell
        for i, (tab, label) in enumerate(((p95, "p95 latency vs. FIFO K=4"), (tput, "throughput vs. FIFO K=4"))):
            ax = axes[i, j]
            rel = (tab.div(tab[BASE], axis=0) - 1) * 100
            for x, (cfg, name) in enumerate(CONFIGS[1:]):
                v = rel[cfg]
                good = (v.mean() < 0) if i == 0 else (v.mean() > 0)
                color = "#1f5fbf" if cfg == "base park K8" else "0.6"
                ax.bar(x, v.mean(), color=color, width=0.62, alpha=0.9 if cfg == "base park K8" else 0.7)
                for seed, y in v.items():
                    ax.plot(x, y, "o", ms=2.6, mfc="white" if seed == 1000 else "black", mec="black", mew=0.5)
            ax.axhline(0, color="0.3", lw=0.6)
            ax.set_xticks(range(len(CONFIGS) - 1), [n for _, n in CONFIGS[1:]])
            if j == 0:
                ax.set_ylabel(f"{'p95 latency' if i == 0 else 'throughput'}\nvs. FIFO K=4 (%)")
            if i == 0:
                ax.set_title(title)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.6)
    fig.savefig(HERE / "fig_final.pdf", bbox_inches="tight")
    (HERE / "numbers_final.json").write_text(json.dumps(nums, indent=1, default=float))
    print("wrote fig_final.pdf, numbers_final.json")


if __name__ == "__main__":
    main()
