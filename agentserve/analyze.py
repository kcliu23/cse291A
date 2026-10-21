"""Summarize sweep runs and draw the Phase 1 figures.

Outputs (in --out, default <root>/analysis):
    runs.csv       one row per run: completion, latency, queueing, throughput,
                   memory, cache reuse, idle-hold fraction, failure reasons
    summary.csv    mean, std and n across seeds for every config
    k_sweep_<server>.png      p95 latency, throughput, failures, idle-hold vs K
    timeline_<run>.png        slots busy, deferred requests, KV/GPU memory over time

Crashed runs (CRASHED file, no DONE) are listed in runs.csv with status=crashed.

Usage:
    python -m agentserve.analyze runs/phase1
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

# Ordinal blue ramp (low -> high arrival rate) and text inks.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK_2, GRID = "#1f1f1e", "#5c5b55", "#e4e3dc"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _pct(x, q):
    x = pd.Series(x).dropna()
    return float(np.percentile(x, q)) if len(x) else np.nan


def _err_kind(err: str | None) -> str:
    if not isinstance(err, str) or not err:
        return ""
    e = err.lower()
    if "context" in e:
        return "context_exceeded"
    if e.startswith("timeout") or e.startswith("task exceeded"):
        return "timeout"
    return err.split(":")[0]


def summarize_run(d: Path) -> dict:
    cfg = json.loads((d / "run_config.json").read_text()) if (d / "run_config.json").exists() else {}
    srv = cfg.get("server_config", {})
    row = {
        "run": str(d), "status": "done" if (d / "DONE").exists() else "crashed",
        "ctx": srv.get("ctx"), "parallel": srv.get("parallel"),
        "policy": cfg.get("policy"), "slot_policy": cfg.get("slot_policy"),
        "k": cfg.get("k"), "rate": cfg.get("rate"), "seed": cfg.get("workload_seed"),
        "server_startup_s": cfg.get("server_startup_s"),
    }
    tasks = pd.DataFrame(_jsonl(d / "tasks.jsonl"))
    reqs = pd.DataFrame(_jsonl(d / "requests.jsonl"))
    sysd = _jsonl(d / "system.jsonl")
    if tasks.empty:
        return row

    ok = tasks[tasks.status == "ok"]
    lat = ok.t_end - ok.t_arrival
    queue = (tasks.t_admit - tasks.t_arrival).dropna()
    makespan = tasks.t_end.max() - tasks.t_arrival.min()
    row.update({
        "n_tasks": len(tasks), "n_ok": len(ok),
        "n_failed": int((tasks.status == "failed").sum()),
        "n_timeout": int((tasks.status == "timeout").sum()),
        "fail_rate": 1 - len(ok) / len(tasks),
        "lat_p50": _pct(lat, 50), "lat_p95": _pct(lat, 95), "lat_p99": _pct(lat, 99),
        "lat_mean": float(lat.mean()) if len(lat) else np.nan,
        "queue_p50": _pct(queue, 50), "queue_p95": _pct(queue, 95),
        "makespan_s": makespan,
        "throughput_tasks_s": len(ok) / makespan if makespan > 0 else np.nan,
    })
    for cls, g in ok.groupby("cls"):
        row[f"lat_p95_{cls}"] = _pct(g.t_end - g.t_arrival, 95)
    kinds = tasks.error.map(_err_kind).value_counts()
    for k, v in kinds.items():
        if k:
            row[f"err_{k}"] = int(v)

    if not reqs.empty:
        r_ok = reqs[reqs.status == "ok"]
        row["n_requests"] = len(reqs)
        row["gen_tokens"] = float(r_ok.tokens_predicted.sum())
        row["gen_tokens_s"] = row["gen_tokens"] / makespan if makespan > 0 else np.nan
        row["ttft_p50"] = _pct(r_ok.t_first - r_ok.t_send, 50)
        row["ttft_p95"] = _pct(r_ok.t_first - r_ok.t_send, 95)
        ev = r_ok.tokens_evaluated.sum()
        row["prompt_tokens"] = float(ev)
        row["cache_hit_ratio"] = float(r_ok.tokens_cached.sum() / ev) if ev else np.nan
        t = pd.json_normalize(r_ok.timings)
        if "predicted_ms" in t and "predicted_n" in t:
            row["decode_tok_s_mean"] = float((t.predicted_n / (t.predicted_ms / 1000)).mean())
        if "prompt_ms" in t:
            row["prefill_ms_p50"] = _pct(t.prompt_ms, 50)

        # Idle-hold: share of admitted task time spent waiting on tools, i.e.
        # holding an admission slot while the model is not working for the task.
        r_ok = r_ok.sort_values(["task_id", "step"])
        prev_end = r_ok.groupby("task_id").t_end.shift()
        tool_wait = (r_ok.t_ready - prev_end).clip(lower=0).groupby(r_ok.task_id).sum()
        held = (tasks.set_index("task_id").t_end - tasks.set_index("task_id").t_admit).dropna()
        common = held.index.intersection(tool_wait.index)
        if held.loc[common].sum() > 0:
            row["idle_hold_frac"] = float(tool_wait.loc[common].sum() / held.loc[common].sum())

    if sysd:
        s = pd.json_normalize(sysd)
        for col, name in [("rss_mb", "peak_rss_mb"), ("gpu_mem_mb", "peak_gpu_mem_mb"),
                          ("metrics.llamacpp:kv_cache_usage_ratio", "peak_kv_usage"),
                          ("metrics.llamacpp:requests_deferred", "peak_deferred")]:
            if col in s:
                row[name] = float(pd.to_numeric(s[col], errors="coerce").max())
        if "slots_busy" in s and "n_slots" in s:
            row["slot_util_mean"] = float((s.slots_busy / s.n_slots).mean())
    return row


def load(root: Path) -> pd.DataFrame:
    dirs = sorted({p.parent for p in root.rglob("run_config.json")} | {p.parent for p in root.rglob("CRASHED")})
    return pd.DataFrame([summarize_run(d) for d in dirs])


GROUP = ["ctx", "parallel", "policy", "slot_policy", "k", "rate"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    done = df[df.status == "done"]
    num = [c for c in done.columns if c not in GROUP + ["run", "status", "seed"]
           and pd.api.types.is_numeric_dtype(done[c])]
    g = done.groupby(GROUP, dropna=False)[num]
    out = g.agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    crashed = df[df.status == "crashed"].groupby(GROUP, dropna=False).size()
    out["n_crashed"] = crashed.reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


def _style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("K (max concurrent tasks)", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_2)


def plot_k_sweep(summary: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    panels = [("lat_p95", "p95 task completion latency", "seconds"),
              ("throughput_tasks_s", "Completed tasks per second", "tasks/s"),
              ("fail_rate", "Failed or timed-out tasks", "fraction of tasks"),
              ("idle_hold_frac", "Admitted time spent waiting on tools", "fraction")]
    for (ctx, par, pol, sp), g in summary.groupby(["ctx", "parallel", "policy", "slot_policy"], dropna=False):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        rates = sorted(g.rate.unique())
        for ax, (m, title, yl) in zip(axes.flat, panels):
            _style(ax, title, yl)
            if f"{m}_mean" not in g:
                ax.text(0.5, 0.5, "not available", ha="center", transform=ax.transAxes, color=INK_2)
                continue
            for i, rate in enumerate(rates):
                h = g[g.rate == rate].sort_values("k")
                c = RAMP[min(i + (len(RAMP) - len(rates)), len(RAMP) - 1)] if len(rates) <= len(RAMP) else RAMP[-1]
                ax.errorbar(h.k, h[f"{m}_mean"], yerr=h[f"{m}_std"].fillna(0), color=c,
                            marker="o", markersize=6, linewidth=2, capsize=3,
                            label=f"{rate} tasks/s")
            ax.set_xscale("log", base=2)
            ks = sorted(g.k.unique())
            ax.set_xticks(ks)
            ax.set_xticklabels([str(int(k)) for k in ks])
            if m == "fail_rate":
                ax.set_ylim(bottom=0)
        axes.flat[0].legend(title="Arrival rate", frameon=False, fontsize=9, title_fontsize=9)
        n = int(g.n_seeds.max())
        fig.suptitle(f"FIFO, fixed K sweep: -c {ctx}, -np {par} ({pol}, slots={sp}); "
                     f"mean ± std over {n} seeds", color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout()
        p = out / f"k_sweep_c{ctx}_np{par}_{pol}_{sp}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_server_grid(summary: pd.DataFrame, out: Path) -> list[Path]:
    """For sweeps over -c x -np at a fixed K and rate: metrics vs slots, one line per total context."""
    paths = []
    for (pol, sp, k, rate), g in summary.groupby(["policy", "slot_policy", "k", "rate"], dropna=False):
        if g.parallel.nunique() < 2:
            continue
        panels = [("lat_p95", "p95 task completion latency", "seconds"),
                  ("throughput_tasks_s", "Completed tasks per second", "tasks/s"),
                  ("fail_rate", "Failed or timed-out tasks", "fraction of tasks"),
                  ("peak_gpu_mem_mb" if "peak_gpu_mem_mb_mean" in g else "peak_rss_mb",
                   "Peak memory", "MB")]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
        ctxs = sorted(g.ctx.unique())
        for ax, (m, title, yl) in zip(axes.flat, panels):
            _style(ax, title, yl)
            ax.set_xlabel("server slots (-np)", color=INK_2)
            if f"{m}_mean" not in g:
                continue
            for i, ctx in enumerate(ctxs):
                h = g[g.ctx == ctx].sort_values("parallel")
                c = RAMP[min(i + (len(RAMP) - len(ctxs)), len(RAMP) - 1)] if len(ctxs) <= len(RAMP) else RAMP[-1]
                ax.errorbar(h.parallel, h[f"{m}_mean"], yerr=h[f"{m}_std"].fillna(0), color=c,
                            marker="o", markersize=6, linewidth=2, capsize=3, label=f"-c {int(ctx)}")
            ps = sorted(g.parallel.unique())
            ax.set_xscale("log", base=2)
            ax.set_xticks(ps)
            ax.set_xticklabels([str(int(p)) for p in ps])
            if m == "fail_rate":
                ax.set_ylim(bottom=0)
        axes.flat[0].legend(title="Total KV context", frameon=False, fontsize=9, title_fontsize=9)
        fig.suptitle(f"Context per slot vs. slot count: {pol}, K={k}, {rate} tasks/s; "
                     f"mean ± std over {int(g.n_seeds.max())} seeds", color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout()
        p = out / f"server_grid_{pol}_{sp}_k{k}_rate{rate}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_timeline(run: Path, out: Path) -> Path | None:
    sysd = _jsonl(run / "system.jsonl")
    if not sysd:
        return None
    s = pd.json_normalize(sysd)
    series = [("slots_busy", "Busy server slots"),
              ("metrics.llamacpp:requests_deferred", "Requests waiting for a slot"),
              ("metrics.llamacpp:kv_cache_usage_ratio", "KV cache usage ratio"),
              ("gpu_mem_mb", "GPU memory used (MB)"),
              ("rss_mb", "Server RSS (MB)")]
    series = [(c, t) for c, t in series if c in s and pd.to_numeric(s[c], errors="coerce").notna().any()]
    fig, axes = plt.subplots(len(series), 1, figsize=(9, 1.9 * len(series)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (c, t) in zip(axes, series):
        ax.plot(s.t, pd.to_numeric(s[c], errors="coerce"), color=RAMP[1], linewidth=2)
        ax.set_title(t, color=INK, fontsize=10, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK_2)
    axes[-1].set_xlabel("time since run start (s)", color=INK_2)
    rel = run.parts[-4:]
    fig.suptitle("Run timeline: " + "/".join(rel), color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    p = out / ("timeline_" + "_".join(rel) + ".png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = load(root)
    if df.empty:
        raise SystemExit(f"no runs under {root}")
    df.to_csv(out / "runs.csv", index=False)
    summ = aggregate(df)
    summ.to_csv(out / "summary.csv", index=False)
    print(f"{len(df)} runs ({(df.status == 'crashed').sum()} crashed) -> {out}/runs.csv, summary.csv")
    multi_k = summ.groupby(["ctx", "parallel"]).k.nunique().max() > 1
    for p in (plot_k_sweep(summ, out) if multi_k else []) + plot_server_grid(summ, out):
        print("  plot", p)
    # timeline for the most loaded config (largest K, highest rate, first seed) per server config
    done = df[df.status == "done"]
    for _, g in done.groupby(["ctx", "parallel"]):
        top = g.sort_values(["k", "rate", "seed"], ascending=[False, False, True]).iloc[0]
        p = plot_timeline(Path(top.run), out)
        if p:
            print("  plot", p)
    cols = ["ctx", "parallel", "k", "rate", "n_seeds", "n_ok_mean", "fail_rate_mean", "lat_p95_mean",
            "lat_p95_std", "throughput_tasks_s_mean", "idle_hold_frac_mean", "cache_hit_ratio_mean"]
    print(summ[[c for c in cols if c in summ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
