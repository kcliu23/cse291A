"""Figures and numbers for the Phase 1 paper, read from the sweep analyses.

Usage (repo root, after `python -m agentserve.analyze runs/<sweep>` for each sweep):
    python paper/make_figures.py                  # Mac runs in runs/, AWS runs in runs_aws/
    python paper/make_figures.py --root /path     # a different copy of the run folders
Writes paper/fig_ksweep.pdf, paper/fig_retention.pdf and paper/numbers.json (every number the
paper quotes, so the text can be checked against the data).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
COL_W = 3.45  # IEEE column width, inches
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 3.5})
MAC, AWS = "#7f7f7f", "#1f5fbf"


def runs(root: Path, sweep: str) -> pd.DataFrame:
    return pd.read_csv(root / sweep / "analysis" / "runs.csv")


def agg(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE.parent)
    a = ap.parse_args()
    mac_root, aws_root = a.root / "runs", a.root / "runs_aws"
    cols = ["lat_p95", "throughput_tasks_s", "cache_hit_ratio", "prefill_ms_p50", "decode_tok_s_mean",
            "ttft_p95", "idle_hold_frac", "n_failed", "n_timeout", "peak_footprint_mb"]
    ks = {}
    for name, root in (("mac", mac_root), ("aws", aws_root)):
        df = runs(root, "ksweep")
        ks[name] = agg(df, ["k", "rate"], [c for c in cols if c in df.columns])
    nums: dict = {"ksweep": {n: d.round(4).to_dict("records") for n, d in ks.items()}}

    # Figure 1: p95 latency and prompt-cache reuse vs K, both machines, two arrival rates.
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85))
    K = [1, 2, 4, 8, 16]
    for ax, metric, ylabel in ((axes[0], "lat_p95", "p95 latency (s)"),
                               (axes[1], "cache_hit_ratio", "cache reuse (%)")):
        scale = 100 if metric == "cache_hit_ratio" else 1
        for name, color, label in (("mac", MAC, "M2 Mac"), ("aws", AWS, "AWS L4")):
            for rate, style in ((0.1, "--"), (0.2, "-")):
                d = ks[name][ks[name].rate == rate].set_index("k").loc[K]
                x = range(len(K))
                ax.errorbar(x, d[f"{metric}_mean"] * scale,
                            yerr=d[f"{metric}_std"] * scale if metric == "lat_p95" else None,
                            color=color, ls=style, marker="o", capsize=1.5, elinewidth=0.6,
                            label=f"{label}, {rate}/s")
        ax.axvline(2, color="0.75", lw=0.6, ls=":", zorder=0)
        ax.set_xticks(range(len(K)), [str(k) for k in K])
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.4, columnspacing=1.2)
    fig.supxlabel("admission limit K (dotted line: K = 4 = slots)", fontsize=7.5, y=0.02)
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.84))
    fig.savefig(HERE / "fig_ksweep.pdf", bbox_inches="tight")

    # Figure 2: K = 8 on the Mac (rate 0.1/s) under different memory configurations.
    cache = runs(mac_root, "cache")
    cache = cache[cache.slot_policy == "auto"]
    off = runs(mac_root, "cache_off")
    slots = runs(mac_root, "slots")
    mac_k = runs(mac_root, "ksweep")
    model_mib = float(cache.model_buffer_mib.iloc[0])

    def row(label, df, over=False):
        d = df[["lat_p95", "cache_hit_ratio", "throughput_tasks_s", "peak_footprint_mb", "n_failed"]]
        return {"label": label, "p95": d.lat_p95.mean(), "p95_std": d.lat_p95.std(),
                "reuse": d.cache_hit_ratio.mean(), "tput": d.throughput_tasks_s.mean(),
                "mem_gb": (model_mib + d.peak_footprint_mb.mean()) / 1024, "fail": d.n_failed.mean(),
                "over": over}

    rows = [
        row("K=4, f16 KV (best fixed K)", mac_k[(mac_k.k == 4) & (mac_k.rate == 0.1)]),
        row("K=8, prompt cache off", off),
        row("K=8, f16 KV (baseline)", cache[cache.variant == "cram0"]),
        row("K=8 + 1 GB host cache", cache[cache.variant == "cram1024"]),
        row("K=8 + 8 GB host cache", cache[cache.variant == "cram8192"], over=True),
        row("K=8, 8 slots, 8-bit KV", slots[(slots.ctx == 32768) & (slots.variant == "kvq8")]),
    ]
    nums["retention"] = rows
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        color = "0.55" if r["label"].startswith("K=8, f16") or "off" in r["label"] else AWS
        if r["label"].startswith("K=4"):
            color = "0.3"
        ax.barh(yi, r["p95"], xerr=r["p95_std"], color=color, height=0.62, capsize=1.5,
                error_kw={"elinewidth": 0.6}, hatch="////" if r["over"] else None,
                edgecolor="white" if r["over"] else None, linewidth=0)
        ax.text(r["p95"] + r["p95_std"] + 8, yi,
                f"reuse {r['reuse'] * 100:.0f}%, {r['mem_gb']:.1f} GB" + (" (over budget)" if r["over"] else ""),
                va="center", fontsize=6.5)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlabel("p95 task latency (s), M2 Mac, 0.1 tasks/s")
    ax.set_xlim(0, 640)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "fig_retention.pdf", bbox_inches="tight")

    # Mechanism numbers, rate 0.2, K = 4 -> 8, both machines.
    mech = {}
    for name in ("mac", "aws"):
        d = ks[name][ks[name].rate == 0.2].set_index("k")
        mech[name] = {k: {c: round(float(d.loc[k, f"{c}_mean"]), 4) for c in
                          ["lat_p95", "throughput_tasks_s", "cache_hit_ratio", "prefill_ms_p50",
                           "decode_tok_s_mean", "ttft_p95"]} for k in (4, 8)}
    nums["mechanism_rate0.2"] = mech
    slot_fail = slots.groupby(["ctx", "parallel", "variant"]).n_failed.mean().round(2)
    nums["slots_failures_mean"] = {f"{c}x{p}_{v}": f for (c, p, v), f in slot_fail.items()}
    (HERE / "numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    print("wrote fig_ksweep.pdf, fig_retention.pdf, numbers.json")


if __name__ == "__main__":
    main()
