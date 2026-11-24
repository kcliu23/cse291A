"""Tool-delay sweep figure: where does parking start to pay off?

Usage (repo root, after `python -m agentserve.analyze runs/delay/d<X>` for every delay on each machine):
    python paper/make_delay_figure.py                # Mac runs in runs/delay, AWS runs in runs_aws/delay
    python paper/make_delay_figure.py --root /path
Writes paper/fig_delay.pdf and paper/numbers_delay.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                     "lines.markersize": 3.5})
CONFIGS = [("fifo", 4, "FIFO K=4", "0.6", "--"), ("slots", 4, "Affinity K=4", "0.3", "-"),
           ("park", 8, "Parking K=8", "#1f5fbf", "-")]
MACHINES = [("runs", "M2 Mac"), ("runs_aws", "AWS L4")]


def load(root: Path, sub: str) -> pd.DataFrame:
    rows = []
    for d in sorted((root / sub / "delay").glob("d*")):
        f = d / "analysis" / "runs.csv"
        m = re.fullmatch(r"d(\d+)", d.name)
        if f.exists() and m:
            df = pd.read_csv(f)
            df["delay"] = int(m.group(1))
            rows.append(df[df.status == "done"])
    return pd.concat(rows) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=HERE.parent)
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=False)
    nums = {}
    for ax, (sub, title) in zip(axes, MACHINES):
        df = load(a.root, sub)
        if df.empty:
            ax.set_title(f"{title}: no data")
            continue
        p95 = df.pivot_table(index=["delay", "seed"], columns=["policy", "k"], values="lat_p95")
        tput = df.pivot_table(index=["delay", "seed"], columns=["policy", "k"], values="throughput_tasks_s")
        base = p95[("slots", 4)]
        rel = {f"{pol}_k{k}": ((p95[(pol, k)] / base - 1) * 100).round(1) for pol, k, *_ in CONFIGS
               if (pol, k) in p95}
        nums[sub] = {
            "p95_mean": {f"{pol}_k{k}": p95[(pol, k)].groupby("delay").mean().round(1).to_dict()
                         for pol, k, *_ in CONFIGS if (pol, k) in p95},
            "tput_mean": {f"{pol}_k{k}": tput[(pol, k)].groupby("delay").mean().round(4).to_dict()
                          for pol, k, *_ in CONFIGS if (pol, k) in tput},
            "p95_vs_affinity_k4_pct": {key: {f"{d}_{s}": v for (d, s), v in ser.items()} for key, ser in rel.items()},
        }
        for pol, k, label, color, ls in CONFIGS:
            if (pol, k) not in p95:
                continue
            m = p95[(pol, k)].groupby("delay").mean()
            ax.plot(m.index, m.values, ls=ls, marker="o", color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("mean tool-call time (s)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("p95 task latency (s)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "fig_delay.pdf", bbox_inches="tight")
    (HERE / "numbers_delay.json").write_text(json.dumps(nums, indent=1, default=float))
    print("wrote fig_delay.pdf, numbers_delay.json")


if __name__ == "__main__":
    main()
