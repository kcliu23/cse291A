"""Seeded agent workload generator.

A workload is a list of agent tasks. Each task has an initial context and one or
more steps. A step sends (history + step input) to the model, appends the model
output, then waits a controlled "tool delay" and appends a tool result before
the next step. Context therefore grows across steps, as in a real agent loop.

Text is filler words drawn from a small vocabulary of common English words, so
~1 word ~= 1 token for Llama-family tokenizers. The runner logs the server's
actual token counts; analysis uses those, not these estimates.

Usage:
    python -m agentserve.workload --config configs/workload_dev.json \
        --seed 0 --rate 0.5 --n-tasks 60 --out workloads/dev_s0_r0.5.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

GENERATOR_VERSION = 1

VOCAB = (
    "the system agent model memory cache request task step tool result data "
    "user query answer context token server queue slot batch latency time "
    "file code search table value record list item report check update read "
    "write plan goal state event error retry output input prompt summary note "
    "node graph path route city price order product review score rank match "
    "a an of to in on for with by from at as is are was be has have can will "
    "first next last new old large small fast slow high low open close start end "
    "red blue green north south east west morning evening week month year day "
    "water light power energy market budget team project design test build run "
    "simple clear short long early late local remote public private shared main"
).split()


def _words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(max(0, n)))


def _uniform_int(rng: random.Random, lo_hi) -> int:
    lo, hi = lo_hi
    return rng.randint(int(lo), int(hi))


def _uniform(rng: random.Random, lo_hi) -> float:
    lo, hi = lo_hi
    return rng.uniform(float(lo), float(hi))


def generate(cfg: dict, seed: int, rate: float, n_tasks: int) -> dict:
    """Generate a workload dict. Deterministic for a given (cfg, seed, rate, n_tasks)."""
    rng = random.Random(seed)
    classes = cfg["classes"]
    names = list(classes)
    weights = [classes[c]["weight"] for c in names]

    # Shared system-prompt prefix: identical across tasks, so it is reusable
    # across requests if the runtime keeps it in a slot's KV cache.
    prefix_rng = random.Random(cfg.get("prefix_seed", 12345))
    shared_prefix = "System: " + _words(prefix_rng, cfg.get("shared_prefix_tokens", 0)) + "\n"

    tasks = []
    t = 0.0
    for i in range(n_tasks):
        if rate > 0:
            t += rng.expovariate(rate)
        cls = rng.choices(names, weights=weights)[0]
        c = classes[cls]
        context = "Document: " + _words(rng, _uniform_int(rng, c["context_tokens"])) + "\n"
        steps = []
        for s in range(c["steps"]):
            last = s == c["steps"] - 1
            step = {
                "input": f"Step {s + 1}: " + _words(rng, _uniform_int(rng, c["step_input_tokens"])) + "\nAssistant:",
                "n_predict": int(c["output_tokens"]),
                "tool_delay_s": 0.0 if last else round(_uniform(rng, c["tool_delay_s"]), 3),
                "tool_output": "" if last else
                "\nTool result: " + _words(rng, _uniform_int(rng, c["tool_output_tokens"])) + "\n",
            }
            steps.append(step)
        tasks.append({"id": i, "cls": cls, "arrival_s": round(t, 4), "context": context, "steps": steps})

    wl = {
        "meta": {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "rate_tasks_per_s": rate,
            "n_tasks": n_tasks,
            "config": cfg,
        },
        "shared_prefix": shared_prefix,
        "tasks": tasks,
    }
    wl["meta"]["sha256"] = hashlib.sha256(
        json.dumps({"p": shared_prefix, "t": tasks}, sort_keys=True).encode()
    ).hexdigest()
    return wl


def estimate_peak_context(wl: dict) -> list[int]:
    """Rough per-task peak context in words (prefix + context + all steps + outputs)."""
    base = len(wl["shared_prefix"].split())
    out = []
    for t in wl["tasks"]:
        n = base + len(t["context"].split())
        for s in t["steps"]:
            n += len(s["input"].split()) + s["n_predict"] + len(s["tool_output"].split())
        out.append(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, required=True, help="Poisson arrival rate, tasks/s")
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    wl = generate(cfg, a.seed, a.rate, a.n_tasks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(wl))
    peaks = estimate_peak_context(wl)
    by_cls: dict[str, int] = {}
    for t in wl["tasks"]:
        by_cls[t["cls"]] = by_cls.get(t["cls"], 0) + 1
    print(f"wrote {a.out}: {len(wl['tasks'])} tasks, classes={by_cls}, "
          f"last arrival={wl['tasks'][-1]['arrival_s']:.1f}s, max est. context={max(peaks)} words")


if __name__ == "__main__":
    main()
