"""Phase 2 feasibility probe: can llama-server park a slot's context and bring it back cheaply?

The Phase 2 policy parks an agent's KV cache while it waits on a tool and restores it when the
agent returns. This measures, for contexts of ~1k / 2k / 3.5k tokens:

  recompute_ms  processing the full prompt again (what eviction costs today)
  save_ms       writing the slot's KV state to a file  (POST /slots/0?action=save)
  restore_ms    reading it back into the slot          (POST /slots/0?action=restore)
  reused        prompt tokens served from the slot after the restore (should be ~all of them)
  file_mb       size of the saved state

Parking is worth it when save + restore is much cheaper than recompute.

Usage (repo root, ~2 min on the M2; uses port 8091):
    python tools/slot_probe.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentserve.server import Server  # noqa: E402
from agentserve.workload import _words  # noqa: E402


def post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(base: str, prompt: str) -> dict:
    r = post(base + "/completion", {"prompt": prompt, "n_predict": 8, "temperature": 0,
                                    "cache_prompt": True, "id_slot": 0})
    t = r.get("timings", {})
    return {"prompt_ms": t.get("prompt_ms"), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="../llama.cpp/build/bin/llama-server")
    ap.add_argument("--model", default="../models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--out", type=Path, default=Path("runs/slot_probe"))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--lengths", default="1000,2000,3500")
    ap.add_argument("--extra", default="", help="extra server flags, e.g. '-fa on -ctk q8_0 -ctv q8_0'")
    args = ap.parse_args()

    save_dir = (args.out / "slots").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"binary": args.binary, "model": args.model, "ctx": 16384, "parallel": 4, "port": args.port,
           "extra": ["--cache-ram", "0", "--slot-save-path", str(save_dir), *args.extra.split()]}
    rng = random.Random(7)
    rows = []
    with Server(cfg, args.out / "server.log") as srv:
        base = srv.url
        complete(base, "warm up " + _words(rng, 50))
        for n in map(int, args.lengths.split(",")):
            ctx = _words(rng, n)            # the agent's conversation so far
            follow = " " + _words(rng, 40)  # what it sends after the tool call
            other = _words(rng, n)          # another agent that takes the slot meanwhile

            cold = complete(base, ctx)                          # first visit: full prefill
            t = time.monotonic()
            s = post(f"{base}/slots/0?action=save", {"filename": f"ctx{n}.bin"})
            save_wall = (time.monotonic() - t) * 1000
            complete(base, other)                               # slot overwritten
            t = time.monotonic()
            r = post(f"{base}/slots/0?action=restore", {"filename": f"ctx{n}.bin"})
            restore_wall = (time.monotonic() - t) * 1000
            back = complete(base, ctx + follow)                 # agent returns after restore
            complete(base, other)                               # overwrite again, no restore
            evicted = complete(base, ctx + follow)              # agent returns after eviction

            size = (save_dir / f"ctx{n}.bin").stat().st_size / 2**20
            rows.append({
                "tokens": cold["prompt_n"],
                "recompute_ms": evicted["prompt_ms"], "evicted_reused": evicted["cache_n"],
                "save_ms": (s.get("timings") or {}).get("save_ms", round(save_wall, 1)),
                "restore_ms": (r.get("timings") or {}).get("restore_ms", round(restore_wall, 1)),
                "restored_prompt_ms": back["prompt_ms"], "restored_reused": back["cache_n"],
                "file_mb": round(size, 1)})
            print(json.dumps(rows[-1]))
    (args.out / "probe.json").write_text(json.dumps(rows, indent=1))
    print(f"\n{'tokens':>7} {'recompute':>10} {'save':>7} {'restore':>8} {'after restore':>14} {'reused':>7} {'file':>7}")
    for x in rows:
        print(f"{x['tokens']:>7} {x['recompute_ms']:>8.0f}ms {x['save_ms']:>5.0f}ms {x['restore_ms']:>6.0f}ms "
              f"{x['restored_prompt_ms']:>12.0f}ms {x['restored_reused']:>7} {x['file_mb']:>5.1f}MB")
    print(f"-> {args.out}/probe.json")


if __name__ == "__main__":
    main()
