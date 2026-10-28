"""Does an 8-bit KV cache change what the model says?

Phase 1 proposes `-ctk q8_0 -ctv q8_0` (8 slots of 4,096 tokens in the memory of 4) as a
Phase 2 baseline. The sweeps used fixed-length filler, so they measured time and memory only.
This script checks output quality with the same model and build:

  f16_a  flash attention, f16 KV      reference
  q8     flash attention, 8-bit KV    the configuration under test
  f16_b  flash attention, f16 KV      fresh repeat of the reference = the noise floor

Each config gets a fresh llama-server with one 4,096-token slot (the per-slot context of the
8 x 4,096 configuration) and answers the same prompts with greedy decoding:

  needle   30 long-context retrieval questions: a 6-digit code hidden at 5 depths in
           ~1k / ~2k / ~3k tokens of filler (where KV precision matters most); scored
  arith    20 short arithmetic questions; scored
  open     15 open-ended questions, 128 tokens each; compared with the reference
  summary  5 two-sentence summaries of ~3k-token documents; compared with the reference

Reported per config: accuracy on scored tasks, and against f16_a: identical outputs, where the
first differing token appears, and the mean |difference| in the chosen tokens' log-probabilities
before that point. q8 is "no measurable change" if it matches f16_a about as well as f16_b does.

Usage (from the repo root, ~10-15 min on the M2):
    python tools/kv_quality.py
    python tools/kv_quality.py --binary ../llama.cpp/build/bin/llama-server --out runs/kvq
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentserve.server import Server  # noqa: E402

CONFIGS = [
    ("f16_a", ["-fa", "on"]),
    ("q8", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]),
    ("f16_b", ["-fa", "on"]),
]
REFERENCE = "f16_a"

FILLER = [
    "The committee met on Tuesday to review the quarterly maintenance schedule.",
    "Rainfall in the northern valley was slightly above the ten-year average.",
    "Several volunteers repainted the fence around the community garden.",
    "The library extended its weekend hours during the exam period.",
    "A new bus route now connects the harbor with the old market square.",
    "Engineers inspected the bridge supports and reported no visible damage.",
    "The bakery on the corner started selling rye bread again in March.",
    "Most of the orchard trees were pruned before the first frost.",
    "The museum added a small exhibit about early printing presses.",
    "Traffic on the coastal road increased during the summer festival.",
    "The school replaced the windows in its east wing over the holidays.",
    "A local choir performed at the town hall on the last Sunday of the month.",
    "The river ferry was out of service for two days for routine repairs.",
    "Farmers reported a good harvest of potatoes and late cabbage.",
    "The city council approved funds for new streetlights near the station.",
    "Two new benches were installed along the lakeside walking path.",
    "The weather station recorded strong winds early on Thursday morning.",
    "The hardware store moved to a larger building across the street.",
    "Students from the technical college visited the regional power plant.",
    "The annual book fair attracted visitors from neighboring towns.",
    "Workers resurfaced the parking lot behind the sports center.",
    "The post office began offering package lockers near the entrance.",
    "A family of swans has nested near the old mill again this spring.",
    "The clinic hired an additional nurse for the evening shift.",
    "The hiking club cleared fallen branches from the forest trail.",
    "The cinema showed a series of classic films throughout October.",
    "A small fire in a storage shed was put out quickly by neighbors.",
    "The town's website was redesigned to make forms easier to find.",
    "Fishing boats returned early because of rising waves offshore.",
    "The garden center offered a workshop on growing herbs indoors.",
]
COLORS = ["red", "blue", "green", "silver", "amber", "violet", "black", "white", "golden", "gray"]
ANIMALS = ["falcon", "otter", "badger", "heron", "lynx", "marten", "osprey", "ibex", "walrus", "bison"]
OPEN = [
    "Explain how a hash table handles collisions.",
    "Describe the water cycle to a ten-year-old.",
    "What are the trade-offs between TCP and UDP?",
    "Give three tips for writing clear technical documentation.",
    "Explain what a mutex is and when you would use one.",
    "Why does the Moon show phases?",
    "Describe how a bicycle's gears make pedaling easier.",
    "What is the difference between a process and a thread?",
    "Explain the idea of compound interest with an example.",
    "How does a refrigerator keep food cold?",
    "What makes a good unit test?",
    "Explain why the sky appears blue.",
    "Describe the main steps of making bread from scratch.",
    "What is caching, and why does it speed up programs?",
    "Explain how vaccines train the immune system.",
]
WORDS_PER_LEN = {"1k": 750, "2k": 1500, "3k": 2400}  # ~1.3 tokens/word; 3k leaves room in a 4,096 slot
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]


def haystack(rng: random.Random, n_words: int) -> list[str]:
    out, words = [], 0
    while words < n_words:
        s = rng.choice(FILLER)
        out.append(s)
        words += len(s.split())
    return out


def build_items(seed: int = 291) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for rep in range(2):
        for name, n_words in WORDS_PER_LEN.items():
            for depth in DEPTHS:
                key = f"{rng.choice(COLORS)} {rng.choice(ANIMALS)}"
                code = f"{rng.randrange(100000, 1000000)}"
                hay = haystack(rng, n_words)
                hay.insert(int(depth * len(hay)), f"The access code for the {key} vault is {code}.")
                prompt = (" ".join(hay) + f"\n\nWhat is the access code for the {key} vault? "
                          "Reply with the 6-digit number only.")
                items.append({"id": f"needle_{name}_d{depth}_r{rep}", "kind": "needle",
                              "prompt": prompt, "max_tokens": 16, "answer": code})
    for i in range(20):
        a, b, c = rng.randrange(11, 99), rng.randrange(11, 99), rng.randrange(11, 99)
        items.append({"id": f"arith_{i}", "kind": "arith", "max_tokens": 16, "answer": str(a * b + c),
                      "prompt": f"Compute {a} * {b} + {c}. Reply with the number only."})
    for i, q in enumerate(OPEN):
        items.append({"id": f"open_{i}", "kind": "open", "prompt": q, "max_tokens": 128})
    for i in range(5):
        doc = " ".join(haystack(rng, WORDS_PER_LEN["3k"]))
        items.append({"id": f"summary_{i}", "kind": "summary", "max_tokens": 96,
                      "prompt": doc + "\n\nSummarize the text above in two sentences."})
    return items


def post(url: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(base_url: str, item: dict, want_logprobs: bool) -> dict:
    body = {"messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0, "max_tokens": item["max_tokens"],
            "cache_prompt": False}  # every prompt computed from scratch, same as every config
    if want_logprobs:
        body.update(logprobs=True, top_logprobs=1)
    t = time.monotonic()
    resp = post(base_url + "/v1/chat/completions", body)
    choice = resp["choices"][0]
    lp = ((choice.get("logprobs") or {}).get("content")) or []
    return {"text": choice["message"]["content"],
            "tokens": [x.get("token") for x in lp], "logprobs": [x.get("logprob") for x in lp],
            "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
            "seconds": round(time.monotonic() - t, 3)}


def score(item: dict, text: str) -> bool | None:
    if "answer" not in item:
        return None
    nums = re.findall(r"-?\d[\d,]*", text)
    return bool(nums) and nums[0].replace(",", "") == item["answer"]


def run_config(name: str, extra: list[str], args, items: list[dict]) -> list[dict]:
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"binary": args.binary, "model": args.model, "ctx": 4096, "parallel": 1,
           "port": args.port, "extra": [*extra, "--cache-ram", "0"]}
    rows = []
    with Server(cfg, out_dir / "server.log") as srv:
        want_lp = True
        for i, item in enumerate(items):
            try:
                r = ask(srv.url, item, want_lp)
            except urllib.error.HTTPError as e:
                if not want_lp:
                    raise
                print(f"  [{name}] logprobs not supported ({e.code}); continuing without them")
                want_lp = False
                r = ask(srv.url, item, want_lp)
            row = {"id": item["id"], "kind": item["kind"], **r, "correct": score(item, r["text"])}
            rows.append(row)
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                print(f"  [{name}] {i + 1}/{len(items)}")
    with open(out_dir / "responses.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def compare(ref: list[dict], other: list[dict]) -> dict:
    same, first_diff, lp_diffs, prefix_frac = 0, [], [], []
    for a, b in zip(ref, other):
        assert a["id"] == b["id"]
        if a["text"] == b["text"]:
            same += 1
        n = 0
        while n < min(len(a["text"]), len(b["text"])) and a["text"][n] == b["text"][n]:
            n += 1
        prefix_frac.append(n / max(len(a["text"]), 1))
        if a["tokens"] and b["tokens"]:
            k = 0
            while k < min(len(a["tokens"]), len(b["tokens"])) and a["tokens"][k] == b["tokens"][k]:
                lp_diffs.append(abs(a["logprobs"][k] - b["logprobs"][k]))
                k += 1
            if a["tokens"] != b["tokens"]:
                first_diff.append(k)
    return {"n": len(ref), "identical": same, "identical_frac": round(same / len(ref), 3),
            "mean_shared_prefix_frac": round(sum(prefix_frac) / len(prefix_frac), 3),
            "first_diff_token_median": sorted(first_diff)[len(first_diff) // 2] if first_diff else None,
            "mean_abs_logprob_diff": round(sum(lp_diffs) / len(lp_diffs), 5) if lp_diffs else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="../llama.cpp/build/bin/llama-server")
    ap.add_argument("--model", default="../models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--out", type=Path, default=Path("runs/kvq"))
    ap.add_argument("--port", type=int, default=8090)  # not 8080, so it never meets a sweep
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    args = ap.parse_args()

    items = build_items()
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, extra in CONFIGS:
        print(f"[kvq] {name}: {' '.join(extra)}")
        results[name] = run_config(name, extra, args, items)

    kinds = sorted({it["kind"] for it in items})
    summary = {"items": len(items), "accuracy": {}, "vs_reference": {}}
    for name, rows in results.items():
        acc = {}
        for kind in kinds:
            scored = [r["correct"] for r in rows if r["kind"] == kind and r["correct"] is not None]
            if scored:
                acc[kind] = f"{sum(scored)}/{len(scored)}"
        summary["accuracy"][name] = acc
    for name in results:
        if name == REFERENCE:
            continue
        summary["vs_reference"][name] = {"all": compare(results[REFERENCE], results[name])}
        for kind in kinds:
            ref_k = [r for r in results[REFERENCE] if r["kind"] == kind]
            oth_k = [r for r in results[name] if r["kind"] == kind]
            summary["vs_reference"][name][kind] = compare(ref_k, oth_k)
    pts = [r["prompt_tokens"] for r in results[REFERENCE] if r["kind"] == "needle" and r["prompt_tokens"]]
    if pts:
        summary["needle_prompt_tokens"] = {"min": min(pts), "max": max(pts)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\naccuracy (correct/scored):")
    for name, acc in summary["accuracy"].items():
        print(f"  {name:6s} " + "  ".join(f"{k} {v}" for k, v in acc.items()))
    print(f"\nagreement with {REFERENCE} (identical outputs; median first differing token; "
          "mean |logprob diff| before it):")
    for name, by_kind in summary["vs_reference"].items():
        for kind, c in by_kind.items():
            print(f"  {name:6s} {kind:8s} {c['identical']:>3}/{c['n']:<3} "
                  f"first diff @ {c['first_diff_token_median']}  |dlogp| {c['mean_abs_logprob_diff']}")
    if pts:
        print(f"\nneedle prompts: {min(pts)}-{max(pts)} tokens (slot holds 4096)")
    print(f"\n-> {args.out}/summary.json")


if __name__ == "__main__":
    main()
