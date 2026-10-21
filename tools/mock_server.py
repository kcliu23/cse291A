"""Mock llama-server for testing the pipeline without a model or GPU.

Implements the subset of llama-server used by agentserve: /health, /props,
/slots, /metrics and streaming /completion. It mimics the behaviours that
matter for the scheduler: a fixed number of slots, per-slot context limit
(ctx / parallel), prefix caching per slot, queueing when all slots are busy,
and slower decoding when more slots are active. Numbers it produces are NOT
measurements; use it only to check that the pipeline works.

Tokens are approximated as whitespace-separated words.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web


class Slot:
    def __init__(self, i: int, n_ctx: int):
        self.id = i
        self.n_ctx = n_ctx
        self.busy = False
        self.cache: list[str] = []
        self.last_used = 0.0


class Mock:
    def __init__(self, a):
        self.a = a
        self.n_ctx_slot = a.ctx // a.parallel
        self.slots = [Slot(i, self.n_ctx_slot) for i in range(a.parallel)]
        self.cond = asyncio.Condition()
        self.deferred = 0
        self.prompt_tokens_total = 0
        self.predicted_total = 0

    def busy(self) -> int:
        return sum(s.busy for s in self.slots)

    @staticmethod
    def common_prefix(a: list[str], b: list[str]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def pick(self, words: list[str], id_slot: int) -> Slot | None:
        if id_slot >= 0:
            s = self.slots[id_slot]
            return None if s.busy else s
        idle = [s for s in self.slots if not s.busy]
        if not idle:
            return None
        best = max(idle, key=lambda s: self.common_prefix(s.cache, words))
        if words and self.common_prefix(best.cache, words) / len(words) > 0.5:
            return best
        return min(idle, key=lambda s: s.last_used)

    async def completion(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        words = body["prompt"].split()
        n_predict = int(body.get("n_predict", 16))
        id_slot = int(body.get("id_slot", -1))
        if len(words) + n_predict > self.n_ctx_slot:
            return web.json_response({"error": {
                "code": 400, "type": "exceed_context_size_error",
                "message": f"the request exceeds the available context size ({len(words) + n_predict} > {self.n_ctx_slot})"}},
                status=400)

        async with self.cond:
            self.deferred += 1
            slot = None
            while slot is None:
                slot = self.pick(words, id_slot)
                if slot is None:
                    await self.cond.wait()
            self.deferred -= 1
            slot.busy = True

        try:
            cached = self.common_prefix(slot.cache, words) if body.get("cache_prompt", True) else 0
            cached = min(cached, len(words) - 1)
            n_eval = len(words) - cached
            t0 = time.monotonic()
            await asyncio.sleep(n_eval * self.a.prefill_ms / 1000 * (1 + 0.5 * (self.busy() - 1)))
            prompt_ms = (time.monotonic() - t0) * 1000
            self.prompt_tokens_total += n_eval

            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            out: list[str] = []
            t1 = time.monotonic()
            for i in range(n_predict):
                await asyncio.sleep(self.a.decode_ms / 1000 * (1 + self.a.contention * (self.busy() - 1)))
                tok = f" w{i % 50}"
                out.append(tok.strip())
                await resp.write(f"data: {json.dumps({'content': tok, 'stop': False})}\n\n".encode())
            predicted_ms = (time.monotonic() - t1) * 1000
            self.predicted_total += n_predict
            slot.cache = words + out
            final = {"content": "", "stop": True, "id_slot": slot.id, "stop_type": "limit",
                     "tokens_predicted": n_predict, "tokens_evaluated": len(words),
                     "tokens_cached": cached, "truncated": False,
                     "timings": {"cache_n": cached, "prompt_n": n_eval, "prompt_ms": prompt_ms,
                                 "predicted_n": n_predict, "predicted_ms": predicted_ms}}
            await resp.write(f"data: {json.dumps(final)}\n\n".encode())
            await resp.write_eof()
            return resp
        finally:
            async with self.cond:
                slot.busy = False
                slot.last_used = time.monotonic()
                self.cond.notify_all()

    async def health(self, _):
        return web.json_response({"status": "ok"})

    async def props(self, _):
        return web.json_response({"total_slots": len(self.slots), "model_path": "MOCK",
                                  "default_generation_settings": {"n_ctx": self.n_ctx_slot}})

    async def slots_ep(self, _):
        return web.json_response([{"id": s.id, "n_ctx": s.n_ctx, "is_processing": s.busy} for s in self.slots])

    async def metrics(self, _):
        kv = sum(len(s.cache) for s in self.slots)
        lines = [
            f"llamacpp:prompt_tokens_total {self.prompt_tokens_total}",
            f"llamacpp:tokens_predicted_total {self.predicted_total}",
            f"llamacpp:requests_processing {self.busy()}",
            f"llamacpp:requests_deferred {self.deferred}",
            f"llamacpp:kv_cache_tokens {kv}",
            f"llamacpp:kv_cache_usage_ratio {kv / (self.n_ctx_slot * len(self.slots)):.4f}",
        ]
        return web.Response(text="\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--prefill-ms", type=float, default=0.2, help="ms per uncached prompt token")
    ap.add_argument("--decode-ms", type=float, default=15.0, help="ms per generated token, 1 slot busy")
    ap.add_argument("--contention", type=float, default=0.15, help="decode slowdown per extra busy slot")
    a = ap.parse_args()
    m = Mock(a)
    app = web.Application(client_max_size=2**26)
    app.add_routes([web.post("/completion", m.completion), web.get("/health", m.health),
                    web.get("/props", m.props), web.get("/slots", m.slots_ep), web.get("/metrics", m.metrics)])
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
