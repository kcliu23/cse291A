"""Streaming client for llama-server's /completion endpoint.

Outputs are forced to a fixed length (ignore_eos + n_predict) so that every
policy generates exactly the same number of tokens per step. Speedups can then
never come from shorter outputs.
"""
from __future__ import annotations

import json
import time

import aiohttp


class CompletionError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind          # "http_4xx", "http_5xx", "server_error", "connection", "timeout"
        self.message = message


async def complete(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    n_predict: int,
    *,
    id_slot: int | None = None,
    cache_prompt: bool = True,
    seed: int = 0,
    timeout_s: float | None = None,
) -> dict:
    """Send one streaming completion request. Returns timing and token-count fields.

    Times are time.monotonic() values; the runner converts them to run-relative times.
    """
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": cache_prompt,
        "stream": True,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot

    t_send = time.monotonic()
    t_first = None
    n_chunks = 0
    parts: list[str] = []
    final: dict = {}
    try:
        async with session.post(
            f"{base_url}/completion",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                kind = "http_4xx" if 400 <= resp.status < 500 else "http_5xx"
                raise CompletionError(kind, f"HTTP {resp.status}: {body[:400]}")
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise CompletionError("server_error", json.dumps(obj["error"])[:400])
                content = obj.get("content", "")
                if content:
                    if t_first is None:
                        t_first = time.monotonic()
                    parts.append(content)
                    n_chunks += 1
                if obj.get("stop"):
                    final = obj
                    break
    except TimeoutError as e:  # aiohttp raises asyncio.TimeoutError (alias of TimeoutError)
        raise CompletionError("timeout", str(e) or "request timed out") from e
    except aiohttp.ClientError as e:
        raise CompletionError("connection", repr(e)) from e

    t_end = time.monotonic()
    timings = final.get("timings", {}) or {}
    return {
        "t_send": t_send,
        "t_first": t_first if t_first is not None else t_end,
        "t_end": t_end,
        "content": "".join(parts),
        "n_chunks": n_chunks,
        "tokens_predicted": final.get("tokens_predicted", timings.get("predicted_n")),
        "tokens_evaluated": final.get("tokens_evaluated"),
        "tokens_cached": final.get("tokens_cached", timings.get("cache_n")),
        "id_slot": final.get("id_slot"),
        "stop_type": final.get("stop_type"),
        "truncated": final.get("truncated"),
        "timings": timings,
    }
