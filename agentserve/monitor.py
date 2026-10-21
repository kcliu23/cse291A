"""System sampler: polls llama-server metrics, slot state, process memory and GPU memory.

Writes one JSON line per sample to system.jsonl. Every source is optional:
a source that is unavailable (e.g. nvidia-smi on a Mac) is skipped.
"""
from __future__ import annotations

import asyncio
import json
import platform
import re
import shutil
import time

import aiohttp
import psutil


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        try:
            out[name] = float(parts[-1])
        except ValueError:
            pass
    return out


async def _gpu_mem_mb() -> float | None:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        return sum(float(x) for x in out.decode().split())
    except ValueError:
        return None


_FOOTPRINT_RE = re.compile(r"(?:phys_footprint|Footprint):\s*([\d.]+)\s*([KMG])B", re.I)


async def _footprint_mb(pid: int) -> float | None:
    """macOS only: physical footprint, which includes Metal (unified-memory) buffers that RSS misses.
    Best effort: returns None if `footprint` is missing or its output format differs."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "footprint", "-p", str(pid),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
    except (OSError, TimeoutError):
        return None
    m = _FOOTPRINT_RE.search(out.decode(errors="replace"))
    if not m:
        return None
    return float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2).upper()]


def _rss_mb(proc: psutil.Process | None) -> float | None:
    if proc is None:
        return None
    try:
        total = proc.memory_info().rss
        for ch in proc.children(recursive=True):
            total += ch.memory_info().rss
        return total / 2**20
    except psutil.Error:
        return None


class Monitor:
    def __init__(self, base_url: str, out_path, t0: float, interval_s: float = 0.5,
                 server_pid: int | None = None):
        self.base_url = base_url
        self.out_path = out_path
        self.t0 = t0
        self.interval_s = interval_s
        self.proc = psutil.Process(server_pid) if server_pid else None
        self.has_gpu = shutil.which("nvidia-smi") is not None
        self.has_footprint = (platform.system() == "Darwin" and server_pid is not None
                              and shutil.which("footprint") is not None)
        self.n = 0
        self.slots_ok = True
        self.metrics_ok = True

    async def run(self, stop: asyncio.Event) -> None:
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            with open(self.out_path, "w") as f:
                while not stop.is_set():
                    sample = await self._sample(s)
                    f.write(json.dumps(sample) + "\n")
                    f.flush()
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
                    except TimeoutError:
                        pass

    async def _sample(self, s: aiohttp.ClientSession) -> dict:
        rec: dict = {"t": time.monotonic() - self.t0, "rss_mb": _rss_mb(self.proc)}
        if self.has_gpu:
            rec["gpu_mem_mb"] = await _gpu_mem_mb()
        if self.has_footprint and self.n % 4 == 0:  # footprint is slow; sample every 4th tick
            rec["footprint_mb"] = await _footprint_mb(self.proc.pid)
        self.n += 1
        if self.metrics_ok:
            try:
                async with s.get(f"{self.base_url}/metrics") as r:
                    if r.status == 200:
                        rec["metrics"] = parse_prometheus(await r.text())
                    else:
                        self.metrics_ok = False
            except (aiohttp.ClientError, TimeoutError):
                pass
        if self.slots_ok:
            try:
                async with s.get(f"{self.base_url}/slots") as r:
                    if r.status == 200:
                        slots = await r.json()
                        rec["slots_busy"] = sum(1 for x in slots if x.get("is_processing"))
                        rec["n_slots"] = len(slots)
                    else:
                        self.slots_ok = False
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
        return rec
