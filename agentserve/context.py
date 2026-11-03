"""Phase 2: client-side management of the server's slots and of agents' context.

llama-server has a fixed number of slots, each holding one conversation's KV cache. In
Phase 1 the server picked slots by itself, so when more agents were admitted than there
are slots, a returning agent usually found its context overwritten and recomputed it.

The SlotManager takes over that choice. Every model call asks it for a slot; it knows
which agent's context is resident in which slot and decides what happens to a waiting
agent's context when its slot is needed by someone else:

    drop      evict it; the agent recomputes its history when it returns (Phase 1 behaviour)
    park      save the slot's KV state to a file on the SSD first (llama-server
              POST /slots/{id}?action=save) and restore it when the agent returns
              (action=restore), so the history is read back instead of recomputed

Policies (selected by the sweep's "policies" list):

    slots       slot affinity + returning agents first; evicted context is dropped
    park        as slots, but every evicted context is parked (LRU victim)
    park_cost   as park, but contexts shorter than min_park_tokens are dropped
                (recomputing them is cheaper than a save + restore), and the victim is the
                agent expected to be away longest (KVFlow-style), not the least recently used

Admission (how many tasks may be active) is still FIFO with a fixed K; the manager only
decides who runs in which slot. At most one request runs per slot.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

POLICIES = ("slots", "park", "park_cost")


@dataclass
class SlotState:
    id: int
    owner: int | None = None      # task whose context is resident
    busy: bool = False            # a request or a save/restore is using the slot
    last_used: float = 0.0


@dataclass
class TaskCtx:
    id: int
    cls: str
    tokens: int = 0               # context length after its last request (prompt + output)
    resident: int | None = None   # slot holding its context
    parked: str | None = None     # file name of its parked context
    parked_bytes: int = 0
    away_until: float = 0.0       # expected end of the current tool wait (monotonic)
    done: bool = False


@dataclass
class Waiter:
    task: TaskCtx
    returning: bool
    seq: int
    fut: asyncio.Future = field(repr=False)


class SlotManager:
    def __init__(self, n_slots: int, policy: str, *, base_url: str, save_dir: str | None,
                 run_tag: str, t0_fn, min_park_tokens: int = 600, default_tool_s: float = 2.0):
        if policy not in POLICIES:
            raise ValueError(f"unknown context policy {policy!r}")
        if policy != "slots" and not save_dir:
            raise ValueError(f"policy {policy!r} needs a slot save directory (--slot-save-path)")
        self.policy = policy
        self.base_url = base_url
        self.save_dir = Path(save_dir) if save_dir else None
        self.run_tag = run_tag
        self.rel = t0_fn
        self.min_park_tokens = min_park_tokens if policy == "park_cost" else 0
        self.slots = [SlotState(i) for i in range(n_slots)]
        self.tasks: dict[int, TaskCtx] = {}
        self.waiters: list[Waiter] = []
        self.seq = 0
        self.events: list[dict] = []
        self.parked_bytes = 0
        self.peak_parked_bytes = 0
        self.tool_est: dict[str, float] = {}   # running mean of tool delays per task class
        self.tool_n: dict[str, int] = {}
        self.default_tool_s = default_tool_s
        self.session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------ public API
    def task(self, task: dict) -> TaskCtx:
        t = self.tasks.get(task["id"])
        if t is None:
            t = self.tasks[task["id"]] = TaskCtx(task["id"], task["cls"])
        return t

    async def acquire(self, t: TaskCtx, returning: bool) -> tuple[int, dict]:
        """Wait for a slot for t's next request. Returns (slot id, info for the request record)."""
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        w = Waiter(t, returning, self.seq, loop.create_future())
        self.seq += 1
        self.waiters.append(w)
        self._dispatch()
        try:
            slot_id, victim = await w.fut
        except asyncio.CancelledError:
            if w in self.waiters:
                self.waiters.remove(w)
            elif w.fut.done() and not w.fut.cancelled():
                s = self.slots[w.fut.result()[0]]
                s.busy = False
                self._dispatch()
            raise
        info = {"slot_wait_ms": round((time.monotonic() - t0) * 1000, 1), "id_slot_mgr": slot_id}
        try:
            info["ctx_action"] = await self._prepare(slot_id, t, victim)
        except Exception:
            self.slots[slot_id].busy = False
            self._dispatch()
            raise
        return slot_id, info

    def release(self, t: TaskCtx, slot_id: int, tokens: int | None, last: bool,
                tool_delay_s: float = 0.0) -> None:
        """Called after t's request on slot_id finished (or failed)."""
        s = self.slots[slot_id]
        s.busy = False
        s.last_used = time.monotonic()
        if tokens:
            t.tokens = tokens
        if last:
            self.finish(t)
        else:
            t.away_until = time.monotonic() + self._expected_tool_s(t.cls)
        self._dispatch()

    def tool_done(self, t: TaskCtx, delay_s: float) -> None:
        n = self.tool_n.get(t.cls, 0)
        self.tool_est[t.cls] = (self.tool_est.get(t.cls, 0.0) * n + delay_s) / (n + 1)
        self.tool_n[t.cls] = n + 1

    def finish(self, t: TaskCtx) -> None:
        """Task finished or failed: free its slot and delete its parked context."""
        if t.done:
            return
        t.done = True
        if t.resident is not None:
            s = self.slots[t.resident]
            if s.owner == t.id:
                s.owner = None
            t.resident = None
        self._discard_parked(t)
        self._dispatch()

    # ------------------------------------------------------------ scheduling
    def _expected_tool_s(self, cls: str) -> float:
        return self.tool_est.get(cls, self.default_tool_s)

    def _dispatch(self) -> None:
        """Hand free slots to waiters: own resident slot first, then returning agents, then new ones."""
        progress = True
        while progress and self.waiters:
            progress = False
            # 1. a waiter whose context is resident in an idle slot takes it at once
            for w in list(self.waiters):
                r = w.task.resident
                if r is not None and not self.slots[r].busy and self.slots[r].owner == w.task.id:
                    self._grant(w, r, None)
                    progress = True
            if not self.waiters:
                break
            # 2. otherwise the highest-priority waiter takes the best free slot, if any
            w = min(self.waiters, key=lambda x: (not x.returning, x.seq))
            choice = self._choose_slot(w.task)
            if choice is not None:
                self._grant(w, *choice)
                progress = True

    def _choose_slot(self, t: TaskCtx) -> tuple[int, int | None] | None:
        idle = [s for s in self.slots if not s.busy]
        if not idle:
            return None
        empty = [s for s in idle if s.owner is None]
        if empty:
            return empty[0].id, None
        # every idle slot holds the context of an agent that is waiting on a tool
        victims = [s for s in idle if s.owner is not None and not self._owner_waiting_for(s.owner)]
        if not victims:
            return None
        if self.policy == "park_cost":
            # evict the agent expected to come back last
            s = max(victims, key=lambda s: self.tasks[s.owner].away_until)
        else:
            s = min(victims, key=lambda s: s.last_used)
        return s.id, s.owner

    def _owner_waiting_for(self, owner: int) -> bool:
        return any(w.task.id == owner for w in self.waiters)

    def _grant(self, w: Waiter, slot_id: int, victim: int | None) -> None:
        self.waiters.remove(w)
        self.slots[slot_id].busy = True
        if not w.fut.done():
            w.fut.set_result((slot_id, victim))

    # ------------------------------------------------------------ data movement
    async def _prepare(self, slot_id: int, t: TaskCtx, victim: int | None) -> str:
        s = self.slots[slot_id]
        if s.owner == t.id:
            return "hit"
        if victim is not None:
            v = self.tasks[victim]
            if self.policy != "slots" and v.tokens >= self.min_park_tokens and not v.done:
                await self._save(slot_id, v)
            else:
                self._log("drop", slot_id, v, tokens=v.tokens)
            v.resident = None
        s.owner = t.id
        t.resident = slot_id
        if t.parked:
            await self._restore(slot_id, t)
            return "restore"
        return "cold"

    async def _post(self, path: str, body: dict) -> dict:
        async with self.session.post(self.base_url + path, json=body,
                                     timeout=aiohttp.ClientTimeout(total=120)) as r:
            text = await r.text()
            if r.status != 200:
                raise RuntimeError(f"{path}: HTTP {r.status}: {text[:300]}")
            return json.loads(text)

    async def _save(self, slot_id: int, v: TaskCtx) -> None:
        self._discard_parked(v)
        name = f"{self.run_tag}_t{v.id}_{int(time.monotonic() * 1000)}.bin"
        t = time.monotonic()
        r = await self._post(f"/slots/{slot_id}?action=save", {"filename": name})
        ms = (r.get("timings") or {}).get("save_ms", (time.monotonic() - t) * 1000)
        size = self._size(name)
        v.parked, v.parked_bytes = name, size
        self.parked_bytes += size
        self.peak_parked_bytes = max(self.peak_parked_bytes, self.parked_bytes)
        self._log("save", slot_id, v, ms=ms, bytes=size, tokens=r.get("n_saved", v.tokens))

    async def _restore(self, slot_id: int, t: TaskCtx) -> None:
        start = time.monotonic()
        r = await self._post(f"/slots/{slot_id}?action=restore", {"filename": t.parked})
        ms = (r.get("timings") or {}).get("restore_ms", (time.monotonic() - start) * 1000)
        self._log("restore", slot_id, t, ms=ms, bytes=t.parked_bytes, tokens=r.get("n_restored"))
        self._discard_parked(t)

    def _size(self, name: str) -> int:
        try:
            return (self.save_dir / name).stat().st_size
        except OSError:
            return 0

    def _discard_parked(self, t: TaskCtx) -> None:
        if not t.parked:
            return
        try:
            os.remove(self.save_dir / t.parked)
        except OSError:
            pass
        self.parked_bytes -= t.parked_bytes
        t.parked, t.parked_bytes = None, 0

    def _log(self, action: str, slot_id: int, t: TaskCtx, **kw) -> None:
        self.events.append({"t": self.rel(time.monotonic()), "action": action, "slot": slot_id,
                            "task_id": t.id, "cls": t.cls, "parked_bytes_total": self.parked_bytes,
                            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()}})
