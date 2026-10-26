"""Start and stop llama-server (or the mock server) for one run.

Server config keys:
    binary   path to llama-server (ignored when mock is true); relative = to the repo root
    model    path to the GGUF file; relative = to the repo root
    ctx      total context (-c); split evenly across slots
    parallel number of slots (-np)
    ngl      layers offloaded to GPU (-ngl), default 99
    port     default 8080
    extra    list of extra CLI args
    log_verbosity llama.cpp -lv level, default 4 (needed for buffer sizes in server.log)
    mock     true to launch tools/mock_server.py instead (pipeline testing only)
    mock_args  extra args for the mock server
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def resolve(path: str) -> str:
    """Expand ~ and $VARS; a relative path is relative to the repo root, so the same
    config works on any machine with the layout <root>/{agentserve,llama.cpp,models}."""
    p = Path(os.path.expandvars(path)).expanduser()
    return str(p if p.is_absolute() else (REPO / p).resolve())


def build_cmd(cfg: dict) -> list[str]:
    port = str(cfg.get("port", 8080))
    if cfg.get("mock"):
        return [sys.executable, str(REPO / "tools" / "mock_server.py"),
                "--port", port, "--ctx", str(cfg["ctx"]), "--parallel", str(cfg["parallel"]),
                *map(str, cfg.get("mock_args", []))]
    return [resolve(cfg["binary"]), "-m", resolve(cfg["model"]),
            "-c", str(cfg["ctx"]), "-np", str(cfg["parallel"]),
            "-ngl", str(cfg.get("ngl", 99)),
            "--host", "127.0.0.1", "--port", port, "--metrics",
            # -lv 4 logs the model/KV/compute buffer sizes that analyze.py parses;
            # 5+ logs per-token debug output and a dry-run pass with 0.00 MiB sizes.
            "-lv", str(cfg.get("log_verbosity", 4)),
            *map(str, cfg.get("extra", []))]


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def verify_own_server(url: str, proc: subprocess.Popen, cfg: dict) -> None:
    """Fail unless the server answering at `url` is the process we just started, with our slots.

    Guards against a stray server already listening on the port: ours would fail to bind and
    exit, while the stray one answers /health, so every request would silently go to it.
    """
    if proc.poll() is not None:
        raise RuntimeError(f"server exited with code {proc.returncode} during startup "
                           "(see server.log; e.g. port already in use)")
    try:
        with urllib.request.urlopen(url + "/props", timeout=5) as r:
            props = json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError) as e:
        raise RuntimeError(f"could not read {url}/props to verify the server: {e!r}") from e
    slots = props.get("total_slots")
    if slots is not None and int(slots) != int(cfg["parallel"]):
        raise RuntimeError(f"server at {url} reports {slots} slots but this run started one with "
                           f"-np {cfg['parallel']}: another server is answering on this port")


def wait_healthy(url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} during startup")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


class Server:
    def __init__(self, cfg: dict, log_path: Path):
        self.cfg = cfg
        self.log_path = log_path
        self.url = f"http://127.0.0.1:{cfg.get('port', 8080)}"
        self.proc: subprocess.Popen | None = None
        self.startup_s: float | None = None
        self.exit_code_before_stop: int | None = None  # set if the server had already exited on its own

    def __enter__(self) -> "Server":
        port = int(self.cfg.get("port", 8080))
        if port_in_use(port):
            raise RuntimeError(f"port {port} is already in use, probably by a leftover llama-server. "
                               f"Find it with `lsof -i :{port}` and stop it, then rerun the sweep.")
        cmd = build_cmd(self.cfg)
        self.log = open(self.log_path, "w")
        self.log.write("# " + " ".join(cmd) + "\n")
        self.log.flush()
        t = time.monotonic()
        # start_new_session: the server gets its own process group, so a Ctrl-C or hang-up
        # aimed at the terminal reaches only this Python process, which then stops the
        # server cleanly in __exit__.
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            wait_healthy(self.url, self.proc, self.cfg.get("startup_timeout_s", 600))
            verify_own_server(self.url, self.proc, self.cfg)
        except Exception:
            self.__exit__(None, None, None)
            raise
        self.startup_s = round(time.monotonic() - t, 3)  # cold start: process launch to healthy
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            # None = still running (normal). A number means it died before we stopped it:
            # negative = killed by that signal (-15 SIGTERM, -2 SIGINT, -9 SIGKILL).
            self.exit_code_before_stop = self.proc.poll()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None
