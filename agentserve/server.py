"""Start and stop llama-server (or the mock server) for one run.

Server config keys:
    binary   path to llama-server (ignored when mock is true)
    model    path to the GGUF file
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

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def build_cmd(cfg: dict) -> list[str]:
    port = str(cfg.get("port", 8080))
    if cfg.get("mock"):
        return [sys.executable, str(REPO / "tools" / "mock_server.py"),
                "--port", port, "--ctx", str(cfg["ctx"]), "--parallel", str(cfg["parallel"]),
                *map(str, cfg.get("mock_args", []))]
    return [str(Path(cfg["binary"]).expanduser()), "-m", str(Path(cfg["model"]).expanduser()),
            "-c", str(cfg["ctx"]), "-np", str(cfg["parallel"]),
            "-ngl", str(cfg.get("ngl", 99)),
            "--host", "127.0.0.1", "--port", port, "--metrics",
            # -lv 4 logs the model/KV/compute buffer sizes that analyze.py parses;
            # 5+ logs per-token debug output and a dry-run pass with 0.00 MiB sizes.
            "-lv", str(cfg.get("log_verbosity", 4)),
            *map(str, cfg.get("extra", []))]


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
