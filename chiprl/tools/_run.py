"""Subprocess runner shared by every tool wrapper.

Every tool returns a ToolResult that serializes to JSON. Raw logs are kept on disk
(log_path) but never handed to a policy: the model only ever sees parsed fields.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"


@dataclasses.dataclass
class ToolResult:
    tool: str
    ok: bool
    infra_error: bool = False  # tool missing, crash, OOM: never scored as 0
    timeout: bool = False
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    elapsed_s: float = 0.0
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def which(name: str) -> str | None:
    return shutil.which(name)


TERM_GRACE_S = 2.0


def _limit_prefix(cpu_s: int, mem_mb: int) -> list[str]:
    """Apply rlimits in a shell that then execs the tool.

    Not `preexec_fn`: Python documents it as unsafe in a multithreaded process, and the grader runs
    many jobs on threads. CPU soft limit sends SIGXCPU, the hard limit 5 s later sends SIGKILL.
    """
    lines = [f"ulimit -Ht {cpu_s + 5}", f"ulimit -St {cpu_s}"]
    if sys.platform.startswith("linux"):  # macOS does not enforce RLIMIT_AS
        lines.append(f"ulimit -v {mem_mb << 10}")
    return ["/bin/sh", "-c", "; ".join(lines) + '; exec "$@"', "chiprl-limits"]


def _kill_group(p: subprocess.Popen) -> None:
    """SIGTERM the whole process group, then SIGKILL whatever is left after a grace period.

    TERM first so wrappers can clean up (the docker client stops its container on TERM); the group
    so grandchildren die too (a shell wrapper's children would otherwise outlive the timeout).
    """
    for sig, wait_s in ((signal.SIGTERM, TERM_GRACE_S), (signal.SIGKILL, None)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        if wait_s is None:
            return
        # Even if the leader exits here, its children may still be in the group: KILL follows.
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(timeout=wait_s)


def run(
    cmd: list[str], cwd: Path, log_name: str, timeout_s: int = 120, cpu_s: int = 300, mem_mb: int = 4096
) -> tuple[int | None, str, float]:
    """Run cmd in cwd with rlimits. Returns (returncode or None on timeout, combined output, elapsed).

    The tool runs in its own session (process group). On timeout the whole group is terminated, so
    nothing it spawned keeps running. This is the in-process sandbox (rlimits + separate workdir).
    """
    t0 = time.time()
    p = subprocess.Popen(
        _limit_prefix(cpu_s, mem_mb) + cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=timeout_s)
        rc: int | None = p.returncode
    except subprocess.TimeoutExpired:
        _kill_group(p)
        try:
            out, _ = p.communicate(timeout=TERM_GRACE_S + 3)
        except subprocess.TimeoutExpired:  # a descendant still holds the pipe open
            out = ""
        out = (out or "") + "\n<TIMEOUT>"
        rc = None
    elapsed = time.time() - t0
    (cwd / log_name).write_text(out)
    return rc, out, elapsed


def tool_versions() -> dict[str, str]:
    """Versions go into every cache key so a tool upgrade invalidates cached results.

    Computed once per process (it runs each tool, and `sta` is a container start); returns a copy.
    """
    return dict(_tool_versions())


@lru_cache(maxsize=1)
def _tool_versions() -> dict[str, str]:
    out = {}
    for name, args in [
        ("yosys", ["-V"]),
        ("iverilog", ["-V"]),
        ("verilator", ["--version"]),
        ("sta", ["-version"]),
    ]:
        exe = which(name)
        if not exe:
            out[name] = "missing"
            continue
        try:
            p = subprocess.run(
                [exe, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,
            )
            out[name] = (p.stdout or p.stderr).strip().splitlines()[0]
        except Exception as e:
            out[name] = f"error:{e}"
    return out


@lru_cache(maxsize=1)
def has_yosys_slang() -> bool:
    """True if the yosys-slang plugin (full SystemVerilog frontend) loads. Ships with OSS CAD Suite."""
    exe = which("yosys")
    if not exe or os.environ.get("CHIPRL_NO_SLANG"):
        return False
    try:
        p = subprocess.run(
            [exe, "-q", "-p", "plugin -i slang; help read_slang"],
            check=False,
            capture_output=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return p.returncode == 0


def yosys_read_sv(path: Path | str, top: str | None = None) -> str:
    """Yosys script lines that read a SystemVerilog file: yosys-slang if present, else read_verilog -sv."""
    if has_yosys_slang():
        return f"plugin -i slang\nread_slang {path}" + (f" --top {top}" if top else "")
    return f"read_verilog -sv {path}"
