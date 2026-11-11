"""Subprocess runner shared by every tool wrapper.

Every tool returns a ToolResult that serializes to JSON. Raw logs are kept on disk
(log_path) but never handed to a policy: the model only ever sees parsed fields.
"""
from __future__ import annotations

import dataclasses
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"


@dataclasses.dataclass
class ToolResult:
    tool: str
    ok: bool
    infra_error: bool = False          # tool missing, crash, OOM: never scored as 0
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


def _limits(cpu_s: int, mem_mb: int):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        os.setsid()
    return fn


def run(cmd: list[str], cwd: Path, log_name: str, timeout_s: int = 120,
        cpu_s: int = 300, mem_mb: int = 4096) -> tuple[int | None, str, float]:
    """Run cmd in cwd with rlimits. Returns (returncode or None on timeout, combined output, elapsed).

    This is the in-process sandbox (rlimits + separate workdir). The Docker sandbox
    (network none, read-only grader mounts) wraps this same call in envs/sandbox.py.
    """
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s,
                           preexec_fn=_limits(cpu_s, mem_mb), errors="replace")
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc = None
        out = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n<TIMEOUT>"
    elapsed = time.time() - t0
    (cwd / log_name).write_text(out)
    return rc, out, elapsed


def tool_versions() -> dict[str, str]:
    """Versions go into every cache key so a tool upgrade invalidates cached results."""
    out = {}
    for name, args in [("yosys", ["-V"]), ("iverilog", ["-V"]), ("verilator", ["--version"]),
                       ("sta", ["-version"])]:
        exe = which(name)
        if not exe:
            out[name] = "missing"
            continue
        try:
            p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=10,
                               stdin=subprocess.DEVNULL)
            out[name] = (p.stdout or p.stderr).strip().splitlines()[0]
        except Exception as e:  # noqa: BLE001
            out[name] = f"error:{e}"
    return out
