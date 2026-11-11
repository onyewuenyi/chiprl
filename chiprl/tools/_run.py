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


def _limits(cpu_s: int, mem_mb: int):
    def fn():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        if sys.platform.startswith("linux"):  # macOS does not enforce RLIMIT_AS
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))

    return fn


def run(
    cmd: list[str], cwd: Path, log_name: str, timeout_s: int = 120, cpu_s: int = 300, mem_mb: int = 4096
) -> tuple[int | None, str, float]:
    """Run cmd in cwd with rlimits. Returns (returncode or None on timeout, combined output, elapsed).

    This is the in-process sandbox (rlimits + separate workdir). The Docker sandbox
    (network none, read-only grader mounts) wraps this same call in envs/sandbox.py.
    """
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            check=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=_limits(cpu_s, mem_mb),
            errors="replace",
        )
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc = None
        out = (
            (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        ) + "\n<TIMEOUT>"
    elapsed = time.time() - t0
    (cwd / log_name).write_text(out)
    return rc, out, elapsed


def tool_versions() -> dict[str, str]:
    """Versions go into every cache key so a tool upgrade invalidates cached results."""
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
