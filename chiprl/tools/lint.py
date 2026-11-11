"""Lint = static policy checks (anti-hack) + Verilator --lint-only."""

from __future__ import annotations

import re
from pathlib import Path

from ._run import ToolResult, run, which

# System tasks a synthesizable DUT never needs. $finish / $display / final in the DUT are
# the classic VerilogEval reward hacks (end sim early, spoof the "Mismatches: 0" line).
_ALLOWED_SYSTASKS = {"$signed", "$unsigned", "$clog2", "$bits"}
_SYSTASK_RE = re.compile(r"\$[a-zA-Z_]\w*")
_RESERVED_MODULES = {"RefModule", "tb", "stimulus_gen"}


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", " ", src)


def policy_check(src: str, top: str) -> list[dict]:
    diags = []
    code = _strip_comments(src)
    for t in sorted(set(_SYSTASK_RE.findall(code)) - _ALLOWED_SYSTASKS):
        diags.append({"rule": "banned_systask", "msg": f"{t} not allowed in DUT"})
    if re.search(r"\bfinal\b", code):
        diags.append({"rule": "banned_final", "msg": "final blocks not allowed in DUT"})
    if re.search(r"`include|`timescale", code):
        diags.append({"rule": "banned_directive", "msg": "`include/`timescale not allowed"})
    mods = re.findall(r"\bmodule\s+(\w+)", code)
    for m in mods:
        if m in _RESERVED_MODULES:
            diags.append({"rule": "reserved_module", "msg": f"module {m} shadows grader module"})
    if top not in mods:
        diags.append({"rule": "missing_top", "msg": f"module {top} not defined"})
    return diags


_VL_RE = re.compile(r"%(Error|Warning)(?:-(\w+))?:\s*(?:[^:]+:(\d+):(?:\d+:)?)?\s*(.*)")


def lint(sub_path: Path, top: str, workdir: Path) -> ToolResult:
    src = sub_path.read_text()
    diags = policy_check(src, top)
    if diags:
        return ToolResult("lint", ok=False, diagnostics=diags, metrics={"policy_violations": len(diags)})
    exe = which("verilator")
    if not exe:
        return ToolResult("lint", ok=False, infra_error=True, diagnostics=[{"msg": "verilator missing"}])
    rc, out, dt = run(
        [
            exe,
            "--lint-only",
            "-Wall",
            "-Wno-DECLFILENAME",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-fatal",
            "--top-module",
            top,
            str(sub_path),
        ],
        workdir,
        "lint.log",
        timeout_s=60,
    )
    if rc is None:
        return ToolResult("lint", ok=False, infra_error=True, timeout=True, elapsed_s=dt)
    for sev, code, line, msg in _VL_RE.findall(out):
        diags.append(
            {
                "severity": sev.lower(),
                "code": code or None,
                "line": int(line) if line else None,
                "msg": msg.strip()[:200],
            }
        )
    errors = sum(d["severity"] == "error" for d in diags)
    return ToolResult(
        "lint",
        ok=(rc == 0 and errors == 0),
        diagnostics=diags[:50],
        metrics={"errors": errors, "warnings": len(diags) - errors},
        elapsed_s=dt,
        log_path=str(workdir / "lint.log"),
    )
