"""Yosys equivalence check (equiv_make / equiv_simple -seq / equiv_induct).

Bounded + inductive; 'proven' only when every $equiv cell is proven. Unproven is NOT a
failure for grading purposes (it just withholds the +0.1 bonus) because induction is
incomplete for designs whose state encodings differ.
"""
from __future__ import annotations

import re
from pathlib import Path

from ._run import ToolResult, run, which

_STATUS_RE = re.compile(r"Found (\d+) \$equiv cells.*?Of those cells (\d+) are proven and (\d+) are unproven", re.S)


def equiv(sub_path: Path, ref_rtl: str, top: str, workdir: Path, seq: int = 8) -> ToolResult:
    exe = which("yosys")
    if not exe:
        return ToolResult("equiv", ok=False, infra_error=True)
    (workdir / "equiv_gold.sv").write_text(re.sub(r"\bRefModule\b", "gold", ref_rtl))
    (workdir / "equiv_gate.sv").write_text(re.sub(rf"\b{top}\b", "gate", sub_path.read_text()))
    script = f"""
read_verilog -sv equiv_gold.sv
read_verilog -sv equiv_gate.sv
proc
opt_clean
async2sync
equiv_make gold gate equiv
hierarchy -top equiv
equiv_simple -seq {seq}
equiv_induct -seq {seq}
tee -o equiv_status.txt equiv_status
"""
    (workdir / "equiv.ys").write_text(script)
    rc, out, dt = run([exe, "-q", "-s", "equiv.ys"], workdir, "equiv.log", timeout_s=120)
    if rc is None:
        return ToolResult("equiv", ok=False, timeout=True, elapsed_s=dt, metrics={"proven": False})
    status = (workdir / "equiv_status.txt").read_text() if (workdir / "equiv_status.txt").exists() else ""
    m = _STATUS_RE.search(status)
    if "Equivalence successfully proven" in status:
        return ToolResult("equiv", ok=True, elapsed_s=dt, metrics={"proven": True})
    if m:
        tot, prov, unprov = map(int, m.groups())
        return ToolResult("equiv", ok=True, elapsed_s=dt,
                          metrics={"proven": unprov == 0, "cells": tot, "unproven": unprov})
    return ToolResult("equiv", ok=False, elapsed_s=dt, metrics={"proven": False},
                      log_path=str(workdir / "equiv.log"))
