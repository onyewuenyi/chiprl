"""Yosys + ABC synthesis to Sky130 HD. Returns area, cell counts, logic depth, netlist paths."""
from __future__ import annotations

import gzip
import re
import shutil
from pathlib import Path

from ._run import THIRD_PARTY, ToolResult, run, which

LIB_GZ = THIRD_PARTY / "sky130/sky130hd_tt.lib.gz"


def liberty_path() -> Path:
    """Unzipped Sky130 HD tt_025C_1v80 liberty (cached next to the .gz)."""
    out = LIB_GZ.with_suffix("")  # .lib
    if not out.exists():
        tmp = out.with_suffix(".lib.tmp")
        with gzip.open(LIB_GZ, "rb") as f, open(tmp, "wb") as g:
            shutil.copyfileobj(f, g)
        tmp.rename(out)
    return out


_AREA_RE = re.compile(r"Chip area for (?:top )?module '\\?(\w+)':\s*([\d.]+)")
_CELLS_RE = re.compile(r"Number of cells:\s*(\d+)")
_LTP_RE = re.compile(r"Longest topological path in \\?(\w+) \(length=(\d+)\)")
_CELLTYPE_RE = re.compile(r"^\s+(sky130_fd_sc_hd__\w+)\s+(\d+)\s*$", re.M)


def synth(sub_path: Path, top: str, workdir: Path, flags: str = "", timeout_s: int = 300) -> ToolResult:
    exe = which("yosys")
    if not exe:
        return ToolResult("synth", ok=False, infra_error=True, diagnostics=[{"msg": "yosys missing"}])
    lib = liberty_path()
    script = f"""
read_verilog -sv {sub_path}
synth -top {top} -flatten {flags}
tee -o ltp.txt ltp -noff
dfflibmap -liberty {lib}
abc -liberty {lib}
opt_clean -purge
hilomap -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO
setundef -zero
rename -enumerate
tee -o stat.txt stat -liberty {lib}
write_verilog -noattr -noexpr -nohex -nodec netlist.v
read_liberty -lib {lib}
write_json netlist.json
"""
    (workdir / "synth.ys").write_text(script)
    rc, out, dt = run([exe, "-q", "-s", "synth.ys"], workdir, "synth.log", timeout_s=timeout_s)
    if rc is None:
        return ToolResult("synth", ok=False, timeout=True, elapsed_s=dt)
    if rc != 0:
        errs = [ln.strip()[:200] for ln in out.splitlines() if "ERROR" in ln][:10]
        return ToolResult("synth", ok=False, elapsed_s=dt, diagnostics=[{"msg": e} for e in errs],
                          log_path=str(workdir / "synth.log"))
    stat = (workdir / "stat.txt").read_text()
    ltp = (workdir / "ltp.txt").read_text()
    area = _AREA_RE.findall(stat)
    cells = _CELLS_RE.findall(stat)
    depth = _LTP_RE.findall(ltp)
    types = {k: int(v) for k, v in _CELLTYPE_RE.findall(stat)}
    n_ff = sum(v for k, v in types.items() if re.search(r"__(df|dl|sdf|edf)", k))
    return ToolResult("synth", ok=True, elapsed_s=dt, log_path=str(workdir / "synth.log"), metrics={
        "area_um2": float(area[-1][1]) if area else None,
        "cells": int(cells[-1]) if cells else None,
        "flops": n_ff,
        "logic_depth": int(depth[-1][1]) if depth else None,
        "cell_types": types,
        "netlist": str(workdir / "netlist.v"),
        "netlist_json": str(workdir / "netlist.json"),
        "liberty": str(lib),
    })
