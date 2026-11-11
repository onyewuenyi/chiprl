"""Yosys + ABC synthesis to Sky130 HD. Returns area, cell counts, logic depth, netlist paths."""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path

from ._run import THIRD_PARTY, ToolResult, run, which

LIB_GZ = THIRD_PARTY / "sky130/sky130hd_tt.lib.gz"


# Cells the ORFS sky130hd platform marks dont_use (power-gating, probes, delay cells).
# Filtering the liberty instead of `abc -dont_use` keeps this portable across Yosys versions.
DONT_USE = re.compile(r"__(lpflow_|probe|probec|macro_sparecell|dlygate|dlymetal|clkdlybuf|clkdlyinv)")


def _filter_liberty(text: str) -> str:
    out, i = [], 0
    for m in re.finditer(r'\n\s*cell\s*\(\s*"?(\w+)"?\s*\)\s*\{', text):
        if m.start() < i or not DONT_USE.search(m.group(1)):
            continue
        depth, j = 0, m.end() - 1
        while True:
            c = text[j]
            depth += (c == "{") - (c == "}")
            j += 1
            if depth == 0:
                break
        out.append(text[i : m.start()])
        i = j
    out.append(text[i:])
    return "".join(out)


def liberty_path() -> Path:
    """Sky130 HD tt_025C_1v80 liberty, unzipped and dont_use-filtered (cached next to the .gz)."""
    out = LIB_GZ.parent / "sky130hd_tt.chiprl.lib"
    if not out.exists():
        with gzip.open(LIB_GZ, "rt") as f:
            text = _filter_liberty(f.read())
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.rename(out)
    return out


_LTP_RE = re.compile(r"Longest topological path in \\?(\w+) \(length=(\d+)\)")


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
tee -q -o stat.json stat -json -liberty {lib}
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
        return ToolResult(
            "synth",
            ok=False,
            elapsed_s=dt,
            diagnostics=[{"msg": e} for e in errs],
            log_path=str(workdir / "synth.log"),
        )
    st = json.loads((workdir / "stat.json").read_text())["modules"]
    mod = st.get(f"\\{top}") or st.get(top) or next(iter(st.values()))
    depth = _LTP_RE.findall((workdir / "ltp.txt").read_text())
    types = {k.lstrip("\\"): int(v) for k, v in mod.get("num_cells_by_type", {}).items()}
    n_ff = sum(v for k, v in types.items() if re.search(r"__(df|dl|sdf|edf)", k))
    area = mod.get("area")
    if isinstance(area, dict):  # newer yosys nests area stats
        area = area.get("total", area.get("chip_area"))
    return ToolResult(
        "synth",
        ok=True,
        elapsed_s=dt,
        log_path=str(workdir / "synth.log"),
        metrics={
            "area_um2": float(area) if area is not None else None,
            "cells": int(mod.get("num_cells", 0)),
            "flops": n_ff,
            "logic_depth": int(depth[-1][1]) if depth else None,
            "cell_types": types,
            "netlist": str(workdir / "netlist.v"),
            "netlist_json": str(workdir / "netlist.json"),
            "liberty": str(lib),
        },
    )
