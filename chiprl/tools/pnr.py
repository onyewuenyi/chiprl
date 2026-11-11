"""Slow-path ground truth: LibreLane (OpenROAD) RTL-to-GDS, then VCD-annotated post-route power.

LibreLane's own power numbers use default activity (it has no VCD/SAIF input). So the flow here is:
  1. LibreLane Classic flow (Yosys via slang, OpenROAD floorplan/place/CTS/route, RCX)
  2. gate-level sim of the *post-route* netlist (includes CTS buffers) on the hidden stimulus
  3. OpenSTA: post-route netlist + RCX SPEF + that VCD -> report_power, WNS

Minutes per design, so it runs on a sampled audit slice, never inside the RL loop.
Runs containerized (`--dockerized`) so the Mac needs only Docker; ciel fetches the PDK.
Install: uv sync --extra pd
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..tasks.task import Task
from ._run import ToolResult, run
from .power import opensta_power
from .sim import simulate

PDK, SCL = "sky130A", "sky130_fd_sc_hd"


def librelane_config(task: Task, rtl_name: str, period_ns: float, die_um: float) -> dict:
    cfg: dict = {
        "DESIGN_NAME": task.top,
        "VERILOG_FILES": [f"dir::{rtl_name}"],
        "USE_SLANG": True,
        "CLOCK_PERIOD": period_ns,
        "FP_SIZING": "absolute",
        "DIE_AREA": [0, 0, die_um, die_um],
        "PL_TARGET_DENSITY_PCT": 55,
        "RUN_LINTER": False,  # our lint gate already ran
    }
    cfg["CLOCK_PORT"] = task.clock  # None -> virtual clock for combinational tasks
    return cfg


def _find(final: Path, sub: str, pattern: str) -> Path | None:
    hits = sorted((final / sub).rglob(pattern)) if (final / sub).exists() else []
    nom = [h for h in hits if "nom" in str(h)]
    return (nom or hits or [None])[0]


def pnr(
    task: Task,
    sub_rtl: str,
    workdir: Path,
    period_ns: float = 10.0,
    die_um: float = 120.0,
    dockerized: bool = True,
    timeout_s: int = 1800,
) -> ToolResult:
    exe = shutil.which("librelane")
    if not exe:
        return ToolResult(
            "pnr", ok=False, infra_error=True, diagnostics=[{"msg": "librelane missing (uv sync --extra pd)"}]
        )
    d = workdir / "pnr"
    d.mkdir(exist_ok=True)
    (d / "dut.sv").write_text(sub_rtl)
    (d / "config.json").write_text(json.dumps(librelane_config(task, "dut.sv", period_ns, die_um), indent=2))
    cmd = [
        exe,
        *(["--dockerized"] if dockerized else []),
        "--pdk",
        PDK,
        "--scl",
        SCL,
        "--run-tag",
        "chiprl",
        "--overwrite",
        "--hide-progress-bar",
        "config.json",
    ]
    rc, out, dt = run(cmd, d, "librelane.log", timeout_s=timeout_s, cpu_s=timeout_s, mem_mb=16384)
    final = d / "runs" / "chiprl" / "final"
    if rc is None:
        return ToolResult("pnr", ok=False, timeout=True, elapsed_s=dt)
    if rc != 0 or not final.exists():
        tail = out.strip().splitlines()[-10:]
        return ToolResult(
            "pnr",
            ok=False,
            elapsed_s=dt,
            diagnostics=[{"msg": ln[:200]} for ln in tail],
            log_path=str(d / "librelane.log"),
        )
    m = json.loads((final / "metrics.json").read_text()) if (final / "metrics.json").exists() else {}
    metrics = {
        "area_um2": m.get("design__instance__area"),
        "wns_ns": m.get("timing__setup__ws"),
        "power_default_activity_w": m.get("power__total"),
        "instances": m.get("design__instance__count"),
        "flow_s": round(dt, 1),
    }
    nl, spef = _find(final, "nl", "*.nl.v"), _find(final, "spef", "*.spef")
    lib = _find(Path.home() / ".ciel", "", f"{SCL}__tt_025C_1v80.lib")
    if nl and spef and lib:
        gl = simulate(
            nl, task.ref_rtl, task.testbench, d, vcd="pr.vcd", gate_level=True, timescale_ns=True, tag="prsim"
        )
        metrics["post_route_sim_pass"] = gl.ok
        if gl.metrics.get("samples"):
            p = opensta_power(
                nl, d / "pr.vcd", str(lib), task.top, d, task.clock, period_ns, "tb/top_module1", spef=spef
            )
            metrics["vcd_power"] = p.metrics if p.ok else {"error": p.diagnostics}
    else:
        metrics["vcd_power"] = {"error": f"missing views nl={nl} spef={spef} lib={lib}"}
    return ToolResult("pnr", ok=True, elapsed_s=dt, metrics=metrics, log_path=str(d / "librelane.log"))


__all__ = ["librelane_config", "pnr", "run"]
