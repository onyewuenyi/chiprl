"""Static timing at several process corners (OpenSTA), so slack is checked where it is worst.

    ss  sky130_fd_sc_hd__ss_100C_1v60   slow transistors, hot, low voltage: setup-critical
    tt  the vendored tt_025C_1v80 (the library synthesis used)
    ff  sky130_fd_sc_hd__ff_n40C_1v95   fast, cold, high voltage: hold-critical

One OpenSTA run per corner, each reading that corner's liberty against the same netlist (the cell
names match across Sky130 HD corners). Setup slack (report_wns, max paths) and hold slack (min
paths) are both reported; `met` needs setup and hold slack >= 0 at every corner. Corner libraries
come from the full PDK (ciel, installed with LibreLane), found under CHIPRL_SKY130_LIB_DIR or
~/.ciel; a missing corner is an infra error, never a pass.

This is a measurement. The optimization reward's timing gate (docs/REWARD.md §3) uses it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ._run import ToolResult, run, which
from .synth import liberty_path

CORNER_LIBS = {"ss": "sky130_fd_sc_hd__ss_100C_1v60.lib", "ff": "sky130_fd_sc_hd__ff_n40C_1v95.lib"}
DEFAULT_CORNERS = ("ss", "tt", "ff")
_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def pdk_lib_dir() -> Path | None:
    env = os.environ.get("CHIPRL_SKY130_LIB_DIR")
    if env:
        return Path(env) if Path(env).is_dir() else None
    hits = sorted(Path.home().glob(".ciel/ciel/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/lib"))
    return hits[-1] if hits else None


def corner_lib(corner: str) -> Path | None:
    if corner == "tt":
        return liberty_path()
    d = pdk_lib_dir()
    p = d / CORNER_LIBS[corner] if d and corner in CORNER_LIBS else None
    return p if p and p.exists() else None


def sta_corner(
    netlist_v: Path, lib: Path, top: str, clock: str | None, period_ns: float, workdir: Path, corner: str
) -> ToolResult:
    exe = which("sta")
    if not exe:
        return ToolResult("timing", ok=False, infra_error=True, diagnostics=[{"msg": "sta missing"}])
    clk = (
        f"create_clock -name clk -period {period_ns} [get_ports {clock}]"
        if clock
        else f"create_clock -name vclk -period {period_ns}"
    )
    tcl = f"""read_liberty {lib}
read_verilog {netlist_v}
link_design {top}
{clk}
set_input_delay 0 -clock [get_clocks *] [all_inputs]
set_output_delay 0 -clock [get_clocks *] [all_outputs]
puts "CHIPRL_SETUP [sta::format_time [sta::worst_slack_cmd max] 4]"
puts "CHIPRL_HOLD [sta::format_time [sta::worst_slack_cmd min] 4]"
report_tns -digits 4
exit
"""
    tcl_name = f"timing_{corner}.tcl"
    (workdir / tcl_name).write_text(tcl)
    rc, out, dt = run([exe, "-no_splash", "-exit", tcl_name], workdir, f"timing_{corner}.log", timeout_s=300)
    setup = re.search(rf"CHIPRL_SETUP\s+{_NUM}", out)
    hold = re.search(rf"CHIPRL_HOLD\s+{_NUM}", out)
    tns = re.search(rf"tns\s+(?:max\s+)?{_NUM}", out)
    if rc != 0 or not setup or not hold:
        return ToolResult(
            "timing",
            ok=False,
            infra_error=True,
            elapsed_s=dt,
            diagnostics=[{"msg": f"OpenSTA failed at corner {corner}"}],
            log_path=str(workdir / f"timing_{corner}.log"),
        )
    return ToolResult(
        "timing",
        ok=True,
        elapsed_s=dt,
        metrics={
            "corner": corner,
            "setup_slack_ns": float(setup.group(1)),
            # Single clock, ideal: slack moves one for one with the period, so this is the corner's
            # minimum period (1 / Fmax). R3 sets each task's clock at 90-95% of Fmax.
            "min_period_ns": round(period_ns - float(setup.group(1)), 4),
            "hold_slack_ns": float(hold.group(1)),
            "tns_ns": float(tns.group(1)) if tns else None,
            "lib": lib.name,
        },
        log_path=str(workdir / f"timing_{corner}.log"),
    )


def multi_corner(
    netlist_v: Path,
    top: str,
    clock: str | None,
    period_ns: float,
    workdir: Path,
    corners: tuple[str, ...] = DEFAULT_CORNERS,
) -> ToolResult:
    """metrics: corners {name: {setup_slack_ns, min_period_ns, hold_slack_ns, tns_ns, lib}}, wns_ns,
    whs_ns, worst_setup_corner, min_period_ns (at the worst corner), met."""
    per: dict[str, dict] = {}
    elapsed = 0.0
    for c in corners:
        lib = corner_lib(c)
        if lib is None:
            return ToolResult(
                "timing", ok=False, infra_error=True, diagnostics=[{"msg": f"no liberty for corner {c}"}]
            )
        r = sta_corner(netlist_v, lib, top, clock, period_ns, workdir, c)
        elapsed += r.elapsed_s
        if not r.ok:
            return ToolResult("timing", ok=False, infra_error=True, elapsed_s=elapsed, diagnostics=r.diagnostics,
                              log_path=r.log_path)  # fmt: skip
        per[c] = {k: v for k, v in r.metrics.items() if k != "corner"}
    worst = min(per, key=lambda c: per[c]["setup_slack_ns"])
    wns = per[worst]["setup_slack_ns"]
    whs = min(p["hold_slack_ns"] for p in per.values())
    return ToolResult(
        "timing",
        ok=True,
        elapsed_s=elapsed,
        metrics={
            "period_ns": period_ns,
            "corners": per,
            "wns_ns": wns,
            "whs_ns": whs,
            "worst_setup_corner": worst,
            "min_period_ns": per[worst]["min_period_ns"],
            "met": wns >= 0 and whs >= 0,
        },
    )
