"""Switching-activity power from a gate-level VCD.

Two backends, same JSON shape:
  * opensta: `read_vcd` + `report_power` (ground truth for this repo; needs `sta` on PATH,
    e.g. inside the iic-osic-tools / ORFS container).
  * proxy:   pure Python. P_sw = sum_net 0.5 * C_net * V^2 * toggles / T, C_net = sum of
    liberty input-pin caps on the net's fanout; P_leak = sum of cell_leakage_power.
    No internal power, no wire cap. This is the "toggle-weighted capacitance" proxy that
    the Phase 5 proxy study correlates against OpenSTA.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from ._run import ToolResult, run, which

VDD = 1.8


# ---------------------------------------------------------------- liberty (minimal)
@lru_cache(maxsize=4)
def parse_liberty(lib_path: str) -> dict:
    """Returns {cell: {"area", "leak_nw", "pins": {pin: {"dir", "cap_pf"}}}}. Cached to JSON."""
    cache = Path(lib_path + ".chiprl.json")
    if cache.exists():
        return json.loads(cache.read_text())
    cells: dict = {}
    cell = pin = None
    depth = cell_depth = pin_depth = 0
    with open(lib_path) as fh:
        lines = [raw.strip() for raw in fh]
    for ln in lines:
        m = re.match(r'cell\s*\(\s*"?([\w]+)"?\s*\)\s*\{', ln)
        if m:
            cell = m.group(1)
            cells[cell] = {"area": 0.0, "leak_nw": 0.0, "pins": {}}
            cell_depth = depth + 1
        elif cell and (m := re.match(r'pin\s*\(\s*"?([\w]+)"?\s*\)\s*\{', ln)) and depth == cell_depth:
            pin = m.group(1)
            cells[cell]["pins"][pin] = {"dir": None, "cap_pf": 0.0}
            pin_depth = depth + 1
        elif cell and depth == cell_depth:
            if m := re.match(r"area\s*:\s*([\d.eE+-]+)", ln):
                cells[cell]["area"] = float(m.group(1))
            elif m := re.match(r"cell_leakage_power\s*:\s*([\d.eE+-]+)", ln):
                cells[cell]["leak_nw"] = float(m.group(1))
        elif pin and depth == pin_depth:
            if m := re.match(r'direction\s*:\s*"?(\w+)"?', ln):
                cells[cell]["pins"][pin]["dir"] = m.group(1)
            elif m := re.match(r"capacitance\s*:\s*([\d.eE+-]+)", ln):
                cells[cell]["pins"][pin]["cap_pf"] = float(m.group(1))
        depth += ln.count("{") - ln.count("}")
        if pin and depth < pin_depth:
            pin = None
        if cell and depth < cell_depth:
            cell = None
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cells))
    tmp.rename(cache)
    return cells


# ---------------------------------------------------------------- VCD toggle counts
def vcd_toggles(vcd_path: Path) -> tuple[dict[str, int], float, float]:
    """Per-bit toggle counts keyed 'name' or 'name[i]'. Returns (toggles, t_end, timescale_s)."""
    ids: dict[str, list[tuple[str, int, int]]] = {}  # id -> [(name, width, lsb)]
    last: dict[str, str] = {}
    tog: dict[str, int] = {}
    ts_s, t = 1e-12, 0
    unit = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}
    in_defs = True
    with open(vcd_path) as f:
        text_iter = iter(f)
        for raw in text_iter:
            ln = raw.strip()
            if in_defs:
                if ln.startswith("$timescale"):
                    body = ln if "$end" in ln else ln + " " + next(text_iter).strip()
                    m = re.search(r"(\d+)\s*(s|ms|us|ns|ps|fs)", body)
                    if m:
                        ts_s = int(m.group(1)) * unit[m.group(2)]
                elif ln.startswith("$var"):
                    parts = ln.split()
                    width, vid, name = int(parts[2]), parts[3], parts[4]
                    lsb = 0
                    if len(parts) > 6 and parts[5].startswith("["):
                        rng = parts[5].strip("[]").split(":")
                        lsb = int(rng[-1])
                    ids.setdefault(vid, []).append((name, width, lsb))
                elif ln.startswith("$enddefinitions"):
                    in_defs = False
                continue
            if not ln:
                continue
            c = ln[0]
            if c == "#":
                t = int(ln[1:])
                continue
            if c in "01xzXZ":
                val, vid = c, ln[1:]
            elif c in "bB":
                val, vid = ln[1:].split()
            else:
                continue
            aliases = ids.get(vid)
            if not aliases:
                continue
            width = aliases[0][1]
            pad = "0" if val[0] in "01" else val[0]
            v = val.rjust(width, pad)[-width:]
            old = last.get(vid)
            last[vid] = v
            if old is None:
                continue
            for i in range(width):
                a, b = old[width - 1 - i], v[width - 1 - i]
                if a != b and a in "01" and b in "01":
                    for name, _, lsb in aliases:
                        key = name if width == 1 else f"{name}[{lsb + i}]"
                        tog[key] = tog.get(key, 0) + 1
    return tog, t * ts_s, ts_s


# ---------------------------------------------------------------- proxy backend
def proxy_power(netlist_json: Path, vcd_path: Path, lib_path: str, top: str) -> dict:
    lib = parse_liberty(lib_path)
    mod = json.loads(netlist_json.read_text())["modules"][top]
    bit_name: dict[int, str] = {}
    for name, nn in mod["netnames"].items():
        bits = nn["bits"]
        off = nn.get("offset", 0)
        for i, b in enumerate(bits):
            if isinstance(b, int) and (b not in bit_name or not nn.get("hide_name")):
                bit_name[b] = name if len(bits) == 1 else f"{name}[{off + i}]"
    cap: dict[int, float] = {}
    leak_nw = 0.0
    for c in mod["cells"].values():
        lc = lib.get(c["type"])
        if not lc:
            continue
        leak_nw += lc["leak_nw"]
        for pname, bits in c["connections"].items():
            p = lc["pins"].get(pname)
            if p and p["dir"] == "input":
                for b in bits:
                    if isinstance(b, int):
                        cap[b] = cap.get(b, 0.0) + p["cap_pf"]
    tog, t_end, _ = vcd_toggles(vcd_path)
    if t_end <= 0:
        raise ValueError("empty VCD")
    e_sw = 0.0  # joules
    annotated = 0
    for b, c_pf in cap.items():
        n = tog.get(bit_name.get(b, ""), None)
        if n is None:
            continue
        annotated += 1
        e_sw += 0.5 * c_pf * 1e-12 * VDD**2 * n
    p_sw = e_sw / t_end
    return {
        "backend": "proxy",
        "switching_w": p_sw,
        "internal_w": None,
        "leakage_w": leak_nw * 1e-9,
        "total_w": p_sw + leak_nw * 1e-9,
        "nets_with_cap": len(cap),
        "nets_annotated": annotated,
        "annotation_frac": annotated / max(1, len(cap)),
        "sim_time_s": t_end,
        "toggle_weighted_cap_pf": sum(c * tog.get(bit_name.get(b, ""), 0) for b, c in cap.items()),
    }


# ---------------------------------------------------------------- OpenSTA backend
_PWR_TOTAL_RE = re.compile(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", re.M)


def opensta_power(
    netlist_v: Path,
    vcd_path: Path,
    lib_path: str,
    top: str,
    workdir: Path,
    clock: str | None,
    period_ns: float,
    scope: str,
    spef: Path | None = None,
) -> ToolResult:
    exe = which("sta")
    if not exe:
        return ToolResult("power", ok=False, infra_error=True, diagnostics=[{"msg": "sta missing"}])
    clk = (
        f"create_clock -name clk -period {period_ns} [get_ports {clock}]"
        if clock
        else f"create_clock -name vclk -period {period_ns}"
    )
    tcl = f"""read_liberty {lib_path}
read_verilog {netlist_v}
link_design {top}
{clk}
set_input_delay 0 -clock [get_clocks *] [all_inputs]
set_output_delay 0 -clock [get_clocks *] [all_outputs]
{f"read_spef {spef}" if spef else ""}
read_vcd -scope {scope} {vcd_path}
report_activity_annotation
report_power -digits 6
report_checks -path_delay max -digits 4 -format end
report_wns -digits 4
exit
"""
    (workdir / "power.tcl").write_text(tcl)
    rc, out, dt = run([exe, "-no_splash", "-exit", "power.tcl"], workdir, "sta.log", timeout_s=300)
    if rc is None or rc != 0:
        return ToolResult(
            "power", ok=False, infra_error=True, elapsed_s=dt, log_path=str(workdir / "sta.log")
        )
    m = _PWR_TOTAL_RE.search(out)
    wns = re.search(r"wns\s+(?:max\s+)?([\d.eE+-]+)", out)
    if not m:
        return ToolResult("power", ok=False, infra_error=True, diagnostics=[{"msg": "no Total line"}])
    internal, switching, leakage, total = (float(x) for x in m.groups())
    return ToolResult(
        "power",
        ok=True,
        elapsed_s=dt,
        log_path=str(workdir / "sta.log"),
        metrics={
            "backend": "opensta",
            "internal_w": internal,
            "switching_w": switching,
            "leakage_w": leakage,
            "total_w": total,
            "wns_ns": float(wns.group(1)) if wns else None,
            "period_ns": period_ns,
            "parasitics": "spef" if spef else "none",
        },
    )


def power(
    netlist_v: Path,
    netlist_json: Path,
    vcd_path: Path,
    lib_path: str,
    top: str,
    workdir: Path,
    clock: str | None,
    period_ns: float = 10.0,
    scope: str = "tb/top_module1",
    backend: str = "auto",
) -> ToolResult:
    """backend: auto (opensta if available, plus proxy), opensta, proxy."""
    metrics: dict = {}
    if backend in ("auto", "proxy"):
        try:
            metrics["proxy"] = proxy_power(netlist_json, vcd_path, lib_path, top)
        except Exception as e:
            return ToolResult("power", ok=False, infra_error=True, diagnostics=[{"msg": f"proxy: {e}"}])
    if backend in ("auto", "opensta") and which("sta"):
        r = opensta_power(netlist_v, vcd_path, lib_path, top, workdir, clock, period_ns, scope)
        metrics["opensta"] = r.metrics if r.ok else {"error": r.diagnostics or "failed"}
    elif backend == "opensta":
        return ToolResult("power", ok=False, infra_error=True, diagnostics=[{"msg": "sta missing"}])
    sta_m = metrics.get("opensta") or {}
    best = sta_m if sta_m.get("total_w") is not None else metrics.get("proxy")
    if best is None:
        return ToolResult("power", ok=False, infra_error=True, diagnostics=[{"msg": "opensta failed"}])
    return ToolResult(
        "power", ok=True, metrics={"total_w": best["total_w"], "backend": best["backend"], **metrics}
    )
