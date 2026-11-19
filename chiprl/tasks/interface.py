"""A design's port list, read by Yosys rather than by regex.

`task.parse_ports` handles only `input [hi:lo] name`; it misreads `input signed [..] x` and raises on
parameterized widths. Anything that has to drive a design from outside (differential simulation,
behavior signatures) needs the real interface, so it comes from Yosys's elaborated netlist.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from functools import lru_cache
from pathlib import Path

from ..tools._run import run, which, yosys_read_sv

Port = tuple[str, str, int]  # (direction, name, width)

RESET_NAMES = ("reset", "rst", "areset", "aresetn", "resetn", "rst_n", "clear", "clr")
CLOCK_NAMES = ("clk", "clock")


@lru_cache(maxsize=512)
def _ports_cached(digest: str, rtl: str, top: str) -> tuple[Port, ...] | None:
    exe = which("yosys")
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="chiprl_ports_") as d:
        wd = Path(d)
        (wd / "d.sv").write_text(rtl)
        (wd / "p.ys").write_text(f"{yosys_read_sv('d.sv', top)}\nhierarchy -top {top}\nwrite_json p.json\n")
        rc, _, _ = run([exe, "-q", "-s", "p.ys"], wd, "p.log", timeout_s=60)
        if rc != 0 or not (wd / "p.json").exists():
            return None
        mods = json.loads((wd / "p.json").read_text())["modules"]
    mod = mods.get(top) or next((m for m in mods.values() if m.get("attributes", {}).get("top")), None)
    if mod is None:
        return None
    out: list[Port] = []
    for name, p in mod["ports"].items():
        if p["direction"] not in ("input", "output"):
            return None  # inout: not drivable from a plain testbench
        out.append((p["direction"], name, len(p["bits"])))
    return tuple(out)


def ports(rtl: str, top: str) -> tuple[Port, ...] | None:
    """Ports of module `top` in `rtl`, in declaration order. None if Yosys cannot elaborate it."""
    return _ports_cached(hashlib.sha256(rtl.encode()).hexdigest(), rtl, top)


def clock_of(ps: tuple[Port, ...]) -> str | None:
    return next((n for d, n, w in ps if d == "input" and w == 1 and n.lower() in CLOCK_NAMES), None)


def resets_of(ps: tuple[Port, ...]) -> list[str]:
    return [n for d, n, w in ps if d == "input" and w == 1 and n.lower() in RESET_NAMES]
