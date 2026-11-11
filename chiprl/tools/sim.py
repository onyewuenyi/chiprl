"""Compile + simulate a submission against the task's hidden VerilogEval testbench.

Anti-hack measures live here because the testbench is the reward source:
  * a per-run nonce is injected into the tb's result line; only nonce lines are parsed,
    so a DUT cannot print a fake "Mismatches: 0".
  * the sample count must match the reference run, so an early $finish cannot pass.
  * the tb's own $dumpvars is replaced with one scoped to the DUT for power analysis.
"""
from __future__ import annotations

import re
import secrets
from pathlib import Path

from ._run import THIRD_PARTY, ToolResult, run, which

_DUMP_RE = re.compile(r"initial\s+begin\s*\$dumpfile\([^;]*;\s*\$dumpvars\([^;]*;\s*end", re.S)
_RESULT_RE = re.compile(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples")


def prepare_tb(tb: str, nonce: str, vcd: str | None, timescale_ns: bool) -> str:
    dump = (f'initial begin $dumpfile("{vcd}"); $dumpvars(1, tb.top_module1); end'
            if vcd else "")
    tb, n = _DUMP_RE.subn(dump, tb, count=1)
    if n == 0 and vcd:
        tb = tb.replace("endmodule", dump + "\nendmodule", 1) if "module tb" not in tb else \
            re.sub(r"(module\s+tb\s*\([^)]*\)\s*;)", r"\1\n" + dump, tb, count=1)
    tb = tb.replace('$display("Mismatches:', f'$display("{nonce} Mismatches:')
    if timescale_ns:
        # VerilogEval uses 1ps; #5 half-period would be a 100 GHz clock. For power we want a
        # physical frequency: 1ns units -> 10 ns period -> 100 MHz.
        tb = re.sub(r"`timescale[^\n]*", "`timescale 1ns/1ps", tb)
    return tb


def simulate(sub_path: Path, ref_rtl: str, tb: str, workdir: Path, *, vcd: str | None = None,
             gate_level: bool = False, timescale_ns: bool = False, timeout_s: int = 120,
             tag: str = "sim") -> ToolResult:
    iverilog, vvp = which("iverilog"), which("vvp")
    if not (iverilog and vvp):
        return ToolResult(tag, ok=False, infra_error=True, diagnostics=[{"msg": "iverilog missing"}])
    nonce = "CHIPRL_" + secrets.token_hex(8)
    (workdir / f"{tag}_tb.sv").write_text(prepare_tb(tb, nonce, vcd, timescale_ns))
    (workdir / f"{tag}_ref.sv").write_text(ref_rtl)
    srcs = [f"{tag}_tb.sv", f"{tag}_ref.sv", str(sub_path)]
    flags = ["-g2012"]
    if gate_level:
        flags += ["-DFUNCTIONAL", "-DUNIT_DELAY=#0"]
        srcs += [str(THIRD_PARTY / "sky130/sky130_hd_primitives.v"), str(THIRD_PARTY / "sky130/sky130_hd.v")]
    rc, out, dt1 = run([iverilog, *flags, "-s", "tb", "-o", f"{tag}.vvp", *srcs], workdir,
                       f"{tag}_compile.log", timeout_s=60)
    if rc is None:
        return ToolResult(f"{tag}_compile", ok=False, infra_error=True, timeout=True)
    if rc != 0:
        diags = [{"msg": ln.strip()[:200]} for ln in out.splitlines() if "error" in ln.lower()][:20]
        return ToolResult(f"{tag}_compile", ok=False, diagnostics=diags, elapsed_s=dt1,
                          log_path=str(workdir / f"{tag}_compile.log"))
    rc, out, dt2 = run([vvp, "-n", f"{tag}.vvp"], workdir, f"{tag}.log", timeout_s=timeout_s)
    if rc is None:
        # wall-clock timeout of a DUT that hangs the simulator is behavior, not infra.
        return ToolResult(tag, ok=False, timeout=True, elapsed_s=dt1 + dt2,
                          metrics={"mismatches": None, "samples": None, "pass_frac": 0.0})
    if rc < 0:  # killed by signal (OOM / rlimit): infra
        return ToolResult(tag, ok=False, infra_error=True, elapsed_s=dt1 + dt2,
                          diagnostics=[{"msg": f"vvp killed by signal {-rc}"}])
    m = [r for ln in out.splitlines() if ln.startswith(nonce) for r in _RESULT_RE.findall(ln)]
    if not m:
        return ToolResult(tag, ok=False, elapsed_s=dt1 + dt2, diagnostics=[{"msg": "no result line"}],
                          metrics={"mismatches": None, "samples": None, "pass_frac": 0.0},
                          log_path=str(workdir / f"{tag}.log"))
    mism, samples = (int(x) for x in m[-1])
    hints = [ln.strip()[6:200] for ln in out.splitlines() if ln.startswith("Hint:")][:10]
    return ToolResult(tag, ok=(mism == 0), elapsed_s=dt1 + dt2,
                      metrics={"mismatches": mism, "samples": samples,
                               "pass_frac": (1 - mism / samples) if samples else 0.0,
                               "timed_out_in_tb": "TIMEOUT" in out},
                      diagnostics=[{"hint": h} for h in hints],
                      log_path=str(workdir / f"{tag}.log"))
