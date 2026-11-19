"""Differential simulation: the candidate against a golden design on random inputs, cycle by cycle.

The hidden testbench checks the stimulus its author wrote. This checks thousands of random cycles
that nobody chose, with the same comparison rule as VerilogEval (an x on a golden output matches
anything; a defined golden output must match exactly). It is cheap (one Icarus run) and finds
behavior the testbench never exercises. Two uses:

  * refutation: a mismatch is a concrete input trace on which the designs differ, so `equiv()`
    reports `refuted` without a solver. No mismatch proves nothing: it is evidence, not a proof.
  * behavior signature: a hash of the candidate's outputs on a fixed random stimulus, used to group
    behaviorally identical candidates (`graders/cluster.py`).

Stimulus per cycle: every input gets fresh random bits, except resets, which are held for the first
`reset_cycles` cycles and then asserted with probability `reset_p` (uniform 50% resets would keep
the state machine near reset forever). Inputs change, outputs are compared 1 ns later, then the
clock rises: combinational and Mealy outputs are checked against the new inputs, registered outputs
one cycle after the edge that produced them.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path

from ..tasks.interface import Port, clock_of, ports, resets_of
from ._run import ToolResult, run, which

GOLD = "chiprl_gold"
_DIFF_RE = re.compile(r"DIFF (\d+) (-?\d+) (\d+)")


def _rename_module(rtl: str, old: str, new: str) -> str:
    return re.sub(rf"\bmodule\s+{re.escape(old)}\b", f"module {new}", rtl)


def gold_as_top(gold_rtl: str, top: str) -> str:
    """The golden design as module `top` (a VerilogEval reference is module RefModule)."""
    return re.sub(r"\bRefModule\b", top, gold_rtl)


def _rand_expr(width: int) -> str:
    words = (width + 31) // 32
    expr = "$random(seed)" if words == 1 else "{" + ", ".join(["$random(seed)"] * words) + "}"
    return expr


def testbench(ps: tuple[Port, ...], top: str, nonce: str, cycles: int, seeds: tuple[int, ...],
              reset_cycles: int, reset_p: float) -> str:  # fmt: skip
    clk = clock_of(ps)
    resets = set(resets_of(ps))
    ins = [(n, w) for d, n, w in ps if d == "input" and n != clk]
    outs = [(n, w) for d, n, w in ps if d == "output"]
    decl = [f"  reg [{w - 1}:0] i_{n};" for n, w in ins]
    decl += [f"  wire [{w - 1}:0] g_{n}, c_{n};" for n, w in outs]
    if clk:
        decl.append(f"  reg i_{clk} = 1'b0;")

    def conns(prefix: str) -> str:
        cs = [f".{n}(i_{n})" for d, n, _ in ps if d == "input"]
        cs += [f".{n}({prefix}_{n})" for d, n, _ in ps if d == "output"]
        return ", ".join(cs)

    thresh = int(reset_p * 1024)
    drive = []
    for n, w in ins:
        if n in resets:
            active_low = n.lower().endswith("n")
            on, off = ("1'b0", "1'b1") if active_low else ("1'b1", "1'b0")
            drive.append(
                f"      i_{n} = (cyc < {reset_cycles} || ($random(seed) & 1023) < {thresh}) ? {on} : {off};"
            )
        else:
            drive.append(f"      i_{n} = {_rand_expr(w)};")
    cmp = " || ".join(f"!(g_{n} === (g_{n} ^ c_{n} ^ g_{n}))" for n, _ in outs) or "1'b0"
    trace = " ".join("%h %h" for _ in outs)
    trace_args = ", ".join(f"g_{n}, c_{n}" for n, _ in outs)
    tick = [f"      #4 i_{clk} = 1'b1;", f"      #5 i_{clk} = 1'b0;"] if clk else ["      #9;"]
    seed_init = "\n".join(
        f"    seed = {s};\n    for (k = 0; k < {cycles}; k = k + 1) begin\n      cyc = cyc + 1;\n"
        + "\n".join(drive)
        + "\n      #1;\n"
        + f"      if ({cmp}) begin mism = mism + 1; if (first < 0) first = cyc; end\n"
        + (f'      $display("T {trace}", {trace_args});\n' if outs else "")
        + "\n".join(tick)
        + "\n    end"
        for s in seeds
    )
    return f"""`timescale 1ns/1ps
module chiprl_diff_tb;
{chr(10).join(decl)}
  integer seed, cyc, k, mism, first;
  {GOLD} g({conns("g")});
  {top} c({conns("c")});
  initial begin
    mism = 0; first = -1; cyc = -1;
{seed_init}
    $display("{nonce} DIFF %0d %0d %0d", mism, first, cyc + 1);
    $finish;
  end
endmodule
"""


def signature_of(trace_lines: list[str]) -> str:
    """Hash of candidate outputs per cycle, with nibbles where the golden output is x/z masked."""
    h = hashlib.sha256()
    for ln in trace_lines:
        vals = ln.split()[1:]
        for g, c in zip(vals[0::2], vals[1::2], strict=True):
            masked = "".join(
                "-" if gc in "xXzZ" else cc for gc, cc in zip(g.rjust(len(c), "0"), c, strict=False)
            )
            h.update(masked.encode() + b" ")
        h.update(b"\n")
    return h.hexdigest()[:16]


def diffsim(
    sub_path: Path,
    gold_rtl: str,
    top: str,
    workdir: Path,
    *,
    cycles: int = 1000,
    seeds: tuple[int, ...] = (1, 2, 3),
    reset_cycles: int = 2,
    reset_p: float = 0.05,
    timeout_s: int = 60,
    tag: str = "diff",
) -> ToolResult:
    """metrics: verdict (refuted | clean | unknown), mismatches, first_mismatch_cycle, cycles, signature."""
    iverilog, vvp = which("iverilog"), which("vvp")
    if not (iverilog and vvp):
        return ToolResult(tag, ok=False, infra_error=True, diagnostics=[{"msg": "iverilog missing"}])
    gold_top = gold_as_top(gold_rtl, top)
    gp, cp = ports(gold_top, top), ports(sub_path.read_text(), top)
    base = {"verdict": "unknown", "cycles": 0}
    if gp is None:
        return ToolResult(
            tag, ok=False, metrics=base, diagnostics=[{"msg": "could not read the golden port list"}]
        )
    # A candidate Yosys cannot elaborate (e.g. an out-of-range part-select, which Icarus accepts) is
    # driven through the golden interface; if its ports really differ, the compile below fails.
    if cp is not None and gp != cp:
        return ToolResult(
            tag,
            ok=True,
            metrics={**base, "verdict": "refuted", "reason": "interface differs"},
            diagnostics=[{"msg": f"ports differ: gold {gp} vs candidate {cp}"}],
        )
    nonce = "CHIPRL_" + secrets.token_hex(8)
    (workdir / f"{tag}_gold.sv").write_text(_rename_module(gold_top, top, GOLD))
    (workdir / f"{tag}_tb.sv").write_text(testbench(gp, top, nonce, cycles, seeds, reset_cycles, reset_p))
    srcs = [f"{tag}_tb.sv", f"{tag}_gold.sv", str(sub_path)]
    rc, out, dt1 = run([iverilog, "-g2012", "-s", "chiprl_diff_tb", "-o", f"{tag}.vvp", *srcs], workdir,
                       f"{tag}_compile.log", timeout_s=60)  # fmt: skip
    if rc is None or rc < 0:
        return ToolResult(tag, ok=False, infra_error=True, timeout=rc is None, metrics=base, elapsed_s=dt1)
    if rc != 0:
        errs = [ln.strip()[:200] for ln in out.splitlines() if "error" in ln.lower()][:5]
        return ToolResult(tag, ok=False, metrics=base, diagnostics=[{"msg": e} for e in errs], elapsed_s=dt1)
    rc, out, dt2 = run([vvp, "-n", f"{tag}.vvp"], workdir, f"{tag}.log", timeout_s=timeout_s)
    if rc is None:
        return ToolResult(tag, ok=False, timeout=True, metrics=base, elapsed_s=dt1 + dt2)
    m = next((_DIFF_RE.search(ln) for ln in out.splitlines() if ln.startswith(nonce)), None)
    if rc != 0 or m is None:
        return ToolResult(tag, ok=False, infra_error=rc < 0, metrics=base, elapsed_s=dt1 + dt2,
                          diagnostics=[{"msg": "no result line"}])  # fmt: skip
    mism, first, n = (int(x) for x in m.groups())
    trace = [ln for ln in out.splitlines() if ln.startswith("T ")]
    return ToolResult(
        tag,
        ok=True,
        elapsed_s=dt1 + dt2,
        metrics={
            "verdict": "refuted" if mism else "clean",
            "mismatches": mism,
            "first_mismatch_cycle": first if mism else None,
            "cycles": n,
            "signature": signature_of(trace),
        },
        log_path=str(workdir / f"{tag}.log"),
    )
