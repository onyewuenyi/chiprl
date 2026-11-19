"""Equivalence check of a submission against the hidden reference. Fails closed.

`equiv()` combines three checks into one verdict:

  * differential simulation (`tools/diffsim.py`): thousands of random cycles against the reference
    under RTL semantics. Cheapest, runs first, refutes only.
  * bounded check (`equiv_bmc`): a miter of the two whole designs from power-up, every input free,
    every flop starting at x (ASIC semantics: RTL initializers stripped). A counterexample is a
    real trace from primary inputs. Vacuous when the reference's outputs depend on an initializer
    (they stay x, so every comparison passes); differential simulation covers that case. Depth comes from the reference's flop bits (`bmc_depth`), not from a
    clock, so the verdict is the same on a loaded machine and can be cached.
  * inductive prover: Yosys `equiv_*` by default (equiv_make / equiv_simple -seq / equiv_induct).
    eqy (SBY k-induction) is opt-in with prover="eqy".

Verdict:
  proven   the prover proved it AND the bounded check found no counterexample AND the bounded
           problem was satisfiable (not vacuous)
  refuted  differential simulation or the bounded check found a counterexample
  unknown  anything else: unproven, timeout, tool error, a prover that fails its self-test

Why Yosys `equiv_*` and not eqy by default (measured 2026-11-18, 155 VerilogEval references that
pass simulation, M5 Mac): `equiv_*` proved 147 at p50 0.04 s (22 s in total); eqy proved 139, every
one also proven by `equiv_*`, at p50 1.1 s and up to 88 s (503 s in total).

Why eqy has a self-test: its `sat` strategy, on Yosys 0.70 / eqy 0.71, combines `-set-init-undef`
with `-set-def-formal` on a `$anyinit` flop. Those constraints contradict each other, no trace exists,
and every sequential partition is "proven" vacuously (docs/failure_log.md #1). eqy now uses its
`sby` strategy, and `eqy_sound()` runs it on a known-inequivalent pair once per process; a prover
that proves that pair is not used.

For RTL generation `proven` only gates the +0.1 bonus: induction is incomplete for designs whose
state encoding differs from the reference, so unproven is not a failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from ._run import ToolResult, run, which, yosys_read_sv
from .diffsim import diffsim

BMC_DEPTH = 12  # for small designs; see bmc_depth()
_SAT_TIMEOUT_S = 20  # per solve; hitting it marks the result as a timeout (not cached)
_SMT_SOLVERS = ("bitwuzla", "yices", "boolector", "z3")

_STATUS_RE = re.compile(
    r"Found (\d+) \$equiv cells.*?Of those cells (\d+) are proven and (\d+) are unproven", re.S
)

# A toggle flop against a hold flop: inequivalent one cycle after reset. Any sound prover refutes it.
_CANARY_GOLD = """module {top} (input clk, input rst, output logic q);
  always @(posedge clk) q <= rst ? 1'b0 : ~q;
endmodule
"""
_CANARY_GATE = """module {top} (input clk, input rst, output logic q);
  always @(posedge clk) q <= rst ? 1'b0 : q;
endmodule
"""


def _gold_text(ref_rtl: str, top: str) -> str:
    return re.sub(r"\bRefModule\b", top, ref_rtl)


def _smt_solver() -> str | None:
    for s in _SMT_SOLVERS:
        if which("yices-smt2" if s == "yices" else s):
            return s
    return None


def eqy_available() -> bool:
    return bool(which("eqy") and which("sby") and _smt_solver())


def _eqy_prove(gold_text: str, gate_text: str, top: str, workdir: Path, depth: int) -> tuple[str, dict]:
    """Run eqy with the SBY strategy. Returns (proven | unproven | timeout, info)."""
    gold = workdir / "equiv_gold.sv"
    gate = workdir / "equiv_gate.sv"
    gold.write_text(gold_text)
    gate.write_text(gate_text)
    cfg = f"""[gold]
{yosys_read_sv(gold, top)}
prep -top {top}
async2sync

[gate]
{yosys_read_sv(gate, top)}
prep -top {top}
async2sync

[strategy sby]
use sby
depth {depth}
engine smtbmc {_smt_solver()}
"""
    (workdir / "equiv.eqy").write_text(cfg)
    exe = which("eqy")
    assert exe is not None
    rc, out, dt = run([exe, "-f", "equiv.eqy", "-d", "eqy_run"], workdir, "eqy.log", timeout_s=180)
    if rc is None:
        return "timeout", {"elapsed_s": dt}
    tail = "\n".join(out.strip().splitlines()[-5:])
    statuses = sorted(
        p.read_text().strip() for p in (workdir / "eqy_run" / "strategies").glob("*/*/status") if p.is_file()
    )
    proven = rc == 0 and "PASS" in tail and "FAIL" not in tail
    return ("proven" if proven else "unproven"), {"rc": rc, "elapsed_s": dt, "partitions": statuses}


@lru_cache(maxsize=1)
def eqy_sound() -> bool:
    """False if eqy, as configured here, proves a known-inequivalent pair (or cannot run at all)."""
    if not eqy_available():
        return False
    top = "canary"
    with tempfile.TemporaryDirectory(prefix="chiprl_eqy_canary_") as d:
        status, _ = _eqy_prove(_CANARY_GOLD.format(top=top), _CANARY_GATE.format(top=top), top, Path(d), 5)
    return status == "unproven"


def equiv_eqy(sub_path: Path, ref_rtl: str, top: str, workdir: Path, depth: int = 10) -> ToolResult:
    if not eqy_available():
        missing = [n for n in ("eqy", "sby") if not which(n)] or ["SMT solver"]
        return ToolResult(
            "equiv", ok=False, infra_error=True, diagnostics=[{"msg": f"{', '.join(missing)} missing"}]
        )
    base = {"backend": "eqy"}
    if not eqy_sound():
        return ToolResult(
            "equiv",
            ok=False,
            metrics={"proven": False, "verdict": "unknown", **base},
            diagnostics=[
                {"msg": "eqy self-test failed: it proves a known-inequivalent pair; verdict ignored"}
            ],
        )
    status, info = _eqy_prove(_gold_text(ref_rtl, top), sub_path.read_text(), top, workdir, depth)
    dt = info.pop("elapsed_s", 0.0)
    if status == "timeout":
        return ToolResult(
            "equiv",
            ok=False,
            timeout=True,
            elapsed_s=dt,
            metrics={"proven": False, "verdict": "unknown", **base},
        )
    proven = status == "proven"
    return ToolResult(
        "equiv",
        ok=True,
        elapsed_s=dt,
        metrics={"proven": proven, "verdict": "proven" if proven else "unknown", **base, **info},
        log_path=str(workdir / "eqy.log"),
    )


def parse_equiv_status(status: str) -> tuple[bool, int, int]:
    """(proven, total $equiv cells, unproven) from `equiv_status` output.

    Zero cells is never a proof: with nothing to compare, "0 unproven" says nothing about the designs.
    """
    m = _STATUS_RE.search(status)
    if not m:
        return False, 0, 0
    tot, _prov, unprov = map(int, m.groups())
    return (tot > 0 and unprov == 0), tot, unprov


def equiv_yosys(sub_path: Path, ref_rtl: str, top: str, workdir: Path, seq: int = 8) -> ToolResult:
    exe = which("yosys")
    if not exe:
        return ToolResult("equiv", ok=False, infra_error=True)
    (workdir / "equiv_gold.sv").write_text(re.sub(r"\bRefModule\b", "gold", ref_rtl))
    (workdir / "equiv_gate.sv").write_text(re.sub(rf"\b{top}\b", "gate", sub_path.read_text()))
    script = f"""
{yosys_read_sv("equiv_gold.sv")}
{yosys_read_sv("equiv_gate.sv").replace("plugin -i slang", "")}
proc
attrmap -remove init
opt_clean
async2sync
equiv_make gold gate equiv
hierarchy -top equiv
equiv_simple -seq {seq}
equiv_induct -seq {seq}
tee -o equiv_status.txt equiv_status
"""
    (workdir / "equiv.ys").write_text(script)
    status_file = workdir / "equiv_status.txt"
    status_file.unlink(missing_ok=True)
    rc, _, dt = run([exe, "-q", "-s", "equiv.ys"], workdir, "equiv.log", timeout_s=120)
    base = {"backend": "yosys_equiv"}
    if rc is None:
        return ToolResult(
            "equiv",
            ok=False,
            timeout=True,
            elapsed_s=dt,
            metrics={"proven": False, "verdict": "unknown", **base},
        )
    status = status_file.read_text() if status_file.exists() else ""
    proven, tot, unprov = parse_equiv_status(status)
    if tot == 0:
        return ToolResult(
            "equiv",
            ok=False,
            elapsed_s=dt,
            metrics={"proven": False, "verdict": "unknown", **base},
            log_path=str(workdir / "equiv.log"),
        )
    return ToolResult(
        "equiv",
        ok=True,
        elapsed_s=dt,
        metrics={
            "proven": proven,
            "verdict": "proven" if proven else "unknown",
            "cells": tot,
            "unproven": unprov,
            **base,
        },
    )


def bmc_depth(state_bits: int | None) -> int:
    """Bounded-check depth from the reference's flop bits.

    Combinational: 1 cycle is a complete check. Otherwise cost grows steeply with depth x state
    (measured: conwaylife, 256 bits, 0.7 s at depth 1, 30 s at 4, solver timeout at 8), so wide
    designs get a shallow check. Unknown size: the safe small depth.
    """
    if state_bits is None:
        return 4
    if state_bits == 0:
        return 1
    if state_bits <= 64:
        return BMC_DEPTH
    if state_bits <= 128:
        return 6
    return 2


_STATE_BITS: dict[str, int | None] = {}


def state_bits(ref_rtl: str, top: str) -> int | None:
    """Flop bits in the reference after proc/flatten/memory_map. Cached per reference text."""
    k = hashlib.sha256(ref_rtl.encode()).hexdigest()
    if k in _STATE_BITS:
        return _STATE_BITS[k]
    exe = which("yosys")
    n: int | None = None
    if exe:
        with tempfile.TemporaryDirectory(prefix="chiprl_ffbits_") as d:
            wd = Path(d)
            (wd / "gold.sv").write_text(_gold_text(ref_rtl, top))
            (wd / "ff.ys").write_text(
                f"{yosys_read_sv('gold.sv', top)}\nhierarchy -top {top}\n"
                "proc; flatten; memory_map; opt_clean\nwrite_json ff.json\n"
            )
            rc, _, _ = run([exe, "-q", "-s", "ff.ys"], wd, "ff.log", timeout_s=60)
            if rc == 0 and (wd / "ff.json").exists():
                mods = json.loads((wd / "ff.json").read_text())["modules"]
                n = sum(
                    len(c["connections"].get("Q", []))
                    for m in mods.values()
                    for c in m.get("cells", {}).values()
                    if re.search(r"ff|dlatch", c["type"])
                )
    _STATE_BITS[k] = n
    return n


def equiv_bmc(sub_path: Path, ref_rtl: str, top: str, workdir: Path, depth: int | None = None) -> ToolResult:
    """Bounded check of the whole design from power-up. metrics.verdict: clean | refuted | unknown.

    Semantics follow the VerilogEval testbench: an x on a reference output matches anything, a
    defined reference output must be matched exactly. Both designs start with every flop at x.
    The same constraints are solved once without the proof goal ("live"): if they admit no trace
    the proof would be vacuous, and the verdict is `unknown`.
    """
    exe = which("yosys")
    if not exe:
        return ToolResult("equiv_bmc", ok=False, infra_error=True)
    if depth is None:
        depth = bmc_depth(state_bits(ref_rtl, top))
    (workdir / "bmc_gold.sv").write_text(_gold_text(ref_rtl, top))
    (workdir / "bmc_gate.sv").write_text(sub_path.read_text())
    # attrmap -remove init: ASIC semantics. A standard-cell flop has no power-up value, so an RTL
    # initializer must not count (docs/hack_log.md #6). Flops start at x via -set-init-undef.
    prep = f"hierarchy -top {top}\nproc; attrmap -remove init; flatten; memory_map; opt_clean; async2sync"
    constraints = f"-seq {depth} -enable_undef -set-init-undef -set-def-inputs -timeout {_SAT_TIMEOUT_S}"
    script = f"""
{yosys_read_sv("bmc_gold.sv", top)}
{prep}
design -stash gold
{yosys_read_sv("bmc_gate.sv", top).replace("plugin -i slang", "")}
{prep}
design -stash gate
design -copy-from gold -as gold {top}
design -copy-from gate -as gate {top}
miter -equiv -flatten -make_assert -ignore_gold_x gold gate miter
hierarchy -top miter
tee -o bmc_live.txt sat {constraints} miter
tee -o bmc_prove.txt sat {constraints} -prove-asserts -show-inputs -show-outputs miter
"""
    (workdir / "bmc.ys").write_text(script)
    for name in ("bmc_live.txt", "bmc_prove.txt"):
        (workdir / name).unlink(missing_ok=True)
    rc, _, dt = run([exe, "-q", "-s", "bmc.ys"], workdir, "bmc.log", timeout_s=120)
    base = {"backend": "yosys_bmc", "depth": depth}
    if rc is None:
        return ToolResult(
            "equiv_bmc", ok=False, timeout=True, elapsed_s=dt, metrics={"verdict": "unknown", **base}
        )

    def read(name: str) -> str:
        p = workdir / name
        return p.read_text() if p.exists() else ""

    live_txt, prove_txt = read("bmc_live.txt"), read("bmc_prove.txt")
    solver_timeout = "TIMEOUT" in live_txt or "TIMEOUT" in prove_txt
    live = "SAT solving finished - model found" in live_txt
    if "model found: FAIL!" in prove_txt:
        verdict = "refuted"
    elif live and "no model found: SUCCESS!" in prove_txt:
        verdict = "clean"
    else:
        verdict = "unknown"  # script error, solver timeout, or a vacuous problem
    return ToolResult(
        "equiv_bmc",
        ok=verdict != "unknown",
        timeout=solver_timeout and verdict == "unknown",
        elapsed_s=dt,
        metrics={"verdict": verdict, "live": live, **base},
        log_path=str(workdir / "bmc.log"),
    )


def equiv(sub_path: Path, ref_rtl: str, top: str, workdir: Path, prover: str = "yosys") -> ToolResult:
    """Combined verdict (see module docstring). metrics: proven, verdict, backend, diff, bmc, prover."""
    # Cheapest refuter first: random-stimulus differential simulation under RTL semantics. It also
    # covers a hole in the bounded check: a reference whose outputs depend on an RTL initializer is
    # all-x once initializers are stripped, so every bounded comparison passes vacuously.
    diff = diffsim(sub_path, ref_rtl, top, workdir)
    if diff.metrics.get("verdict") == "refuted":
        return ToolResult(
            "equiv",
            ok=True,
            elapsed_s=diff.elapsed_s,
            metrics={"proven": False, "verdict": "refuted", "backend": "diffsim", "diff": diff.metrics},
            log_path=diff.log_path,
        )
    bmc = equiv_bmc(sub_path, ref_rtl, top, workdir)
    if bmc.infra_error:
        return ToolResult(
            "equiv", ok=False, infra_error=True, metrics={"proven": False, "verdict": "unknown"}
        )
    if bmc.metrics["verdict"] == "refuted":
        return ToolResult(
            "equiv",
            ok=True,
            elapsed_s=bmc.elapsed_s,
            metrics={
                "proven": False,
                "verdict": "refuted",
                "backend": "yosys_bmc",
                "diff": diff.metrics,
                "bmc": bmc.metrics,
            },
            log_path=bmc.log_path,
        )
    use_eqy = prover == "eqy" and eqy_available() and eqy_sound()
    pr = (equiv_eqy if use_eqy else equiv_yosys)(sub_path, ref_rtl, top, workdir)
    proven = bool(pr.metrics.get("proven")) and bmc.metrics["verdict"] == "clean"
    return ToolResult(
        "equiv",
        ok=pr.ok and bmc.ok,
        infra_error=pr.infra_error,
        timeout=pr.timeout or bmc.timeout,
        elapsed_s=diff.elapsed_s + bmc.elapsed_s + pr.elapsed_s,
        metrics={
            "proven": proven,
            "verdict": "proven" if proven else "unknown",
            "backend": pr.metrics.get("backend"),
            "diff": diff.metrics,
            "bmc": bmc.metrics,
            "prover": pr.metrics,
        },
        log_path=pr.log_path,
    )
