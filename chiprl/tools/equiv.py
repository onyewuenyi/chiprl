"""Equivalence check of a submission against the hidden reference. Fails closed.

Three checks, combined by `equiv()` into one verdict:

  * bounded check (`equiv_bmc`, Yosys only, runs everywhere): a miter of the two whole designs from
    power-up, every input free, flops starting at x, for `BMC_DEPTH` cycles. A counterexample here is
    a real trace from primary inputs, so it is the only source of `refuted`.
  * eqy (preferred prover): partitions on matched names and proves each partition with SBY
    (x-propagation + k-induction over SMT).
  * yosys equiv_* (prover when eqy is unusable, e.g. apt Yosys): equiv_make / equiv_simple -seq /
    equiv_induct.

Verdict:
  proven   the prover proved it AND the bounded check found no counterexample AND the bounded
           problem was satisfiable (not vacuous)
  refuted  the bounded check found a counterexample
  unknown  anything else: unproven, timeout, tool error, a prover that fails its self-test

Why the self-test exists: eqy's `sat` strategy, on Yosys 0.70 / eqy 0.71, combines
`-set-init-undef` with `-set-def-formal` on a `$anyinit` flop. Those constraints contradict each
other, no trace exists, and every sequential partition is "proven" vacuously (docs/failure_log.md
#1). `eqy_sound()` runs the configured strategy on a known-inequivalent pair once per process; a
prover that proves it is ignored.

For RTL generation `proven` only gates the +0.1 bonus: induction is incomplete for designs whose
state encoding differs from the reference, so unproven is not a failure.
"""

from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path

from ._run import ToolResult, run, which, yosys_read_sv

BMC_DEPTH = 12
_SAT_TIMEOUT_S = 30
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


def equiv_bmc(sub_path: Path, ref_rtl: str, top: str, workdir: Path, depth: int = BMC_DEPTH) -> ToolResult:
    """Bounded check of the whole design from power-up. metrics.verdict: clean | refuted | unknown.

    Semantics follow the VerilogEval testbench: an x on a reference output matches anything, a
    defined reference output must be matched exactly. Both designs start with every flop at x.
    The same constraints are solved once without the proof goal ("live"): if they admit no trace
    the proof would be vacuous, and the verdict is `unknown`.
    """
    exe = which("yosys")
    if not exe:
        return ToolResult("equiv_bmc", ok=False, infra_error=True)
    (workdir / "bmc_gold.sv").write_text(_gold_text(ref_rtl, top))
    (workdir / "bmc_gate.sv").write_text(sub_path.read_text())
    prep = f"hierarchy -top {top}\nproc; flatten; memory_map; opt_clean; async2sync"
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
        elapsed_s=dt,
        metrics={"verdict": verdict, "live": live, **base},
        log_path=str(workdir / "bmc.log"),
    )


def equiv(sub_path: Path, ref_rtl: str, top: str, workdir: Path) -> ToolResult:
    """Combined verdict (see module docstring). metrics: proven, verdict, backend, bmc, prover."""
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
            metrics={"proven": False, "verdict": "refuted", "backend": "yosys_bmc", "bmc": bmc.metrics},
            log_path=bmc.log_path,
        )
    prover = equiv_eqy if eqy_available() and eqy_sound() else equiv_yosys
    pr = prover(sub_path, ref_rtl, top, workdir)
    proven = bool(pr.metrics.get("proven")) and bmc.metrics["verdict"] == "clean"
    return ToolResult(
        "equiv",
        ok=pr.ok and bmc.ok,
        infra_error=pr.infra_error,
        timeout=pr.timeout or bmc.timeout,
        elapsed_s=bmc.elapsed_s + pr.elapsed_s,
        metrics={
            "proven": proven,
            "verdict": "proven" if proven else "unknown",
            "backend": pr.metrics.get("backend"),
            "bmc": bmc.metrics,
            "prover": pr.metrics,
        },
        log_path=pr.log_path,
    )
