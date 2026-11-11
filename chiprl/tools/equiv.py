"""Equivalence check of a submission against the hidden reference.

Backends:
  * eqy (YosysHQ, preferred): partitions both designs on matched names and proves each partition
    with the `sat` strategy (bounded + temporal induction).
  * yosys equiv_* (fallback when eqy is absent, e.g. apt Yosys): equiv_make / equiv_simple -seq /
    equiv_induct.

'proven' gates only the +0.1 bonus. Unproven is not a failure, because induction is incomplete for
designs whose state encodings differ from the reference.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._run import ToolResult, run, which, yosys_read_sv

_STATUS_RE = re.compile(
    r"Found (\d+) \$equiv cells.*?Of those cells (\d+) are proven and (\d+) are unproven", re.S
)


def _write_designs(sub_path: Path, ref_rtl: str, top: str, workdir: Path) -> tuple[Path, Path]:
    gold = workdir / "equiv_gold.sv"
    gate = workdir / "equiv_gate.sv"
    gold.write_text(re.sub(r"\bRefModule\b", top, ref_rtl))
    gate.write_text(sub_path.read_text())
    return gold, gate


def equiv_eqy(sub_path: Path, ref_rtl: str, top: str, workdir: Path, depth: int = 10) -> ToolResult:
    exe = which("eqy")
    if not exe:
        return ToolResult("equiv", ok=False, infra_error=True, diagnostics=[{"msg": "eqy missing"}])
    gold, gate = _write_designs(sub_path, ref_rtl, top, workdir)
    cfg = f"""[gold]
{yosys_read_sv(gold, top)}
prep -top {top}
async2sync

[gate]
{yosys_read_sv(gate, top)}
prep -top {top}
async2sync

[strategy sat]
use sat
depth {depth}
"""
    (workdir / "equiv.eqy").write_text(cfg)
    rc, out, dt = run([exe, "-f", "equiv.eqy", "-d", "eqy_run"], workdir, "eqy.log", timeout_s=180)
    if rc is None:
        return ToolResult(
            "equiv", ok=False, timeout=True, elapsed_s=dt, metrics={"proven": False, "backend": "eqy"}
        )
    tail = "\n".join(out.strip().splitlines()[-5:])
    proven = rc == 0 and "PASS" in tail and "FAIL" not in tail
    return ToolResult(
        "equiv",
        ok=True,
        elapsed_s=dt,
        metrics={"proven": proven, "backend": "eqy", "rc": rc},
        log_path=str(workdir / "eqy.log"),
    )


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
    rc, _, dt = run([exe, "-q", "-s", "equiv.ys"], workdir, "equiv.log", timeout_s=120)
    base = {"backend": "yosys_equiv"}
    if rc is None:
        return ToolResult("equiv", ok=False, timeout=True, elapsed_s=dt, metrics={"proven": False, **base})
    status = (workdir / "equiv_status.txt").read_text() if (workdir / "equiv_status.txt").exists() else ""
    if "Equivalence successfully proven" in status:
        return ToolResult("equiv", ok=True, elapsed_s=dt, metrics={"proven": True, **base})
    if m := _STATUS_RE.search(status):
        tot, _prov, unprov = map(int, m.groups())
        return ToolResult(
            "equiv",
            ok=True,
            elapsed_s=dt,
            metrics={"proven": unprov == 0, "cells": tot, "unproven": unprov, **base},
        )
    return ToolResult(
        "equiv",
        ok=False,
        elapsed_s=dt,
        metrics={"proven": False, **base},
        log_path=str(workdir / "equiv.log"),
    )


def equiv(sub_path: Path, ref_rtl: str, top: str, workdir: Path) -> ToolResult:
    if which("eqy"):
        return equiv_eqy(sub_path, ref_rtl, top, workdir)
    return equiv_yosys(sub_path, ref_rtl, top, workdir)
