"""One task end to end: lint -> sim -> equiv -> synth -> gate-level sim (VCD) -> power -> JSON.

    python -m chiprl.pipeline tests/fixtures/verilogeval/Prob068_countbcd [--rtl my.sv] [--out run.json]

With no --rtl, the reference implementation is the submission (pipeline smoke test).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from . import cache
from .graders.rtl_gen import grade_rtl, reference_samples
from .tasks.task import Task, load_verilogeval_dir
from .tools._run import tool_versions
from .tools.power import power
from .tools.sim import simulate
from .tools.synth import synth


def _synth_blamed_on_submission(task: Task) -> bool:
    """Control for a synth timeout: if the reference synthesizes fine right now, the submission did it."""
    with tempfile.TemporaryDirectory(prefix="chiprl_ctl_") as d:
        ref = Path(d) / "dut.sv"
        ref.write_text(task.ref_rtl.replace("RefModule", task.top))
        return synth(ref, task.top, Path(d)).ok


def run_task(
    task: Task,
    sub_rtl: str,
    workdir: Path,
    power_backend: str = "auto",
    period_ns: float = 10.0,
    use_cache: bool = True,
) -> dict:
    """Grade, then measure PPA only for a design that passed every gate before it.

    `gates` records each gate in order; a later gate is absent if an earlier one failed.
    `infra_error` is True if any stage failed for a reason that is not the submission's fault;
    such a result is never cached and its numbers must not be used.
    """
    t0 = time.time()
    ck = cache.key("pipeline/v3", task.content_hash(), sub_rtl, power_backend, str(period_ns))
    if use_cache and (hit := cache.get(ck)):
        return {**hit, "cache_hit": True}
    gates: dict[str, bool] = {}
    out: dict = {
        "task_id": task.task_id,
        "family": task.family,
        "tool_versions": tool_versions(),
        "infra_error": False,
        "gates": gates,
        "ppa": None,
    }

    def done() -> dict:
        out["elapsed_s"] = round(time.time() - t0, 3)
        if use_cache and not out["infra_error"]:
            cache.put(ck, out)
        return out

    g = grade_rtl(sub_rtl, task, workdir)
    out["grade"] = {"reward": g.reward, "stage": g.stage, **g.detail}
    out["infra_error"] = g.reward is None
    gates["grade"] = g.stage == "ok" and g.reward is not None and g.reward >= 1.0
    if not gates["grade"]:
        return done()

    sub = workdir / "dut.sv"
    s = synth(sub, task.top, workdir)
    out["synth"] = s.to_dict()
    gates["synth"] = s.ok
    if not s.ok:
        out["infra_error"] = s.infra_error or (s.timeout and not _synth_blamed_on_submission(task))
        return done()

    # Gate-level sim on the same hidden stimulus: (1) post-synth functional check,
    # (2) full-netlist switching activity for power.
    gl = simulate(
        Path(s.metrics["netlist"]),
        task.ref_rtl,
        task.testbench,
        workdir,
        vcd="gl.vcd",
        gate_level=True,
        timescale_ns=True,
        tag="glsim",
    )
    out["gate_sim"] = gl.to_dict()
    if gl.infra_error:
        out["infra_error"] = True
        return done()
    exp = reference_samples(task)
    gates["glsim"] = bool(gl.ok and exp is not None and gl.metrics.get("samples") == exp)

    # Power is only meaningful for a netlist that behaves like the RTL: no gate-level pass, no number.
    p = None
    if gates["glsim"]:
        p = power(
            Path(s.metrics["netlist"]),
            Path(s.metrics["netlist_json"]),
            workdir / "gl.vcd",
            s.metrics["liberty"],
            task.top,
            workdir,
            task.clock,
            period_ns,
            backend=power_backend,
        )
        out["power"] = p.to_dict()
        gates["power"] = p.ok
        out["infra_error"] = p.infra_error
    out["ppa"] = {
        "area_um2": s.metrics["area_um2"],
        "cells": s.metrics["cells"],
        "flops": s.metrics["flops"],
        "logic_depth": s.metrics["logic_depth"],
        "power_w": p.metrics.get("total_w") if p and p.ok else None,
        "power_backend": p.metrics.get("backend") if p and p.ok else None,
        "post_synth_sim_pass": gates["glsim"],
        "clock_period_ns": period_ns,
        "valid": all(gates.values()) and "power" in gates,
    }
    return done()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--rtl", type=Path, help="submission; default = reference RTL")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--power-backend", default="auto", choices=["auto", "proxy", "opensta"])
    ap.add_argument("--period-ns", type=float, default=10.0)
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--pnr", action="store_true", help="also run the LibreLane slow path (minutes)")
    a = ap.parse_args(argv)
    task = load_verilogeval_dir(a.task_dir)
    rtl = a.rtl.read_text() if a.rtl else task.ref_rtl.replace("RefModule", task.top)
    wd = Path(tempfile.mkdtemp(prefix=f"chiprl_{task.task_id}_"))
    res = run_task(task, rtl, wd, a.power_backend, a.period_ns, use_cache=not a.no_cache)
    if a.pnr and (res.get("ppa") or {}).get("area_um2") is not None:
        from .tools.pnr import pnr

        res["pnr"] = pnr(task, rtl, wd, period_ns=a.period_ns).to_dict()
    res["workdir"] = str(wd) if a.keep else None
    js = json.dumps(res, indent=2, default=str)
    if a.out:
        a.out.write_text(js)
    print(js)
    if not a.keep:
        import shutil

        shutil.rmtree(wd, ignore_errors=True)
    return 2 if res.get("infra_error") else 0


if __name__ == "__main__":
    sys.exit(main())
