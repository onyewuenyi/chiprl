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
from .graders.rtl_gen import grade_rtl
from .tasks.task import Task, load_verilogeval_dir
from .tools._run import tool_versions
from .tools.power import power
from .tools.sim import simulate
from .tools.synth import synth


def run_task(
    task: Task,
    sub_rtl: str,
    workdir: Path,
    power_backend: str = "auto",
    period_ns: float = 10.0,
    use_cache: bool = True,
) -> dict:
    t0 = time.time()
    ck = cache.key("pipeline/v2", task.content_hash(), sub_rtl, power_backend, str(period_ns))
    if use_cache and (hit := cache.get(ck)):
        return {**hit, "cache_hit": True}
    out: dict = {"task_id": task.task_id, "family": task.family, "tool_versions": tool_versions()}
    g = grade_rtl(sub_rtl, task, workdir)
    out["grade"] = {"reward": g.reward, "stage": g.stage, **g.detail}
    if g.stage != "ok" or not g.reward or g.reward < 1.0:
        out["ppa"] = None
        out["elapsed_s"] = round(time.time() - t0, 3)
        return out
    sub = workdir / "dut.sv"
    s = synth(sub, task.top, workdir)
    out["synth"] = s.to_dict()
    if not s.ok:
        out["ppa"] = None
        return out
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
    p = None
    if gl.metrics.get("samples"):
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
    out["ppa"] = {
        "area_um2": s.metrics["area_um2"],
        "cells": s.metrics["cells"],
        "flops": s.metrics["flops"],
        "logic_depth": s.metrics["logic_depth"],
        "power_w": p.metrics.get("total_w") if p and p.ok else None,
        "power_backend": p.metrics.get("backend") if p and p.ok else None,
        "post_synth_sim_pass": gl.ok,
        "clock_period_ns": period_ns,
    }
    out["elapsed_s"] = round(time.time() - t0, 3)
    if use_cache:
        cache.put(ck, out)
    return out


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
    return 0 if res["grade"]["reward"] is not None else 2


if __name__ == "__main__":
    sys.exit(main())
