"""RTL-generation grader: a sequence of gates. Infra errors return reward None (excluded), never 0."""
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from ..tasks.task import Task
from ..tools.equiv import equiv
from ..tools.lint import lint
from ..tools.sim import simulate


@dataclasses.dataclass
class R:
    reward: float | None
    stage: str                       # lint | compile | sim | samples | ok | infra
    detail: dict = dataclasses.field(default_factory=dict)


_REF_SAMPLES: dict[str, int] = {}


def reference_samples(task: Task) -> int | None:
    """Sample count of the reference-vs-reference run. A DUT that ends sim early is caught here."""
    key = task.content_hash()
    if key not in _REF_SAMPLES:
        with tempfile.TemporaryDirectory(prefix="chiprl_ref_") as d:
            wd = Path(d)
            p = wd / "dut.sv"
            p.write_text(task.ref_rtl.replace("RefModule", task.top))
            r = simulate(p, task.ref_rtl, task.testbench, wd)
            if not r.ok:
                return None
            _REF_SAMPLES[key] = r.metrics["samples"]
    return _REF_SAMPLES[key]


def grade_rtl(sub_rtl: str, task: Task, workdir: Path | None = None, equiv_bonus: bool = True,
              dense: bool = False) -> R:
    """dense=False: binary pass (VerilogEval semantics). dense=True: pass_frac shaping."""
    own = workdir is None
    tmp = tempfile.TemporaryDirectory(prefix="chiprl_") if own else None
    wd = Path(tmp.name) if tmp else workdir
    try:
        sub = wd / "dut.sv"
        sub.write_text(sub_rtl)
        lr = lint(sub, task.top, wd)
        if lr.infra_error:
            return R(None, "infra", {"lint": lr.to_dict()})
        if not lr.ok:
            return R(0.0, "lint", {"lint": lr.to_dict()})
        sr = simulate(sub, task.ref_rtl, task.testbench, wd)
        if sr.infra_error:
            return R(None, "infra", {"sim": sr.to_dict()})
        if sr.tool.endswith("_compile"):
            return R(0.0, "compile", {"sim": sr.to_dict()})
        exp = reference_samples(task)
        if exp is None:
            return R(None, "infra", {"msg": "reference run failed"})
        if sr.metrics.get("samples") != exp:
            return R(0.0, "samples", {"sim": sr.to_dict(), "expected_samples": exp})
        pf = sr.metrics["pass_frac"]
        r = pf if dense else float(pf == 1.0)
        detail = {"lint": lr.to_dict(), "sim": sr.to_dict()}
        if r == 1.0 and equiv_bonus:
            er = equiv(sub, task.ref_rtl, task.top, wd)
            detail["equiv"] = er.to_dict()
            if er.metrics.get("proven"):
                r += 0.1
        return R(r, "ok", detail)
    finally:
        if tmp:
            tmp.cleanup()
