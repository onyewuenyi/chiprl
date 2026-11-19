"""RTL-generation grader: a sequence of gates. Infra errors return reward None (excluded), never 0.

Stages (exactly one per result):
  lint      policy violation or Verilator error                      reward 0
  compile   the simulator rejects the source                         reward 0
  samples   the run ended with the wrong sample count (early finish) reward 0
  sim       functional mismatch against the hidden testbench         reward 0 (pass_frac if dense)
  timeout   the submission hangs the simulator                       reward 0
  resource  the submission kills or stalls a tool (OOM, compile bomb) reward 0
  ok        passed                                                   reward 1.0, +0.1 if proven
  infra     the machine or toolchain failed                          reward None

`timeout` and `resource` need care. The same symptom, a tool timing out or being killed, comes from
a hostile or broken submission and from an overloaded machine. Scoring it None lets a policy raise
its pass rate by crashing on hard tasks (rows with None are dropped by evals); scoring it 0 turns
real infra failures into wrong answers. So the grader runs a control: the reference design through
the same tools, under the same limits, right then. Reference healthy means the submission is to
blame (0). Reference also failing means infra (None). See docs/hack_log.md #5.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from ..tasks.task import Task
from ..tools._run import ToolResult
from ..tools.equiv import equiv
from ..tools.lint import lint
from ..tools.sim import simulate

# The reference must lint and simulate in under this many seconds for a resource failure to be
# blamed on the submission. Fixture references take about 0.3 s; the limits are 60 to 120 s.
CONTROL_MAX_S = 30.0


@dataclasses.dataclass
class R:
    reward: float | None
    stage: str  # lint | compile | samples | sim | timeout | resource | ok | infra
    detail: dict = dataclasses.field(default_factory=dict)


_REF_SAMPLES: dict[str, int] = {}


def _ref_as_top(task: Task) -> str:
    return task.ref_rtl.replace("RefModule", task.top)


def reference_samples(task: Task) -> int | None:
    """Sample count of the reference-vs-reference run. A DUT that ends sim early is caught here."""
    key = task.content_hash()
    if key not in _REF_SAMPLES:
        with tempfile.TemporaryDirectory(prefix="chiprl_ref_") as d:
            wd = Path(d)
            p = wd / "dut.sv"
            p.write_text(_ref_as_top(task))
            r = simulate(p, task.ref_rtl, task.testbench, wd)
            if not r.ok:
                return None
            _REF_SAMPLES[key] = r.metrics["samples"]
    return _REF_SAMPLES[key]


def reference_is_healthy(task: Task) -> bool:
    """Control run, never cached: does the reference lint and simulate cleanly right now, with margin?"""
    with tempfile.TemporaryDirectory(prefix="chiprl_ctl_") as d:
        wd = Path(d)
        p = wd / "dut.sv"
        p.write_text(_ref_as_top(task))
        lr = lint(p, task.top, wd)
        if not lr.ok or lr.elapsed_s > CONTROL_MAX_S:
            return False
        sr = simulate(p, task.ref_rtl, task.testbench, wd)
        return sr.ok and sr.elapsed_s <= CONTROL_MAX_S


def _resource_failure(tr: ToolResult) -> bool:
    return bool(tr.timeout or "killed_signal" in tr.metrics)


def _blame(task: Task, stage: str, detail: dict) -> R:
    """Settle a resource failure: 0 if the reference is healthy under the same limits, else None."""
    if reference_is_healthy(task):
        return R(0.0, stage, {**detail, "blamed": "submission"})
    return R(None, "infra", {**detail, "msg": "reference control also failed under the same limits"})


def grade_rtl(
    sub_rtl: str, task: Task, workdir: Path | None = None, equiv_bonus: bool = True, dense: bool = False
) -> R:
    """dense=False: binary pass (VerilogEval semantics). dense=True: pass_frac shaping."""
    own = workdir is None
    tmp = tempfile.TemporaryDirectory(prefix="chiprl_") if own else None
    wd = Path(tmp.name) if tmp else workdir
    assert wd is not None
    try:
        sub = wd / "dut.sv"
        sub.write_text(sub_rtl)
        lr = lint(sub, task.top, wd)
        if lr.infra_error:
            if _resource_failure(lr):
                return _blame(task, "resource", {"lint": lr.to_dict()})
            return R(None, "infra", {"lint": lr.to_dict()})
        if not lr.ok:
            return R(0.0, "lint", {"lint": lr.to_dict()})
        sr = simulate(sub, task.ref_rtl, task.testbench, wd)
        if sr.infra_error:
            if _resource_failure(sr):
                return _blame(task, "resource", {"sim": sr.to_dict()})
            return R(None, "infra", {"sim": sr.to_dict()})
        if sr.tool.endswith("_compile"):
            return R(0.0, "compile", {"sim": sr.to_dict()})
        if sr.timeout:
            return _blame(task, "timeout", {"sim": sr.to_dict()})
        exp = reference_samples(task)
        if exp is None:
            return R(None, "infra", {"msg": "reference run failed"})
        if sr.metrics.get("samples") != exp:
            return R(0.0, "samples", {"sim": sr.to_dict(), "expected_samples": exp})
        pf = sr.metrics["pass_frac"]
        detail = {"lint": lr.to_dict(), "sim": sr.to_dict()}
        if pf != 1.0:
            return R(pf if dense else 0.0, "sim", detail)
        r = 1.0
        if equiv_bonus:
            er = equiv(sub, task.ref_rtl, task.top, wd)
            detail["equiv"] = er.to_dict()
            if er.metrics.get("proven"):
                r += 0.1
            elif er.metrics.get("verdict") == "refuted":
                # Passes the hidden testbench, differs from the reference on some input sequence: the
                # testbench is weaker than the spec. The score keeps VerilogEval semantics; flag it.
                detail["audit"] = "testbench_weaker_than_spec"
        return R(r, "ok", detail)
    finally:
        if tmp:
            tmp.cleanup()
