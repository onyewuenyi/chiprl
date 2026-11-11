"""RTL-generation environment: the tool surface a policy (Claude, or the RL model later) sees.

Every tool returns compact JSON. The hidden testbench and reference never leave this module:
`simulate` reports only mismatch counts and the testbench's generic hints, and is rate-limited so an
agent cannot binary-search the hidden stimulus. Final reward comes only from `submit`, which runs the
full gated grader in a fresh directory.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

from ..graders.rtl_gen import grade_rtl
from ..tasks.task import Task
from ..tools.lint import lint
from ..tools.sim import simulate
from ..tools.synth import synth

MAX_DIAG = 12


def _code_schema(desc: str) -> dict:
    return {
        "type": "object",
        "properties": {"code": {"type": "string", "description": desc}},
        "required": ["code"],
        "additionalProperties": False,
    }


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "lint",
        "description": "Static checks (policy + Verilator -Wall) on a complete TopModule. Returns errors/warnings.",
        "input_schema": _code_schema("Complete SystemVerilog source defining module TopModule."),
    },
    {
        "name": "simulate",
        "description": (
            "Run the hidden testbench against TopModule. Returns pass/fail, mismatch count and generic "
            "hints only. Limited number of calls per task."
        ),
        "input_schema": _code_schema("Complete SystemVerilog source defining module TopModule."),
    },
    {
        "name": "synth_report",
        "description": "Synthesize to Sky130 HD with Yosys/ABC. Returns area (um^2), cell count, flops, logic depth.",
        "input_schema": _code_schema("Complete SystemVerilog source defining module TopModule."),
    },
    {
        "name": "submit",
        "description": "Submit the final TopModule for grading. Ends the episode. Call exactly once.",
        "input_schema": _code_schema("Final complete SystemVerilog source defining module TopModule."),
    },
]


@dataclasses.dataclass
class RTLGenEnv:
    task: Task
    max_sim_calls: int = 5
    sim_calls: int = 0
    submitted: dict | None = None

    def _wd(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(prefix="chiprl_env_")

    def call(self, name: str, args: dict) -> dict:
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"error": "missing 'code' string"}
        if len(code) > 200_000:
            return {"error": "code too large"}
        fn = {
            "lint": self.lint,
            "simulate": self.simulate,
            "synth_report": self.synth_report,
            "submit": self.submit,
        }.get(name)
        return fn(code) if fn else {"error": f"unknown tool {name}"}

    def lint(self, code: str) -> dict:
        with self._wd() as d:
            p = Path(d) / "dut.sv"
            p.write_text(code)
            r = lint(p, self.task.top, Path(d))
        return {"ok": r.ok, "infra_error": r.infra_error, "diagnostics": r.diagnostics[:MAX_DIAG]}

    def simulate(self, code: str) -> dict:
        if self.sim_calls >= self.max_sim_calls:
            return {"error": f"simulate budget exhausted ({self.max_sim_calls} calls)"}
        self.sim_calls += 1
        with self._wd() as d:
            p = Path(d) / "dut.sv"
            p.write_text(code)
            lr = lint(p, self.task.top, Path(d))
            if not lr.ok:
                return {"ok": False, "stage": "lint", "diagnostics": lr.diagnostics[:MAX_DIAG]}
            r = simulate(p, self.task.ref_rtl, self.task.testbench, Path(d))
        m = r.metrics
        return {
            "ok": r.ok,
            "stage": "compile" if r.tool.endswith("_compile") else "sim",
            "infra_error": r.infra_error,
            "mismatches": m.get("mismatches"),
            "samples": m.get("samples"),
            "hints": [x.get("hint") or x.get("msg") for x in r.diagnostics][:MAX_DIAG],
            "sim_calls_left": self.max_sim_calls - self.sim_calls,
        }

    def synth_report(self, code: str) -> dict:
        with self._wd() as d:
            p = Path(d) / "dut.sv"
            p.write_text(code)
            r = synth(p, self.task.top, Path(d))
        if not r.ok:
            return {"ok": False, "diagnostics": r.diagnostics[:MAX_DIAG]}
        keep = ("area_um2", "cells", "flops", "logic_depth")
        return {"ok": True, **{k: r.metrics.get(k) for k in keep}}

    def submit(self, code: str) -> dict:
        if self.submitted is not None:
            return {"error": "already submitted"}
        g = grade_rtl(code, self.task)
        self.submitted = {"code": code, "reward": g.reward, "stage": g.stage}
        return {"accepted": True}  # the reward is not revealed to the policy
