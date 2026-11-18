"""Pipeline gating: PPA numbers exist only for a design that passed every gate before them.

Also: infra results are flagged and never cached (AGENTS rules 3 and 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import chiprl.pipeline as pl
from chiprl import cache
from chiprl.tasks.task import load_verilogeval_dir
from chiprl.tools._run import ToolResult

FIX = Path(__file__).parent / "fixtures/verilogeval"

# hack_log #6. `live` starts at 1 and can never clear, so RTL simulation passes. A standard-cell flop
# has no initial value: at gate level `live` powers up x and the output is x for the whole run.
INITIAL_VALUE_HACK = """module TopModule (
  input clk,
  input reset,
  output reg [3:0] q
);
  reg [3:0] cnt;
  reg live = 1'b1;
  always @(posedge clk) begin
    if (reset || cnt == 10) cnt <= 1;
    else cnt <= cnt + 1;
    live <= live & ~(cnt === 4'd15);
  end
  always @(*) q = live ? cnt : 4'd0;
endmodule
"""


@pytest.fixture
def task():
    return load_verilogeval_dir(FIX / "Prob035_count1to10")


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB", tmp_path / "results.sqlite")
    return tmp_path


def _ref(task) -> str:
    return task.ref_rtl.replace("RefModule", task.top)


def test_reference_passes_every_gate(task, tmp_path):
    out = pl.run_task(task, _ref(task), tmp_path, power_backend="proxy", use_cache=False)
    assert out["gates"] == {"grade": True, "synth": True, "glsim": True, "power": True}, out["gates"]
    assert out["infra_error"] is False
    assert out["ppa"]["valid"] and out["ppa"]["power_w"] > 0 and out["ppa"]["post_synth_sim_pass"]


def test_hack_initial_value_gets_no_power(task, tmp_path):
    out = pl.run_task(task, INITIAL_VALUE_HACK, tmp_path, power_backend="proxy", use_cache=False)
    assert out["grade"]["reward"] == 1.0, "RTL sim passes; the equivalence bonus must be withheld"
    assert out["gates"] == {"grade": True, "synth": True, "glsim": False}, out["gates"]
    assert "power" not in out and out["ppa"]["power_w"] is None and out["ppa"]["valid"] is False
    assert out["infra_error"] is False, "a gate-level mismatch is the design's fault, not infra"


def test_wrong_answer_gets_no_ppa_and_is_cached(task, tmp_path, tmp_cache):
    wrong = _ref(task).replace("q <= 1;", "q <= 2;")
    out = pl.run_task(task, wrong, tmp_path, power_backend="proxy")
    assert out["gates"] == {"grade": False} and out["ppa"] is None and out["infra_error"] is False
    assert out["grade"]["stage"] == "sim" and "cache_hit" not in out
    again = pl.run_task(task, wrong, tmp_path, power_backend="proxy")
    assert again.get("cache_hit") is True and again["grade"]["reward"] == 0.0


def test_infra_result_is_flagged_and_never_cached(task, tmp_path, tmp_cache, monkeypatch):
    monkeypatch.setattr(pl, "synth", lambda *a, **k: ToolResult("synth", ok=False, infra_error=True))
    out = pl.run_task(task, _ref(task), tmp_path, power_backend="proxy")
    assert out["infra_error"] is True and out["ppa"] is None and "elapsed_s" in out
    monkeypatch.undo()
    monkeypatch.setattr(cache, "DB", tmp_cache / "results.sqlite")
    healthy = pl.run_task(task, _ref(task), tmp_path, power_backend="proxy")
    assert "cache_hit" not in healthy and healthy["ppa"]["valid"], (
        "the infra result must not have been stored"
    )


def test_synth_timeout_is_blamed_by_reference_control(task, tmp_path, monkeypatch):
    real = pl.synth

    def timeout_on_marker(sub_path, *a, **k):
        if "CHIPRL_SLOW" in Path(sub_path).read_text():
            return ToolResult("synth", ok=False, timeout=True)
        return real(sub_path, *a, **k)

    monkeypatch.setattr(pl, "synth", timeout_on_marker)
    out = pl.run_task(
        task, _ref(task) + "\n// CHIPRL_SLOW\n", tmp_path, power_backend="proxy", use_cache=False
    )
    assert out["gates"]["synth"] is False and out["infra_error"] is False, (
        "reference synthesizes: submission's fault"
    )

    monkeypatch.setattr(pl, "synth", lambda *a, **k: ToolResult("synth", ok=False, timeout=True))
    out = pl.run_task(task, _ref(task), tmp_path, power_backend="proxy", use_cache=False)
    assert out["infra_error"] is True, "reference times out too: infra"
