"""Multi-corner timing: physically ordered corners, Fmax per corner, and a missing corner is infra."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chiprl import pipeline
from chiprl.tasks.task import load_verilogeval_dir
from chiprl.tools import timing
from chiprl.tools.synth import synth

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASK = load_verilogeval_dir(FIX / "Prob068_countbcd")


@pytest.fixture(scope="module")
def netlist(tmp_path_factory):
    wd = tmp_path_factory.mktemp("timing")
    p = wd / "dut.sv"
    p.write_text(TASK.ref_rtl.replace("RefModule", TASK.top))
    s = synth(p, TASK.top, wd)
    assert s.ok
    probe = timing.multi_corner(Path(s.metrics["netlist"]), TASK.top, TASK.clock, 10.0, wd, ("tt",))
    if probe.infra_error:
        pytest.skip("OpenSTA not runnable here (scripts/bin/sta needs Docker)")
    if timing.pdk_lib_dir() is None:
        pytest.skip("Sky130 corner libraries not installed (ciel via LibreLane)")
    return Path(s.metrics["netlist"]), wd


def test_corners_are_physically_ordered(netlist):
    nl, wd = netlist
    m = timing.multi_corner(nl, TASK.top, TASK.clock, 10.0, wd).metrics
    c = m["corners"]
    assert c["ss"]["setup_slack_ns"] < c["tt"]["setup_slack_ns"] < c["ff"]["setup_slack_ns"]
    assert c["ff"]["hold_slack_ns"] <= c["tt"]["hold_slack_ns"] <= c["ss"]["hold_slack_ns"]
    assert m["worst_setup_corner"] == "ss" and m["met"] is True
    assert m["min_period_ns"] == pytest.approx(10.0 - c["ss"]["setup_slack_ns"], abs=1e-3)


def test_min_period_predicts_pass_and_fail(netlist):
    nl, wd = netlist
    pmin = timing.multi_corner(nl, TASK.top, TASK.clock, 10.0, wd).metrics["min_period_ns"]
    assert timing.multi_corner(nl, TASK.top, TASK.clock, round(pmin * 1.05, 3), wd).metrics["met"] is True
    tight = timing.multi_corner(nl, TASK.top, TASK.clock, round(pmin * 0.9, 3), wd).metrics
    assert tight["met"] is False and tight["worst_setup_corner"] == "ss"


def test_missing_corner_library_is_infra(monkeypatch):
    monkeypatch.setattr(timing, "corner_lib", lambda c: None if c == "ss" else timing.liberty_path())
    with tempfile.TemporaryDirectory() as d:
        r = timing.multi_corner(Path(d) / "none.v", TASK.top, TASK.clock, 10.0, Path(d))
    assert r.infra_error and not r.ok and "ss" in r.diagnostics[0]["msg"]


def test_pipeline_reports_timing_when_asked(netlist, tmp_path):
    out = pipeline.run_task(TASK, TASK.ref_rtl.replace("RefModule", TASK.top), tmp_path, power_backend="proxy",
                            use_cache=False, corners=timing.DEFAULT_CORNERS)  # fmt: skip
    assert out["ppa"]["timing_met"] is True and out["ppa"]["worst_corner"] == "ss"
    assert out["ppa"]["min_period_ns"] > 0 and set(out["timing"]["metrics"]["corners"]) == {"ss", "tt", "ff"}
    plain = pipeline.run_task(TASK, TASK.ref_rtl.replace("RefModule", TASK.top), tmp_path, power_backend="proxy",
                              use_cache=False)  # fmt: skip
    assert "timing" not in plain and plain["ppa"]["timing_met"] is None
