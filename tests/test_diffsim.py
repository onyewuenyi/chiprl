"""Differential simulation: references are clean against themselves, wrong designs are refuted with a
cycle, behavior signatures ignore formatting, and it caught a broken upstream testbench."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chiprl.graders.rtl_gen import grade_rtl
from chiprl.tasks.task import QUARANTINE, load_verilogeval_dir
from chiprl.tools import diffsim as ds

FIX = Path(__file__).parent / "fixtures/verilogeval"
BROKEN = Path(__file__).parent / "fixtures/broken_tb/Prob053_m2014_q4d"
TASKS = sorted(p for p in FIX.iterdir() if p.is_dir())


def _ref(task) -> str:
    return task.ref_rtl.replace("RefModule", task.top)


def _diff(src: str, task, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dut.sv"
        p.write_text(src)
        return ds.diffsim(p, task.ref_rtl, task.top, Path(d), **kw)


@pytest.mark.parametrize("d", TASKS, ids=lambda p: p.name)
def test_reference_is_clean_against_itself(d):
    t = load_verilogeval_dir(d)
    m = _diff(_ref(t), t).metrics
    assert m["verdict"] == "clean" and m["mismatches"] == 0 and m["cycles"] == 3000


def test_wrong_design_is_refuted_at_a_cycle():
    t = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    m = _diff(_ref(t).replace("state <= 1;", "state <= 2;"), t).metrics
    assert m["verdict"] == "refuted" and m["mismatches"] > 0 and m["first_mismatch_cycle"] is not None


def test_signature_ignores_formatting_and_separates_behaviors():
    t = load_verilogeval_dir(FIX / "Prob035_count1to10")
    ref = _ref(t)
    reformatted = "// same circuit, different text\n" + ref.replace("q+1", "q + 4'd1").replace("    ", "\t")
    wrong = ref.replace("q == 10", "q == 9")
    s_ref, s_fmt, s_wrong = (_diff(x, t).metrics["signature"] for x in (ref, reformatted, wrong))
    assert s_ref == s_fmt and s_ref != s_wrong
    assert _diff(ref, t, seeds=(7,)).metrics["signature"] != s_ref, "the signature depends on the stimulus"


def test_interface_mismatch_is_refuted():
    t = load_verilogeval_dir(FIX / "Prob009_popcount3")
    bad = "module TopModule(input [2:0] in, output [2:0] out);\n  assign out = in;\nendmodule\n"
    m = _diff(bad, t).metrics
    assert m["verdict"] == "refuted" and m["reason"] == "interface differs"


def test_candidate_yosys_cannot_read_still_runs(monkeypatch):
    """Icarus accepts some designs Yosys will not elaborate; drive them through the golden ports."""
    t = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    real = ds.ports

    def ports_but_not_the_candidate(rtl, top):
        return None if rtl.startswith("// CAND") else real(rtl, top)

    monkeypatch.setattr(ds, "ports", ports_but_not_the_candidate)
    m = _diff("// CAND\n" + _ref(t), t).metrics
    assert m["verdict"] == "clean"


def test_broken_upstream_testbench_is_caught():
    """failure_log #14: Prob053's reference is x after the first edge, so its testbench accepts anything."""
    t = load_verilogeval_dir(BROKEN)
    wrong = _ref(t).replace("in ^ out", "in & out")
    g = grade_rtl(wrong, t)
    assert g.reward == 1.0, "the upstream testbench passes it"
    assert g.detail["equiv"]["metrics"]["verdict"] == "refuted"
    assert g.detail["equiv"]["metrics"]["backend"] == "diffsim"
    assert g.detail["audit"] == "testbench_weaker_than_spec"
    assert "Prob053_m2014_q4d" in QUARANTINE
