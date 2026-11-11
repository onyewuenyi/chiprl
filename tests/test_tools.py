"""Tool-wrapper unit tests that do not need the slow path."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from chiprl.tasks.task import load_verilogeval_dir
from chiprl.tools.equiv import equiv_eqy, equiv_yosys
from chiprl.tools.pnr import librelane_config

FIX = Path(__file__).parent / "fixtures/verilogeval"


def test_librelane_config_seq_and_comb():
    seq = librelane_config(load_verilogeval_dir(FIX / "Prob068_countbcd"), "dut.sv", 10.0, 120.0)
    assert seq["CLOCK_PORT"] == "clk" and seq["USE_SLANG"] and seq["DIE_AREA"] == [0, 0, 120.0, 120.0]
    comb = librelane_config(load_verilogeval_dir(FIX / "Prob009_popcount3"), "dut.sv", 10.0, 120.0)
    assert comb["CLOCK_PORT"] is None


@pytest.mark.parametrize("backend", ["yosys_equiv", "eqy"])
def test_equiv_backends_prove_reference(backend):
    if backend == "eqy" and not shutil.which("eqy"):
        pytest.skip("eqy not installed (OSS CAD Suite only)")
    task = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    fn = equiv_eqy if backend == "eqy" else equiv_yosys
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dut.sv"
        p.write_text(task.ref_rtl.replace("RefModule", task.top))
        assert fn(p, task.ref_rtl, task.top, Path(d)).metrics["proven"]
        p.write_text(task.ref_rtl.replace("RefModule", task.top).replace("state <= 1;", "state <= 2;"))
        assert not fn(p, task.ref_rtl, task.top, Path(d)).metrics["proven"]
