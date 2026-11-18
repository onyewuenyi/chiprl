"""Equivalence gate soundness. The rule under test: nothing the testbench rejects is ever `proven`.

Regression for docs/failure_log.md #1 (eqy's `sat` strategy proved every sequential design).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chiprl.graders.mutate import mutants
from chiprl.graders.rtl_gen import grade_rtl
from chiprl.tasks.task import load_verilogeval_dir
from chiprl.tools import equiv as eq

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASKS = sorted(p for p in FIX.iterdir() if p.is_dir())


def _ref(task) -> str:
    return task.ref_rtl.replace("RefModule", task.top)


def _run(fn, src: str, task):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dut.sv"
        p.write_text(src)
        return fn(p, task.ref_rtl, task.top, Path(d))


@pytest.fixture(params=TASKS, ids=lambda p: p.name)
def task(request):
    return load_verilogeval_dir(request.param)


def test_reference_is_proven(task):
    r = _run(eq.equiv, _ref(task), task)
    assert r.metrics["verdict"] == "proven" and r.metrics["proven"], r
    assert r.metrics["bmc"]["verdict"] == "clean" and r.metrics["bmc"]["live"], r


def test_no_mutant_the_testbench_kills_is_proven(task):
    killed = refuted = 0
    for name, src in list(mutants(_ref(task)))[:8]:
        g = grade_rtl(src, task, equiv_bonus=False)
        if g.reward is None or g.reward >= 1.0:
            continue  # a survivor can be a truly equivalent mutant
        killed += 1
        v = _run(eq.equiv, src, task)
        assert not v.metrics["proven"], (name, v.metrics)
        refuted += v.metrics["verdict"] == "refuted"
    print(f"{task.task_id}: {refuted}/{killed} killed mutants refuted by the bounded check")
    assert killed >= 5 and refuted >= 0.8 * killed


def test_reset_value_and_logic_mutants_are_refuted():
    task = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    for old, new in [("state <= 1;", "state <= 2;"), ("state <= state + 1;", "state <= state;")]:
        src = _ref(task).replace(old, new)
        assert src != _ref(task)
        assert _run(eq.equiv, src, task).metrics["verdict"] == "refuted", (old, new)
        assert _run(eq.equiv_bmc, src, task).metrics["verdict"] == "refuted", (old, new)


def test_zero_equiv_cells_is_not_a_proof():
    degenerate = (
        "Found 0 $equiv cells in equiv:\n  Of those cells 0 are proven and 0 are unproven.\n"
        "  Equivalence successfully proven!\n"
    )
    assert eq.parse_equiv_status(degenerate) == (False, 0, 0)
    real = "Found 2 $equiv cells in equiv:\n  Of those cells 2 are proven and 0 are unproven.\n"
    assert eq.parse_equiv_status(real) == (True, 2, 0)
    assert (
        eq.parse_equiv_status("Found 2 $equiv cells:\n Of those cells 1 are proven and 1 are unproven.")[0]
        is False
    )
    assert eq.parse_equiv_status("") == (False, 0, 0)


def test_prover_that_proves_the_canary_is_ignored(monkeypatch):
    """Simulates the vacuous-proof bug without needing the broken toolchain."""
    task = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    monkeypatch.setattr(eq, "eqy_available", lambda: True)
    monkeypatch.setattr(eq, "_eqy_prove", lambda *a, **k: ("proven", {"rc": 0, "elapsed_s": 0.0}))
    eq.eqy_sound.cache_clear()
    try:
        assert eq.eqy_sound() is False
        r = _run(eq.equiv_eqy, _ref(task), task)
        assert r.metrics["proven"] is False and r.metrics["verdict"] == "unknown"
        # asked for eqy, equiv() falls back to the yosys prover instead of trusting the unsound one
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "dut.sv"
            p.write_text(_ref(task))
            r = eq.equiv(p, task.ref_rtl, task.top, Path(d), prover="eqy")
        assert r.metrics["backend"] == "yosys_equiv" and r.metrics["proven"]
    finally:
        eq.eqy_sound.cache_clear()


def test_vacuous_bounded_check_is_unknown(monkeypatch):
    """If the bounded problem has no trace at all, a 'no counterexample' result proves nothing."""
    task = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    real_run = eq.run

    def run_then_break_liveness(cmd, cwd, log_name, **kw):
        out = real_run(cmd, cwd, log_name, **kw)
        if log_name == "bmc.log":
            (cwd / "bmc_live.txt").write_text("SAT solving finished - no model found.\n")
        return out

    monkeypatch.setattr(eq, "run", run_then_break_liveness)
    r = _run(eq.equiv_bmc, _ref(task), task)
    assert r.metrics["verdict"] == "unknown" and r.metrics["live"] is False


def test_bmc_depth_follows_reference_size():
    assert eq.bmc_depth(0) == 1 and eq.bmc_depth(16) == eq.BMC_DEPTH
    assert eq.bmc_depth(100) == 6 and eq.bmc_depth(256) == 2 and eq.bmc_depth(None) == 4
    counter = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    popcount = load_verilogeval_dir(FIX / "Prob009_popcount3")
    assert eq.state_bits(counter.ref_rtl, counter.top) == 2
    assert eq.state_bits(popcount.ref_rtl, popcount.top) == 0


def test_eqy_prover_is_opt_in_and_agrees():
    task = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
    if not eq.eqy_available():
        pytest.skip("eqy not installed")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "dut.sv"
        p.write_text(_ref(task))
        assert eq.equiv(p, task.ref_rtl, task.top, Path(d)).metrics["backend"] == "yosys_equiv"
        r = eq.equiv(p, task.ref_rtl, task.top, Path(d), prover="eqy")
        assert r.metrics["backend"] == "eqy" and r.metrics["proven"]
