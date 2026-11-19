"""Fault injection: broken tools and a contended cache land in the right class, and runs finish.

The rule (AGENTS rule 3): a broken machine or toolchain is `reward=None`, never 0; a broken
submission is 0, never None. Shim executables on PATH stand in for the failures.
"""

from __future__ import annotations

import functools
import os
import shutil
import sqlite3
import stat
import threading
import time
from pathlib import Path

import pytest

import chiprl.graders.rtl_gen as g
from chiprl import cache
from chiprl.graders import service
from chiprl.runtime.executor import imap
from chiprl.tasks.task import load_verilogeval_dir
from chiprl.tools.sim import simulate

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASK = load_verilogeval_dir(FIX / "Prob035_count1to10")
REF = TASK.ref_rtl.replace("RefModule", TASK.top)


def _shim(tmp_path: Path, monkeypatch, name: str, body: str) -> None:
    """Put an executable `name` ahead of the real one on PATH. $REAL is the real tool."""
    real = shutil.which(name)
    d = tmp_path / "shims"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text(f'#!/bin/sh\nREAL="{real}"\n{body}\n')
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def fast_sim_timeout(monkeypatch):
    monkeypatch.setattr(g, "simulate", functools.partial(simulate, timeout_s=3))


def test_missing_simulator_is_infra(tmp_path, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir()
    for tool in ("verilator", "yosys"):  # everything lint needs, nothing sim needs
        (d / tool).symlink_to(shutil.which(tool))
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}/usr/bin{os.pathsep}/bin")
    r = g.grade_rtl(REF, TASK)
    assert r.reward is None and r.stage == "infra", r


def test_hanging_simulator_is_infra_not_zero(tmp_path, monkeypatch, fast_sim_timeout):
    """The toolchain hangs on every design, the reference included: that is the machine, not the model."""
    _shim(tmp_path, monkeypatch, "vvp", "sleep 600")
    t0 = time.time()
    r = g.grade_rtl(REF, TASK)
    assert r.reward is None and r.stage == "infra", r
    assert time.time() - t0 < 30


def test_killed_simulator_is_infra_not_zero(tmp_path, monkeypatch):
    _shim(tmp_path, monkeypatch, "vvp", "kill -9 $$")
    r = g.grade_rtl(REF, TASK)
    assert r.reward is None and r.stage == "infra", r


def test_slow_but_working_tool_grades_normally(tmp_path, monkeypatch):
    _shim(tmp_path, monkeypatch, "vvp", 'sleep 1; exec "$REAL" "$@"')
    r = g.grade_rtl(REF, TASK, equiv_bonus=False)
    assert r.reward == 1.0 and r.stage == "ok", r


def test_run_finishes_with_a_hanging_toolchain(tmp_path, monkeypatch, fast_sim_timeout):
    """Every job comes back (as infra, after one retry), none is lost, nothing hangs the run."""
    _shim(tmp_path, monkeypatch, "vvp", "sleep 600")
    tasks = [load_verilogeval_dir(p) for p in sorted(FIX.iterdir()) if p.is_dir()]

    def job(t):
        return g.grade_rtl(t.ref_rtl.replace("RefModule", t.top), t, equiv_bonus=False)

    t0 = time.time()
    out = list(imap(job, tasks, backend="local", workers=4, retries=1, is_infra=lambda r: r.reward is None))
    assert len(out) == len(tasks) and all(
        o.ok and o.value is not None and o.value.reward is None for o in out
    )
    assert all(o.attempts == 2 for o in out)
    assert time.time() - t0 < 120


def test_write_locked_cache_still_grades(monkeypatch):
    """Another process holds the write lock: grading proceeds and returns the right answer."""
    monkeypatch.setattr(cache, "BUSY_TIMEOUT_S", 0.3)
    cache.put("warm-up", 1)
    holder = sqlite3.connect(cache.DB, isolation_level=None)
    holder.execute("begin exclusive")
    try:
        result: dict = {}
        th = threading.Thread(target=lambda: result.setdefault("r", service.grade(REF, TASK)))
        th.start()
        th.join(timeout=60)
        assert not th.is_alive()
        r = result["r"]
        assert r.reward == 1.1 and r.stage == "ok" and r.detail["cache_hit"] is False
    finally:
        holder.execute("rollback")
        holder.close()
