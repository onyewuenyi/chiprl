"""Cache honesty (AGENTS rule 5): the key moves with every input that can change a result, infra is
never stored, concurrent graders share one computation, and many writers do not lock each other out.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chiprl import cache
from chiprl.graders import service
from chiprl.graders.rtl_gen import R
from chiprl.tasks.task import load_verilogeval_dir

FIX = Path(__file__).parent / "fixtures/verilogeval"
TASK = load_verilogeval_dir(FIX / "Prob075_counter_2bc")
REF = TASK.ref_rtl.replace("RefModule", TASK.top)


def test_key_moves_with_each_input(monkeypatch):
    base = cache.key("t", "rtl")
    assert cache.key("t", "rtl") == base
    assert cache.key("t", "rtl2") != base

    monkeypatch.setenv("CHIPRL_SIM_BACKEND", "verilator")
    assert cache.key("t", "rtl") != base, "sim backend must be in the key"
    monkeypatch.delenv("CHIPRL_SIM_BACKEND")
    monkeypatch.setenv("CHIPRL_NO_SLANG", "1")
    assert cache.key("t", "rtl") != base, "frontend switch must be in the key"
    monkeypatch.delenv("CHIPRL_NO_SLANG")
    assert cache.key("t", "rtl") == base

    real = cache.environment()
    monkeypatch.setattr(cache, "environment", lambda: {**real, "code": "edited"})
    assert cache.key("t", "rtl") != base, "grader code must be in the key"
    monkeypatch.setattr(cache, "environment", lambda: {**real, "tools": {**real["tools"], "eqy": "EQY v9"}})
    assert cache.key("t", "rtl") != base, "eqy version must be in the key"


def test_code_fingerprint_covers_the_grading_path():
    files = {str(p) for p in cache._CODE_PATHS}
    assert any(f.endswith("graders") for f in files) and any(f.endswith("tools") for f in files)
    assert any(f.endswith("pipeline.py") for f in files)
    env = cache.environment()
    assert {"eqy", "sby", "yosys"} <= set(env["tools"]) and len(env["code"]) == 16


def test_grade_hits_on_second_call_and_matches_uncached():
    first = service.grade(REF, TASK)
    second = service.grade(REF, TASK)
    assert first.detail["cache_hit"] is False and second.detail["cache_hit"] is True
    assert (first.reward, first.stage) == (second.reward, second.stage) == (1.1, "ok")


def test_infra_is_never_stored(monkeypatch):
    calls = []

    def infra(*a, **k):
        calls.append(1)
        return R(None, "infra", {})

    monkeypatch.setattr(service, "grade_rtl", infra)
    assert service.grade(REF, TASK).reward is None
    assert service.grade(REF, TASK).detail["cache_hit"] is False
    assert len(calls) == 2


def test_equiv_timeout_is_not_stored(monkeypatch):
    calls = []

    def slow_equiv(*a, **k):
        calls.append(1)
        return R(1.0, "ok", {"equiv": {"timeout": True, "metrics": {"proven": False}}})

    monkeypatch.setattr(service, "grade_rtl", slow_equiv)
    service.grade(REF, TASK)
    service.grade(REF, TASK)
    assert len(calls) == 2


def test_single_flight():
    n = 0
    lock = threading.Lock()

    def compute():
        nonlocal n
        with lock:
            n += 1
        time.sleep(0.2)
        return {"v": 1}

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(lambda _: cache.get_or_compute("same-key", compute), range(8)))
    assert n == 1
    assert [v for v, _ in out] == [{"v": 1}] * 8 and sum(not hit for _, hit in out) == 1


def test_single_flight_propagates_errors_and_recovers():
    def boom():
        time.sleep(0.1)
        raise RuntimeError("tool crashed")

    with ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(cache.get_or_compute, "k-err", boom) for _ in range(4)]
    for f in futs:
        with pytest.raises(RuntimeError):
            f.result()
    assert cache.get_or_compute("k-err", lambda: {"ok": 1}) == ({"ok": 1}, False)


def test_concurrent_writers_do_not_lock():
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda i: cache.put(f"k{i}", {"i": i}), range(500)))
    assert all(cache.get(f"k{i}") == {"i": i} for i in range(0, 500, 37))


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CHIPRL_NO_CACHE", "1")
    assert service.grade(REF, TASK).detail["cache_hit"] is False
    assert service.grade(REF, TASK).detail["cache_hit"] is False


def test_fresh_db_survives_a_concurrent_first_touch(tmp_path, monkeypatch):
    """Regression: 16 threads opening a new file at once raised "database is locked"."""
    monkeypatch.setattr(cache, "DB", tmp_path / "fresh.sqlite")
    with ThreadPoolExecutor(16) as pool:
        out = list(pool.map(lambda i: cache.get_or_compute(f"k{i}", lambda i=i: {"i": i}), range(64)))
    assert [v for v, _ in out] == [{"i": i} for i in range(64)]


def test_broken_cache_degrades_to_compute(tmp_path, monkeypatch):
    import sqlite3

    def broken(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "get", broken)
    monkeypatch.setattr(cache, "put", broken)
    r = service.grade(REF, TASK)
    assert r.reward == 1.1 and r.detail["cache_hit"] is False
