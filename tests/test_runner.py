"""Subprocess runner: limits hold, and a timeout leaves nothing running behind it.

At thousands of jobs a leaked process per timeout exhausts the machine, so these are correctness
tests for the grader service, not hygiene.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from chiprl.tools._run import run, tool_versions


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, within_s: float = 3.0) -> bool:
    end = time.time() + within_s
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_timeout_kills_grandchildren(tmp_path: Path):
    pidfile = tmp_path / "grandchild.pid"
    script = f"sleep 60 & echo $! > {pidfile}; wait"
    rc, out, dt = run(["sh", "-c", script], tmp_path, "t.log", timeout_s=1)
    assert rc is None and "<TIMEOUT>" in out and dt < 10
    pid = int(pidfile.read_text())
    assert _wait_dead(pid), f"grandchild {pid} outlived the timeout"


def test_timeout_sends_term_first(tmp_path: Path):
    """A child that cleans up on SIGTERM (the docker client does) gets the chance to."""
    marker = tmp_path / "terminated"
    script = f"trap 'touch {marker}; exit 0' TERM; sleep 60 & wait"
    rc, _, _ = run(["sh", "-c", script], tmp_path, "t.log", timeout_s=1)
    assert rc is None
    end = time.time() + 3
    while not marker.exists() and time.time() < end:
        time.sleep(0.05)
    assert marker.exists(), "child never saw SIGTERM"


def test_cpu_limit_is_enforced(tmp_path: Path):
    rc, _, dt = run([sys.executable, "-c", "while True: pass"], tmp_path, "t.log", timeout_s=30, cpu_s=1)
    assert rc is not None and rc < 0, f"busy loop should die by signal, rc={rc}"
    assert dt < 15


def test_normal_command_and_exit_code(tmp_path: Path):
    rc, out, _ = run(["sh", "-c", "echo hi; echo err >&2; exit 3"], tmp_path, "t.log")
    assert rc == 3 and "hi" in out and "err" in out
    assert (tmp_path / "t.log").read_text() == out


def test_many_threads_do_not_deadlock(tmp_path: Path):
    """preexec_fn is documented as unsafe with threads; the runner must not depend on it."""

    def job(i: int) -> int | None:
        d = tmp_path / str(i)
        d.mkdir()
        return run(["sh", "-c", "exit 0"], d, "t.log", timeout_s=20)[0]

    with ThreadPoolExecutor(32) as pool:
        assert list(pool.map(job, range(200))) == [0] * 200


def test_tool_versions_is_computed_once(monkeypatch):
    import chiprl.tools._run as r

    first = tool_versions()
    monkeypatch.setattr(r, "which", lambda name: (_ for _ in ()).throw(AssertionError("recomputed")))
    assert tool_versions() == first
    tool_versions()["yosys"] = "mutated"
    assert tool_versions() == first, "callers must not be able to mutate the cached value"
